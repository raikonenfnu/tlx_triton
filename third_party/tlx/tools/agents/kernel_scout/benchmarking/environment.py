"""Device admission and provenance. Import this before backend kernels."""

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=30).strip()


def source_identity(repository):
    repository = Path(repository)
    # The scout branch can change without changing the measured kernel sources.
    paths = [
        "python/triton", "third_party/tlx/ops", "third_party/tlx/tutorials", "third_party/tlx/language",
        "third_party/amd/backend"
    ]
    git_sha = command(["git", "rev-parse", "HEAD"], repository)
    files = command(["git", "ls-files", "--", *paths], repository).splitlines()
    hasher = hashlib.sha256()
    for name in files:
        path = repository / name
        if path.is_file():
            hasher.update(name.encode())
            hasher.update(path.read_bytes())
    return {
        "git_sha": git_sha, "kernel_tree_sha256": hasher.hexdigest(), "dirty":
        bool(command(["git", "status", "--porcelain", "--", *paths], repository))
    }


def inspect_device():
    import torch
    from triton.runtime import driver

    if not torch.cuda.is_available():
        raise RuntimeError("a GPU runtime is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    props = torch.cuda.get_device_properties(0)
    target = driver.active.get_current_target()
    physical = int(os.environ.get("SCOUT_PHYSICAL_GPU", "0"))
    if target.backend == "hip":
        entries = json.loads(command(["/opt/rocm/bin/amd-smi", "list", "--json"]))
        entry = next(row for row in entries if int(row["gpu"]) == physical)
        uuid, bdf = entry["uuid"], entry["bdf"]
    else:
        uuid, bdf = command(["nvidia-smi", "-i",
                             str(physical), "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"]).split(",")
    return {
        "vendor": "amd" if target.backend == "hip" else "nvidia", "arch": str(target.arch), "name": props.name,
        "total_memory": props.total_memory, "host": platform.node(), "physical_gpu": physical, "uuid": uuid.strip(),
        "bdf": bdf.strip(), "kfd_id": entry["kfd_id"] if target.backend == "hip" else None
    }


def conditions(device):
    if device["vendor"] == "amd":
        return json.loads(
            command([
                "/opt/rocm/bin/amd-smi", "metric", "-g",
                str(device["physical_gpu"]), "--clock", "--power", "--temperature", "--json"
            ]))
    return {
        "nvidia_smi":
        command([
            "nvidia-smi", "-i",
            str(device["physical_gpu"]), "--query-gpu=clocks.sm,clocks.mem,power.limit,temperature.gpu",
            "--format=csv,noheader"
        ])
    }


def kfd_owners(process_root, gpu_id):
    owners = set()
    for process in process_root.iterdir():
        if not process.name.isdigit():
            continue
        files = [*(process / "queues").glob("*/gpuid"), *process.glob("vram_*")]
        for path in files:
            try:
                value = int(path.read_text().strip())
            except FileNotFoundError:
                continue  # Process/queue exited during observation.
            owner_gpu = int(path.name.removeprefix("vram_")) if path.name.startswith("vram_") else value
            if owner_gpu == gpu_id and (path.name == "gpuid" or value > 0):
                owners.add(int(process.name))
    return owners


def assert_idle(device):
    """Refuse an occupied device; do not change another job's clocks or processes."""
    physical = device["physical_gpu"]
    if device["vendor"] == "amd":
        entries = json.loads(command(["/opt/rocm/bin/amd-smi", "process", "--json"]))
        matches = [row for row in entries if int(row["gpu"]) == physical]
        if len(matches) != 1:
            raise RuntimeError("cannot inspect selected GPU ownership")
        pids = []
        for row in matches[0]["process_list"]:
            if row.get("process_info") == "No running processes detected":
                continue
            info = row.get("process_info", row)
            if not isinstance(info, dict) or "pid" not in info:
                raise RuntimeError(f"unrecognized AMD process record: {row}")
            pids.append(int(info["pid"]))
        # KFD also exposes owners when the management API reports no processes.
        kfd = Path("/sys/class/kfd/kfd/proc")
        if not kfd.is_dir():
            raise RuntimeError("KFD process ownership is unavailable")
        pids.extend(kfd_owners(kfd, device["kfd_id"]))
    else:
        uuid = command(["nvidia-smi", "-i", str(physical), "--query-gpu=uuid", "--format=csv,noheader"])
        rows = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]).splitlines()
        pids = [int(row.split(",")[1]) for row in rows if row.split(",")[0].strip() == uuid]
    foreign = [pid for pid in pids if pid != os.getpid()]
    if foreign:
        raise RuntimeError(f"selected GPU has external processes: {foreign}")


def runtime_identity(repository):
    import torch
    import triton
    import triton._C.libtriton as native
    import triton.testing

    native_path = Path(native.__file__)
    identity = {
        "python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version, "hip":
        torch.version.hip, "cuda": torch.version.cuda, "triton": triton.__version__, "compiler_sha256":
        hashlib.sha256(native_path.read_bytes()).hexdigest(), "timing_sha256":
        hashlib.sha256(Path(triton.testing.__file__).read_bytes()).hexdigest(), "performance_environment": {
            key: value
            for key, value in os.environ.items()
            if key in {
                "TORCH_BLAS_PREFER_HIPBLASLT", "PYTORCH_TUNABLEOP_ENABLED", "PYTORCH_TUNABLEOP_TUNING",
                "NVIDIA_TF32_OVERRIDE", "HSA_OVERRIDE_GFX_VERSION", "TRITON_ALWAYS_COMPILE", "TRITON_DEFAULT_FP_FUSION",
                "TRITON_ENABLE_C_CACHE"
            }
        }, "triton_module": str(Path(triton.__file__).resolve()), "source": source_identity(repository)
    }
    return identity


def package_identity(name):
    """Do not infer a source commit from a package version."""
    import importlib.metadata
    import importlib.util

    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"status": "unavailable"}
    root = Path(spec.origin).resolve().parent
    try:
        sha = command(["git", "-C", str(root), "rev-parse", "HEAD"])
        dirty = bool(command(["git", "-C", str(root), "status", "--porcelain"]))
    except subprocess.CalledProcessError:
        sha, dirty = None, None
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    hasher = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.suffix in {".py", ".so"}):
        hasher.update(str(path.relative_to(root)).encode())
        hasher.update(path.read_bytes())
    return {
        "status": "available", "git_sha": sha, "version": version, "dirty": dirty, "source_sha256": hasher.hexdigest(),
        "module": str(root)
    }
