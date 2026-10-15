"""Device admission and provenance. Import this before backend kernels."""

import hashlib
import json
import os
import platform
import re
import subprocess
from functools import cache
from pathlib import Path


class GPUOccupiedError(RuntimeError):
    """A selected GPU has another process that prevents exclusive measurement."""


def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=30).strip()


@cache
def rocr_device(physical):
    """Map an AMD SMI device to its HSA UUID; their ordinal orders can differ."""
    env = os.environ.copy()
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        env.pop(name, None)
    entries = json.loads(
        subprocess.check_output(["/opt/rocm/bin/amd-smi", "list", "--json"], env=env, text=True, timeout=30))
    entry = next((row for row in entries if int(row["gpu"]) == physical), None)
    if entry is None:
        raise RuntimeError(f"AMD SMI device {physical} is unavailable")
    info = subprocess.check_output(["/opt/rocm/bin/rocminfo"], env=env, text=True, timeout=30)
    matches = []
    for agent in re.split(r"(?m)^Agent\s+\d+\s*$", info):
        node = re.search(r"(?m)^\s*Node:\s*(\d+)\s*$", agent)
        uuid = re.search(r"(?m)^\s*Uuid:\s*(GPU-[0-9a-fA-F]+)\s*$", agent)
        if node and uuid and int(node[1]) == int(entry["node_id"]):
            matches.append(uuid[1])
    if len(matches) != 1:
        raise RuntimeError(f"cannot map AMD SMI device {physical} to one HSA GPU UUID")
    return matches[0]


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


def configure_cpu_threads():
    """Bound CPU preparation threads consistently across probes, workers, and reproducers."""
    import torch

    count = int(os.environ.get("SCOUT_CPU_THREADS", "8"))
    if count < 1:
        raise ValueError("SCOUT_CPU_THREADS must be positive")
    if torch.get_num_threads() != count:
        torch.set_num_threads(count)


def inspect_device():
    import torch
    from triton.runtime import driver

    if not torch.cuda.is_available():
        raise RuntimeError("a GPU runtime is required")
    configure_cpu_threads()
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
        check_pci_device(props, bdf)
    else:
        uuid, bdf = command(["nvidia-smi", "-i",
                             str(physical), "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"]).split(",")
    return {
        "vendor": "amd" if target.backend == "hip" else "nvidia", "arch": str(target.arch), "name": props.name,
        "total_memory": props.total_memory, "host": platform.node(), "physical_gpu": physical, "uuid": uuid.strip(),
        "bdf": bdf.strip(), "kfd_id": entry["kfd_id"] if target.backend == "hip" else None
    }


def check_pci_device(properties, expected_bdf):
    domain, bus, device = expected_bdf.split(":")
    expected = (int(domain, 16), int(bus, 16), int(device.split(".")[0], 16))
    actual = tuple(getattr(properties, name, None) for name in ("pci_domain_id", "pci_bus_id", "pci_device_id"))
    if actual != expected:
        raise RuntimeError(f"Torch selected PCI {actual}, expected {expected_bdf}; device visibility is inconsistent")


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


def assert_idle(device, *, allowed_pids=()):
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
    foreign = [pid for pid in pids if pid != os.getpid() and pid not in allowed_pids]
    if foreign:
        raise GPUOccupiedError(f"selected GPU has external processes: {foreign}")


def compiler_source_identity(module_path):
    """Pin the imported compiler, including a separate backend checkout."""
    root = Path(module_path).resolve().parent
    repository = root.parent.parent
    if root == repository / "python/triton" and (repository / ".git").exists():
        return source_identity(repository)
    return package_identity("triton")


def runtime_identity():
    import torch
    import triton
    import triton._C.libtriton as native
    import triton.testing

    native_path = Path(native.__file__)
    with native_path.open("rb") as source:
        compiler_hash = hashlib.file_digest(source, "sha256").hexdigest()
    identity = {
        "python": platform.python_version(), "torch": torch.__version__, "torch_git": torch.version.git_version,
        "torch_num_threads": torch.get_num_threads(), "hip": torch.version.hip, "cuda": torch.version.cuda, "triton":
        triton.__version__, "compiler_sha256": compiler_hash, "timing_sha256":
        hashlib.sha256(Path(triton.testing.__file__).read_bytes()).hexdigest(), "performance_environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("AITER_", "FLYDSL_")) or key in {
                "TORCH_BLAS_PREFER_HIPBLASLT", "PYTORCH_TUNABLEOP_ENABLED", "PYTORCH_TUNABLEOP_TUNING",
                "NVIDIA_TF32_OVERRIDE", "HSA_OVERRIDE_GFX_VERSION", "TRITON_ALWAYS_COMPILE", "TRITON_DEFAULT_FP_FUSION",
                "TRITON_ENABLE_C_CACHE", "ENABLE_CK", "CK_DIR", "GPU_ARCHS", "ROCM_PATH", "HIPCC_FLAGS_APPEND"
            }
        }, "configuration_sha256": {
            key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
            for key, value in os.environ.items()
            if key.startswith("AITER_CONFIG_") and Path(value).is_file()
        }, "triton_module": str(Path(triton.__file__).resolve()), "torch_module": str(Path(torch.__file__).resolve()),
        "source": compiler_source_identity(triton.__file__)
    }
    return identity


def git_source_files(repository):
    """Include tuning data and submodule sources, but exclude ignored JIT output."""
    names = command(["git", "ls-files", "--cached", "--others", "--exclude-standard"], repository).splitlines()
    for name in sorted(set(names)):
        path = repository / name
        if path.is_file():
            yield path
        elif path.is_dir() and (path / ".git").exists():
            yield from git_source_files(path)


def package_identity(name):
    """Do not infer a source commit from a package version."""
    import importlib.metadata
    import importlib.util

    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"status": "unavailable"}
    if spec.origin is None:
        return {"status": "unavailable", "error": f"{name} resolves to a namespace; add its checkout to PYTHONPATH"}
    root = Path(spec.origin).resolve().parent
    try:
        repository = Path(command(["git", "-C", str(root), "rev-parse", "--show-toplevel"]))
        # A wheel inside another repository is not that repository's source package.
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--",
             str(Path(spec.origin).resolve().relative_to(repository))], cwd=repository, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        sha = command(["git", "-C", str(root), "rev-parse", "HEAD"])
        dirty = bool(command(["git", "-C", str(root), "status", "--porcelain"]))
    except subprocess.CalledProcessError:
        repository, sha, dirty = None, None, None
    try:
        version = importlib.metadata.version("amd-aiter" if name == "aiter" else name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    dependencies = {}
    if name == "aiter":
        for dependency in ("flydsl", "numpy", "pandas", "einops", "ninja", "pybind11"):
            try:
                dependencies[dependency] = importlib.metadata.version(dependency)
            except importlib.metadata.PackageNotFoundError:
                dependencies[dependency] = None
    hasher = hashlib.sha256()
    files = git_source_files(repository) if repository else sorted(
        p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and "build" not in p.parts)
    for path in files:
        hasher.update(str(path.relative_to(repository or root)).encode() + b"\0")
        hasher.update(path.read_bytes())
    return {
        "status": "available", "git_sha": sha, "version": version, "dirty": dirty, "source_sha256": hasher.hexdigest(),
        "module": str(root), "repository": str(repository) if repository else None, "dependencies": dependencies
    }
