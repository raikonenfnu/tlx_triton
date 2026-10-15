"""CPU tests for controller contracts and external publication state transitions."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from appctl import baseline_key, validate_baseline, worker_environment
from coding_agents import query_json
from contracts import applicable, backends, digest, load_target, read_json, select_winner, work, write_json
from discovery import validate_proposals
from publication import publish


@pytest.fixture
def target():
    return load_target(ROOT / "targets/amd/gfx950/varlen_f16_fa.json")


def test_targets_and_vendor_eligibility(target):
    registry = backends()
    for path in (ROOT / "targets").rglob("*.json"):
        load_target(path)
    assert applicable(registry["aiter"], target, {"vendor": "amd", "arch": "gfx950"})
    assert not applicable(registry["aiter"], target, {"vendor": "nvidia", "arch": "90"})
    assert not applicable(registry["aiter"], target, {"vendor": "amd", "arch": "gfx942"})
    assert target["cases"][0]["q_lengths"] == [1000] * 10
    assert len(target["cases"]) == 6


def test_mxfp8_focus_shapes_and_reference_capabilities():
    target = load_target(ROOT / "targets/amd/gfx950/mxfp8_fa.json")
    assert target["operation"] == "flash_attn_mxfp8"
    assert target["correctness"] == {"seeds": [17, 29], "atol": 0.2, "rtol": 0.0, "max_relative_rms_error": 0.1}
    assert {(c["q_lengths"][0], c["causal"])
            for c in target["cases"]} == {(n, causal)
                                          for n in (1024, 2048, 4096, 8192)
                                          for causal in (False, True)}
    for case in target["cases"]:
        assert case["q_lengths"] == case["kv_lengths"] == [case["q_lengths"][0]] * 4
        assert (case["heads"], case["kv_heads"], case["head_dim"], case["dtype"]) == (32, 32, 128, "bfloat16")
        assert case["sm_scale"] == case["qkv_std"] == 0.5
    registry = backends()
    device = {"vendor": "amd", "arch": "gfx950"}
    assert all(applicable(registry[b], target, device) for b in ("tlx_ops", "aiter", "pytorch"))
    assert not applicable(registry["tlx_tutorials"], target, device)


@pytest.mark.parametrize("change", [
    lambda c: c.update(dtype="float16"),
    lambda c: c.update(head_dim=64),
    lambda c: c.update(kv_heads=16),
    lambda c: c.update(q_lengths=[384] * 4, kv_lengths=[384] * 4),
    lambda c: c.update(sm_scale=-0.5),
    lambda c: c.update(qkv_std=float("nan")),
])
def test_mxfp8_contract_rejects_incompatible_shapes(change, tmp_path):
    target = load_target(ROOT / "targets/amd/gfx950/mxfp8_fa.json")
    change(target["cases"][0])
    path = tmp_path / "target.json"
    write_json(path, target)
    with pytest.raises(ValueError):
        load_target(path)


def test_issue_gemm_targets_preserve_operations_layouts_and_dtypes():
    counts = {}
    for filename, operation in [("a16w16_gemm.json", "mm"), ("a16w16_addmm.json", "addmm"), ("a16w16_bmm.json", "bmm")]:
        target = load_target(ROOT / "targets/amd/gfx950" / filename)
        assert target["operation"] == operation
        for case in target["cases"]:
            if "source_issue" not in case:
                continue
            issue = int(case["source_issue"].rsplit("/", 1)[1])
            counts[(issue, operation, case["dtype"])] = counts.get((issue, operation, case["dtype"]), 0) + 1
            if operation == "addmm":
                assert (case["a_layout"], case["b_layout"], case["bias_layout"]) == ("row_major", "column_major",
                                                                                     "vector")
            if operation == "bmm":
                assert (case["batch"], case["m"], case["n"], case["k"], case["shared_a"]) == (3072, 448, 160, 931, True)
            if operation != "bmm":
                assert case["a_std"] == case["k"]**-0.5 and case["b_std"] == case["n"]**-0.5
    assert counts == {
        (issue, operation, dtype): count
        for (issue, operation
             ), count in {(20, "mm"): 8, (20, "addmm"): 3, (21, "addmm"): 7, (21, "bmm"): 1, (22, "mm"): 2}.items()
        for dtype in ("bfloat16", "float16")
    }


def test_gemm_work_counts_shared_storage_and_bias():
    case = {"m": 2, "n": 3, "k": 4, "batch": 5, "shared_a": True}
    assert work(case, "addmm") == (48, 58)
    assert work(case, "bmm") == (240, 196)
    assert work({**case, "shared_a": False}, "bmm") == (240, 260)


@pytest.mark.parametrize("filename,change", [
    ("a16w16_addmm.json", lambda c: c.update(alpha=2)),
    ("a16w16_addmm.json", lambda c: c.update(bias_layout="matrix")),
    ("a16w16_bmm.json", lambda c: c.update(batch=0)),
    ("a16w16_bmm.json", lambda c: c.update(shared_a="true")),
])
def test_matrix_contract_rejects_changed_bias_or_batch_semantics(tmp_path, filename, change):
    target = load_target(ROOT / "targets/amd/gfx950" / filename)
    change(target["cases"][0])
    path = tmp_path / "target.json"
    write_json(path, target)
    with pytest.raises(ValueError):
        load_target(path)


def test_aiter_bmm_does_not_convert_fp16_to_bf16(monkeypatch):
    from backends.seeds import build

    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    with pytest.raises(NotImplementedError, match="BF16 inputs only"):
        build("aiter", "bmm", {"dtype": "float16"}, {})


@pytest.mark.parametrize("operation", ["mm", "addmm", "bmm"])
def test_cpu_inputs_keep_matrix_layout_and_shared_batch(operation):
    torch = pytest.importorskip("torch")
    from benchmarking.inputs import input_digest, make_inputs

    case = {
        "dtype": "bfloat16", "m": 2, "n": 3, "k": 4, "a_layout": "column_major", "b_layout": "row_major", "batch": 3,
        "shared_a": True
    }
    inputs = make_inputs(case, operation, 17, device="cpu")
    again = make_inputs(case, operation, 17, device="cpu")
    assert inputs["a"].dtype == inputs["b"].dtype == torch.bfloat16
    assert inputs["a"].stride(-2) == 1 and inputs["b"].stride(-1) == 1
    if operation == "addmm":
        assert inputs["bias"].shape == (3, )
    if operation == "bmm":
        assert inputs["a"].shape == (3, 2, 4) and inputs["a"].stride(0) == 0
        assert inputs["a"].untyped_storage().nbytes() == 2 * 4 * 2
    assert input_digest(inputs) == input_digest(again)
    if operation == "bmm":
        again["a"][0, 0, 0] += 1
    else:
        again["a"][0, 0] += 1
    assert input_digest(inputs) != input_digest(again)


def test_cpu_oracle_keeps_vector_bias_and_batch_semantics():
    torch = pytest.importorskip("torch")
    from benchmarking.inputs import reference

    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float16)
    bias = torch.tensor([1, -1], dtype=torch.float16)
    assert torch.equal(reference({}, "addmm", {"a": a, "b": b, "bias": bias}), torch.tensor([[20, 21], [44, 49]]))
    batched = {"a": a.expand(2, -1, -1), "b": torch.stack([b, torch.eye(2, dtype=torch.float16)])}
    expected = torch.tensor([[[19, 22], [43, 50]], [[1, 2], [3, 4]]])
    assert torch.equal(reference({}, "bmm", batched), expected)


def test_relative_rms_guard_rejects_zero_outputs_despite_loose_absolute_tolerance():
    torch = pytest.importorskip("torch")
    from benchmarking.inputs import relative_rms_error

    reference = torch.tensor([1e-5, -2e-5])
    zeros = torch.zeros_like(reference)
    torch.testing.assert_close(zeros, reference, atol=0.02, rtol=0.02)
    assert relative_rms_error(zeros, reference) == pytest.approx(1.0)
    assert relative_rms_error(reference, reference) == 0
    assert relative_rms_error(zeros, zeros) == 0
    assert relative_rms_error(reference, zeros) == float("inf")


def test_mxfp8_seed_keeps_quantization_inside_the_measured_call(monkeypatch):
    from backends.seeds import build

    calls = []
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    ops = SimpleNamespace(flash_attn_mxfp8=lambda *a, **kw: calls.append((a, kw)) or "output")
    monkeypatch.setitem(sys.modules, "triton.tlx", SimpleNamespace(ops=ops))
    case = {"head_dim": 128, "causal": True, "sm_scale": 0.5}
    run = build("tlx_ops", "flash_attn_mxfp8", case, {})
    assert not calls
    inputs = {key: key for key in ("q", "k", "v")}
    assert run(inputs) == "output"
    assert calls == [(("q", "k", "v"), {"causal": True, "sm_scale": 0.5, "space": "full"})]


@pytest.mark.parametrize("change", [
    lambda t: t.update(output_policy="preallocated"),
    lambda t: t["timing"].update(rep_ms=float("nan")),
    lambda t: t["correctness"].update(seeds=[1, 1]),
    lambda t: t["cases"][0].update(kv_lengths=[100]),
    lambda t: t["cases"][0].update(kv_heads=3),
    lambda t: t["cases"][0].update(q_lengths=[-1] * 10),
    lambda t: t["cases"][0].update(id="../escape"),
])
def test_invalid_target_rejected(target, change, tmp_path):
    change(target)
    path = tmp_path / "target.json"
    write_json(path, target)
    with pytest.raises(ValueError):
        load_target(path)


def test_work_counts_rectangular_causal_and_gqa():
    case = {"q_lengths": [4, 2], "kv_lengths": [2, 4], "heads": 4, "kv_heads": 2, "head_dim": 128, "causal": True}
    # First sequence: 0+0+1+2 pairs. Second: 3+4 pairs.
    flops, byte_count = work(case, "flash_attn_varlen")
    assert flops == 4 * 4 * 128 * 10
    assert byte_count == 2 * 128 * (2 * 6 * 4 + 2 * 6 * 2)
    assert work({"m": 2, "n": 3, "k": 4}, "mm") == (48, 52)


def measurement(latency, low=None, high=None, status="ok"):
    return {
        "status": status, "latency_ms": latency, "p20_ms": latency if low is None else low, "p80_ms":
        latency if high is None else high
    }


def test_winner_uses_correct_stable_results_and_uncertainty():
    rows = {
        "wrong": measurement(0.1, status="failed"), "noisy": measurement(0.2, status="unstable"), "a": measurement(1),
        "b": measurement(1.2)
    }
    assert select_winner(rows, 0.03) == "a"
    rows["b"] = measurement(1.1, 1.01, 1.2)
    assert select_winner(rows, 0.03) == "a / b (tie)"
    assert select_winner({}, 0.03) == "none"


def test_baseline_key_accepts_tlx_changes_rejects_device_or_protocol(target):
    device = {"vendor": "amd", "arch": "gfx950", "physical_gpu": 0, "host": "machine"}
    runtime = {"torch": "2", "torch_git": "a", "hip": "7", "cuda": None, "python": "3.12", "compiler_sha256": "old"}
    key = baseline_key([target], device, runtime)
    assert baseline_key([target], device, {**runtime, "compiler_sha256": "new"}) == key
    assert baseline_key([target], {**device, "physical_gpu": 1}, runtime) != key
    changed = copy.deepcopy(target)
    changed["timing"]["rep_ms"] += 1
    assert baseline_key([changed], device, runtime) != key


def test_baseline_reproducer_digest_checked(tmp_path):
    path = tmp_path / "request.json"
    write_json(path, {"expected_identity": {"source": "a"}})
    baseline = {
        "campaign":
        "c", "week":
        "2026-10-12", "baseline_key":
        "k", "rows": [{
            "results": {
                "torch":
                {"status": "ok", "reproducer": str(path), "request_sha256": digest(json.loads(path.read_text()))}
            }
        }]
    }
    validate_baseline(baseline, campaign="c", week="2026-10-12", key="k")
    with pytest.raises(ValueError, match="does not match"):
        validate_baseline(baseline, campaign="c", week="2026-10-19", key="k")
    write_json(path, {"expected_identity": {"source": "changed"}})
    with pytest.raises(ValueError, match="changed"):
        validate_baseline(baseline, campaign="c", week="2026-10-12", key="k")


def test_worker_environment_selects_same_physical_gpu(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr("appctl.rocr_device", lambda gpu: "GPU-abcd" if gpu == 3 else None)
    env = worker_environment(3, {"python": "/aiter-venv/bin/python", "pythonpath": "/source:/torch"})
    assert env["ROCR_VISIBLE_DEVICES"] == "GPU-abcd"
    assert env["SCOUT_PHYSICAL_GPU"] == "3"
    assert env["HIP_VISIBLE_DEVICES"] == env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PYTHONPATH"] == "/source:/torch"
    assert env["PATH"] == "/aiter-venv/bin:/usr/bin"
    nvidia = worker_environment(3, {"vendor": "nvidia"})
    assert nvidia["CUDA_VISIBLE_DEVICES"] == "3" and "ROCR_VISIBLE_DEVICES" not in nvidia


@pytest.mark.parametrize("info,expected", [
    ("Agent 1\n Node: 2\n Uuid: GPU-1111\nAgent 2\n Node: 5\n Uuid: GPU-abcd\n", "GPU-abcd"),
    ("Agent 1\n Node: 2\n Uuid: GPU-1111\n", None),
    ("Agent 1\n Node: 5\n Uuid: GPU-abcd\nAgent 2\n Node: 5\n Uuid: GPU-2222\n", None),
])
def test_rocr_device_matches_node_and_rejects_ambiguous_mapping(monkeypatch, info, expected):
    from benchmarking.environment import rocr_device

    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "7")

    def output(args, **kwargs):
        assert not {"ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"} & kwargs["env"].keys()
        return json.dumps([{"gpu": 3, "node_id": 5}]) if "amd-smi" in args[0] else info

    monkeypatch.setattr(subprocess, "check_output", output)
    rocr_device.cache_clear()
    try:
        if expected:
            assert rocr_device(3) == expected
        else:
            with pytest.raises(RuntimeError, match="one HSA GPU UUID"):
                rocr_device(3)
    finally:
        rocr_device.cache_clear()


@pytest.mark.parametrize("backend", [None, "triton", "gluon"])
@pytest.mark.parametrize("operation", ["flash_attn", "flash_attn_varlen"])
def test_aiter_attention_adapters_preserve_contract(monkeypatch, backend, operation):
    from backends.seeds import build

    calls = []

    class Tensor:

        def __init__(self, name):
            self.name = name

        def transpose(self, a, b):
            return Tensor(f"{self.name}.transpose({a},{b})")

        def contiguous(self):
            return Tensor(f"{self.name}.contiguous()")

    def attention(*args, **kwargs):
        calls.append((args, kwargs))
        return Tensor("output")

    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    path = "aiter.ops.triton.attention.mha" if backend else "aiter"
    module = ModuleType(path)
    module.flash_attn_func = module.flash_attn_varlen_func = attention
    monkeypatch.setitem(sys.modules, path, module)
    case = {"head_dim": 128, "causal": True, "q_lengths": [3, 5], "kv_lengths": [4, 7]}
    inputs = {key: Tensor(key) for key in ("q", "k", "v", "cu_q", "cu_k")}
    params = {"backend": backend} if backend else {}
    result = build("aiter", operation, case, params)(inputs)
    args, options = calls[0]
    assert options == {"dropout_p": 0.0, "softmax_scale": 128**-0.5, "causal": True, **params}
    if operation == "flash_attn_varlen":
        assert args == (inputs["q"], inputs["k"], inputs["v"], inputs["cu_q"], inputs["cu_k"], 5, 7)
        assert result.name == "output"
    else:
        assert [arg.name for arg in args] == [f"{key}.transpose(1,2)" for key in ("q", "k", "v")]
        assert result.name == "output.transpose(1,2).contiguous()"


def test_aiter_gemm_adapter_keeps_output_dtype_and_measured_packing(monkeypatch):
    from backends.seeds import build

    calls, packed = [], []
    weight = SimpleNamespace(contiguous=lambda: packed.append("packed") or "weight")
    module = ModuleType("aiter.ops.triton.gemm.basic.gemm_a16w16")
    module.gemm_a16w16 = lambda *a, **kw: calls.append((a, kw)) or "output"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    run = build("aiter", "mm", {}, {"implementation": "triton", "persistent": True})
    assert not packed
    a = SimpleNamespace(dtype="float16")
    assert run({"a": a, "b": SimpleNamespace(T=weight)}) == "output"
    assert packed == ["packed"]
    assert calls == [((a, "weight"), {"dtype": "float16", "persistent": True})]


def test_aiter_public_gemm_records_dispatch_and_preserves_contract(monkeypatch):
    from backends.seeds import build

    calls, packing = [], []
    torch = ModuleType("torch")
    torch.float16 = "float16"
    monkeypatch.setitem(sys.modules, "torch", torch)
    module = ModuleType("aiter.tuned_gemm")
    module.get_GEMM_A16W16_config = lambda **kw: {"libtype": "torch", "kernelName": ""}
    module.gemm_a16w16 = lambda *a, **kw: calls.append((a, kw)) or "output"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    run = build("aiter", "mm", {"dtype": "float16", "m": 2, "n": 3, "k": 4}, {})
    assert not packing
    assert run.scout_metadata["implementation"] == "torch"
    a = SimpleNamespace(dtype="float16")
    weight = SimpleNamespace(contiguous=lambda: packing.append(True) or "weight")
    assert run({"a": a, "b": SimpleNamespace(T=weight)}) == "output"
    assert packing == [True] and calls == [((a, "weight"), {"otype": "float16"})]


@pytest.mark.parametrize("operation", ["mm", "flash_attn", "flash_attn_varlen"])
def test_aiter_defaults_to_one_public_operator(operation):
    from backends.seeds import candidates

    assert candidates("aiter", operation) == [{"id": "public_api", "params": {}}]
    assert len(candidates("aiter", operation, search=True)) == 3


@pytest.mark.parametrize("search", [False, True])
def test_aiter_discovery_needs_explicit_search_flag(tmp_path, monkeypatch, target, search):
    import appctl

    calls = []
    args = SimpleNamespace(provider="claude", aiter_search=search, max_candidates=6, rounds=1, model=None, context=None,
                           agent_timeout=5)

    def evaluate(*args):
        return {"status": "ok", "latency_ms": 1, "candidate": args[3]}

    def discover(*a, **kw):
        calls.append("discovery")
        return {"status": "complete", "summary": "source inspected", "candidates": []}, []

    monkeypatch.setattr(appctl, "evaluate", evaluate)
    monkeypatch.setattr(appctl, "discover", discover)
    result = appctl.scout("aiter", target, target["cases"][0], tmp_path, {}, args)
    assert result["status"] == "ok"
    assert result["evaluated_candidates"] == (3 if search else 1)
    assert calls == (["discovery"] if search else [])


def test_aiter_discovery_uses_installed_checkout(tmp_path, monkeypatch, target):
    import discovery

    checkout = tmp_path / "aiter-source"
    checkout.mkdir()
    research = {"package": {"status": "available", "repository": str(checkout), "git_sha": "aiter-sha"}}

    def query(provider, prompt, **kwargs):
        assert kwargs["workspace"] == checkout
        assert "aiter-sha" in prompt
        assert "Measured feedback" in prompt
        return {"status": "complete", "summary": "source inspected", "candidates": []}

    monkeypatch.setattr(discovery, "query_json", query)
    response, proposals = discovery.discover("claude", "aiter", target, target["cases"][0], [], tmp_path, 2,
                                             research=research)
    assert response["status"] == "complete" and proposals == []


def test_pytorch_discovery_reads_the_installed_package(tmp_path, monkeypatch, target):
    import discovery

    package = tmp_path / "torch"
    package.mkdir()
    research = {"runtime": {"torch_module": str(package / "__init__.py")}}

    def query(provider, prompt, **kwargs):
        assert kwargs["workspace"] == package
        return {"status": "complete", "summary": "Torch source inspected", "candidates": []}

    monkeypatch.setattr(discovery, "query_json", query)
    response, _ = discovery.discover("claude", "pytorch", target, target["cases"][0], [], tmp_path, 1,
                                     research=research)
    assert response["status"] == "complete"


def test_package_identity_tracks_tuning_and_excludes_jit_output(tmp_path, monkeypatch):
    import importlib.metadata

    from benchmarking.environment import package_identity

    repository = tmp_path / "source"
    module = repository / "aiter"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("")
    config = repository / "configs/tuned.csv"
    config.parent.mkdir()
    config.write_text("shape,kernel\n512,first\n")
    (repository / ".gitignore").write_text("aiter/jit/\n")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run([
        "git", "-C",
        str(repository), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "test fixture"
    ], check=True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(module / "__init__.py")))

    def version(name):
        if name == "amd-aiter":
            return "test-version"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    initial = package_identity("aiter")
    assert initial["version"] == "test-version" and initial["repository"] == str(repository)
    assert "flydsl" in initial["dependencies"]
    cache = module / "jit/build"
    cache.mkdir(parents=True)
    (cache / "kernel.so").write_bytes(b"compiled cache")
    assert package_identity("aiter") == initial
    config.write_text("shape,kernel\n512,second\n")
    changed = package_identity("aiter")
    assert changed["source_sha256"] != initial["source_sha256"] and changed["dirty"]


def test_namespace_package_is_unavailable(monkeypatch):
    from benchmarking.environment import package_identity

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(origin=None))
    result = package_identity("aiter")
    assert result["status"] == "unavailable" and "namespace" in result["error"]


def test_compiler_source_uses_its_own_checkout(tmp_path):
    from benchmarking.environment import compiler_source_identity

    repository = tmp_path / "external-triton"
    module = repository / "python/triton/__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("# compiler source\n")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run([
        "git", "-C",
        str(repository), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm",
        "compiler fixture"
    ], check=True)
    sha = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    initial = compiler_source_identity(module)
    assert initial["git_sha"] == sha and not initial["dirty"]
    module.write_text("# changed compiler source\n")
    changed = compiler_source_identity(module)
    assert changed["git_sha"] == sha and changed["dirty"]
    assert changed["kernel_tree_sha256"] != initial["kernel_tree_sha256"]


def test_package_wheel_does_not_borrow_parent_repository(tmp_path, monkeypatch):
    import importlib.metadata

    from benchmarking.environment import package_identity

    (tmp_path / "tracked.py").write_text("")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run([
        "git", "-C",
        str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "unrelated fixture"
    ], check=True)
    wheel = tmp_path / "venv/site-packages/triton/__init__.py"
    wheel.parent.mkdir(parents=True)
    wheel.write_text("# installed wheel\n")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(wheel)))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "wheel-version")
    result = package_identity("triton")
    assert result["git_sha"] is None and result["repository"] is None
    assert result["version"] == "wheel-version"


def test_proposal_source_and_budget_validation(tmp_path):
    response = {
        "status":
        "complete", "summary":
        "one variant", "candidates": [{
            "id": "one", "params": {}, "adapter_source": "def build(case, params):\n    return lambda x: x['q']",
            "rationale": "source caller"
        }]
    }
    proposals = validate_proposals(response, tmp_path, 1)
    assert proposals[0]["adapter_sha256"]
    assert Path(proposals[0]["adapter"]).is_file()
    with pytest.raises(ValueError, match="budget"):
        validate_proposals(response, tmp_path, 0)
    response["candidates"][0]["id"] = "../../escape"
    with pytest.raises(ValueError, match="ids"):
        validate_proposals(response, tmp_path, 1)


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_fake_provider_protocol(provider, tmp_path):
    cli = tmp_path / provider
    cli.write_text('''#!/usr/bin/env python3
import json,sys
prompt=sys.stdin.read()
answer={"summary":prompt,"candidates":[]}
if "--output-last-message" in sys.argv:
    from pathlib import Path
    Path(sys.argv[sys.argv.index("--output-last-message")+1]).write_text(json.dumps(answer))
else:
    print(json.dumps({"subtype":"success","is_error":False,"structured_output":answer}))
''')
    cli.chmod(0o700)
    prompt = "literal $(do-not-execute) `backticks`"
    result = query_json(provider, prompt, workspace=tmp_path, schema={"type": "object"}, executable=str(cli))
    assert result["summary"] == prompt


def test_claude_error_envelope_rejected(tmp_path):
    cli = tmp_path / "claude"
    cli.write_text('#!/usr/bin/env python3\nimport json\nprint(json.dumps({"subtype":"error","is_error":True}))\n')
    cli.chmod(0o700)
    with pytest.raises(RuntimeError, match="unsuccessful"):
        query_json("claude", "query", workspace=tmp_path, schema={"type": "object"}, executable=str(cli))


class FakeGitHub:

    def __init__(self):
        self.issues = {}
        self.comments = {}
        self.events = []
        self.corrupt_verification = False

    def all(self, resource):
        if resource.startswith("issues?"):
            return list(copy.deepcopy(self.issues).values())
        number = int(resource.split("/")[1])
        return [copy.deepcopy(c) for c in self.comments.values() if c["issue"] == number]

    def request(self, method, resource, payload=None):
        self.events.append((method, resource))
        parts = resource.split("/")
        if method == "POST" and resource == "issues":
            number = len(self.issues) + 1
            self.issues[number] = {"number": number, "state": "open", "html_url": f"issue/{number}", **payload}
            return copy.deepcopy(self.issues[number])
        if len(parts) == 3 and method == "POST":
            ident = len(self.comments) + 1
            self.comments[ident] = {"id": ident, "issue": int(parts[1]), "html_url": f"comment/{ident}", **payload}
            return copy.deepcopy(self.comments[ident])
        collection = self.comments if parts[1] == "comments" else self.issues
        ident = int(parts[-1])
        if method == "PATCH":
            collection[ident].update(payload)
        result = copy.deepcopy(collection[ident])
        if method == "GET" and self.corrupt_verification:
            result["body"] = "corrupted"
        return result


def report(mode="weekly", week="2026-10-12", date="2026-10-14"):
    return {"campaign": "abc", "week": week, "date": date, "mode": mode, "device": {"arch": "gfx950"}}


def test_long_report_split_preserves_unicode_rows_and_table_headers():
    from publication import split_report

    header = "| Problem | Command |\n| --- | --- |"
    rows = [f"| case_{i} | `reproduce {'µ' * 20} {i}` |" for i in range(10)]
    chunks = split_report("# Report\n\n" + header + "\n" + "\n".join(rows), max_bytes=240)
    assert len(chunks) > 1
    assert all(len(chunk.encode()) <= 240 for chunk in chunks)
    joined = "\n".join(chunks)
    assert all(joined.count(row) == 1 for row in rows)
    assert all(header in chunk for chunk in chunks if "| case_" in chunk)


def test_chunked_publication_is_idempotent_and_retires_stale_parts(monkeypatch):
    import publication

    monkeypatch.setattr(publication, "BODY_BYTES", 240)
    header = "| Problem | Command |\n| --- | --- |"
    markdown = "# Report\n\n" + header + "\n" + "\n".join(f"| case_{i} | {'x' * 70} |" for i in range(10))
    api = FakeGitHub()
    issue = publish(api, report(), markdown)
    comments = len(api.comments)
    assert comments > 1 and "continues" in issue["body"]
    publish(api, report(), markdown)
    assert len(api.issues) == 1 and len(api.comments) == comments
    publish(api, report(mode="nightly"), markdown)
    nightly_count = len(api.comments)
    publish(api, report(mode="nightly"), markdown)
    assert len(api.comments) == nightly_count
    publish(api, report(mode="nightly"), "short report")
    nightly_parts = [c for c in api.comments.values() if "mode=nightly;" in c["body"]]
    assert all("superseded" in c["body"] for c in nightly_parts)
    assert all("case_" in c["body"] for c in api.comments.values() if "mode=weekly;" in c["body"])


def test_failed_continuation_check_does_not_close_previous_week(monkeypatch):
    import publication

    monkeypatch.setattr(publication, "BODY_BYTES", 50)
    api = FakeGitHub()
    old = publish(api, report(week="2026-10-05"), "old")
    original = api.request

    def request(method, resource, payload=None):
        result = original(method, resource, payload)
        if method == "GET" and resource.startswith("issues/comments/"):
            result["body"] = "corrupted"
        return result

    monkeypatch.setattr(api, "request", request)
    with pytest.raises(RuntimeError, match="read-after-write"):
        publish(api, report(), "a" * 40 + "\n\n" + "b" * 40)
    assert api.issues[old["number"]]["state"] == "open"


def test_publication_rotation_and_daily_upsert():
    api = FakeGitHub()
    previous = publish(api, report(week="2026-10-05"), "old week")
    current = publish(api, report(), "baseline")
    assert api.issues[previous["number"]]["state"] == "closed"
    # Verification of the new issue must precede closure of the previous one.
    assert api.events.index(("GET", "issues/2")) < api.events.index(("PATCH", "issues/1"))
    publish(api, report(), "baseline retry")
    assert len(api.issues) == 2
    publish(api, report(mode="nightly"), "day one")
    publish(api, report(mode="nightly"), "day one retry")
    daily = [c for c in api.comments.values() if "date=2026-10-14" in c["body"]]
    assert len(daily) == 1 and "day one retry" in daily[0]["body"]
    assert current["number"] == 2


def test_failed_publication_verification_preserves_previous_week():
    api = FakeGitHub()
    previous = publish(api, report(week="2026-10-05"), "old week")
    api.corrupt_verification = True
    with pytest.raises(RuntimeError, match="not verified"):
        publish(api, report(), "new week")
    assert api.issues[previous["number"]]["state"] == "open"


def test_nightly_requires_owned_open_weekly_issue():
    with pytest.raises(RuntimeError, match="weekly baseline"):
        publish(FakeGitHub(), report(mode="nightly"), "night")


@pytest.mark.skipif(os.environ.get("SCOUT_TEST_GPU") != "1", reason="set SCOUT_TEST_GPU=1 for source-built GPU smoke")
def test_gpu_same_inputs_and_mutation_rejected(tmp_path, target):
    # Run through the actual worker, not a mock timer.
    case = {**target["cases"][0], "id": "small", "q_lengths": [33, 65], "kv_lengths": [33, 65]}
    timing = {**target["timing"], "warmup_ms": 5, "rep_ms": 10, "repeats": 3}
    target = {**target, "timing": timing}
    request = {"backend": "tlx_ops", "target": target, "case": case, "candidate": {"id": "public_api", "params": {}}}
    path, output = tmp_path / "request.json", tmp_path / "out.json"
    write_json(path, request)
    subprocess.run([sys.executable, str(ROOT / "benchmarking/worker.py"), str(path), str(output)], check=True)
    result = json.loads(output.read_text())
    assert result["status"] in {"ok", "unstable"}
    assert result["correctness_seeds"] == [17, 29]
    assert len(result["samples_ms"]) == 3
    bad = {
        "status":
        "complete", "summary":
        "mutation", "candidates": [{
            "id":
            "bad", "params": {}, "rationale":
            "negative test", "adapter_source":
            "def build(case, params):\n    def run(x):\n        x['q'].zero_()\n        return x['q']\n    return run"
        }]
    }
    request["candidate"] = validate_proposals(bad, tmp_path, 1)[0]
    write_json(path, request)
    failed = subprocess.run([sys.executable, str(ROOT / "benchmarking/worker.py"), str(path), str(output)], check=False)
    assert failed.returncode == 1
    assert json.loads(output.read_text())["status"] == "failed"


def test_blocked_discovery_is_not_unsupported(tmp_path):
    response = {"status": "blocked", "summary": "sandbox denied source access", "candidates": []}
    assert validate_proposals(response, tmp_path, 2) == []
    response["candidates"] = [{"id": "unverified"}]
    with pytest.raises(ValueError, match="blocked"):
        validate_proposals(response, tmp_path, 2)


@pytest.mark.parametrize("aiter_override", [False, True])
def test_controller_freezes_controls_and_preserves_venv(tmp_path, monkeypatch, target, aiter_override):
    import argparse

    import appctl

    monkeypatch.setattr(appctl, "wait_for_gpu", lambda *args, **kwargs: kwargs["ready"]())

    target_path = tmp_path / "target.json"
    write_json(target_path, target)
    device = {"vendor": "amd", "arch": "gfx950", "physical_gpu": 0, "name": "GPU"}
    identity = {
        "device": device, "runtime": {"compiler_sha256": "compiler", "torch": "test"}, "aiter":
        {"status": "unavailable"}
    }
    identity["runtime"]["triton_module"] = str(appctl.REPOSITORY / "python/triton/__init__.py")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)
    override = {**identity, "aiter": {"status": "available", "repository": "/aiter", "git_sha": "aiter-source"}}
    aiter_python = tmp_path / "aiter/bin/python"
    monkeypatch.setattr(appctl, "probe", lambda runtime, *args: override
                        if runtime["python"] == str(aiter_python) else identity)
    runtimes = tmp_path / "runtimes.json"
    write_json(runtimes, {"aiter": {"python": str(aiter_python)}})
    calls = []

    def fake_scout(backend, t, case, directory, runtime, args, *, research=None, cache=None, pool=None):
        expected_python = aiter_python if backend == "aiter" else venv_python
        assert runtime["python"] == str(expected_python)  # Preserve the virtual environment's interpreter path.
        if backend == "aiter":
            assert research["package"] == override["aiter"]
        calls.append(backend)
        directory.mkdir(parents=True)
        request = directory / "request.json"
        write_json(request, {"expected_identity": identity})
        return {
            "status": "ok", "input_sha256": "same-inputs", "latency_ms": 1, "p20_ms": 0.9, "p80_ms": 1.1, "tflops": 1,
            "logical_tbps": 1, "runtime": identity["runtime"], "backend_pin": {"git_sha": "source"}, "candidate":
            {"id": "seed"}, "reproducer": str(request), "request_sha256": digest(json.loads(request.read_text())),
            "reproducer_command": "command"
        }

    monkeypatch.setattr(appctl, "scout", fake_scout)
    args = argparse.Namespace(targets=[target_path], case=["one_x_1000"], state=tmp_path / "state",
                              python=str(venv_python), pythonpath="/source",
                              runtimes=runtimes if aiter_override else None, date="2026-10-14", execute=True,
                              mode="weekly", gpu=0, provider="builtin", max_candidates=3, rounds=0, agent_timeout=10,
                              worker_timeout=10, report_pointer=tmp_path / "pointer.json")
    appctl.run(args)
    assert calls == (["aiter"] if aiter_override else []) + ["pytorch", "tlx_ops", "tlx_tutorials"]
    calls.clear()
    args.mode, args.date = "nightly", "2026-10-15"
    appctl.run(args)
    assert calls == ["tlx_ops", "tlx_tutorials"]
    report_path = json.loads(args.report_pointer.read_text())["report"]
    nightly = json.loads(Path(report_path).read_text())
    progress = json.loads((Path(report_path).parent / "progress.json").read_text())
    assert progress == nightly
    controls = nightly["rows"][0]["results"]
    assert all(controls[name]["status"] == "ok" for name in ("pytorch", "tlx_ops", "tlx_tutorials", "tlx_ops_baseline"))
    assert controls["pytorch"]["frozen_date"] == controls["tlx_ops_baseline"]["frozen_date"] == "2026-10-14"
    assert controls["aiter"]["status"] == ("ok" if aiter_override else "unavailable")
    assert controls["aiter"]["frozen_date"] == "2026-10-14"
    args.mode = "weekly"
    with pytest.raises(ValueError, match="immutable"):
        appctl.run(args)
    args.mode, args.case = "nightly", ["ten_x_1000"]
    with pytest.raises(FileNotFoundError):
        appctl.run(args)


@pytest.mark.parametrize("changed", [None, "source", "request", "rows", "mode", "complete", "policy"])
def test_controller_resumes_complete_cases_and_rejects_drift(tmp_path, monkeypatch, target, changed):
    import datetime as dt

    import appctl

    target = {**target, "cases": target["cases"][:2]}
    target_path = tmp_path / "target.json"
    write_json(target_path, target)
    identity = {
        "device": {"vendor": "amd", "arch": "gfx950", "physical_gpu": 0, "name": "GPU"}, "runtime":
        {"compiler_sha256": "original", "triton_module":
         str(appctl.REPOSITORY / "python/triton/__init__.py")}, "aiter": {"status": "unavailable"}
    }
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(appctl, "wait_for_gpu", lambda *a, **kw: kw["ready"]())
    monkeypatch.setattr(appctl, "probe", lambda *a: identity)
    monkeypatch.setattr(appctl, "backends", lambda: {"tlx_ops": backends()["tlx_ops"]})
    calls = []
    interrupted = True

    def scout(backend, t, case, directory, runtime, args, **kwargs):
        calls.append(case["id"])
        if interrupted and case == target["cases"][1]:
            raise appctl.GPUOccupiedError("external job arrived")
        directory.mkdir(parents=True)
        request = directory / "request.json"
        write_json(request, {"expected_identity": identity})
        return {
            **measurement(1), "input_sha256": "same-inputs", "candidate": {"id": "seed"}, "backend_pin":
            {"git_sha": "source"}, "runtime": identity["runtime"], "tflops": 1, "logical_tbps": 1, "reproducer":
            str(request), "request_sha256": digest(read_json(request)), "reproducer_command": "command"
        }

    monkeypatch.setattr(appctl, "scout", scout)
    args = SimpleNamespace(targets=[target_path], case=None, state=tmp_path / "state", python=sys.executable,
                           pythonpath="/source", runtimes=None, date="2026-10-15", execute=True, mode="weekly", gpu=0,
                           provider="builtin", max_candidates=3, rounds=0, agent_timeout=10, worker_timeout=10,
                           report_pointer=tmp_path / "pointer.json")
    with pytest.raises(appctl.GPUOccupiedError):
        appctl.run(args)
    assert not args.report_pointer.exists() and not list(args.state.glob("baselines/**/*.json"))
    partial = next(args.state.glob("runs/*/progress.json"))
    previous = read_json(partial)
    assert not previous["complete"] and len(previous["rows"]) == 1
    previous["started_utc"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=120)).isoformat()
    if changed == "source":
        identity["runtime"]["compiler_sha256"] = "changed"
    elif changed == "request":
        write_json(Path(previous["rows"][0]["results"]["tlx_ops"]["reproducer"]), {})
    elif changed == "rows":
        previous["rows"][0]["case"] = target["cases"][1]
    elif changed == "mode":
        previous["mode"] = "nightly"
    elif changed == "complete":
        previous["complete"] = True
    elif changed == "policy":
        args.max_candidates += 1
    write_json(partial, previous)
    args.resume_from = partial
    interrupted = False
    calls.clear()
    if changed:
        with pytest.raises(ValueError, match="resume|reproducer"):
            appctl.run(args)
        assert not calls and not args.report_pointer.exists()
        return
    appctl.run(args)
    result = read_json(read_json(args.report_pointer)["report"])
    assert calls == [target["cases"][1]["id"]]
    assert result["complete"] and result["resumed_cases"] == 1 and len(result["rows"]) == 2
    assert result["rows"][0] == previous["rows"][0]
    assert result["elapsed_seconds"] >= 120  # Include the interruption rather than reporting only resumed work.


def test_report_escapes_table_cells():
    from reporting import render
    r = report()
    r.update(
        device={"name": "GPU",
                "arch": "gfx950", "physical_gpu": 0}, provider="builtin", columns=["pytorch"], budget={}, rows=[{
                    "description":
                    "shape | value", "case": {"id":
                                              "one"}, "target": {"metric": "compute", "timing": {"tie_fraction": 0.03}},
                    "results": {"pytorch": {"status": "failed", "error": "a | b\nline"}}
                }])
    text = render(r)
    assert "shape \\| value" in text and "a \\| b line" in text
    assert "| none |" in text


def test_report_labels_public_gemm_torch_fallback():
    from reporting import render

    r = report()
    result = {
        **measurement(1), "tflops": 1, "candidate": {"id": "public_api"}, "backend_pin": {"git_sha": "aiter"},
        "runtime": {"compiler_sha256": "compiler"}, "adapter_metadata": {"implementation": "torch"}
    }
    r.update(
        device={"name": "GPU",
                "arch": "gfx950", "physical_gpu": 0}, provider="builtin", columns=["aiter"], budget={}, rows=[{
                    "description":
                    "GEMM", "case": {"id": "one"}, "target": {"metric": "compute", "timing": {"tie_fraction": 0.03}},
                    "results": {"aiter": result}
                }])
    text = render(r)
    assert "Torch fallback" in text and "| Dispatch |" in text


def test_report_search_cost_excludes_frozen_controls_and_baseline_copy():
    from reporting import render

    r = report()
    fresh = {"status": "failed", "search": {"agent_calls": 2, "discovery_seconds": 3, "worker_seconds": 4}}
    frozen = {**fresh, "frozen_date": "2026-10-14"}
    r.update(
        device={"name": "GPU", "arch": "gfx950", "physical_gpu": 0}, provider="claude", budget={},
        discovery_scope="target", columns=["tlx_ops", "pytorch", "tlx_ops_baseline"], rows=[{
            "description": "shape", "case": {"id":
                                             "one"}, "target": {"metric": "compute", "timing": {"tie_fraction": 0.03}},
            "results": {"tlx_ops": fresh, "pytorch": frozen, "tlx_ops_baseline": fresh}
        }])
    text = render(r)
    assert "2 agent calls, 3.0s in discovery, 4.0s in measurement workers" in text
    assert "Discovery scope: target" in text


def test_weekly_publication_does_not_close_future_or_unrelated_issue():
    api = FakeGitHub()
    future = publish(api, report(week="2026-10-19"), "future")
    api.issues[2] = {"number": 2, "state": "open", "body": "ordinary user issue"}
    publish(api, report(), "older")
    assert api.issues[future["number"]]["state"] == "open"
    assert api.issues[2]["state"] == "open"


def test_timeout_terminates_worker(tmp_path):
    from appctl import execute
    with pytest.raises(RuntimeError, match="exceeded"):
        execute([sys.executable, "-c", "import time; time.sleep(10)"], env=os.environ.copy(),
                log=tmp_path / "worker.log", timeout=0.1)


def test_kfd_ownership_finds_queues_and_vram(tmp_path):
    from benchmarking.environment import kfd_owners
    for pid in (101, 102, 103):
        (tmp_path / str(pid)).mkdir()
    queue = tmp_path / "101/queues/0"
    queue.mkdir(parents=True)
    (queue / "gpuid").write_text("999")
    (tmp_path / "102/vram_999").write_text("4096")
    (tmp_path / "103/vram_999").write_text("0")
    assert kfd_owners(tmp_path, 999) == {101, 102}
    assert kfd_owners(tmp_path, 111) == set()


def test_scout_confirms_selected_candidate_and_records_blocked_search(tmp_path, monkeypatch, target):
    from types import SimpleNamespace

    import appctl
    args = SimpleNamespace(provider="codex", max_candidates=3, rounds=1, model=None, context=None, agent_timeout=5)
    requests = []

    def evaluate(backend, target, case, candidate, directory, runtime, args):
        requests.append((candidate["id"], directory.name))
        return {"status": "ok", "latency_ms": 1, "candidate": candidate}

    monkeypatch.setattr(appctl, "evaluate", evaluate)
    monkeypatch.setattr(
        appctl, "discover", lambda *a, **kw:
        ({"status": "blocked", "summary": "source tools denied", "candidates": []}, []))
    result = appctl.scout("tlx_ops", target, target["cases"][0], tmp_path / "ops", {}, args)
    assert requests == [("public_api", "r0_public_api"), ("public_api", "confirmation")]
    assert result["discovery_status"] == "blocked"
    assert result["status"] == "ok"  # Measured seed remains valid, search coverage is partial.
    result = appctl.scout("tlx_tutorials", target, target["cases"][0], tmp_path / "tutorials", {}, args)
    assert result["status"] == "blocked"


@pytest.fixture
def cached_scout(tmp_path, monkeypatch, target):
    import appctl
    from discovery import DiscoveryCache

    calls, evaluations = [], []
    args = SimpleNamespace(provider="claude", discovery="target", max_candidates=3, rounds=1, model=None, context=None,
                           agent_timeout=5)

    def discover(*a, **kwargs):
        calls.append(kwargs["scope"])
        response = {
            "status":
            "complete", "summary":
            "checked all shapes", "candidates": [{
                "id": "generic", "params": {}, "adapter_source":
                "def build(case, params):\n    return lambda inputs: inputs['q']\n", "rationale": "test source"
            }]
        }
        return response, validate_proposals(response, a[5], a[6])

    def evaluate(backend, target, case, candidate, directory, runtime, args):
        evaluations.append((case["id"], candidate["id"], directory.name))
        # A cached proposal must still fail when it is invalid for a different shape.
        if "adapter" in candidate and case.get("causal", False):
            return {"status": "failed", "error": "incorrect output", "worker_seconds": 2}
        return {
            "status": "ok", "latency_ms": 0.5 if "adapter" in candidate else 1, "candidate": candidate,
            "worker_seconds": 2
        }

    monkeypatch.setattr(appctl, "discover", discover)
    monkeypatch.setattr(appctl, "evaluate", evaluate)
    cache = DiscoveryCache(tmp_path / "cache")
    case = target["cases"][0]

    def run(name, *, case=case, cache=cache, research=None):
        return appctl.scout("tlx_ops", target, case, tmp_path / name, {}, args, research=research, cache=cache)

    return SimpleNamespace(run=run, calls=calls, evaluations=evaluations, args=args, cache=cache, target=target)


def test_target_discovery_reuses_adapters_but_rechecks_each_shape(cached_scout, tmp_path):
    from discovery import DiscoveryCache

    s = cached_scout
    first = s.run("first")
    second = s.run("second", case={**s.target["cases"][0], "id": "causal", "causal": True})
    assert s.calls == ["target"]
    assert first["candidate"]["id"] == "agent_generic"
    assert first["search"]["worker_seconds"] == 6  # Two candidates and independent confirmation.
    assert second["search"]["cache_hit"] and second["search"]["agent_calls"] == 0
    assert second["candidate"]["id"] == "public_api"  # Reject the cached proposal on this shape.
    assert any(case == "causal" and candidate.startswith("agent_cached_") for case, candidate, _ in s.evaluations)
    # A later process reuses the saved source without another agent call.
    s.args.discovery = "off"
    third = s.run("later", cache=DiscoveryCache(tmp_path / "cache"))
    assert third["search"]["cache_hit"] and third["evaluated_candidates"] == 2
    assert s.calls == ["target"]
    assert Path(third["candidate"]["adapter"]).is_relative_to(tmp_path / "later")


def test_target_discovery_reuses_a_completed_empty_search(cached_scout, monkeypatch):
    import appctl

    calls = []

    def empty(*a, **kwargs):
        calls.append(1)
        return {"status": "complete", "summary": "public API has no tuning surface", "candidates": []}, []

    monkeypatch.setattr(appctl, "discover", empty)
    cached_scout.run("first")
    second = cached_scout.run("second")
    assert calls == [1] and second["search"]["cache_hit"]
    assert second["evaluated_candidates"] == 1


def test_blocked_target_discovery_retries_next_run_not_each_shape(cached_scout, tmp_path, monkeypatch):
    import appctl
    from discovery import DiscoveryCache

    calls = []

    def blocked(*a, **kwargs):
        calls.append(1)
        return {"status": "blocked", "summary": "permission denied", "candidates": []}, []

    monkeypatch.setattr(appctl, "discover", blocked)
    cached_scout.run("first")
    assert cached_scout.run("second")["discovery_status"] == "blocked"
    assert calls == [1] and not list((tmp_path / "cache").glob("*.json"))
    cached_scout.run("later", cache=DiscoveryCache(tmp_path / "cache"))
    assert calls == [1, 1]


@pytest.mark.parametrize("scope", ["off", "case"])
def test_discovery_scope_controls_research(cached_scout, scope):
    s = cached_scout
    s.args.discovery = scope
    # Case discovery deliberately bypasses the target cache, as the controller does.
    cache = None if scope == "case" else s.cache
    for i in range(2):
        s.run(str(i), cache=cache)
    assert len(s.calls) == (2 if scope == "case" else 0)


def test_refresh_discovery_replaces_cache_once_per_target(cached_scout, tmp_path):
    from discovery import DiscoveryCache

    s = cached_scout
    s.run("original")
    refreshed = DiscoveryCache(tmp_path / "cache", refresh=True)
    assert not s.run("refresh", cache=refreshed)["search"]["cache_hit"]
    assert s.run("same_run", cache=refreshed)["search"]["cache_hit"]
    assert s.calls == ["target", "target"]


def test_discovery_cache_invalidates_changed_target_runtime_and_device(tmp_path, target):
    from discovery import DiscoveryCache

    cache = DiscoveryCache(tmp_path)
    research = {"runtime": {"compiler_sha256": "old"}, "device": {"arch": "gfx950"}, "context_sha256": "old"}
    key = cache.key("tlx_ops", target, research)
    cache.save(key, "complete", "no additional variants", [])
    assert cache.load(key) is not None
    for field in research:
        changed = {**research, field: "changed"}
        assert cache.load(cache.key("tlx_ops", target, changed)) is None
    changed_target = copy.deepcopy(target)
    changed_target["cases"][0]["heads"] += 1
    assert cache.load(cache.key("tlx_ops", changed_target, research)) is None
    assert cache.load(cache.key("pytorch", target, research)) is None


def test_discovery_cache_rejects_changed_content(cached_scout, tmp_path):
    from discovery import DiscoveryCache

    cached_scout.run("first")
    path = next((tmp_path / "cache").glob("*.json"))
    saved = json.loads(path.read_text())
    saved["response"]["candidates"][0]["adapter_source"] = "raise RuntimeError('changed')"
    write_json(path, saved)
    with pytest.raises(ValueError, match="content differs"):
        cached_scout.run("later", cache=DiscoveryCache(tmp_path / "cache"))


def test_target_discovery_prompt_is_compact_and_covers_all_cases(monkeypatch, tmp_path, target):
    import discovery

    def query(provider, prompt, **kwargs):
        assert "Research all cases in target.cases" in prompt
        assert "conditions" not in prompt and "UNNECESSARY_DEVICE_DUMP" not in prompt
        assert "GPUOccupiedError" in prompt and "latency_ms" in prompt
        assert target["cases"][-1]["id"] in prompt
        return {"status": "complete", "summary": "no alternatives", "candidates": []}

    monkeypatch.setattr(discovery, "query_json", query)
    feedback = [{
        "candidate": {"id": "seed"}, "result": {
            "status": "failed", "error": "GPUOccupiedError", "latency_ms": 1, "conditions": "UNNECESSARY_DEVICE_DUMP",
            "runtime": {"source": "huge"}
        }
    }]
    discovery.discover("claude", "tlx_ops", target, target["cases"][0], feedback, tmp_path, 2, scope="target")


def test_full_corpus_research_count_scales_with_targets_not_shapes(cached_scout, tmp_path):
    import appctl
    from discovery import DiscoveryCache

    s = cached_scout
    s.args.max_candidates = 6
    registry = backends()
    targets = [load_target(p) for p in sorted((ROOT / "targets/amd/gfx950").glob("*.json"))]
    research_groups = 0
    for scope in ("target", "off", "case"):
        s.args.discovery = scope
        cache = None if scope == "case" else DiscoveryCache(tmp_path / "corpus_cache")
        before = len(s.calls)
        for target in targets:
            for backend, contract in registry.items():
                if backend == "aiter" or not applicable(contract, target, {"vendor": "amd", "arch": "gfx950"}):
                    continue
                if scope == "target":
                    research_groups += 1
                for case in target["cases"]:
                    appctl.scout(backend, target, case, tmp_path / scope / target["id"] / backend / case["id"], {},
                                 s.args, cache=cache)
        assert len(s.calls) - before == {"target": 17, "off": 0, "case": 172}[scope]
    assert research_groups == 17
    assert sum(len(t["cases"]) for t in targets) == 60


@pytest.mark.parametrize("operation", ["mm", "addmm", "bmm"])
def test_shared_inputs_preserve_bits_layouts_and_zero_stride_batches(tmp_path, monkeypatch, operation):
    torch = pytest.importorskip("torch")
    from benchmarking import inputs

    case = {
        "dtype": "bfloat16", "m": 3, "n": 4, "k": 5, "batch": 2, "shared_a": True, "a_layout": "column_major",
        "b_layout": "column_major"
    }
    expected = inputs.make_inputs(case, operation, 17, device="cpu")
    first, fingerprint = inputs.cached_inputs(case, operation, 17, tmp_path, device="cpu")
    assert inputs.input_digest(expected) == fingerprint == inputs.input_digest(first)
    monkeypatch.setattr(inputs, "make_inputs", lambda *a, **kw: pytest.fail("cache hit regenerated random data"))
    second, actual_hash = inputs.cached_inputs(case, operation, 17, tmp_path, device="cpu")
    assert actual_hash == fingerprint
    for name in expected:
        assert second[name].stride() == expected[name].stride()
        assert torch.equal(second[name], expected[name])
    if operation == "bmm":
        assert second["a"].stride(0) == 0
    host = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    host["a"].flatten()[0] += 1
    # Modify the unique storage even for a column-major or expanded view.
    host["b"].zero_()
    torch.save(host, next(tmp_path.glob("*.pt")))
    with pytest.raises(ValueError, match="cache content differs"):
        inputs.cached_inputs(case, operation, 17, tmp_path, device="cpu")


def test_prepared_case_reuses_oracles_and_discards_mutated_inputs(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from benchmarking import inputs, worker

    original = inputs.cached_inputs
    calls = []

    def cached(*args, **kwargs):
        calls.append(args[2])
        return original(*args, device="cpu", **kwargs)

    monkeypatch.setattr(inputs, "cached_inputs", cached)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    target = load_target(ROOT / "targets/amd/gfx950/a16w16_gemm.json")
    request = {"target": target, "case": {**target["cases"][0], "m": 3, "n": 4, "k": 5}, "input_cache": str(tmp_path)}
    session = worker.MeasurementSession()
    prepared = session.prepare(request)
    expected = prepared[17]["expected"]
    assert calls == [17, 29, 0]
    assert session.prepare(request)[17]["expected"] is expected
    assert calls == [17, 29, 0]
    prepared[17]["inputs"]["a"].zero_()
    session.discard_mutated_inputs()
    assert session.case_key is None and not session.data
    recovered = session.prepare(request)
    torch.testing.assert_close(recovered[17]["expected"], expected)


def test_session_rejects_changed_source_and_harness(monkeypatch):
    from benchmarking import worker

    session = worker.MeasurementSession()
    session.runtime, session.harness = {"compiler": "pinned"}, "harness"
    monkeypatch.setattr(worker, "harness_digest", lambda: "harness")
    monkeypatch.setattr(worker, "runtime_identity", lambda: {"compiler": "changed"})
    with pytest.raises(ValueError, match="runtime changed"):
        session.verify()
    monkeypatch.setattr(worker, "harness_digest", lambda: "changed")
    with pytest.raises(ValueError, match="harness changed"):
        session.verify()


def test_probe_initializes_backend_defaults_before_fingerprinting(tmp_path, monkeypatch):
    from benchmarking import worker

    events = []
    monkeypatch.setattr(sys, "argv", ["worker.py", "--probe", str(tmp_path / "device.json")])
    monkeypatch.setattr(worker, "inspect_device", dict)
    monkeypatch.setattr(worker, "assert_idle", lambda *a: None)
    monkeypatch.setattr(worker, "initialize_packages", lambda: events.append("initialized"))
    monkeypatch.setattr(worker, "runtime_identity", lambda: events.append("fingerprinted") or {})
    monkeypatch.setattr(worker, "package_identity", lambda name: {})
    assert worker.main() == 0
    assert events == ["initialized", "fingerprinted"]


@pytest.mark.parametrize("configured, expected", [(None, "1"), ("2", "2")])
def test_compiler_thread_limit_precedes_backend_initialization(monkeypatch, configured, expected):
    import importlib.util
    import os

    from benchmarking import worker

    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    if configured is not None:
        monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", configured)

    def find_backend(name):
        assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == expected

    monkeypatch.setattr(importlib.util, "find_spec", find_backend)
    worker.initialize_packages()


def test_compile_drain_uses_the_framework_shutdown_api(monkeypatch):
    from benchmarking import worker

    calls = []
    monkeypatch.setitem(sys.modules, "torch._inductor.async_compile",
                        SimpleNamespace(shutdown_compile_workers=lambda: calls.append("drained")))
    worker.drain_compile_workers()
    assert calls == ["drained"]


@pytest.mark.parametrize("error_type", ["provenance", "ownership"])
def test_persistent_infrastructure_failure_aborts_without_a_measurement(tmp_path, monkeypatch, error_type):
    from benchmarking import worker

    error = worker.ProvenanceError("source drift") if error_type == "provenance" else worker.GPUOccupiedError(
        "occupied")

    def fail(*a):
        raise error

    calls = []
    monkeypatch.setattr(worker, "measure", fail)
    monkeypatch.setattr(worker, "drain_compile_workers", lambda: calls.append("drained"))
    with pytest.raises(type(error)):
        worker.measure_to_file({}, tmp_path / "measurement.json", worker.MeasurementSession())
    assert calls == ["drained"]
    assert not (tmp_path / "measurement.json").exists()


def test_gpu_ownership_excludes_only_registered_peer_workers(monkeypatch):
    from benchmarking import environment

    rows = ["gpu,11"]
    monkeypatch.setattr(environment, "command", lambda args: "gpu" if "--query-gpu=uuid" in args else "\n".join(rows))
    device = {"vendor": "nvidia", "physical_gpu": 0}
    with pytest.raises(environment.GPUOccupiedError):
        environment.assert_idle(device)
    environment.assert_idle(device, allowed_pids=[11])
    rows.append("gpu,12")
    with pytest.raises(environment.GPUOccupiedError, match="12"):
        environment.assert_idle(device, allowed_pids=[11])


@pytest.fixture
def fake_worker_server(tmp_path, monkeypatch):
    import worker_pool

    script = tmp_path / "benchmarking/worker.py"
    script.parent.mkdir()
    script.write_text("""import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--serve',action='store_true');p.add_argument('--response-fd',type=int)
args=p.parse_args();responses=os.fdopen(args.response_fd,'w',buffering=1)
print('backend output must not corrupt acknowledgements',flush=True)
for line in sys.stdin:
 msg=json.loads(line)
 if not msg.get('verify'):
  request=json.loads(Path(msg['request']).read_text())
  if request.get('sleep'):time.sleep(request['sleep'])
  Path(msg['output']).write_text(json.dumps({'status':'ok','pid':os.getpid(),'peers':msg['peers']}))
 responses.write(json.dumps({'exit_code':0})+'\\n');responses.flush()
""")
    monkeypatch.setattr(worker_pool, "ROOT", tmp_path)
    return worker_pool.WorkerPool(tmp_path / "workers", lambda *a: os.environ.copy(), 2)


def test_worker_pool_reuses_processes_and_keeps_logs_out_of_protocol(tmp_path, fake_worker_server):
    pool = fake_worker_server
    request, output, log = (tmp_path / name for name in ("request.json", "measurement.json", "worker.log"))
    write_json(request, {"backend": "pytorch"})
    runtime = {"python": sys.executable}
    with pool:
        pool.evaluate(runtime, 0, request, output, log)
        first = json.loads(output.read_text())
        pool.evaluate(runtime, 0, request, output, log)
        assert json.loads(output.read_text())["pid"] == first["pid"]
        pool.evaluate({**runtime, "pythonpath": "different-runtime"}, 0, request, output, log)
        second = json.loads(output.read_text())
        assert second["pid"] != first["pid"]
        assert set(second["peers"]) == {first["pid"], second["pid"]}
        write_json(request, {"backend": "tlx_ops"})
        pool.evaluate(runtime, 0, request, output, log)
        third = json.loads(output.read_text())
        assert third["pid"] not in {first["pid"], second["pid"]}
        assert set(third["peers"]) == {first["pid"], second["pid"], third["pid"]}
        pool.evaluate(runtime, 0, request, output, log)
        assert json.loads(output.read_text())["pid"] == third["pid"]
        write_json(request, {"backend": "pytorch"})
        pool.evaluate(runtime, 0, request, output, log)
        assert json.loads(output.read_text())["pid"] == first["pid"]
    assert pool.verified
    assert all(w["process"].poll() == 0 for w in pool.workers.values())
    assert "backend output" in next((tmp_path / "workers").glob("*.log")).read_text()


def test_worker_pool_timeout_terminates_its_process(tmp_path, fake_worker_server):
    pool = fake_worker_server
    pool.timeout = 0.05
    request, output, log = (tmp_path / name for name in ("request.json", "measurement.json", "worker.log"))
    write_json(request, {"backend": "pytorch", "sleep": 30})
    with pytest.raises(RuntimeError, match="exceeded"), pool:
        pool.evaluate({"python": sys.executable}, 0, request, output, log)
    assert all(w["process"].poll() is not None for w in pool.workers.values())
    assert not output.exists()


def test_screening_promotes_unstable_fast_candidate_and_reports_only_full_timing(tmp_path, monkeypatch, target):
    import appctl

    args = SimpleNamespace(provider="builtin", discovery="off", max_candidates=3, rounds=0, screen_candidates=True,
                           finalists=2)
    calls = []

    def evaluate(backend, target, case, candidate, directory, runtime, args, *, timing_phase="full"):
        calls.append((candidate["id"], timing_phase, directory.name))
        latency = {"sdpa_auto": 1, "sdpa_math": 9, "sdpa_padded": 2}[candidate["id"]]
        return {
            "status": "unstable" if timing_phase == "screen" and latency == 1 else "ok", "latency_ms": latency,
            "candidate": candidate, "timing_phase": timing_phase, "worker_seconds": 1
        }

    monkeypatch.setattr(appctl, "evaluate", evaluate)
    result = appctl.scout("pytorch", target, target["cases"][0], tmp_path, {}, args)
    assert result["status"] == "ok" and result["timing_phase"] == "full"
    assert result["candidate"]["id"] == "sdpa_auto"
    assert [c[0] for c in calls if c[2].startswith("final_")] == ["sdpa_auto", "sdpa_padded"]
    assert calls[-1] == ("sdpa_auto", "full", "confirmation")
    assert result["search"]["worker_seconds"] == 6


def test_baseline_keys_separate_execution_and_screening_protocols(target):
    assert baseline_key([target], {}, {}) != baseline_key([target], {}, {}, execution="persistent")
    assert baseline_key([target], {}, {}) != baseline_key([target], {}, {}, screening=True)


def test_publication_rejects_partial_progress_and_screening_results():
    api = FakeGitHub()
    partial = {**report(), "complete": False}
    with pytest.raises(ValueError, match="partial progress"):
        publish(api, partial, "partial")
    screened = {**report(), "rows": [{"results": {"backend": {"status": "ok", "timing_phase": "screen"}}}]}
    with pytest.raises(ValueError, match="screening timings"):
        publish(api, screened, "screen")
    infrastructure = {
        **report(), "rows": [{"results": {"backend": {"status": "failed", "error": "GPUOccupiedError: occupied"}}}]
    }
    with pytest.raises(ValueError, match="infrastructure failures"):
        publish(api, infrastructure, "invalid")
    assert not api.issues


def test_deployment_uses_argument_lists_and_success_pointer(tmp_path):
    spec = importlib.util.spec_from_file_location("scout_deploy", ROOT / "deploy/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {
        "python": "/venv/bin/python", "targets": ["/source/target with spaces.json"], "state": "/state with spaces",
        "provider": "claude", "model": "model", "rounds": 1, "gpu_wait_seconds": 1800, "gpu_poll_seconds": 60,
        "discovery": "off", "candidate_cache": "/cache with spaces"
    }
    argv = module.commands(config, "nightly", tmp_path / "pointer.json")
    assert "/source/target with spaces.json" in argv
    assert argv[argv.index("--state") + 1] == "/state with spaces"
    assert argv[argv.index("--report-pointer") + 1] == str(tmp_path / "pointer.json")
    assert argv[argv.index("--gpu-wait-seconds") + 1] == "1800"
    assert argv[argv.index("--gpu-poll-seconds") + 1] == "60"
    assert argv[argv.index("--discovery") + 1] == "off"
    assert argv[argv.index("--candidate-cache") + 1] == "/cache with spaces"


@pytest.mark.parametrize("flags",
                         [["--discovery", "off"], ["--provider", "builtin"], ["--provider", "claude", "--rounds", "0"]])
def test_refresh_discovery_requires_an_enabled_target_research_phase(tmp_path, flags):
    result = subprocess.run([
        sys.executable,
        str(ROOT / "appctl.py"), "run", "--mode", "weekly", "--targets",
        str(ROOT / "targets/amd/gfx950/bf16_fa.json"), "--state",
        str(tmp_path / "state"), "--refresh-discovery", *flags
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 2 and "refresh discovery requires" in result.stderr
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("busy_checks,timeout", [(0, False), (2, False), (31, True)])
def test_gpu_scheduler_waits_once_per_minute_with_a_30_minute_limit(tmp_path, monkeypatch, busy_checks, timeout):
    import gpu_scheduler

    now, sleeps, checks = [0.0], [], []

    def busy(physical, vendor):
        assert (physical, vendor) == (1, "amd")
        checks.append(now[0])
        return "selected GPU has external processes: [123]" if len(checks) <= busy_checks else None

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(gpu_scheduler, "busy_reason", busy)
    monkeypatch.setattr(gpu_scheduler.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(gpu_scheduler.time, "sleep", sleep)
    path = tmp_path / "admission.json"
    if timeout:
        with pytest.raises(TimeoutError, match="stayed busy for 1800s"):
            gpu_scheduler.wait_for_gpu(1, "amd", status_path=path)
    else:
        gpu_scheduler.wait_for_gpu(1, "amd", status_path=path)
    assert checks == list(range(0, min(busy_checks, 30) * 60 + 1, 60))
    assert sleeps == [60] * min(busy_checks, 30)
    assert json.loads(path.read_text())["state"] == ("timed_out" if timeout else "ready")


@pytest.mark.parametrize("message,retryable", [("selected GPU has external processes: [123]", True),
                                               ("cannot inspect selected GPU ownership", False)])
def test_gpu_scheduler_retries_ownership_only(monkeypatch, message, retryable):
    import gpu_scheduler

    monkeypatch.setattr(gpu_scheduler, "command", lambda args: '[{"gpu": 1, "kfd_id": 999}]')

    def check(device):
        assert device == {"physical_gpu": 1, "vendor": "amd", "kfd_id": 999}
        raise RuntimeError(message)

    monkeypatch.setattr(gpu_scheduler, "assert_idle", check)
    if retryable:
        assert gpu_scheduler.busy_reason(1, "amd") == message
    else:
        with pytest.raises(RuntimeError, match=message):
            gpu_scheduler.busy_reason(1, "amd")


def test_gpu_scheduler_does_not_wait_on_an_inspection_failure(tmp_path, monkeypatch):
    import gpu_scheduler

    def fail(*args):
        raise RuntimeError("KFD process ownership is unavailable")

    monkeypatch.setattr(gpu_scheduler, "busy_reason", fail)
    monkeypatch.setattr(gpu_scheduler.time, "sleep", lambda seconds: pytest.fail("inspection errors must not retry"))
    path = tmp_path / "admission.json"
    with pytest.raises(RuntimeError, match="KFD process ownership"):
        gpu_scheduler.wait_for_gpu(1, "amd", status_path=path)
    assert json.loads(path.read_text())["state"] == "failed"


def test_gpu_scheduler_recovers_when_the_worker_probe_finds_new_ownership(tmp_path, monkeypatch):
    import gpu_scheduler
    from benchmarking.environment import GPUOccupiedError

    now, calls = [0.0], []
    monkeypatch.setattr(gpu_scheduler, "busy_reason", lambda *args: None)
    monkeypatch.setattr(gpu_scheduler.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(gpu_scheduler.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))

    def ready():
        calls.append(now[0])
        if len(calls) == 1:
            now[0] += 10  # Framework startup takes time before it observes contention.
            raise GPUOccupiedError("selected GPU has external processes: [123]")
        return {"device": "checked"}

    assert gpu_scheduler.wait_for_gpu(1, "amd", ready=ready,
                                      status_path=tmp_path / "state.json") == {"device": "checked"}
    assert calls == [0, 60]


def test_probe_does_not_reuse_a_stale_busy_result(tmp_path, monkeypatch):
    import appctl
    from benchmarking.environment import GPUOccupiedError

    monkeypatch.setattr(appctl, "worker_environment", lambda *args: {})

    def busy(*args, **kwargs):
        write_json(tmp_path / "device.json", {"status": "busy", "error": "another process"})
        return 2

    monkeypatch.setattr(appctl, "execute", busy)
    with pytest.raises(GPUOccupiedError):
        appctl.probe({"python": "python"}, 1, tmp_path)
    monkeypatch.setattr(appctl, "execute", lambda *a, **kw: 1)
    with pytest.raises(RuntimeError, match="GPU probe failed") as failure:
        appctl.probe({"python": "python"}, 1, tmp_path)
    assert not isinstance(failure.value, GPUOccupiedError)


@pytest.mark.parametrize("busy_check", [1, 2])
def test_worker_probe_reports_contention_before_or_after_fingerprinting(tmp_path, monkeypatch, busy_check):
    from benchmarking import worker
    from benchmarking.environment import GPUOccupiedError

    path = tmp_path / "device.json"
    checks = []

    def check(device):
        checks.append(device)
        if len(checks) == busy_check:
            raise GPUOccupiedError("selected GPU has external processes: [123]")

    monkeypatch.setattr(sys, "argv", ["worker.py", "--probe", str(path)])
    monkeypatch.setattr(worker, "inspect_device", lambda: {"physical_gpu": 7})
    monkeypatch.setattr(worker, "assert_idle", check)
    monkeypatch.setattr(worker, "runtime_identity", dict)
    monkeypatch.setattr(worker, "package_identity", lambda name: {})
    assert worker.main() == 2
    assert json.loads(path.read_text()) == {"status": "busy", "error": "selected GPU has external processes: [123]"}
    assert len(checks) == busy_check


@pytest.mark.parametrize("wait,poll", [(-1, 60), (float("nan"), 60), (1800, 0), (1800, float("inf"))])
def test_gpu_scheduler_rejects_invalid_wait_parameters(monkeypatch, wait, poll):
    import gpu_scheduler

    monkeypatch.setattr(gpu_scheduler, "busy_reason",
                        lambda *args: pytest.fail("invalid parameters must not inspect GPUs"))
    with pytest.raises(ValueError):
        gpu_scheduler.wait_for_gpu(1, "amd", wait_seconds=wait, poll_seconds=poll)


def test_pci_guard_rejects_wrong_physical_gpu():
    from types import SimpleNamespace

    from benchmarking.environment import check_pci_device
    props = SimpleNamespace(pci_domain_id=0, pci_bus_id=0x66, pci_device_id=0)
    check_pci_device(props, "0000:66:00.0")
    with pytest.raises(RuntimeError, match="visibility is inconsistent"):
        check_pci_device(props, "0000:16:00.0")
    with pytest.raises(RuntimeError, match="visibility is inconsistent"):
        check_pci_device(SimpleNamespace(), "0000:66:00.0")


@pytest.mark.parametrize("status", ["failed", "unstable", "unavailable", "unsupported"])
def test_scout_keeps_failure_categories(status, tmp_path, monkeypatch, target):
    from types import SimpleNamespace

    import appctl
    monkeypatch.setattr(appctl, "evaluate", lambda *args: {"status": status, "error": "diagnostic"})
    args = SimpleNamespace(provider="builtin", max_candidates=1, rounds=0)
    result = appctl.scout("tlx_ops", target, target["cases"][0], tmp_path / "ops", {}, args)
    assert result["status"] == status and result["error"] == "diagnostic"


@pytest.mark.parametrize("hashes", [("same", "same"), ("a", "b"), ("a", None)])
def test_input_fingerprints_gate_comparisons(hashes):
    from contracts import enforce_equal_inputs
    results = {str(i): {"status": "ok", "input_sha256": value} for i, value in enumerate(hashes)}
    enforce_equal_inputs(results)
    expected = "ok" if hashes == ("same", "same") else "failed"
    assert all(r["status"] == expected for r in results.values())


def test_unstable_search_summary_does_not_reject_measured_inputs():
    from contracts import enforce_equal_inputs

    results = {
        "aiter": {"status": "ok", "latency_ms": 1, "input_sha256":
                  "same"}, "torch": {"status": "ok", "latency_ms": 2, "input_sha256": "same"}, "tlx":
        {"status": "unstable", "error": "no stable correct candidate", "evaluated_candidates": 2}
    }
    enforce_equal_inputs(results)
    assert results["aiter"]["status"] == results["torch"]["status"] == "ok"
    assert results["tlx"]["status"] == "unstable"


@pytest.mark.parametrize("fingerprint", [None, "different"])
def test_unstable_measurement_still_checks_input_fingerprint(fingerprint):
    from contracts import enforce_equal_inputs

    results = {
        "aiter": {"status": "ok", "latency_ms": 1, "input_sha256": "same"}, "tlx":
        {"status": "unstable", "latency_ms": 2, "input_sha256": fingerprint}
    }
    enforce_equal_inputs(results)
    assert all(r["status"] == "failed" for r in results.values())


@pytest.mark.parametrize("operation", ["mm", "addmm", "bmm", "flash_attn", "flash_attn_varlen", "flash_attn_mxfp8"])
def test_target_init_cli_creates_valid_contract(operation, tmp_path):
    output = tmp_path / operation / "target.json"
    subprocess.run([
        sys.executable,
        str(ROOT / "appctl.py"), "target-init", "--id", "my_target", "--operation", operation, "--vendor", "amd",
        "--arch", "gfx950", "--output",
        str(output)
    ], check=True)
    target = load_target(output)
    assert target["operation"] == operation and target["id"] == "my_target"
    assert target["hardware"] == {"vendor": "amd", "architectures": ["gfx950"]}
    assert target["output_policy"] == "returned_tensor"
    assert work(target["cases"][0], operation)[0] > 0


def test_target_init_refuses_overwrite_and_rejects_unknown_operation(tmp_path):
    from target_templates import create_target
    path = tmp_path / "target.json"
    kwargs = {"name": "my_target", "operation": "mm", "vendor": "amd", "architecture": "gfx950", "metric": "compute"}
    create_target(path, **kwargs)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        create_target(path, **kwargs)
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        create_target(tmp_path / "new.json", **{**kwargs, "operation": "layer_norm"})
    assert not (tmp_path / "new.json").exists()


def test_target_templates_are_independent_and_support_memory_metrics(tmp_path):
    from target_templates import create_target
    kwargs = {"operation": "flash_attn_varlen", "vendor": "amd", "architecture": "gfx950", "metric": "memory"}
    first = create_target(tmp_path / "first.json", name="first", **kwargs)
    first["cases"][0]["q_lengths"][0] = 999
    second = create_target(tmp_path / "second.json", name="second", **kwargs)
    assert second["cases"][0]["q_lengths"] == [33, 65]
    assert second["metric"] == "memory"


@pytest.mark.parametrize("status", ["ok", "failed", "unstable", "unavailable", "unsupported"])
def test_registration_records_real_coverage(target, status):
    from registration import registration_result

    measured = {
        "status": "ok", "input_sha256": "identical", "timing_phase": "full", "correctness_seeds": [17, 29],
        "reproducer": "/request.json", "request_sha256": "request"
    }
    results = {name: dict(measured) for name in ("tlx_ops", "pytorch", "aiter")}
    results["aiter"].update(status=status, error="diagnostic")
    value = {
        "rows": [{"target": target, "case": target["cases"][0], "results": results}], "baseline_key": "pins", "device":
        {"arch": "gfx950"}
    }
    readiness = registration_result(value)
    assert readiness["ready"] == (status in {"ok", "unsupported"})
    assert readiness["cases"][0]["target_sha256"] == digest(target)
    assert readiness["cases"][0]["unsupported"] == ({"aiter": "diagnostic"} if status == "unsupported" else {})
    results["pytorch"]["input_sha256"] = "different"
    rejected = registration_result(value)
    assert not rejected["ready"]
    assert any("fingerprints" in error for error in rejected["cases"][0]["errors"])


@pytest.mark.parametrize("change", ["seed", "screen", "reproducer", "tlx", "reference"])
def test_registration_rejects_incomplete_evidence(target, change):
    from registration import registration_result

    measured = {
        "status": "ok", "input_sha256": "identical", "timing_phase": "full", "correctness_seeds": [17, 29],
        "reproducer": "/request.json", "request_sha256": "request"
    }
    results = {name: dict(measured) for name in ("tlx_ops", "pytorch")}
    if change == "seed":
        results["tlx_ops"]["correctness_seeds"] = [17]
    elif change == "screen":
        results["tlx_ops"]["timing_phase"] = "screen"
    elif change == "reproducer":
        results["tlx_ops"].pop("reproducer")
    else:
        results["tlx_ops" if change == "tlx" else "pytorch"] = {"status": "unsupported"}
    value = {
        "rows": [{"target": target, "case": target["cases"][0], "results": results}], "baseline_key": "pins", "device":
        {"arch": "gfx950"}
    }
    assert not registration_result(value)["ready"]


@pytest.mark.parametrize("failed", [False, True])
def test_registration_controller_never_creates_a_baseline(tmp_path, monkeypatch, target, failed):
    import appctl

    path = tmp_path / "target.json"
    write_json(path, {**target, "cases": target["cases"][:1]})
    identity = {
        "device": {"vendor": "amd", "arch": "gfx950", "physical_gpu": 0, "name": "GPU"}, "runtime":
        {"compiler_sha256": "compiler", "triton_module":
         str(appctl.REPOSITORY / "python/triton/__init__.py")}, "aiter": {"status": "unavailable"}
    }
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(appctl, "wait_for_gpu", lambda *a, **kw: kw["ready"]())
    monkeypatch.setattr(appctl, "probe", lambda *a: identity)
    monkeypatch.setattr(appctl, "backends", lambda: {name: backends()[name] for name in ("tlx_ops", "pytorch")})

    def scout(backend, *a, **kw):
        if failed:
            return {"status": "failed", "error": "bad adapter"}
        return {
            **measurement(1), "input_sha256": "same", "timing_phase": "full", "correctness_seeds": [17, 29],
            "candidate": {"id": "seed"}, "backend_pin": {"git_sha": "source"}, "runtime": identity["runtime"], "tflops":
            1, "logical_tbps": 1, "reproducer": "/request.json", "request_sha256": "request"
        }

    monkeypatch.setattr(appctl, "scout", scout)
    args = SimpleNamespace(targets=[path], case=None, state=tmp_path / "state", python=sys.executable,
                           pythonpath="/source", runtimes=None, date="2026-10-15", execute=True, mode="registration",
                           gpu=0, provider="builtin", max_candidates=3, rounds=0, agent_timeout=10, worker_timeout=10,
                           report_pointer=tmp_path / "pointer.json")
    if failed:
        with pytest.raises(RuntimeError, match="registration needs attention"):
            appctl.run(args)
    else:
        appctl.run(args)
    recorded = read_json(read_json(args.report_pointer)["report"])
    assert recorded["complete"] and recorded["registration_ready"] is not failed
    assert "tlx_ops_baseline" not in recorded["columns"]
    assert not list(args.state.glob("baselines/**/*.json"))
    receipt = read_json(next(args.state.glob("runs/*/registration.json")))
    assert receipt["ready"] is not failed
    with pytest.raises(ValueError, match="only weekly and nightly"):
        publish(None, recorded, "registration")
