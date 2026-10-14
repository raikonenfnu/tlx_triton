"""One isolated candidate measurement. The controller never trusts agent timings."""

import argparse
import hashlib
import os
import statistics
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import REPOSITORY, digest, harness_digest, read_json, work, write_json

from benchmarking.environment import assert_idle, conditions, inspect_device, package_identity, runtime_identity


def measure(request):
    import torch
    import triton.testing
    from backends.seeds import build, module

    from benchmarking.inputs import input_digest, make_inputs, reference

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = inspect_device()
    assert_idle(device)
    target, case, candidate = (request[key] for key in ("target", "case", "candidate"))
    operation = target["operation"]
    if candidate.get("adapter"):
        path = Path(candidate["adapter"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != candidate["adapter_sha256"]:
            raise ValueError("candidate adapter digest mismatch")
        fn = module(path, "scout_candidate").build(case, candidate["params"])
    else:
        fn = build(request["backend"], operation, case, candidate["params"])
    runtime = runtime_identity(request.get("repository", REPOSITORY))
    backend_pin = package_identity("aiter") if request["backend"] == "aiter" else (
        {"git_sha": torch.version.git_version, "version": torch.__version__}
        if request["backend"] == "pytorch" else runtime["source"])
    identity = {"device": device, "runtime": runtime, "backend_pin": backend_pin, "harness_sha256": harness_digest()}
    if request.get("expected_identity") and identity != request["expected_identity"]:
        raise ValueError("reproducer source/runtime/device pins differ from the recorded run")
    checks = []
    input_hashes = {}
    for seed in target["correctness"]["seeds"]:
        inputs = make_inputs(case, operation, seed)
        input_hashes[str(seed)] = input_digest(inputs)
        original = {key: value.clone() for key, value in inputs.items()}
        expected = reference(case, operation, inputs)
        actual = fn(inputs)
        torch.cuda.synchronize()
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
            raise ValueError("candidate must return the canonical output tensor")
        if actual.dtype != getattr(torch, case["dtype"]) or actual.device != inputs[next(iter(inputs))].device:
            raise ValueError("output dtype/device violates the contract")
        if not torch.isfinite(actual).all():
            raise ValueError("output contains non-finite values")
        torch.testing.assert_close(actual.float(), expected, rtol=target["correctness"]["rtol"],
                                   atol=target["correctness"]["atol"])
        if any(not torch.equal(inputs[key], value) for key, value in original.items()):
            raise ValueError("candidate changed a harness input")
        if any(actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr() for value in inputs.values()):
            raise ValueError("returned output must not alias an input")
        checks.append(seed)
    # Restore one canonical seed for every backend's measured inputs.
    inputs = make_inputs(case, operation, 0)
    input_hashes["timing_seed_0"] = input_digest(inputs)
    timing = target["timing"]
    before = conditions(device)
    samples = [
        triton.testing.do_bench(lambda: fn(inputs), warmup=timing["warmup_ms"], rep=timing["rep_ms"],
                                quantiles=[0.2, 0.5, 0.8]) for _ in range(timing["repeats"])
    ]
    assert_idle(device)
    after = conditions(device)
    medians = [sample[1] for sample in samples]
    latency = statistics.median(medians)
    spread = (max(medians) - min(medians)) / latency
    flops, logical_bytes = work(case, operation)
    return {
        "status": "ok" if spread <= timing["max_relative_spread"] else "unstable", **identity, "candidate": candidate,
        "contract_sha256": digest({"target": target, "case": case
                                   }), "correctness_seeds": checks, "input_sha256": digest(input_hashes), "latency_ms":
        latency, "p20_ms": statistics.median(s[0] for s in samples), "p80_ms": statistics.median(s[2] for s in samples),
        "samples_ms": samples, "relative_spread": spread, "tflops": flops / (latency * 1e9), "logical_tbps":
        logical_bytes / (latency * 1e9), "conditions": {"before": before, "after": after}, "pid": os.getpid()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--probe", type=Path)
    args = parser.parse_args()
    if args.probe:
        device = inspect_device()
        assert_idle(device)
        write_json(args.probe,
                   {"device": device, "runtime": runtime_identity(REPOSITORY), "aiter": package_identity("aiter")})
        return 0
    if args.request is None or args.output is None:
        parser.error("request and output are required for a measurement")
    try:
        result = measure(read_json(args.request))
    except Exception as error:  # noqa: BLE001 - candidate failures become explicit artifacts
        traceback.print_exc()
        status = "unavailable" if isinstance(error, ModuleNotFoundError) else (
            "unsupported" if type(error).__name__ in {"UnsupportedOp", "InvalidInput"} else "failed")
        result = {"status": status, "error": f"{type(error).__name__}: {error}"}
    write_json(args.output, result)
    return 0 if result["status"] in {"ok", "unstable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
