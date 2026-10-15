"""One isolated candidate measurement. The controller never trusts agent timings."""

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import digest, harness_digest, read_json, work, write_json

from benchmarking.environment import (
    GPUOccupiedError,
    assert_idle,
    conditions,
    configure_cpu_threads,
    inspect_device,
    package_identity,
    runtime_identity,
)


class MeasurementSession:
    """Retain runtime state and one case's immutable inputs and independent oracles."""

    def __init__(self):
        self.device, self.runtime, self.case_key = None, None, None
        self.packages, self.data = {}, {}
        self.allowed_pids = ()
        self.harness = None

    def idle(self):
        assert_idle(self.device, allowed_pids=self.allowed_pids)

    def identity(self, backend):
        import torch

        if self.runtime is None:
            self.runtime = runtime_identity()
            self.harness = harness_digest()
        if backend == "aiter" and backend not in self.packages:
            self.packages[backend] = package_identity(backend)
        pin = self.packages[backend] if backend == "aiter" else (
            {"git_sha": torch.version.git_version, "version": torch.__version__}
            if backend == "pytorch" else self.runtime["source"])
        return {"device": self.device, "runtime": self.runtime, "backend_pin": pin, "harness_sha256": self.harness}

    def verify(self):
        if self.harness is not None and harness_digest() != self.harness:
            raise ValueError("measurement harness changed during the session")
        if self.runtime is not None and runtime_identity() != self.runtime:
            raise ValueError("worker source or runtime changed during the session")
        for name, pin in self.packages.items():
            if package_identity(name) != pin:
                raise ValueError(f"{name} source changed during the session")

    def prepare(self, request):
        import torch

        from benchmarking.inputs import cached_host_inputs, cached_inputs, input_digest, make_inputs, reference

        target, case = request["target"], request["case"]
        key = digest([case, target["operation"], target["correctness"]])
        if key == self.case_key:
            return self.data
        if self.case_key is not None:
            self.verify()
        self.data.clear()
        self.case_key = None
        gc.collect()
        seeds = list(dict.fromkeys([*target["correctness"]["seeds"], 0]))
        host_data = {}
        if request.get("input_cache"):
            with ThreadPoolExecutor(max_workers=min(3, len(seeds))) as executor:
                futures = {
                    seed: executor.submit(cached_host_inputs, case, target["operation"], seed, request["input_cache"])
                    for seed in seeds
                }
                host_data = {seed: future.result() for seed, future in futures.items()}
        for seed in seeds:
            if seed in self.data:
                continue
            if request.get("input_cache"):
                inputs, fingerprint = cached_inputs(case, target["operation"], seed, request["input_cache"],
                                                    host_data=host_data.pop(seed))
            else:
                inputs = make_inputs(case, target["operation"], seed)
                fingerprint = input_digest(inputs)
            self.data[seed] = {
                "inputs": inputs, "fingerprint": fingerprint, "original":
                {name: value.clone()
                 for name, value in inputs.items()}, "expected":
                reference(case, target["operation"], inputs) if seed in target["correctness"]["seeds"] else None
            }
        torch.cuda.synchronize()
        self.case_key = key
        return self.data

    def discard_mutated_inputs(self):
        import torch

        if any(not torch.equal(sample["inputs"][key], value)
               for sample in self.data.values()
               for key, value in sample["original"].items()):
            self.data.clear()
            self.case_key = None


def measure(request, session=None):
    import torch
    import triton.testing
    from backends.seeds import build, module

    from benchmarking.inputs import relative_rms_error

    started = time.monotonic()
    torch.set_grad_enabled(False)
    configure_cpu_threads()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    session = session or MeasurementSession()
    if session.device is None:
        session.device = inspect_device()
    device = session.device
    session.idle()
    target, case, candidate = (request[key] for key in ("target", "case", "candidate"))
    operation = target["operation"]
    if candidate.get("adapter"):
        path = Path(candidate["adapter"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != candidate["adapter_sha256"]:
            raise ValueError("candidate adapter digest mismatch")
        fn = module(path, "scout_candidate").build(case, candidate["params"])
    else:
        fn = build(request["backend"], operation, case, candidate["params"])
    identity = session.identity(request["backend"])
    if request.get("probe_identity"):
        expected_probe = request["probe_identity"]
        if identity["runtime"] != expected_probe["runtime"] or identity["device"] != expected_probe["device"]:
            raise ValueError("worker runtime or device differs from its admission probe")
        if request["backend"] == "aiter" and identity["backend_pin"] != expected_probe["aiter"]:
            raise ValueError("AITER source differs from its admission probe")
    if request.get("expected_identity") and identity != request["expected_identity"]:
        raise ValueError("reproducer source/runtime/device pins differ from the recorded run")
    checks = []
    numerical_errors = {}
    input_hashes = {}
    setup_seconds = time.monotonic() - started
    preparation_started = time.monotonic()
    reused_preparation = session.case_key == digest([case, operation, target["correctness"]])
    prepared = session.prepare(request)
    preparation_seconds = time.monotonic() - preparation_started
    correctness_started = time.monotonic()
    for seed in target["correctness"]["seeds"]:
        sample = prepared[seed]
        inputs, original, expected = (sample[key] for key in ("inputs", "original", "expected"))
        input_hashes[str(seed)] = sample["fingerprint"]
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
        rms_limit = target["correctness"].get("max_relative_rms_error")
        if rms_limit is not None:
            error = relative_rms_error(actual, expected)
            numerical_errors[str(seed)] = {"relative_rms_error": error}
            if not math.isfinite(error) or error > rms_limit:
                raise ValueError(f"relative RMS error {error:g} exceeds {rms_limit:g}")
        if any(not torch.equal(inputs[key], value) for key, value in original.items()):
            raise ValueError("candidate changed a harness input")
        if any(actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr() for value in inputs.values()):
            raise ValueError("returned output must not alias an input")
        checks.append(seed)
    correctness_seconds = time.monotonic() - correctness_started
    benchmark_started = time.monotonic()
    # Restore one canonical seed for every backend's measured inputs.
    inputs = prepared[0]["inputs"]
    input_hashes["timing_seed_0"] = prepared[0]["fingerprint"]
    timing = dict(target["timing"])
    timing_phase = request.get("timing_phase", "full")
    if timing_phase == "screen":
        timing.update(warmup_ms=min(timing["warmup_ms"], 5), rep_ms=min(timing["rep_ms"], 20), repeats=3)
    elif timing_phase != "full":
        raise ValueError("unknown timing phase")
    before = conditions(device)
    samples = [
        triton.testing.do_bench(lambda: fn(inputs), warmup=timing["warmup_ms"], rep=timing["rep_ms"],
                                quantiles=[0.2, 0.5, 0.8]) for _ in range(timing["repeats"])
    ]
    torch.cuda.synchronize()
    session.idle()
    if any(not torch.equal(inputs[key], value) for key, value in prepared[0]["original"].items()):
        raise ValueError("candidate changed a timing input")
    after = conditions(device)
    medians = [sample[1] for sample in samples]
    latency = statistics.median(medians)
    spread = (max(medians) - min(medians)) / latency
    flops, logical_bytes = work(case, operation)
    return {
        "status": "ok" if spread <= timing["max_relative_spread"] else "unstable", **identity, "candidate": candidate,
        "adapter_metadata": getattr(fn, "scout_metadata",
                                    {}), "contract_sha256": digest({"target": target, "case":
                                                                    case}), "correctness_seeds": checks,
        "numerical_errors": numerical_errors, "input_sha256": digest(input_hashes), "latency_ms": latency, "p20_ms":
        statistics.median(s[0] for s in samples), "p80_ms": statistics.median(s[2] for s in samples), "samples_ms":
        samples, "relative_spread": spread, "tflops": flops / (latency * 1e9), "logical_tbps":
        logical_bytes / (latency * 1e9), "timing_phase": timing_phase, "timing": timing, "conditions":
        {"before": before, "after":
         after}, "pid": os.getpid(), "preparation_reused": reused_preparation, "phase_seconds": {
             "setup": setup_seconds, "preparation": preparation_seconds, "correctness": correctness_seconds,
             "benchmark_and_checks": time.monotonic() - benchmark_started
         }
    }


def measure_to_file(request, output, session=None):
    try:
        result = measure(request, session)
    except Exception as error:  # noqa: BLE001 - candidate failures become explicit artifacts
        traceback.print_exc()
        if session is not None:
            session.discard_mutated_inputs()
        status = "unavailable" if isinstance(error, ModuleNotFoundError) else ("unsupported" if isinstance(
            error, NotImplementedError) or type(error).__name__ in {"UnsupportedOp", "InvalidInput"} else "failed")
        result = {"status": status, "error": f"{type(error).__name__}: {error}"}
    write_json(output, result)
    return 0 if result["status"] in {"ok", "unstable"} else 1


def serve(response_fd=None):
    """Execute serialized requests. Only stdout acknowledgements belong to the protocol."""
    protocol = os.fdopen(response_fd, "w", buffering=1) if response_fd is not None else sys.stdout
    session = MeasurementSession()
    for line in sys.stdin:
        message = json.loads(line)
        session.allowed_pids = message["peers"]
        with contextlib.redirect_stdout(sys.stderr):
            if message.get("verify"):
                session.verify()
                code = 0
            else:
                print(f"REQUEST {message['request']}", file=sys.stderr, flush=True)
                request = read_json(message["request"])
                request["probe_identity"] = message.get("expected")
                code = measure_to_file(request, message["output"], session)
                if session.device is not None:
                    import torch

                    torch.cuda.synchronize()
        protocol.write(json.dumps({"exit_code": code}) + "\n")
        protocol.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--response-fd", type=int)
    args = parser.parse_args()
    if args.serve:
        serve(args.response_fd)
        return 0
    if args.probe:
        try:
            device = inspect_device()
            assert_idle(device)
            identity = {"device": device, "runtime": runtime_identity(), "aiter": package_identity("aiter")}
            assert_idle(device)
        except GPUOccupiedError as error:
            write_json(args.probe, {"status": "busy", "error": str(error)})
            return 2
        write_json(args.probe, identity)
        return 0
    if args.request is None or args.output is None:
        parser.error("request and output are required for a measurement")
    request = read_json(args.request)
    with tempfile.TemporaryDirectory(prefix="scout-inputs-") as directory:
        if request.get("input_cache"):
            request["input_cache"] = directory
        return measure_to_file(request, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
