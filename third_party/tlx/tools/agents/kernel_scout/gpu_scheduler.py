"""Wait for GPU ownership to clear before starting a benchmark job."""

import json
import math
import time

from benchmarking.environment import GPUOccupiedError, assert_idle, command
from contracts import write_json


def busy_reason(physical, vendor):
    device = {"vendor": vendor, "physical_gpu": physical}
    if vendor == "amd":
        entries = json.loads(command(["/opt/rocm/bin/amd-smi", "list", "--json"]))
        entry = next((row for row in entries if int(row["gpu"]) == physical), None)
        if entry is None:
            raise RuntimeError(f"AMD SMI device {physical} is unavailable")
        device["kfd_id"] = entry["kfd_id"]
    elif vendor != "nvidia":
        raise ValueError(f"unsupported GPU vendor: {vendor}")
    try:
        assert_idle(device)
    except RuntimeError as error:
        # Only the ownership diagnostic is retryable. Inspection errors fail closed.
        if str(error).startswith("selected GPU has external processes:"):
            return str(error)
        raise
    return None


def wait_for_gpu(physical, vendor, *, wait_seconds=1800, poll_seconds=60, status_path=None, ready=None):
    """Poll ownership and an optional runtime probe until ready or the deadline expires."""
    if not math.isfinite(wait_seconds) or wait_seconds < 0:
        raise ValueError("GPU wait seconds must be finite and nonnegative")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("GPU poll seconds must be finite and positive")
    started = time.monotonic()
    deadline, next_check = started + wait_seconds, started
    checks = 0
    while True:
        checks += 1
        value = None
        try:
            reason = busy_reason(physical, vendor)
            if reason is None and ready is not None:
                try:
                    value = ready()
                except GPUOccupiedError as error:
                    reason = str(error)
        except Exception as error:
            if status_path:
                write_json(status_path,
                           {"state": "failed", "physical_gpu": physical, "checks": checks, "error": str(error)})
            raise
        remaining = max(0, deadline - time.monotonic())
        state = "ready" if reason is None else ("waiting" if remaining else "timed_out")
        if status_path:
            write_json(
                status_path, {
                    "state": state, "physical_gpu": physical, "checks": checks, "remaining_seconds": remaining,
                    "reason": reason
                })
        if reason is None:
            print(f"GPU {physical} is idle. Starting the benchmark job.", flush=True)
            return value
        if not remaining:
            raise TimeoutError(f"GPU {physical} stayed busy for {wait_seconds:g}s: {reason}")
        next_check = min(next_check + poll_seconds, deadline)
        delay = max(0, next_check - time.monotonic())
        print(f"GPU {physical} is busy. Next check in {delay:g}s. {reason}", flush=True)
        time.sleep(delay)
