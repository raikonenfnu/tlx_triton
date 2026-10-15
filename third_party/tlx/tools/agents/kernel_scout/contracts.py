"""Validated target and backend contracts. This module does not import GPU libraries."""

import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[4]
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
OPERATIONS = {"mm", "addmm", "bmm", "flash_attn", "flash_attn_varlen", "flash_attn_mxfp8"}
GEMM_OPERATIONS = {"mm", "addmm", "bmm"}
DENSE_ATTENTION = {"flash_attn", "flash_attn_mxfp8"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def harness_digest():
    paths = [
        ROOT / "contracts.py", ROOT / "worker_pool.py", ROOT / "backends/seeds.py", *sorted(
            (ROOT / "benchmarking").glob("*.py"))
    ]
    return digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


def attention_scale(case):
    return case.get("sm_scale", case["head_dim"]**-0.5)


def load_target(path):
    target = read_json(path)
    if target.get("schema_version") != 1 or not NAME.fullmatch(target.get("id", "")):
        raise ValueError("target needs schema_version=1 and a valid id")
    if target.get("operation") not in OPERATIONS:
        raise ValueError("unsupported operation")
    if target.get("hardware", {}).get("vendor") not in {"amd", "nvidia"}:
        raise ValueError("hardware.vendor must be amd or nvidia")
    if not target["hardware"].get("architectures"):
        raise ValueError("hardware.architectures must not be empty")
    if target.get("output_policy") != "returned_tensor" or target.get("metric") not in {"compute", "memory"}:
        raise ValueError("use returned_tensor output and compute or memory metric")
    timing = target["timing"]
    for key in ("warmup_ms", "rep_ms", "repeats", "max_relative_spread", "tie_fraction"):
        positive(timing[key], key)
    if not isinstance(timing["repeats"], int) or timing["repeats"] < 3:
        raise ValueError("at least three independent timing repeats are required")
    for key in ("atol", "rtol"):
        value = target["correctness"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not max(target["correctness"][key] for key in ("atol", "rtol")):
        raise ValueError("at least one correctness tolerance must be positive")
    if "max_relative_rms_error" in target["correctness"]:
        positive(target["correctness"]["max_relative_rms_error"], "maximum relative RMS error")
    seeds = target["correctness"]["seeds"]
    if len(set(seeds)) < 2 or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("at least two distinct nonnegative correctness seeds are required")
    cases = target.get("cases", [])
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("cases must have distinct ids")
    for case in cases:
        if not NAME.fullmatch(case["id"]) or case["dtype"] not in {"float16", "bfloat16"}:
            raise ValueError("invalid case id or dtype")
        operation = target["operation"]
        keys = ("m", "n", "k") if operation in GEMM_OPERATIONS else ("heads", "kv_heads", "head_dim")
        for key in keys:
            if not isinstance(case[key], int):
                raise TypeError(f"{key} must be an integer")
            positive(case[key], key)
        for key in ("a_std", "b_std", "bias_std", "qkv_std"):
            if key in case:
                positive(case[key], key)
        if operation in GEMM_OPERATIONS:
            if case.get("a_layout", "row_major") not in {"row_major", "column_major"
                                                         } or case["b_layout"] not in {"row_major", "column_major"}:
                raise ValueError("invalid A or B layout")
            if operation == "addmm" and (case.get("bias_layout") != "vector" or any(
                    type(case.get(key, 1)) not in {int, float} or case.get(key, 1) != 1 for key in ("alpha", "beta"))):
                raise ValueError("addmm requires a vector bias with alpha=beta=1")
            if operation == "bmm":
                if type(case.get("batch")) is not int:
                    raise ValueError("bmm batch must be an integer")
                positive(case["batch"], "batch")
                if type(case.get("shared_a")) is not bool:
                    raise ValueError("bmm must declare shared_a")
        else:
            if case["heads"] % case["kv_heads"] or type(case["causal"]) is not bool:
                raise ValueError("invalid GQA or causal contract")
            q, k = case["q_lengths"], case["kv_lengths"]
            if not q or len(q) != len(k) or any(type(n) is not int or n < 0 for n in q + k):
                raise ValueError("sequence lengths must be paired nonnegative integers")
            if not sum(q) or not sum(k):
                raise ValueError("the target must contain query and key tokens")
            positive(attention_scale(case), "attention scale")
            if operation in DENSE_ATTENTION and (len(set(q + k)) != 1 or q != k):
                raise ValueError("dense attention requires equal sequence lengths")
            if operation == "flash_attn_mxfp8" and (case["dtype"] != "bfloat16" or case["head_dim"] != 128
                                                    or case["heads"] != case["kv_heads"] or q[0] % 256):
                raise ValueError("MXFP8 attention requires BF16, D=128, equal heads, and lengths divisible by 256")
    return target


def backends():
    result = {}
    for path in sorted((ROOT / "backends").glob("*/backend.json")):
        backend = read_json(path)
        name = backend["id"]
        if not NAME.fullmatch(name) or name in result or name != path.parent.name:
            raise ValueError("backend ids must be distinct directory names")
        if not set(backend["operations"]) <= OPERATIONS or not backend["vendors"]:
            raise ValueError(f"invalid backend capabilities: {name}")
        if backend.get("cadence", "weekly") not in {"weekly", "nightly"}:
            raise ValueError(f"invalid backend cadence: {name}")
        if not (path.parent / "instructions.md").is_file():
            raise ValueError(f"missing backend instructions: {name}")
        result[name] = backend
    return result


def applicable(backend, target, device):
    return (device["vendor"] in backend["vendors"]
            and (not backend["architectures"] or device["arch"] in backend["architectures"])
            and target["operation"] in backend["operations"])


def work(case, operation):
    """Useful FLOPs and logical input/output bytes, not hardware counters."""
    if operation in GEMM_OPERATIONS:
        m, n, k = (case[key] for key in ("m", "n", "k"))
        batch = case["batch"] if operation == "bmm" else 1
        a_count = m * k if operation != "bmm" or case["shared_a"] else batch * m * k
        bias_count = n if operation == "addmm" else 0
        # Keep the conventional GEMM FLOP count; bias traffic remains included.
        return 2 * batch * m * n * k, 2 * (a_count + batch * (k * n + m * n) + bias_count)
    pairs = 0
    for q, k in zip(case["q_lengths"], case["kv_lengths"]):
        pairs += sum(max(0, min(k, i + k - q + 1)) for i in range(q)) if case["causal"] else q * k
    h, hk, d = (case[key] for key in ("heads", "kv_heads", "head_dim"))
    logical_bytes = 2 * d * (2 * sum(case["q_lengths"]) * h + 2 * sum(case["kv_lengths"]) * hk)
    return 4 * h * d * pairs, logical_bytes


def enforce_equal_inputs(results):
    # An all-unstable search summary has no measurement or input fingerprint.
    measured = [
        r for r in results.values() if r.get("status") == "ok" or (r.get("status") == "unstable" and "latency_ms" in r)
    ]
    hashes = {r.get("input_sha256") for r in measured}
    if None in hashes or len(hashes) > 1:
        for result in measured:
            result.update(status="failed",
                          error="backend input fingerprints differ or are missing. Comparison rejected")


def select_winner(results, tie_fraction):
    valid = [(name, r) for name, r in results.items() if r.get("status") == "ok"]
    if not valid:
        return "none"
    valid.sort(key=lambda entry: entry[1]["latency_ms"])
    _, best = valid[0]
    # Timing distributions and the target's minimum separation both limit a claim.
    limit = max(best["p80_ms"], best["latency_ms"] * (1 + tie_fraction))
    tied = [name for name, r in valid if r["p20_ms"] <= limit]
    return " / ".join(tied) + (" (tie)" if len(tied) > 1 else "")
