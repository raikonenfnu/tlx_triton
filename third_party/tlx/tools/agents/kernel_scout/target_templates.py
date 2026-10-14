"""Create validated starter targets without importing GPU libraries."""

from copy import deepcopy

from contracts import NAME, OPERATIONS, load_target, write_json

CASES = {
    "mm": {"id": "square_512", "dtype": "float16", "m": 512, "n": 512, "k": 512, "b_layout": "column_major"},
    "flash_attn": {
        "id": "dense_1024", "dtype": "bfloat16", "q_lengths": [1024], "kv_lengths": [1024], "heads": 4, "kv_heads": 4,
        "head_dim": 128, "causal": False
    },
    "flash_attn_varlen": {
        "id": "ragged_small", "dtype": "float16", "q_lengths": [33, 65], "kv_lengths": [33, 65], "heads": 1, "kv_heads":
        1, "head_dim": 128, "causal": False
    },
}


def create_target(path, *, name, operation, vendor, architecture, metric):
    if not NAME.fullmatch(name) or operation not in OPERATIONS:
        raise ValueError("use a valid target id and a supported operation")
    if vendor not in {"amd", "nvidia"} or not architecture.strip() or metric not in {"compute", "memory"}:
        raise ValueError("invalid hardware or metric")
    if path.exists():
        raise FileExistsError(f"target already exists: {path}")
    target = {
        "schema_version": 1,
        "id": name,
        "operation": operation,
        "hardware": {"vendor": vendor, "architectures": [architecture]},
        "output_policy": "returned_tensor",
        "metric": metric,
        "timing": {"warmup_ms": 25, "rep_ms": 100, "repeats": 5, "max_relative_spread": 0.15, "tie_fraction": 0.03},
        "correctness": {"seeds": [17, 29], "rtol": 0.02, "atol": 0.02},
        "skills": ["benchmark-fairness"],
        "cases": [deepcopy(CASES[operation])],
    }
    write_json(path, target)
    return load_target(path)
