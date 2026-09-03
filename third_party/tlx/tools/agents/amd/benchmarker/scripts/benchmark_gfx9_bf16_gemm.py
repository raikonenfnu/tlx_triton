#!/usr/bin/env python3
"""Correctness-gated search over TLX gfx9 BF16 GEMM implementations."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import traceback
from typing import Callable

import torch
import triton
from triton.compiler.errors import CompilationError

from triton.language.extra.tlx.tutorials.amd_addmm_gfx950 import (
    addmm as tlx_addmm,
    available_paths as addmm_paths,
)
from triton.language.extra.tlx.tutorials.amd_bmm_shared_a import (
    bmm as shared_a_row_bmm,
    make_bmm_inputs as make_shared_a_row_bmm_inputs,
)
from triton.language.extra.tlx.tutorials.amd_gemm_pipelined import matmul as pipelined_matmul
from triton.language.extra.tlx.tutorials.amd_gemm_warp_pipeline import matmul as warp_pipeline_matmul
from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a16w16.matmul_kernel import (
    _has_streamk_schedule,
    _launch_register,
    matmul as interwave_matmul,
    streamk_matmul,
)
from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a16w16.matmul_kernel_split_m import (
    matmul as pingpong_matmul,
)
from triton.language.extra.tlx.tutorials.gfx9_gemm.a16w16.v9_beyond_hotloop.matmul_kernel import (
    matmul as v9_matmul,
)


@dataclasses.dataclass(frozen=True)
class Shape:
    name: str
    op: str
    m: int
    n: int
    k: int
    layout_a: str
    layout_b: str
    weight: int | float = 1
    batch: int = 1


@dataclasses.dataclass(frozen=True)
class Candidate:
    name: str
    supports: Callable[[Shape], tuple[bool, str]]
    build: Callable[[torch.Tensor | None, torch.Tensor, torch.Tensor], Callable[[], torch.Tensor]]


def _yes(_: Shape) -> tuple[bool, str]:
    return True, ""


def _mm_only(shape: Shape) -> tuple[bool, str]:
    return (shape.op == "mm", "plain GEMM candidate")


def _pipelined_support(shape: Shape) -> tuple[bool, str]:
    return (shape.op == "mm" and shape.layout_a == "row", "requires row-major A")


def _warp_support(shape: Shape) -> tuple[bool, str]:
    valid = (shape.op == "mm" and shape.layout_a == "row" and shape.layout_b == "row" and shape.m % 256 == 0
             and shape.n % 256 == 0 and shape.k % 64 == 0)
    return valid, "requires mm, row/row, and M/N/K divisible by 256/256/64"


def _pingpong_support(shape: Shape) -> tuple[bool, str]:
    valid = (shape.op == "mm" and shape.layout_a == "row" and shape.m % 256 == 0 and shape.n % 256 == 0
             and shape.k >= 128 and shape.k % 64 == 0)
    return valid, "requires mm, row-major A, M/N divisible by 256, and aligned K"


def _streamk_support(shape: Shape) -> tuple[bool, str]:
    return shape.op == "mm", "plain GEMM candidate"


def _interwave_support(shape: Shape) -> tuple[bool, str]:
    return shape.op == "mm", "plain GEMM candidate"


def _v9_support(shape: Shape) -> tuple[bool, str]:
    return shape.op == "mm", "plain GEMM candidate"


def _addmm_path_support(path: str) -> Callable[[Shape], tuple[bool, str]]:
    def supports(shape: Shape) -> tuple[bool, str]:
        if shape.op != "addmm":
            return False, "fused addmm candidate"
        if shape.layout_a != "row" or shape.layout_b != "column":
            return False, "gfx950 addmm requires row-major A and column-major B"
        if path != "register" and max(shape.m * shape.k, shape.k * shape.n) * 2 >= 2**31:
            return False, "inter-wave direct-to-LDS operands must each fit signed 32-bit byte offsets"
        if path == "inter_wave_tail":
            valid = (shape.k > 1536 and shape.k % 64 != 0 and shape.k * 2 % 16 == 0
                     and 2 * 1024 * 1024 < shape.m * shape.n <= 16 * 1024 * 1024)
            return valid, "inter-wave tail path is reserved for large partial-K outputs"
        if path == "stream_k":
            return _has_streamk_schedule(shape.m, shape.n, shape.k), "requires a profitable Stream-K tail"
        if path == "inter_wave" and not (shape.k >= 128 and shape.k % 64 == 0):
            return False, "inter-wave path requires K >= 128 and divisible by 64"
        return True, f"path {path} must be listed by available_paths"

    return supports


def _addmm_builder(path: str):
    def build(bias, a, b):
        assert bias is not None
        paths = addmm_paths(torch.broadcast_to(bias, (a.shape[0], b.shape[1])), a, b)
        if path not in paths:
            raise NotImplementedError(f"available paths are {paths}")
        return lambda: tlx_addmm(bias, a, b, path=path)

    return build


MM_CANDIDATES = (
    Candidate("register", _mm_only, lambda _bias, a, b: lambda: _launch_register(a, b)),
    Candidate("pipelined", _pipelined_support, lambda _bias, a, b: lambda: pipelined_matmul(a, b)),
    Candidate("warp_pipeline", _warp_support, lambda _bias, a, b: lambda: warp_pipeline_matmul(a, b)),
    Candidate("interwave", _interwave_support, lambda _bias, a, b: lambda: interwave_matmul(a, b)),
    Candidate("streamk", _streamk_support, lambda _bias, a, b: lambda: streamk_matmul(a, b)),
    Candidate("v9", _v9_support, lambda _bias, a, b: lambda: v9_matmul(a, b)),
    Candidate("pingpong", _pingpong_support, lambda _bias, a, b: lambda: pingpong_matmul(a, b)),
)
ADDMM_CANDIDATES = (
    Candidate("addmm_register", _addmm_path_support("register"), _addmm_builder("register")),
    Candidate("addmm_interwave", _addmm_path_support("inter_wave"), _addmm_builder("inter_wave")),
    Candidate("addmm_streamk", _addmm_path_support("stream_k"), _addmm_builder("stream_k")),
    Candidate("addmm_interwave_tail", _addmm_path_support("inter_wave_tail"), _addmm_builder("inter_wave_tail")),
)
BMM_CANDIDATES = (
    Candidate("bmm_shared_a_row", lambda shape: (shape.op == "bmm" and shape.layout_a == "shared-row"
                                                  and shape.layout_b == "row",
                                                  "requires shared row-major A and row-major B"),
              lambda _bias, a, b: lambda: shared_a_row_bmm(a, b)),
)


def _run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _make_matrix(rows: int, cols: int, layout: str, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    scale = cols**-0.5
    if layout == "row":
        return torch.randn((rows, cols), device=device, dtype=torch.bfloat16, generator=generator).mul_(scale)
    if layout == "column":
        return torch.randn((cols, rows), device=device, dtype=torch.bfloat16, generator=generator).mul_(scale).T
    raise ValueError(f"unknown layout {layout!r}")


def _reference(shape: Shape, bias, a, b):
    if shape.op == "mm":
        return torch.matmul(a, b)
    if shape.op == "addmm":
        assert bias is not None
        return torch.addmm(bias, a, b)
    if shape.op == "bmm":
        return torch.bmm(a, b)
    raise ValueError(f"unknown operation {shape.op!r}")


def _timing_samples(fn, warmup_ms: int, rep_ms: int, repeats: int, max_warmup_launches: int,
                    max_timed_launches: int) -> tuple[list[float], int, int]:
    estimate_ms = float(triton.testing.do_bench(fn, warmup=5, rep=20, return_mode="median"))
    warmup_launches = min(max_warmup_launches, max(1, round(warmup_ms / estimate_ms)))
    timed_launches = min(max_timed_launches, max(1, round(rep_ms / estimate_ms)))
    driver = triton.runtime.driver.active
    device_interface = driver.get_device_interface()
    cache = driver.get_empty_cache_for_benchmark()
    samples = []
    for _ in range(repeats):
        for _ in range(warmup_launches):
            fn()
        device_interface.synchronize()
        starts = [device_interface.Event(enable_timing=True) for _ in range(timed_launches)]
        ends = [device_interface.Event(enable_timing=True) for _ in range(timed_launches)]
        for start, end in zip(starts, ends):
            driver.clear_cache(cache)
            start.record()
            fn()
            end.record()
        device_interface.synchronize()
        samples.append(float(statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))))
    return samples, warmup_launches, timed_launches


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor, rtol: float) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    required_atol = torch.clamp(delta - rtol * expected.float().abs(), min=0).max()
    return {"max_abs": float(delta.max()), "required_atol_at_rtol": float(required_atol)}


def _tflops(shape: Shape, ms: float) -> float:
    return 2.0 * shape.batch * shape.m * shape.n * shape.k / (ms * 1e9)


def _candidate_config(name: str, shape: Shape) -> dict | None:
    # Triton's autotuner caches are intentionally not a public API. Capture a
    # stable representation when this version exposes one, but never make the
    # benchmark depend on it.
    kernel = None
    if name in {"register", "addmm_register"}:
        from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a16w16 import matmul_kernel
        kernel = matmul_kernel._register_kernel
    elif name == "pipelined":
        from triton.language.extra.tlx.tutorials import amd_gemm_pipelined
        kernel = amd_gemm_pipelined.matmul_kernel_pipelined_mi300
    if name == "warp_pipeline":
        return {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 16, "NUM_BUFFERS": 2,
                "num_warps": 8}
    if name == "pingpong":
        return {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4, "NUM_BUFFERS": 2,
                "num_warps": 8}
    if name in {"interwave", "addmm_interwave"}:
        from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a16w16.matmul_kernel import (
            _aligned_split_tail_plan,
            choose_tile,
        )
        tail_plan = _aligned_split_tail_plan(shape.m, shape.n, shape.k)
        if tail_plan is not None:
            prefix, split_k = tail_plan
            return {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "SPLIT_K": split_k,
                    "K_PREFIX": prefix, "K_TAIL": shape.k - prefix, "num_warps": 8}
        block_m, block_n, split_k = choose_tile(shape.m, shape.n, shape.k)
        return {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": 64, "SPLIT_K": split_k,
                "num_warps": 8}
    if name == "v9":
        return {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 8}
    if name == "bmm_shared_a_row":
        return {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "NUM_BUFFERS": 3, "num_warps": 8,
                "path": "direct" if shape.k % 32 == 0 else "register"}
    if kernel is None:
        return None
    value = getattr(kernel, "best_config", None)
    if value:
        result = dict(value.kwargs)
        for attr in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
            field = getattr(value, attr, None)
            if field is not None:
                result[attr] = field
        return result
    return None


def benchmark_shape(shape: Shape, args, device: torch.device) -> dict:
    batch = f" batch={shape.batch}" if shape.op == "bmm" else ""
    print(f"\n[{shape.name}] {shape.op} {shape.m}x{shape.n}x{shape.k}{batch} "
          f"{shape.layout_a}/{shape.layout_b}", flush=True)
    if shape.op == "bmm":
        a, b = make_shared_a_row_bmm_inputs(shape.batch, shape.m, shape.n, shape.k, device,
                                             dtype=torch.bfloat16, seed=args.seed)
    else:
        a = _make_matrix(shape.m, shape.k, shape.layout_a, device, args.seed)
        b = _make_matrix(shape.k, shape.n, shape.layout_b, device, args.seed + 1)
    print(f"  inputs: A shape={tuple(a.shape)} stride={a.stride()}", flush=True)
    print(f"  inputs: B shape={tuple(b.shape)} stride={b.stride()}", flush=True)
    bias = (_make_matrix(1, shape.n, "row", device, args.seed + 2).reshape(shape.n) if shape.op == "addmm" else None)
    ref_fn = lambda: _reference(shape, bias, a, b)
    expected = ref_fn()
    torch.cuda.synchronize(device)
    print("  reference: correctness output ready", flush=True)
    candidates = {"mm": MM_CANDIDATES, "addmm": ADDMM_CANDIDATES, "bmm": BMM_CANDIDATES}[shape.op]
    if args.providers:
        requested = set(args.providers)
        candidates = tuple(candidate for candidate in candidates if candidate.name in requested)

    results = []
    torch_samples, torch_warmup_launches, torch_timed_launches = _timing_samples(
        ref_fn, args.warmup_ms, args.rep_ms, args.repeats, args.max_warmup_launches, args.max_timed_launches)
    torch_ms = statistics.median(torch_samples)
    torch_result = {
        "name": {"mm": "torch.matmul", "addmm": "torch.addmm", "bmm": "torch.bmm"}[shape.op],
        "status": "correct",
        "samples_ms": torch_samples,
        "warmup_launches": torch_warmup_launches,
        "timed_launches": torch_timed_launches,
        "median_ms": torch_ms,
        "tflops": _tflops(shape, torch_ms),
    }
    print(f"  {torch_result['name']:<22} {torch_ms:9.4f} ms  {torch_result['tflops']:8.2f} TF/s", flush=True)

    for candidate in candidates:
        supported, reason = candidate.supports(shape)
        result = {"name": candidate.name}
        if not supported:
            result.update(status="unsupported", reason=reason)
            results.append(result)
            print(f"  {candidate.name:<22} unsupported: {reason}", flush=True)
            continue
        try:
            fn = candidate.build(bias, a, b)
            actual = fn()
            torch.cuda.synchronize(device)
            metrics = _error_metrics(actual, expected, args.rtol)
            correct = torch.allclose(actual, expected, atol=args.atol, rtol=args.rtol)
            result.update(metrics)
            if not correct:
                result.update(status="incorrect", reason=f"torch.allclose(atol={args.atol}, rtol={args.rtol}) failed")
                results.append(result)
                print(f"  {candidate.name:<22} INCORRECT max_abs={metrics['max_abs']:.6g}", flush=True)
                del actual
                continue
            del actual
            samples, warmup_launches, timed_launches = _timing_samples(
                fn, args.warmup_ms, args.rep_ms, args.repeats, args.max_warmup_launches, args.max_timed_launches)
            median_ms = statistics.median(samples)
            result.update(
                status="correct",
                samples_ms=samples,
                warmup_launches=warmup_launches,
                timed_launches=timed_launches,
                median_ms=median_ms,
                tflops=_tflops(shape, median_ms),
                speedup_vs_torch=torch_ms / median_ms,
                config=_candidate_config(candidate.name, shape),
            )
            print(f"  {candidate.name:<22} {median_ms:9.4f} ms  {result['tflops']:8.2f} TF/s  "
                  f"{result['speedup_vs_torch']:.3f}x", flush=True)
        except (AssertionError, CompilationError, NotImplementedError, RuntimeError, ValueError) as error:
            result.update(status="failed", reason=f"{type(error).__name__}: {error}")
            if args.tracebacks:
                result["traceback"] = traceback.format_exc()
            print(f"  {candidate.name:<22} failed: {result['reason']}", flush=True)
        results.append(result)

    correct = [result for result in results if result["status"] == "correct"]
    best = min(correct, key=lambda result: result["median_ms"]) if correct else None
    torch.cuda.empty_cache()
    return {
        **dataclasses.asdict(shape),
        "a_stride": list(a.stride()),
        "b_stride": list(b.stride()),
        "torch": torch_result,
        "candidates": results,
        "best": best,
    }


def _markdown(report: dict, command: str) -> str:
    env = report["environment"]
    suite_metadata = report.get("suite_metadata", {})
    weighted_torch = sum(item["weight"] * item["torch"]["median_ms"] for item in report["results"])
    weighted_tlx = sum(item["weight"] * item["best"]["median_ms"] for item in report["results"] if item["best"])
    decisive_tlx_wins = sum(1 for item in report["results"] if item["best"] and item["best"]["speedup_vs_torch"] >= 1.01)
    aggregate = weighted_tlx / weighted_torch
    if decisive_tlx_wins:
        recommendation = (f"TLX clears the 1% decision threshold on {decisive_tlx_wins} shape(s); the best "
                          f"correct TLX path takes `{aggregate:.3f}x` PyTorch time on the supplied-weight aggregate.")
    else:
        recommendation = ("Use the PyTorch operation for this suite. No TLX winner clears the 1% decision "
                          f"threshold; the best correct TLX path takes `{aggregate:.3f}x` PyTorch time on the "
                          "supplied-weight aggregate.")
    lines = [
        f"# gfx950 BF16 GEMM search: {report['suite_name']}",
        "",
        "## Environment",
        "",
        f"- GPU: `{env['gpu']}` (`{env['target']}`)",
        f"- TLX/Triton commit: `{env['git_commit']}`",
        f"- Triton: `{env['triton_version']}` from `{env['triton_path']}`",
        f"- PyTorch: `{env['torch_version']}`; HIP: `{env['hip_version']}`",
        f"- GPU clock lock: `{env['clock_locked']}`",
        f"- Timing: median of {report['policy']['repeats']} independent HIP event batches targeting "
        f"{report['policy']['rep_ms']} ms each, capped at {report['policy']['max_timed_launches']} launches; "
        f"L2 is flushed before each timed launch and warmup is capped at "
        f"{report['policy']['max_warmup_launches']} launches",
        f"- Correctness: BF16 `torch.allclose(atol={report['policy']['atol']}, rtol={report['policy']['rtol']})`",
    ]
    if source := suite_metadata.get("source"):
        lines.append(f"- Suite source: `{source}`")
    if conventions := suite_metadata.get("layout_convention"):
        lines.extend(f"- {op} layout convention: {description}" for op, description in conventions.items())
    lines.extend([
        "", "## Recommendation", "", recommendation, "", "## Best result per shape", "",
        "| Shape | Op/layout | PyTorch ms | PyTorch TF/s | Best TLX path | TLX ms | TLX TF/s | Speedup |",
        "|---|---:|---:|---:|---|---:|---:|---:|"
    ])
    for item in report["results"]:
        best = item["best"]
        if best is None:
            best_cols = ("none", "-", "-", "-")
        else:
            best_cols = (best["name"], f"{best['median_ms']:.4f}", f"{best['tflops']:.2f}",
                         f"{best['speedup_vs_torch']:.3f}x")
        batch = f" b={item['batch']}" if item["op"] == "bmm" else ""
        lines.append(
            f"| `{item['name']}` | {item['op']}{batch} {item['layout_a']}/{item['layout_b']} | "
            f"{item['torch']['median_ms']:.4f} | {item['torch']['tflops']:.2f} | {best_cols[0]} | "
            f"{best_cols[1]} | {best_cols[2]} | {best_cols[3]} |")
    lines.extend(["", "## Best TLX configurations", "",
                  "| Shape | Path | Configuration |", "|---|---|---|"])
    for item in report["results"]:
        best = item["best"]
        if best is None:
            lines.append(f"| `{item['name']}` | none | - |")
            continue
        config = best.get("config")
        if config is None:
            config = _candidate_config(best["name"], Shape(**{field.name: item[field.name]
                                                               for field in dataclasses.fields(Shape)}))
        config_text = json.dumps(config, sort_keys=True, separators=(",", ":")) if config else "internal fixed defaults"
        lines.append(f"| `{item['name']}` | {best['name']} | `{config_text}` |")
    lines.extend(["", "## Candidate details", ""])
    for item in report["results"]:
        lines.append(f"### `{item['name']}`")
        lines.append("")
        lines.append("| Candidate | Status | Median ms | TF/s | vs PyTorch | Error / exclusion |")
        lines.append("|---|---|---:|---:|---:|---|")
        for result in item["candidates"]:
            lines.append(
                f"| {result['name']} | {result['status']} | {result.get('median_ms', '-')} | "
                f"{result.get('tflops', '-')} | {result.get('speedup_vs_torch', '-')} | "
                f"{result.get('reason', '')} |")
        lines.append("")
    lines.extend(["## Reproducer", "", "```bash", command, "```", "",
                  "The JSON artifact produced beside this report contains raw timing samples, strides, correctness "
                  "metrics, candidate failures, and any exposed autotuner configuration."])
    return "\n".join(lines) + "\n"


def _clock_locked() -> str:
    output = _run(["amd-smi", "metric", "-g", "0"])
    for line in output.splitlines():
        if "CLK_LOCKED:" in line:
            return line.split("CLK_LOCKED:", 1)[1].strip()
    return "unknown"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shape", action="append", help="run only the named shape (repeatable)")
    parser.add_argument("--providers", nargs="+", help="limit TLX candidates")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-warmup-launches", type=int, default=25)
    parser.add_argument("--max-timed-launches", type=int, default=500)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip shapes already checkpointed in output-dir/results.json")
    parser.add_argument("--tracebacks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.warmup_ms, args.rep_ms, args.repeats = 5, 20, 1
    payload = json.loads(args.suite.read_text())
    if isinstance(payload, dict):
        suite_name = payload.get("name", args.suite.stem)
        suite_metadata = {key: value for key, value in payload.items() if key != "shapes"}
        payload = payload["shapes"]
    else:
        suite_name = args.suite.stem
        suite_metadata = {}
    shapes = [Shape(**item) for item in payload]
    if args.shape:
        selected = set(args.shape)
        shapes = [shape for shape in shapes if shape.name in selected]
        missing = selected - {shape.name for shape in shapes}
        if missing:
            raise SystemExit(f"unknown shape names: {sorted(missing)}")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or not target.arch.startswith("gfx9"):
        raise SystemExit(f"gfx9 HIP target required, got {target}")
    device = triton.runtime.driver.active.get_active_torch_device()
    repo = Path(__file__).resolve().parents[7]
    imported = Path(triton.__file__).resolve()
    if repo not in imported.parents:
        raise SystemExit(f"imported Triton is not from this checkout: {imported}")

    command = " ".join(["ROCR_VISIBLE_DEVICES=" + os.environ.get("ROCR_VISIBLE_DEVICES", "0"),
                        "TRITON_CACHE_DIR=" + os.environ.get("TRITON_CACHE_DIR", "<fresh-cache>"),
                        sys.executable, *sys.argv])
    report = {
        "schema_version": 1,
        "suite_name": suite_name,
        "suite_metadata": suite_metadata,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "gpu": torch.cuda.get_device_name(device),
            "target": target.arch,
            "git_commit": _run(["git", "-C", str(repo), "rev-parse", "HEAD"]),
            "triton_version": triton.__version__,
            "triton_path": str(imported),
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "python": platform.python_version(),
            "clock_locked": _clock_locked(),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", "default"),
        },
        "policy": {
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "repeats": args.repeats,
            "max_warmup_launches": args.max_warmup_launches,
            "max_timed_launches": args.max_timed_launches,
            "atol": args.atol,
            "rtol": args.rtol,
            "seed": args.seed,
        },
        "results": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text())
        comparable = (previous.get("suite_name") == report["suite_name"] and previous.get("policy") == report["policy"]
                      and previous.get("environment", {}).get("git_commit") == report["environment"]["git_commit"])
        if not comparable:
            raise SystemExit("cannot resume: suite, policy, or git commit differs from checkpoint")
        report["results"] = previous["results"]
    completed = {result["name"] for result in report["results"]}
    for shape in shapes:
        if shape.name in completed:
            print(f"[{shape.name}] already complete; skipping", flush=True)
            continue
        report["results"].append(benchmark_shape(shape, args, device))
        result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = _markdown(report, command)
    (args.output_dir / "issue.md").write_text(markdown)
    print(f"\nWrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'issue.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
