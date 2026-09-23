"""Benchmark TLX+Origami mm/addmm and the suite BMM against torch on gfx950.

The named suites reproduce the in-scope shapes and layouts from tlx_triton
issues #20, #21, and #22.  The BMM case exercises the existing optimized
shared-A tutorial kernel; it is coverage for issue #21, not part of the
Origami mm/addmm registry.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import statistics
from pathlib import Path

import torch
import triton
from triton.language.extra.tlx.tutorials.amd_bmm_shared_a import (
    bmm as tlx_shared_a_bmm,
)
from triton.language.extra.tlx.tutorials.amd_bmm_shared_a import (
    make_bmm_inputs,
)
from triton.tlx.ops import addmm as tlx_addmm
from triton.tlx.ops import mm as tlx_mm
from triton.tlx.ops.kernels.mm import gfx950


@dataclasses.dataclass(frozen=True)
class Case:
    issue: str
    name: str
    op: str
    m: int
    n: int
    k: int
    layout_a: str
    layout_b: str
    batch: int = 1


CASES = (
    Case("square", "square_1024", "mm", 1024, 1024, 1024, "row", "row"),
    Case("square", "square_4096", "mm", 4096, 4096, 4096, "row", "row"),
    Case("square", "square_8192", "mm", 8192, 8192, 8192, "row", "row"),
    Case("square", "square_16384", "mm", 16384, 16384, 16384, "row", "row"),
    Case("20", "gemm_addmm_768x851968x256", "addmm", 768, 851968, 256, "row", "column"),
    Case("20", "gemm_768x256x851968", "mm", 768, 256, 851968, "row", "row"),
    Case("20", "gemm_851968x256x768", "mm", 851968, 256, 768, "column", "row"),
    Case("20", "gemm_512x256x98304", "mm", 512, 256, 98304, "column", "row"),
    Case("20", "gemm_98304x256x512", "mm", 98304, 256, 512, "row", "row"),
    Case("20", "gemm_768x256x114688", "mm", 768, 256, 114688, "row", "row"),
    Case("20", "gemm_addmm_768x114688x256", "addmm", 768, 114688, 256, "row", "column"),
    Case("20", "gemm_114688x256x768", "mm", 114688, 256, 768, "column", "row"),
    Case("20", "gemm_addmm_98304x512x256", "addmm", 98304, 512, 256, "row", "column"),
    Case("20", "gemm_43500x1024x1024", "mm", 43500, 1024, 1024, "row", "row"),
    Case("20", "gemm_1024x1024x43500", "mm", 1024, 1024, 43500, "column", "row"),
    Case("21", "priority_addmm_1024x20480x6144", "addmm", 1024, 20480, 6144, "row", "column"),
    Case("21", "priority_addmm_262144x262x294", "addmm", 262144, 262, 294, "row", "column"),
    Case("21", "priority_bmm_b3072_448x160x931", "bmm", 448, 160, 931, "shared-row", "row", batch=3072),
    Case("21", "priority_addmm_2252800x256x512", "addmm", 2252800, 256, 512, "row", "column"),
    Case("21", "priority_addmm_3072x15360x4096", "addmm", 3072, 15360, 4096, "row", "column"),
    Case("21", "priority_addmm_3072x4096x25344", "addmm", 3072, 4096, 25344, "row", "column"),
    Case("21", "priority_addmm_1024x4096x6144", "addmm", 1024, 4096, 6144, "row", "column"),
    Case("21", "priority_addmm_3072x3072x11800", "addmm", 3072, 3072, 11800, "row", "column"),
    Case("22", "pr2850_bf16_1024x20480x6144", "mm", 1024, 20480, 6144, "row", "row"),
    Case("22", "pr2850_bf16_1024x24576x6144", "mm", 1024, 24576, 6144, "row", "row"),
)


def _matrix(rows: int, cols: int, layout: str, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "row":
        value = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16, generator=generator)
    elif layout == "column":
        value = torch.randn((cols, rows), device="cuda", dtype=torch.bfloat16, generator=generator).T
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return value.mul_(cols**-0.5)


def _samples(fn, args) -> tuple[list[float], int, int]:
    estimate = float(triton.testing.do_bench(fn, warmup=5, rep=20, return_mode="median"))
    warmup_launches = min(args.max_warmup_launches, max(1, round(args.warmup_ms / estimate)))
    timed_launches = min(args.max_timed_launches, max(1, round(args.rep_ms / estimate)))
    driver = triton.runtime.driver.active
    device_interface = driver.get_device_interface()
    cache = driver.get_empty_cache_for_benchmark()
    samples = []
    for _ in range(args.repeats):
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


def _selection(case: Case, a: torch.Tensor, b: torch.Tensor) -> tuple[str, object]:
    if case.op == "mm":
        streamk = gfx950._origami_plan(a, b, variant="streamk")
        bm, bn, bk = streamk.tile
        if case.m % bm == 0 and case.n % bn == 0 and case.k >= 2 * bk and case.k % (2 * bk) == 0:
            split_k = gfx950._origami_parallel_split_k(
                case.m, case.n, case.k, a.element_size(), streamk
            )
            return ("interwave_splitk" if split_k is not None else "streamk"), streamk
        if case.k >= 2 * bk:
            tail_data = gfx950._origami_plan(a, b, variant="tail_data")
            tail_m, tail_n, _ = tail_data.tile
            if case.m % tail_m == 0 and case.n % tail_n == 0:
                return "tail_data", tail_data
        if case.k >= 2 * gfx950.BLOCK_K and case.k * a.element_size() % 16 == 0:
            return "tail_lds", gfx950._origami_plan(a, b, variant="tail_lds")
        return "tail", gfx950._origami_plan(a, b, variant="tail")
    streamk = gfx950._origami_plan(a, b, variant="fused_streamk")
    bm, bn, bk = streamk.tile
    if (case.m % bm == 0 and case.n % bn == 0 and case.k >= 2 * bk and case.k % (2 * bk) == 0
            and not gfx950._needs_i64_offsets(a) and not gfx950._needs_i64_offsets(b)):
        split_k = gfx950._origami_parallel_split_k(
            case.m, case.n, case.k, a.element_size(), streamk
        )
        return ("fused_interwave_splitk" if split_k is not None else "fused_streamk"), streamk
    return "fused_addmm", gfx950._origami_plan(a, b, variant="fused_addmm")


def _run_case(case: Case, args) -> dict:
    lhs = None
    rhs = None
    input_bias = None
    if case.op == "bmm":
        lhs, rhs = make_bmm_inputs(
            case.batch, case.m, case.n, case.k, "cuda", dtype=torch.bfloat16, seed=args.seed
        )
        tlx_fn = lambda: tlx_shared_a_bmm(lhs, rhs)
        torch_fn = lambda: torch.bmm(lhs, rhs)
    else:
        lhs = _matrix(case.m, case.k, case.layout_a, args.seed)
        rhs = _matrix(case.k, case.n, case.layout_b, args.seed + 1)
        input_bias = _matrix(1, case.n, "row", args.seed + 2).reshape(case.n) if case.op == "addmm" else None
    if case.op == "mm":
        tlx_fn = lambda: tlx_mm(lhs, rhs, arch="gfx950", space="origami")
        torch_fn = lambda: torch.mm(lhs, rhs)
    elif case.op == "addmm":
        tlx_fn = lambda: tlx_addmm(input_bias, lhs, rhs, arch="gfx950", space="origami")
        torch_fn = lambda: torch.addmm(input_bias, lhs, rhs)

    actual = tlx_fn()
    expected = torch_fn()
    delta = (actual.float() - expected.float()).abs()
    correct = bool(torch.allclose(actual, expected, rtol=0.02, atol=0.02))
    if case.op == "bmm":
        mode = "shared_a_bmm"
        selection = {
            "candidate": "bmm_shared_a_row",
            "tile": (224, 160, 32),
            "grid": case.batch * triton.cdiv(case.m, 224) * triton.cdiv(case.n, 160),
            "reduction": "none",
            "wgm": 256,
        }
    else:
        mode, decision = _selection(case, lhs, rhs)
        selection = {
            "candidate": decision.kernel.name,
            "tile": decision.tile,
            "grid": decision.grid_size,
            "reduction": decision.reduction,
            "wgm": decision.wgm,
        }
    tlx_samples, warmup_launches, timed_launches = _samples(tlx_fn, args)
    torch_samples, _, _ = _samples(torch_fn, args)
    tlx_ms = statistics.median(tlx_samples)
    torch_ms = statistics.median(torch_samples)
    result = {
        **dataclasses.asdict(case),
        "mode": mode,
        **selection,
        "correct": correct,
        "max_abs": float(delta.max()),
        "tlx_samples_ms": tlx_samples,
        "torch_samples_ms": torch_samples,
        "tlx_ms": tlx_ms,
        "torch_ms": torch_ms,
        "speedup": torch_ms / tlx_ms,
        "warmup_launches": warmup_launches,
        "timed_launches": timed_launches,
    }
    tlx_fn = torch_fn = None
    lhs = rhs = input_bias = actual = expected = delta = None
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("all", "square", "20", "21", "22"), default="all")
    parser.add_argument("--case", action="append", help="Run only the named case; may be repeated")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-warmup-launches", type=int, default=100)
    parser.add_argument("--max-timed-launches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = CASES if args.suite == "all" else tuple(case for case in CASES if case.issue == args.suite)
    if args.case:
        requested = set(args.case)
        cases = tuple(case for case in cases if case.name in requested)
        missing = requested - {case.name for case in cases}
        if missing:
            parser.error(f"unknown case(s) for suite {args.suite}: {sorted(missing)}")
    results = []
    for case in cases:
        print(f"[{case.issue}] {case.name}", flush=True)
        try:
            result = _run_case(case, args)
        except Exception as error:  # noqa: BLE001 - keep running and report per-case failures
            result = {**dataclasses.asdict(case), "error": f"{type(error).__name__}: {error}"}
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    document = {
        "protocol": {
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "repeats": args.repeats,
            "max_warmup_launches": args.max_warmup_launches,
            "max_timed_launches": args.max_timed_launches,
            "cache": "Triton benchmark cache flushed before every timed launch",
            "dtype": "bfloat16",
        },
        "excluded": [],
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
