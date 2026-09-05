"""MI350X (gfx950 / CDNA4) GEMM with geometry-based scheduling.

The 256x256x64 inter-wave pipeline is the common compute engine.  The host
policy changes only how output/K work is assigned:

* a regular data-centric grid when M/N already expose enough parallelism;
* split-K when a small output grid would leave most CUs idle;
* an aligned split-K prefix plus a fused masked tail for arbitrary K; and
* the one-A-tile v9 pipeline for large, short-K output grids.

The choice depends on tile count, K depth and physical operand strides, never
on an exact M/N/K tuple.  ``plan_for`` is intentionally pure so its decisions
can be unit-tested and explained without a GPU.
"""

from __future__ import annotations

import dataclasses
import enum

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a16w16.matmul_kernel import (
    _reduce_k_kernel,
    _TORCH_TO_TL,
    matmul as _data_centric_matmul,
    matmul_tile,
)

from ._shapes import GFX950_FOCUS

PERF_SHAPES = GFX950_FOCUS

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
NUM_CUS = 256
MIN_K_STEPS_PER_SPLIT = 16


class Schedule(enum.Enum):
    DATA_CENTRIC = "data-centric"
    SPLIT_K = "split-k"
    SPLIT_K_TAIL = "split-k-tail"
    WIDE_SHORT_K = "wide-short-k"


@dataclasses.dataclass(frozen=True)
class Plan:
    schedule: Schedule
    split_k: int = 1
    k_prefix: int = 0


def _split_cost(grid_mn, k_steps, split):
    """Estimate critical-path compute plus fp32 publication/reduction work."""
    return 16 * (k_steps // split) + 3 * grid_mn * split


def _choose_split_k(grid_mn, K):
    """Choose an even-pipeline K split, allowing a partially filled CU wave."""
    k_steps = K // BLOCK_K
    best = 1
    best_cost = 16 * k_steps + 3 * grid_mn
    # One output tile per split is useful only while the launch fits roughly one
    # device wave.  Divisibility, rather than powers of two, determines the
    # candidates because every split must own an even number of K64 steps.
    for split in range(2, NUM_CUS // grid_mn + 1):
        if K % (split * 2 * BLOCK_K) != 0:
            continue
        steps = k_steps // split
        if steps < MIN_K_STEPS_PER_SPLIT:
            continue
        cost = _split_cost(grid_mn, k_steps, split)
        if cost < best_cost:
            best, best_cost = split, cost
    return best


def _choose_split_k_tail(grid_mn, K):
    """Choose an aligned split prefix and leave at most a short masked tail."""
    best = None
    for split in range(2, NUM_CUS // grid_mn + 1):
        quantum = split * 2 * BLOCK_K
        prefix = K // quantum * quantum
        steps = prefix // split // BLOCK_K
        if steps < MIN_K_STEPS_PER_SPLIT:
            continue
        tail_blocks = triton.cdiv(K - prefix, 4 * BLOCK_K)
        # The fused reducer streams each fp32 workspace value once.  On the
        # 1024-wide reduction tiles that traffic is closer to two, not three,
        # K-step units; K256 tail chunks amortize address/mask overhead.
        cost = 16 * steps + 2 * grid_mn * split + 16 * tail_blocks
        if best is None or cost < best[0]:
            best = cost, prefix, split
    return None if best is None else best[1:]


def plan_for(M, N, K, _stride_am, stride_ak, _stride_bk, _stride_bn):
    """Return an explainable gfx950 schedule for a two-dimensional product."""
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    full_tiles = M % BLOCK_M == 0 and N % BLOCK_N == 0

    # A large output grid and short K favor v9's single 256x64 A allocation: it
    # removes one LDS read and one async transaction per K step.
    if stride_ak == 1 and grid_mn >= 2 * NUM_CUS and 2 * BLOCK_K <= K <= 16 * BLOCK_K and K % (2 * BLOCK_K) == 0:
        return Plan(Schedule.WIDE_SHORT_K)

    if full_tiles and grid_mn < NUM_CUS // 2:
        if K % (2 * BLOCK_K) == 0:
            split = _choose_split_k(grid_mn, K)
            if split > 1:
                return Plan(Schedule.SPLIT_K, split_k=split, k_prefix=K)
        else:
            tail_plan = _choose_split_k_tail(grid_mn, K)
            if tail_plan is not None:
                prefix, split = tail_plan
                return Plan(Schedule.SPLIT_K_TAIL, split_k=split, k_prefix=prefix)

    return Plan(Schedule.DATA_CENTRIC)


def _swizzle_bases(shape, contiguous_dim):
    def basis(dim, bit):
        return [1 << bit, 0] if dim == 0 else [0, 1 << bit]

    free_dim = 1 - contiguous_dim
    contiguous_bits = int(shape[contiguous_dim]).bit_length() - 1
    free_bits = int(shape[free_dim]).bit_length() - 1
    return (
        [basis(contiguous_dim, bit) for bit in range(contiguous_bits)]
        + [basis(free_dim, bit) for bit in range(4, free_bits)]
        + [basis(free_dim, bit) for bit in range(min(4, free_bits))]
    )


_A_ROW_BASES = tl.constexpr(_swizzle_bases([128, 64], 1))
_A_COLUMN_BASES = tl.constexpr(_swizzle_bases([128, 64], 0))
_B_COLUMN_BASES = tl.constexpr(_swizzle_bases([64, 128], 0))
_B_ROW_BASES = tl.constexpr(_swizzle_bases([64, 128], 1))


@triton.jit
def _split_k_kernel(
    a_ptr,
    b_ptr,
    workspace_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K_PREFIX: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    SPLIT_K: tl.constexpr,
    A_COLUMN_MAJOR: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
    STREAM_A: tl.constexpr,
):
    """Run the shared 256x256x64 pipeline on one aligned K slice."""
    block_m: tl.constexpr = 256
    block_n: tl.constexpr = 256
    block_k: tl.constexpr = 64
    half: tl.constexpr = 128
    pid = tl.program_id(0)
    split_id = pid % SPLIT_K
    tile_id = pid // SPLIT_K
    num_pid_n: tl.constexpr = N // block_n
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n

    a_bases: tl.constexpr = _A_COLUMN_BASES if A_COLUMN_MAJOR else _A_ROW_BASES
    b_bases: tl.constexpr = _B_ROW_BASES if B_ROW_MAJOR else _B_COLUMN_BASES
    a_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], a_bases, [half, block_k])
    b_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], b_bases, [block_k, half])
    element_ty: tl.constexpr = a_ptr.dtype.element_ty
    smem_a_top = tlx.local_alloc((half, block_k), element_ty, 2, layout=a_layout)
    smem_a_bot = tlx.local_alloc((half, block_k), element_ty, 2, layout=a_layout)
    smem_b_left = tlx.local_alloc((block_k, half), element_ty, 2, layout=b_layout)
    smem_b_right = tlx.local_alloc((block_k, half), element_ty, 2, layout=b_layout)

    offs_m_top = pid_m * block_m + tl.arange(0, half)
    offs_m_bot = offs_m_top + half
    offs_n_left = pid_n * block_n + tl.arange(0, half)
    offs_n_right = offs_n_left + half
    offs_k = tl.arange(0, block_k)
    a_top_off = offs_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_bot_off = offs_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_left_off = offs_k[:, None] * stride_bk + offs_n_left[None, :] * stride_bn
    b_right_off = offs_k[:, None] * stride_bk + offs_n_right[None, :] * stride_bn
    k_steps: tl.constexpr = K_PREFIX // SPLIT_K // block_k
    split_offset = split_id * k_steps * block_k
    acc_tl, acc_bl, acc_tr, acc_br = matmul_tile(
        a_ptr,
        b_ptr,
        smem_a_top,
        smem_a_bot,
        smem_b_left,
        smem_b_right,
        a_top_off,
        a_bot_off,
        b_left_off,
        b_right_off,
        split_offset * stride_ak,
        split_offset * stride_bk,
        k_steps,
        stride_ak,
        stride_bk,
        block_m,
        block_n,
        block_k,
        STREAM_A,
    )

    acc_layout: tl.constexpr = tlx.amd_mfma_layout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 4]
    )
    split_base = split_id * M * N
    tl.store(
        workspace_ptr + split_base + offs_m_top[:, None] * N + offs_n_left[None, :],
        tlx.require_layout(acc_tl, acc_layout, pin=False),
    )
    tl.store(
        workspace_ptr + split_base + offs_m_bot[:, None] * N + offs_n_left[None, :],
        tlx.require_layout(acc_bl, acc_layout, pin=False),
    )
    tl.store(
        workspace_ptr + split_base + offs_m_top[:, None] * N + offs_n_right[None, :],
        tlx.require_layout(acc_tr, acc_layout, pin=False),
    )
    tl.store(
        workspace_ptr + split_base + offs_m_bot[:, None] * N + offs_n_right[None, :],
        tlx.require_layout(acc_br, acc_layout, pin=False),
    )


@triton.jit
def _reduce_split_k_tail(
    a_ptr,
    b_ptr,
    workspace_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K_OFFSET: tl.constexpr,
    K_TAIL: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    SPLIT_K: tl.constexpr,
    TAIL_BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce fp32 K slices and, when present, accumulate a masked K tail."""
    pid = tl.program_id(0)
    grid_n: tl.constexpr = triton.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    output_off = offs_m[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    if K_TAIL:
        offs_k = tl.arange(0, TAIL_BLOCK_K)
        for kk in tl.range(0, K_TAIL, TAIL_BLOCK_K, num_stages=1):
            tail_k = kk + offs_k
            k_mask = tail_k < K_TAIL
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + (K_OFFSET + tail_k[None, :]) * stride_ak,
                mask=(offs_m[:, None] < M) & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptr + (K_OFFSET + tail_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                mask=k_mask[:, None] & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
    for split in range(SPLIT_K):
        acc += tl.load(workspace_ptr + split * M * N + output_off, mask=output_mask, other=0.0)
    tl.store(c_ptr + output_off, acc.to(c_ptr.dtype.element_ty), mask=output_mask)


def _run_split_k(a, b, plan):
    M, K = a.shape
    N = b.shape[1]
    output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    workspace = torch.empty((plan.split_k * M, N), device=a.device, dtype=torch.float32)
    grid_mn = M // BLOCK_M * (N // BLOCK_N)
    _split_k_kernel[(grid_mn * plan.split_k,)](
        a,
        b,
        workspace,
        M,
        N,
        plan.k_prefix,
        *a.stride(),
        *b.stride(),
        SPLIT_K=plan.split_k,
        A_COLUMN_MAJOR=a.stride(0) == 1,
        B_ROW_MAJOR=b.stride(1) == 1,
        # A column-major view is a one-pass stream.  For row-major A, bypass L1
        # only while each split's slice is small enough that doing so protects
        # the single reused B tile without discarding useful A locality.
        STREAM_A=N == BLOCK_N and (a.stride(0) == 1 or plan.k_prefix // plan.split_k <= 64 * BLOCK_K),
        num_warps=8,
        num_stages=1,
        matrix_instr_nonkdim=16,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
        enable_sched_group_barrier_scheduler=True,
    )
    k_tail = K - plan.k_prefix
    if not k_tail:
        _reduce_k_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 32))](
            workspace,
            output,
            output,
            M,
            N,
            0,
            0,
            SPLIT_K=plan.split_k,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            OUTPUT_DTYPE=_TORCH_TO_TL[a.dtype],
            ADD_BIAS=False,
            num_warps=4,
        )
        return output

    tail_block_k = 1 << max(4, min(8, (k_tail - 1).bit_length()))
    reduce_block = 64
    reduce_grid = triton.cdiv(M, reduce_block) * triton.cdiv(N, reduce_block)
    _reduce_split_k_tail[(reduce_grid,)](
        a,
        b,
        workspace,
        output,
        M,
        N,
        plan.k_prefix,
        k_tail,
        *a.stride(),
        *b.stride(),
        SPLIT_K=plan.split_k,
        TAIL_BLOCK_K=tail_block_k,
        BLOCK_M=reduce_block,
        BLOCK_N=reduce_block,
        num_warps=4,
    )
    return output


def mm(a, b, *, space="heuristic"):
    """Matrix multiply ``a @ b`` on gfx950 using an analytical schedule.

    The current implementation has no empirical autotune space: all three
    standard space names select the same deterministic policy.
    """
    if space not in ("heuristic", "full", "smoke"):
        raise ValueError(f"unknown search space {space!r}")
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16)
    assert a.is_cuda and b.is_cuda
    assert 1 in a.stride() and 1 in b.stride(), "each operand must be contiguous in one dimension"

    M, K = a.shape
    N = b.shape[1]
    plan = plan_for(M, N, K, *a.stride(), *b.stride())
    if plan.schedule is Schedule.WIDE_SHORT_K:
        from triton.language.extra.tlx.tutorials.gfx9_gemm.a16w16.v9_beyond_hotloop.matmul_kernel import (
            matmul as _wide_short_k_matmul,
        )

        return _wide_short_k_matmul(a, b)
    if plan.schedule in (Schedule.SPLIT_K, Schedule.SPLIT_K_TAIL):
        return _run_split_k(a, b, plan)
    return _data_centric_matmul(a, b)


matmul = mm
