"""
TLX port of the gfx950 Gluon `v9_beyond_hotloop` GEMM.

Based on `amd-gemm-pipelined_test.py`, but adopts the v9 optimizations that are
expressible in TLX:

  1. L2 locality (the v9 headline): XCD-aware PID remap so consecutive output
     tiles land on the same XCD, plus GROUP_SIZE_M workgroup swizzle. Minimizes
     per-XCD input footprint ~ GM + ceil(P/GM), P = workgroups/XCD (optimal
     GM ~= sqrt(P)).
  2. Async direct global->shared via AMD buffer ops (`buffer_load_to_local`),
     double-buffered, with MFMA/mem warp-pipeline split (v8 hot-loop style).
  3. BLOCK_K=64 256x256 tiles (fewer K-iters, better compute/mem overlap).

The explicit Gluon layouts (DistributedLinear / PaddedShared / MFMA / DotOperand)
are chosen by the AMD backend in TLX, so we only express the algorithmic levers.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# gfx950 has 8 XCDs per chip.
NUM_XCDS = 8


@triton.jit
def get_pids(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GRID_MN, NUM_XCDS: tl.constexpr,
             GROUP_SIZE_M: tl.constexpr):
    """v9 PID mapping: XCD-aware remap (so consecutive tiles share an XCD's L2)
    followed by GROUP_SIZE_M swizzle to shrink the per-XCD A/B footprint."""
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = tl.where(tall_xcds == 0, NUM_XCDS, tall_xcds)
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def gemm_v9(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    GRID_MN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)

    pid = tl.program_id(0)
    pid_m, pid_n = get_pids(pid, M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_base_off = offs_m[:, None] * stride_am
    b_base_off = offs_n[None, :] * stride_bn
    offs_ak = offs_k[None, :]
    offs_bk = offs_k[:, None]

    K_ITERS = tl.cdiv(K, BLOCK_K)

    smemA = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    smemB = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    # Prologue: prefetch NUM_BUFFERS-1 tiles; the main loop produces one tile per
    # tile it consumes and the epilogue drains the prefetched tail.
    for i in tl.range(0, NUM_BUFFERS - 1, loop_unroll_factor=NUM_BUFFERS - 1):
        k0 = i * BLOCK_K
        a_off = a_base_off + (k0 + offs_ak) * stride_ak
        b_off = (k0 + offs_bk) * stride_bk + b_base_off
        tok_a = tlx.async_load(a_ptr + a_off, tlx.local_view(smemA, i), mask=offs_ak < K - k0)
        tlx.async_load_commit_group([tok_a])
        tok_b = tlx.async_load(b_ptr + b_off, tlx.local_view(smemB, i), mask=offs_bk < K - k0)
        tlx.async_load_commit_group([tok_b])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Wait for the first prefetched tile.
    tlx.async_load_wait_group((NUM_BUFFERS - 2) * 2)

    for k in tl.range(0, K_ITERS - (NUM_BUFFERS - 1)):
        consumer = k % NUM_BUFFERS
        producer = (k + NUM_BUFFERS - 1) % NUM_BUFFERS
        k_pref = (k + NUM_BUFFERS - 1) * BLOCK_K

        with tlx.warp_pipeline_stage("stage0", priority=1):
            a_tile = tlx.local_load(tlx.local_view(smemA, consumer))
            b_tile = tlx.local_load(tlx.local_view(smemB, consumer))

        tlx.async_load_wait_group(0)

        with tlx.warp_pipeline_stage("stage1", priority=0):
            a_off = a_base_off + (k_pref + offs_ak) * stride_ak
            b_off = (k_pref + offs_bk) * stride_bk + b_base_off
            tok_a = tlx.async_load(a_ptr + a_off, tlx.local_view(smemA, producer), mask=offs_ak < K - k_pref)
            tlx.async_load_commit_group([tok_a])
            tok_b = tlx.async_load(b_ptr + b_off, tlx.local_view(smemB, producer), mask=offs_bk < K - k_pref)
            tlx.async_load_commit_group([tok_b])
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    # Epilogue: drain the NUM_BUFFERS-1 prefetched tail tiles.
    for i in tl.static_range(NUM_BUFFERS - 1):
        consumer = (K_ITERS - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS
        with tlx.warp_pipeline_stage("stage0_ep", priority=1):
            a_tile = tlx.local_load(tlx.local_view(smemA, consumer))
            b_tile = tlx.local_load(tlx.local_view(smemB, consumer))
        tlx.async_load_wait_group((NUM_BUFFERS - 2 - i) * 2)
        with tlx.warp_pipeline_stage("stage1_ep", priority=0):
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def gemm_v9_quad(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    GRID_MN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """v9 hot-loop port: M+N slicing into 4 quadrant accumulators with LDS reads
    and global async loads interleaved between MFMAs (nBuffers=2, unroll-by-2).

    Ragged M/N are handled via offset clamping + masked stores (see below), so
    arbitrary M, N are correct with no interior-tile perf cost. K is still assumed
    divisible by 2*BLOCK_K (the unroll-by-2 pipeline has no K-remainder path).
    """
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)

    HM: tl.constexpr = BLOCK_M // 2
    HN: tl.constexpr = BLOCK_N // 2

    pid = tl.program_id(0)
    pid_m, pid_n = get_pids(pid, M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    offs_am = tl.arange(0, HM)
    offs_bn = tl.arange(0, HN)
    offs_k = tl.arange(0, BLOCK_K)

    # Ragged M/N edges: loads keep clean affine offsets (required for the
    # buffer_load_to_local layout analysis) and rely on AMD buffer ops being
    # hardware OOB-safe (out-of-range lanes return 0, no fault); the masked stores
    # below discard the OOB output positions. rem_m/rem_n drive the store masks.
    rem_m = M - pid_m * BLOCK_M
    rem_n = N - pid_n * BLOCK_N

    # Relative element offsets (i32) for the four half-tiles.
    a_top_off = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_bot_off = a_top_off + HM * stride_am
    b_left_off = offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_right_off = b_left_off + HN * stride_bn
    a_top_off_n = a_top_off + BLOCK_K * stride_ak
    a_bot_off_n = a_bot_off + BLOCK_K * stride_ak
    b_left_off_n = b_left_off + BLOCK_K * stride_bk
    b_right_off_n = b_right_off + BLOCK_K * stride_bk

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    sAt = tlx.local_alloc((HM, BLOCK_K), tlx.dtype_of(a_ptr), 2)
    sAb = tlx.local_alloc((HM, BLOCK_K), tlx.dtype_of(a_ptr), 2)
    sBl = tlx.local_alloc((BLOCK_K, HN), tlx.dtype_of(b_ptr), 2)
    sBr = tlx.local_alloc((BLOCK_K, HN), tlx.dtype_of(b_ptr), 2)

    iterMax = tl.cdiv(K, BLOCK_K)

    # Prologue: prefetch tiles 0 (buffer 0) and 1 (buffer 1).
    tlx.async_load_commit_group([tlx.async_load(b_base + b_left_off, tlx.local_view(sBl, 0))])
    tlx.async_load_commit_group([tlx.async_load(a_base + a_top_off, tlx.local_view(sAt, 0))])
    tlx.async_load_commit_group([tlx.async_load(a_base + a_bot_off, tlx.local_view(sAb, 0))])
    tlx.async_load_commit_group([tlx.async_load(b_base + b_right_off, tlx.local_view(sBr, 0))])
    tlx.async_load_commit_group([tlx.async_load(b_base + b_left_off_n, tlx.local_view(sBl, 1))])
    tlx.async_load_commit_group([tlx.async_load(a_base + a_top_off_n, tlx.local_view(sAt, 1))])
    tlx.async_load_commit_group([tlx.async_load(a_base + a_bot_off_n, tlx.local_view(sAb, 1))])
    tlx.async_load_commit_group([tlx.async_load(b_base + b_right_off_n, tlx.local_view(sBr, 1))])
    a_base += BLOCK_K * stride_ak * 2
    b_base += BLOCK_K * stride_bk * 2

    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(tlx.local_view(sBl, 0))
    a_top = tlx.local_load(tlx.local_view(sAt, 0))
    a_bot = tlx.local_load(tlx.local_view(sAb, 0))
    b_right = tlx.local_load(tlx.local_view(sBr, 0))

    acc_tl = tl.zeros((HM, HN), dtype=tl.float32)
    acc_bl = tl.zeros((HM, HN), dtype=tl.float32)
    acc_tr = tl.zeros((HM, HN), dtype=tl.float32)
    acc_br = tl.zeros((HM, HN), dtype=tl.float32)

    for k in tl.range(0, iterMax - 2, 2):
        # ---- sub-iteration 0: consume buffer 0, prefetch into buffer 0 ----
        acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
        tlx.async_load_wait_group(5)
        a_bot = tlx.local_load(tlx.local_view(sAb, 0))
        tlx.async_load_commit_group([tlx.async_load(b_base + b_left_off, tlx.local_view(sBl, 0))])

        acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
        tlx.async_load_wait_group(5)
        b_right = tlx.local_load(tlx.local_view(sBr, 0))
        tlx.async_load_commit_group([tlx.async_load(a_base + a_top_off, tlx.local_view(sAt, 0))])

        acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
        tlx.async_load_wait_group(5)
        b_left = tlx.local_load(tlx.local_view(sBl, 1))
        tlx.async_load_commit_group([tlx.async_load(a_base + a_bot_off, tlx.local_view(sAb, 0))])

        acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)
        tlx.async_load_wait_group(5)
        a_top = tlx.local_load(tlx.local_view(sAt, 1))
        tlx.async_load_commit_group([tlx.async_load(b_base + b_right_off, tlx.local_view(sBr, 0))])

        # ---- sub-iteration 1: consume buffer 1, prefetch into buffer 1 (_next) ----
        acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
        tlx.async_load_wait_group(5)
        a_bot = tlx.local_load(tlx.local_view(sAb, 1))
        tlx.async_load_commit_group([tlx.async_load(b_base + b_left_off_n, tlx.local_view(sBl, 1))])

        acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
        tlx.async_load_wait_group(5)
        b_right = tlx.local_load(tlx.local_view(sBr, 1))
        tlx.async_load_commit_group([tlx.async_load(a_base + a_top_off_n, tlx.local_view(sAt, 1))])

        acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
        tlx.async_load_wait_group(5)
        b_left = tlx.local_load(tlx.local_view(sBl, 0))
        tlx.async_load_commit_group([tlx.async_load(a_base + a_bot_off_n, tlx.local_view(sAb, 1))])

        acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)
        tlx.async_load_wait_group(5)
        a_top = tlx.local_load(tlx.local_view(sAt, 0))
        tlx.async_load_commit_group([tlx.async_load(b_base + b_right_off_n, tlx.local_view(sBr, 1))])

        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    # ---- Epilogue: iters iterMax-2 and iterMax-1 (operands already prefetched) ----
    acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
    tlx.async_load_wait_group(5)
    a_bot = tlx.local_load(tlx.local_view(sAb, 0))
    acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_view(sBr, 0))
    acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
    tlx.async_load_wait_group(3)
    b_left = tlx.local_load(tlx.local_view(sBl, 1))
    acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(tlx.local_view(sAt, 1))

    # Precompute store addresses + masks up front so each quadrant can be written
    # the instant its accumulator is final -> the HBM write latency overlaps the
    # remaining MFMAs instead of being a fully-exposed serial tail (ATT showed the
    # bunched-at-end stores were ~63% of stalls on the K=256 skinny case).
    cdt = tlx.dtype_of(c_ptr)
    offs_cm = tl.arange(0, HM)
    offs_cn = tl.arange(0, HN)
    m_top = offs_cm < rem_m
    m_bot = (HM + offs_cm) < rem_m
    n_left = offs_cn < rem_n
    n_right = (HN + offs_cn) < rem_n
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_tl = c_base + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_tr = c_tl + HN * stride_cn
    c_bl = c_tl + HM * stride_cm
    c_br = c_bl + HN * stride_cn

    acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
    tl.store(c_tl, acc_tl.to(cdt), mask=m_top[:, None] & n_left[None, :])
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(sAb, 1))
    acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
    tl.store(c_bl, acc_bl.to(cdt), mask=m_bot[:, None] & n_left[None, :])
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_view(sBr, 1))
    acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
    tl.store(c_tr, acc_tr.to(cdt), mask=m_top[:, None] & n_right[None, :])
    acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)
    tl.store(c_br, acc_br.to(cdt), mask=m_bot[:, None] & n_right[None, :])


def run_quad(a, b, c, bm=256, bn=256, bk=64, nw=4, gm=8, nonk=16, wpeu=0):
    M, K = a.shape
    _, N = b.shape
    grid_mn = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    gemm_v9_quad[(grid_mn, )](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        grid_mn,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        GROUP_SIZE_M=gm,
        NUM_XCDS=NUM_XCDS,
        num_warps=nw,
        num_stages=1,
        waves_per_eu=wpeu,
        matrix_instr_nonkdim=nonk,
    )
    return c


def run(a, b, c, bm, bn, bk, nb, nw, gm, nonk=16, wpeu=0):
    M, K = a.shape
    _, N = b.shape
    grid_mn = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    gemm_v9[(grid_mn, )](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        grid_mn,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        GROUP_SIZE_M=gm,
        NUM_BUFFERS=nb,
        NUM_XCDS=NUM_XCDS,
        num_warps=nw,
        num_stages=1,
        waves_per_eu=wpeu,
        matrix_instr_nonkdim=nonk,
    )
    return c


if __name__ == "__main__":
    tflops = lambda ms, M, N, K: 2 * M * N * K * 1e-12 / (ms * 1e-3)

    # Quad (v9 hot-loop port) configs: (label, BM, BN, BK, nw, gm, wpe).
    quad_configs = [
        ("quad 256x256x64 nw4 gm8 wpe0", 256, 256, 64, 4, 8, 0),  # default (nw4-safe)
        ("quad 256x256x64 nw8 gm8 wpe2", 256, 256, 64, 8, 8, 2),  # compute-bound peak >1000 TF
        ("quad 256x256x64 nw8 gm8 wpe0", 256, 256, 64, 8, 8, 0),
    ]
    # Monolithic warp-pipeline configs for comparison: (label, BM, BN, BK, nb, nw, gm).
    mono_configs = [
        ("mono 256x256x32 nb4 nw8 gm8 ", 256, 256, 32, 4, 8, 8),
    ]

    for size in [4096, 8192]:
        M = N = K = size
        torch.manual_seed(42)
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        ref = torch.matmul(a, b)
        c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)

        print(f"\n{'='*70}")
        print(f"  M=N=K={size}")
        print(f"{'='*70}")
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b), rep=200)
        print(f"  {'rocBLAS':<30s} {tflops(ms,M,N,K):7.1f} TFLOPS ({ms:.3f} ms)")

        for name, bm, bn, bk, nw, gm, wpe in quad_configs:
            try:
                run_quad(a, b, c, bm, bn, bk, nw, gm, wpeu=wpe)
                torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
                ok = "OK"
            except Exception:
                ok = "FAIL"
            ms = triton.testing.do_bench(
                lambda bm=bm, bn=bn, bk=bk, nw=nw, gm=gm, wpe=wpe: run_quad(a, b, c, bm, bn, bk, nw, gm, wpeu=wpe),
                rep=200)
            print(f"  {name:<30s} {tflops(ms,M,N,K):7.1f} TFLOPS ({ms:.3f} ms) [{ok}]")

        for name, bm, bn, bk, nb, nw, gm in mono_configs:
            try:
                run(a, b, c, bm, bn, bk, nb, nw, gm)
                torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
                ok = "OK"
            except Exception:
                ok = "FAIL"
            ms = triton.testing.do_bench(
                lambda bm=bm, bn=bn, bk=bk, nb=nb, nw=nw, gm=gm: run(a, b, c, bm, bn, bk, nb, nw, gm), rep=200)
            print(f"  {name:<30s} {tflops(ms,M,N,K):7.1f} TFLOPS ({ms:.3f} ms) [{ok}]")
