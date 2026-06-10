"""
TLX v9 GEMM with fused addmm + GLU epilogue (gfx950).

Reuses the v9 quadrant-sliced hot loop from `amd-gemm-v9-tlx_test.py`
(`gemm_v9_quad`) and fuses the GLU epilogue from `test_gemm_glu.py`:

    x   = A @ B + bias          # bias: (N,) broadcast over M
    out = x + x * Y             # GLU gate, Y: (M, N)

The GEMM hot loop (quadrant slicing, warp-pipelined async global->shared loads,
nBuffers=2 unroll-by-2) is identical to the plain GEMM. Only the epilogue
changes: as each quadrant accumulator goes final we add the bias slice, apply the
GLU with a one-quadrant-ahead Y prefetch (hides the gate-load latency behind the
final MFMAs), and store. The masked epilogue makes ragged M/N safe; K is assumed
divisible by 2*BLOCK_K (unroll-by-2 pipeline).

Default tile is 128x256 (not 256x256): these large-N skinny shapes are memory
bound, and the smaller tile triples the grid (340 -> 680 workgroups for
M=1024,N=21568) which lifts occupancy/MLP and bandwidth utilization from ~78-81%
to ~86-89% of the rocBLAS pure-GEMM throughput.
"""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()

NUM_XCDS = 8


@triton.jit
def get_pids(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GRID_MN, NUM_XCDS: tl.constexpr,
             GROUP_SIZE_M: tl.constexpr):
    """v9 PID mapping: XCD-aware remap then GROUP_SIZE_M swizzle."""
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
def gemm_glu_v9_quad(
    a_ptr, b_ptr, bias_ptr, y_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_ym, stride_yn,
    stride_cm, stride_cn,
    GRID_MN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_ym > 0)
    tl.assume(stride_yn > 0)

    HM: tl.constexpr = BLOCK_M // 2
    HN: tl.constexpr = BLOCK_N // 2

    pid = tl.program_id(0)
    pid_m, pid_n = get_pids(pid, M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    offs_am = tl.arange(0, HM)
    offs_bn = tl.arange(0, HN)
    offs_k = tl.arange(0, BLOCK_K)

    rem_m = M - pid_m * BLOCK_M
    rem_n = N - pid_n * BLOCK_N

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

    # ===== Peeled epilogue =====
    # Precompute all epilogue addresses/masks/bias up front, then issue the first
    # two Y gate loads *before* the penultimate (iterMax-2) dot block. This peels
    # the last steady iteration into the epilogue so those gate reads overlap the
    # penultimate MFMAs (the other two Y tiles are fetched one-ahead in the final
    # block). Peak live Y tiles = 2, so no occupancy loss -- prefetching all four
    # was tried but cut occupancy and regressed the compute-bound large-K case.
    cdt = tlx.dtype_of(c_ptr)
    offs_cm = tl.arange(0, HM)
    offs_cn = tl.arange(0, HN)
    m_top = offs_cm < rem_m
    m_bot = (HM + offs_cm) < rem_m
    n_left = offs_cn < rem_n
    n_right = (HN + offs_cn) < rem_n

    col_l = pid_n * BLOCK_N + offs_cn
    col_r = col_l + HN
    bias_l = tl.load(bias_ptr + col_l, mask=n_left, other=0.0).to(tl.float32)
    bias_r = tl.load(bias_ptr + col_r, mask=n_right, other=0.0).to(tl.float32)

    y_base = y_ptr + pid_m * BLOCK_M * stride_ym + pid_n * BLOCK_N * stride_yn
    y_tl = y_base + offs_cm[:, None] * stride_ym + offs_cn[None, :] * stride_yn
    y_tr = y_tl + HN * stride_yn
    y_bl = y_tl + HM * stride_ym
    y_br = y_bl + HN * stride_yn

    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_tl = c_base + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_tr = c_tl + HN * stride_cn
    c_bl = c_tl + HM * stride_cm
    c_br = c_bl + HN * stride_cn

    mask_tl = m_top[:, None] & n_left[None, :]
    mask_bl = m_bot[:, None] & n_left[None, :]
    mask_tr = m_top[:, None] & n_right[None, :]
    mask_br = m_bot[:, None] & n_right[None, :]

    # Kick off the first two gate loads now -> in flight across the penultimate
    # MFMAs; the other two are issued one-ahead in the final block. Peak live Y
    # tiles stays at 2 (no occupancy loss, unlike prefetching all four).
    y0 = tl.load(y_tl, mask=mask_tl, other=0.0)
    y1 = tl.load(y_bl, mask=mask_bl, other=0.0)

    # ---- penultimate iteration (iterMax-2): operands in buffer 0 ----
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

    # ---- final iteration (iterMax-1): operands in buffer 1, fuse GLU + store ----
    acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
    y2 = tl.load(y_tr, mask=mask_tr, other=0.0)
    xq = acc_tl + bias_l[None, :]
    tl.store(c_tl, (xq + xq * y0.to(tl.float32)).to(cdt), mask=mask_tl)

    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(sAb, 1))
    acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
    y3 = tl.load(y_br, mask=mask_br, other=0.0)
    xq = acc_bl + bias_l[None, :]
    tl.store(c_bl, (xq + xq * y1.to(tl.float32)).to(cdt), mask=mask_bl)

    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_view(sBr, 1))
    acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
    xq = acc_tr + bias_r[None, :]
    tl.store(c_tr, (xq + xq * y2.to(tl.float32)).to(cdt), mask=mask_tr)

    acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)
    xq = acc_br + bias_r[None, :]
    tl.store(c_br, (xq + xq * y3.to(tl.float32)).to(cdt), mask=mask_br)


def tlx_fused_addmm_glu(bias, a, b, y, c=None, bm=128, bn=256, bk=64, nw=4, gm=8, nonk=16, wpeu=0):
    M, K = a.shape
    _, N = b.shape
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid_mn = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    gemm_glu_v9_quad[(grid_mn, )](
        a, b, bias, y, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        y.stride(0), y.stride(1),
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


def pytorch_baseline(bias, a, b, y):
    """PyTorch reference: addmm + GLU (two ops)."""
    x = torch.addmm(bias, a, b).to(torch.float32)
    out = (x + x * y.to(torch.float32)).to(torch.float16)
    return out


if __name__ == "__main__":
    M, N = 1024, 21568
    K_VALUES = [256, 512, 1024]
    tflops = lambda ms, M, N, K: 2 * M * N * K * 1e-12 / (ms * 1e-3)

    for K in K_VALUES:
        print(f"\n{'='*64}\n  Shape M={M}, N={N}, K={K} fp16\n{'='*64}")
        torch.manual_seed(0)
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
        bias = torch.randn(N, device=DEVICE, dtype=torch.float16)
        y = torch.randn(M, N, device=DEVICE, dtype=torch.float16)

        ref = pytorch_baseline(bias, a, b, y)
        ours = tlx_fused_addmm_glu(bias, a, b, y)
        try:
            torch.testing.assert_close(ours, ref, atol=2e-2, rtol=2e-2)
            ok = "OK"
        except AssertionError as e:
            ok = "MISMATCH"
            print(e)
        print(f"  correctness: {ok}")

        base_ms = triton.testing.do_bench(lambda: pytorch_baseline(bias, a, b, y), warmup=25, rep=200)
        tlx_ms = triton.testing.do_bench(lambda: tlx_fused_addmm_glu(bias, a, b, y), warmup=25, rep=200)
        # rocBLAS pure GEMM (no bias/GLU) -- GEMM-only floor for reference.
        rocblas_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=25, rep=200)

        print(f"  rocBLAS pure GEMM (no GLU)       : {rocblas_ms*1000:7.2f} us  ({tflops(rocblas_ms,M,N,K):6.1f} TFLOPS)")
        print(f"  Baseline (PyTorch addmm + GLU)   : {base_ms*1000:7.2f} us  ({tflops(base_ms,M,N,K):6.1f} TFLOPS)")
        print(f"  TLX v9-quad fused (1 kernel)     : {tlx_ms*1000:7.2f} us  ({tflops(tlx_ms,M,N,K):6.1f} TFLOPS)")
        print(f"  Speedup (baseline / TLX)         : {base_ms/tlx_ms:.3f}x")
        print(f"  TLX fused vs rocBLAS pure GEMM    : {tlx_ms/rocblas_ms:.3f}x time ({rocblas_ms/tlx_ms*100:.0f}% throughput)")
