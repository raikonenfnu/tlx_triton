"""Cache-correct Stream-K a16w16 GEMM specialized for gfx950.

The exact (M, N, K) = (1024, 20480, 6144) schedule assigns one full
256x256 tile plus one 24-step K quarter to each of 256 CTAs. Four CTAs
cooperate on each tail tile using write-through publication and volatile,
cache-invalidating polling. Quadrant fixup is distributed over all four CTAs.

The synchronization protocol is based on ROCm/tritonBLAS streamk_gemm.py at
main (d0b91605, verified byte-identical to the original 60b7ecd5 baseline).
"""

import os

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

os.environ.setdefault("TRITON_DISABLE_POST_MISCHED", "1")

NUM_CU = 256
M = 1024
N = 20480
K = 6144
_epoch_counter = 0


def _next_epoch():
    # Python's GIL makes this increment unique across host threads. The counter
    # is process-wide so a recycled allocation cannot repeat the prior launch's
    # expected value (the classic ABA failure of a reusable binary lock).
    global _epoch_counter
    _epoch_counter += 1
    if _epoch_counter == 0x7fffffff:
        raise RuntimeError("Stream-K epoch space exhausted; restart the process")
    return _epoch_counter


def _bases(shape, contig_dim):
    def basis(dim, bit):
        return [1 << bit, 0] if dim == 0 else [0, 1 << bit]

    free_dim = 1 - contig_dim
    cb = int(shape[contig_dim]).bit_length() - 1
    fb = int(shape[free_dim]).bit_length() - 1
    return ([basis(contig_dim, i) for i in range(cb)] +
            [basis(free_dim, i) for i in range(4, fb)] +
            [basis(free_dim, i) for i in range(min(4, fb))])


_A_BASES = tl.constexpr(_bases([128, 64], 1))
_B128_BASES = tl.constexpr(_bases([64, 128], 0))
_C_STORE_SIMD_LAYOUT = tlx.layout(
    shape=((16, 4, 8), (8, 4)), stride=((8, 128, 512), (1, 4096)))


@triton.jit
def _compute_256_segment_fine(a_ptr, b_ptr, sa0, sa1, sb0, sb1,
                              ao0, ao1, bo0, bo1, start_step, n_steps,
                              stride_ak, stride_bk):
    """Champion eight-region inter-wave pipeline over a variable K segment."""
    ksa = 64 * stride_ak
    ksb = 64 * stride_bk
    ka = tl.multiple_of(start_step * ksa, 16)
    kb = tl.multiple_of(start_step * ksb, 16)
    ao0n = ao0 + ksa
    ao1n = ao1 + ksa
    bo0n = bo0 + ksb
    bo1n = bo1 + ksb
    x00 = tl.zeros((128, 128), tl.float32)
    x10 = tl.zeros((128, 128), tl.float32)
    x01 = tl.zeros((128, 128), tl.float32)
    x11 = tl.zeros((128, 128), tl.float32)

    tlx.buffer_load_to_local(sb0[0], b_ptr, bo0 + kb)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(sa0[0], a_ptr, ao0 + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(sa1[0], a_ptr, ao1 + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(sb1[0], b_ptr, bo1 + kb)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(sb0[1], b_ptr, bo0n + kb)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(sa0[1], a_ptr, ao0n + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(sa1[1], a_ptr, ao1n + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(sb1[1], b_ptr, bo1n + kb)
    tlx.async_load_commit_group()
    ka += 2 * ksa
    kb += 2 * ksb
    tlx.async_load_wait_group(6)
    b0 = tlx.local_load(sb0[0], relaxed=True)
    a0 = tlx.local_load(sa0[0], relaxed=True)

    for _ in tl.range(0, n_steps - 2, 2, num_stages=1):
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            x00 = tl.dot(a0, b0, x00)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a1 = tlx.local_load(sa1[0], relaxed=True)
            tlx.buffer_load_to_local(sb0[0], b_ptr, bo0 + kb)
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            x10 = tl.dot(a1, b0, x10)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b1 = tlx.local_load(sb1[0], relaxed=True)
            tlx.buffer_load_to_local(sa0[0], a_ptr, ao0 + ka)
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            x01 = tl.dot(a0, b1, x01)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b0 = tlx.local_load(sb0[1], relaxed=True)
            tlx.buffer_load_to_local(sa1[0], a_ptr, ao1 + ka)
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            x11 = tl.dot(a1, b1, x11)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a0 = tlx.local_load(sa0[1], relaxed=True)
            tlx.buffer_load_to_local(sb1[0], b_ptr, bo1 + kb)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            x00 = tl.dot(a0, b0, x00)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a1 = tlx.local_load(sa1[1], relaxed=True)
            tlx.buffer_load_to_local(sb0[1], b_ptr, bo0n + kb)
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            x10 = tl.dot(a1, b0, x10)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b1 = tlx.local_load(sb1[1], relaxed=True)
            tlx.buffer_load_to_local(sa0[1], a_ptr, ao0n + ka)
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            x01 = tl.dot(a0, b1, x01)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b0 = tlx.local_load(sb0[0], relaxed=True)
            tlx.buffer_load_to_local(sa1[1], a_ptr, ao1n + ka)
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            x11 = tl.dot(a1, b1, x11)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a0 = tlx.local_load(sa0[0], relaxed=True)
            tlx.buffer_load_to_local(sb1[1], b_ptr, bo1n + kb)
            tlx.async_load_commit_group()
            ka += 2 * ksa
            kb += 2 * ksb

    x00 = tl.dot(a0, b0, x00)
    tlx.async_load_wait_group(5)
    a1 = tlx.local_load(sa1[0], relaxed=True)
    x10 = tl.dot(a1, b0, x10)
    tlx.async_load_wait_group(4)
    b1 = tlx.local_load(sb1[0], relaxed=True)
    x01 = tl.dot(a0, b1, x01)
    tlx.async_load_wait_group(3)
    b0 = tlx.local_load(sb0[1], relaxed=True)
    x11 = tl.dot(a1, b1, x11)
    tlx.async_load_wait_group(2)
    a0 = tlx.local_load(sa0[1], relaxed=True)
    x00 = tl.dot(a0, b0, x00)
    tlx.async_load_wait_group(1)
    a1 = tlx.local_load(sa1[1], relaxed=True)
    x10 = tl.dot(a1, b0, x10)
    tlx.async_load_wait_group(0)
    b1 = tlx.local_load(sb1[1], relaxed=True)
    x01 = tl.dot(a0, b1, x01)
    x11 = tl.dot(a1, b1, x11)
    return x00, x10, x01, x11


@triton.jit
def streamk_tritonblas_kernel(a_ptr, b_ptr, c_ptr, partials_ptr, locks_ptr,
                              epoch,
                              stride_am, stride_ak, stride_bk, stride_bn,
                              stride_cm, stride_cn):
    """Exact-shape TLX port of tritonBLAS's ordered Stream-K fixup.

    Each of 256 resident CTAs first computes one ordinary 256x256 head tile.
    The 64 remaining spatial tiles contain 96 K64 steps apiece, so flattening
    them over 256 CTAs assigns exactly 24 steps to each CTA.  In every group of
    four consecutive CTAs, each CTA owns one output quadrant and consumes the
    other three CTAs' partials after write-through lock publication.
    """
    pid = tl.program_id(0)
    split = pid % 4
    tail_tile = pid // 4
    acc_layout: tl.constexpr = tlx.amd_mfma_layout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[2, 4])
    a_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 16)], _A_BASES, [128, 64])
    b_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 16)], _B128_BASES, [64, 128])
    et: tl.constexpr = a_ptr.dtype.element_ty
    sa0 = tlx.local_alloc((128, 64), et, 2, layout=a_layout)
    sa1 = tlx.local_alloc((128, 64), et, 2, layout=a_layout)
    sb0 = tlx.local_alloc((64, 128), et, 2, layout=b_layout)
    sb1 = tlx.local_alloc((64, 128), et, 2, layout=b_layout)
    rm = tl.arange(0, 128)
    rn = tl.arange(0, 128)
    rk = tl.arange(0, 64)

    # Never reset locks inside producer CTAs: another CTA can publish before a
    # late producer executes its reset. A unique host epoch also prevents an ABA
    # match against a stale value at a recycled allocation/cache line.

    # One exact 256-CTA spatial wave: N columns [0, 16384).
    # Match the production kernel's XCD remap.  Adjacent hardware PIDs are
    # striped over the eight XCDs, while each XCD receives a contiguous tile
    # interval and can reuse its A/B working set in its local L2 slice.
    xcd = pid % 8
    local_pid = pid // 8
    head_pid = xcd * 32 + local_pid
    hm = head_pid % 4
    hn = head_pid // 4
    hm0 = hm * 256 + rm
    hm1 = hm0 + 128
    hn0 = hn * 256 + rn
    hn1 = hn0 + 128
    ha0 = hm0[:, None] * stride_am + rk[None, :] * stride_ak
    ha1 = hm1[:, None] * stride_am + rk[None, :] * stride_ak
    hb0 = rk[:, None] * stride_bk + hn0[None, :] * stride_bn
    hb1 = rk[:, None] * stride_bk + hn1[None, :] * stride_bn
    h00, h10, h01, h11 = _compute_256_segment_fine(
        a_ptr, b_ptr, sa0, sa1, sb0, sb1, ha0, ha1, hb0, hb1,
        0, 96, stride_ak, stride_bk)
    C: tl.constexpr = _C_STORE_SIMD_LAYOUT
    tl.store(c_ptr + hm0[:, None] * stride_cm + hn0[None, :] * stride_cn,
             tlx.require_layout(h00.to(et), C))
    tl.store(c_ptr + hm1[:, None] * stride_cm + hn0[None, :] * stride_cn,
             tlx.require_layout(h10.to(et), C))
    tl.store(c_ptr + hm0[:, None] * stride_cm + hn1[None, :] * stride_cn,
             tlx.require_layout(h01.to(et), C))
    tl.store(c_ptr + hm1[:, None] * stride_cm + hn1[None, :] * stride_cn,
             tlx.require_layout(h11.to(et), C))

    # Flatten the 64-tile tail over all CUs: four 24-step producers/tile.
    tm = tail_tile % 4
    tn = tail_tile // 4
    tm0 = tm * 256 + rm
    tm1 = tm0 + 128
    tn0 = 16384 + tn * 256 + rn
    tn1 = tn0 + 128
    ta0 = tm0[:, None] * stride_am + rk[None, :] * stride_ak
    ta1 = tm1[:, None] * stride_am + rk[None, :] * stride_ak
    tb0 = rk[:, None] * stride_bk + tn0[None, :] * stride_bn
    tb1 = rk[:, None] * stride_bk + tn1[None, :] * stride_bn
    x00, x10, x01, x11 = _compute_256_segment_fine(
        a_ptr, b_ptr, sa0, sa1, sb0, sb1, ta0, ta1, tb0, tb1,
        split * 24, 24, stride_ak, stride_bk)

    tile_elems: tl.constexpr = 256 * 256
    q00 = rm[:, None] * 256 + rn[None, :]
    q10 = q00 + 128 * 256
    q01 = q00 + 128
    q11 = q10 + 128
    # Normalize both control-flow arms to the same explicit accumulator layout;
    # otherwise TLX fixup would make only the owner arm encoded and leave the
    # scf.if producer arm with unencoded result types.
    x00 = tlx.require_layout(x00, acc_layout, pin=False)
    x10 = tlx.require_layout(x10, acc_layout, pin=False)
    x01 = tlx.require_layout(x01, acc_layout, pin=False)
    x11 = tlx.require_layout(x11, acc_layout, pin=False)
    # Publish all four partial quadrants so the MFMA accumulators die before
    # fixup; retaining a runtime-selected quadrant costs more VGPR/select work
    # than the saved traffic on gfx950.
    base = pid * tile_elems
    p00 = tlx.require_layout(partials_ptr + base + q00, acc_layout, pin=False)
    p10 = tlx.require_layout(partials_ptr + base + q10, acc_layout, pin=False)
    p01 = tlx.require_layout(partials_ptr + base + q01, acc_layout, pin=False)
    p11 = tlx.require_layout(partials_ptr + base + q11, acc_layout, pin=False)
    tl.store(p00, x00, cache_modifier=".wt")
    tl.store(p10, x10, cache_modifier=".wt")
    tl.store(p01, x01, cache_modifier=".wt")
    tl.store(p11, x11, cache_modifier=".wt")
    tl.debug_barrier()
    tl.store(locks_ptr + pid, epoch, cache_modifier=".wt")

    # All producers release before any consumer waits, so there is no cyclic
    # wait.  Each CTA then owns one output quadrant; this keeps all 256 CTAs
    # useful during fixup instead of making 64 CTAs reduce four quadrants each.
    group_pid = pid - split
    for peer in range(4):
        while tl.load(locks_ptr + group_pid + peer,
                      cache_modifier=".cv", volatile=True) != epoch:
            pass
    if split == 0:
        r00 = tlx.require_layout(tl.load(partials_ptr + group_pid * tile_elems + q00,
                                        cache_modifier=".cv"), acc_layout, pin=False)
        for peer in range(1, 4):
            r00 += tlx.require_layout(tl.load(partials_ptr + (group_pid + peer) * tile_elems + q00,
                                              cache_modifier=".cv"), acc_layout, pin=False)
        tl.store(c_ptr + tm0[:, None] * stride_cm + tn0[None, :] * stride_cn,
                 tlx.require_layout(r00.to(et), C))
    elif split == 1:
        r10 = tlx.require_layout(tl.load(partials_ptr + group_pid * tile_elems + q10,
                                        cache_modifier=".cv"), acc_layout, pin=False)
        for peer in range(1, 4):
            r10 += tlx.require_layout(tl.load(partials_ptr + (group_pid + peer) * tile_elems + q10,
                                              cache_modifier=".cv"), acc_layout, pin=False)
        tl.store(c_ptr + tm1[:, None] * stride_cm + tn0[None, :] * stride_cn,
                 tlx.require_layout(r10.to(et), C))
    elif split == 2:
        r01 = tlx.require_layout(tl.load(partials_ptr + group_pid * tile_elems + q01,
                                        cache_modifier=".cv"), acc_layout, pin=False)
        for peer in range(1, 4):
            r01 += tlx.require_layout(tl.load(partials_ptr + (group_pid + peer) * tile_elems + q01,
                                              cache_modifier=".cv"), acc_layout, pin=False)
        tl.store(c_ptr + tm0[:, None] * stride_cm + tn1[None, :] * stride_cn,
                 tlx.require_layout(r01.to(et), C))
    else:
        r11 = tlx.require_layout(tl.load(partials_ptr + group_pid * tile_elems + q11,
                                        cache_modifier=".cv"), acc_layout, pin=False)
        for peer in range(1, 4):
            r11 += tlx.require_layout(tl.load(partials_ptr + (group_pid + peer) * tile_elems + q11,
                                              cache_modifier=".cv"), acc_layout, pin=False)
        tl.store(c_ptr + tm1[:, None] * stride_cm + tn1[None, :] * stride_cn,
                 tlx.require_layout(r11.to(et), C))




class StreamKWorkspace:
    """Reusable partials and epoch locks for one serial execution stream.

    A workspace must not be used concurrently. Unique monotonically increasing
    epochs make stale lock values harmless and remove the per-call reset launch.
    """

    def __init__(self, device):
        self.partials = torch.empty((NUM_CU, 256, 256), device=device, dtype=torch.float32)
        self.locks = torch.zeros((NUM_CU,), device=device, dtype=torch.int32)


def matmul(a, b, workspace=None):
    """Run the exact-shape Stream-K schedule.

    Pass a ``StreamKWorkspace`` for the steady-state one-launch path. Without
    one, this uses a cache-correct explicit reset before the main kernel.
    """
    assert a.shape == (M, K) and b.shape == (K, N)
    assert a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16)
    assert a.is_cuda and b.is_cuda
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    epoch = _next_epoch()
    if workspace is None:
        partials = torch.empty((NUM_CU, 256, 256), device=a.device, dtype=torch.float32)
        # This ordered zero-fill is required. Per-CTA resets inside the main
        # kernel are not a grid barrier and can erase another publication.
        locks = torch.zeros((NUM_CU,), device=a.device, dtype=torch.int32)
    else:
        assert workspace.partials.device == a.device
        partials, locks = workspace.partials, workspace.locks
    streamk_tritonblas_kernel[(NUM_CU,)](
        a, b, c, partials, locks, epoch,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        num_warps=8, num_stages=1, matrix_instr_nonkdim=16,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),))
    return c
