"""
AMD Flash Attention Forward — Async DMA Kernel (CDNA4)
============================================================

Usage:
    # Defaults: -b 1 -hq 64 -sq 1024 8192 16384 -d 64 128 -causal false --kernel async_simple
    python amd-fa-pipelined_test.py

    # Sweep sequence lengths and head dims
    python amd-fa-pipelined_test.py -sq 512 1024 4096 -d 64 128

    # Both causal modes, multiple batch sizes
    python amd-fa-pipelined_test.py -b 1 2 -causal true false

    # Multiple kernels
    python amd-fa-pipelined_test.py --kernel async_simple --dtype fp16
"""

import argparse
import math
import pytest
import torch
import torch.nn.functional as F

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _assume_strides(
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
):
    tl.assume(stride_qz >= 0)
    tl.assume(stride_qh >= 0)
    tl.assume(stride_qm > 0)
    tl.assume(stride_qk >= 0)
    tl.assume(stride_kz >= 0)
    tl.assume(stride_kh >= 0)
    tl.assume(stride_kn > 0)
    tl.assume(stride_kk >= 0)
    tl.assume(stride_vz >= 0)
    tl.assume(stride_vh >= 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_vk >= 0)
    tl.assume(stride_oz >= 0)
    tl.assume(stride_oh >= 0)
    tl.assume(stride_om > 0)
    tl.assume(stride_ok >= 0)


@triton.jit
def _remap_xcd(pid, grid, NUM_XCDS: tl.constexpr = 8):
    pids_per_xcd = tl.cdiv(grid, NUM_XCDS)
    tall_xcds = grid % NUM_XCDS
    tall_xcds = tl.where(tall_xcds == 0, NUM_XCDS, tall_xcds)
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    remapped = tl.where(
        xcd < tall_xcds,
        xcd * pids_per_xcd + local_pid,
        tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid,
    )
    return remapped


@triton.jit
def _attn_fwd_async_simple(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    REMAP_XCD: tl.constexpr,
):
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    if REMAP_XCD:
        pid_hz = _remap_xcd(pid_hz, Z * H)
    off_z = pid_hz // H
    off_h = pid_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX,
                other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    QK_SCALE = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, 1)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, 1)

    k_base = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_base = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    for start_n in tl.range(0, hi, BLOCK_N, num_stages=0):
        kn = start_n + offs_n
        k_mask = kn[:, None] < N_CTX
        v_mask = kn[:, None] < N_CTX

        tok_k = tlx.async_load(k_base + start_n * stride_kn, tlx.local_view(k_buf, 0), mask=k_mask)
        tok_v = tlx.async_load(v_base + start_n * stride_vn, tlx.local_view(v_buf, 0), mask=v_mask)
        tlx.async_load_commit_group([tok_k, tok_v])

        wait_tok = tlx.async_load_wait_group(0)
        # Transpose K at the memdesc level (metadata-only) so local_load lands
        # directly in dot_op(opIdx=1) layout — skips the register-shuffle + LDS
        # round-trip that `tl.dot(q, k_cur.T)` would otherwise emit.
        kt_view = tlx.local_trans(tlx.local_view(k_buf, 0))
        kt_cur = tlx.local_load(kt_view, token=wait_tok, relaxed=True)
        v_cur = tlx.local_load(tlx.local_view(v_buf, 0), token=wait_tok, relaxed=True)

        qk = tl.dot(q, kt_cur)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        qk = qk * QK_SCALE - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


@triton.jit
def _attn_fwd_async_fav3(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    REMAP_XCD: tl.constexpr,
    CAUSAL_SPLIT: tl.constexpr,
):
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    if REMAP_XCD:
        pid_hz = _remap_xcd(pid_hz, Z * H)
    off_z = pid_hz // H
    off_h = pid_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX,
                other=0.0)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    NUM_BUFFERS: tl.constexpr = 4
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_BUFFERS)

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = (hi + BLOCK_N - 1) // BLOCK_N
    n_main = tl.maximum(n_blocks - NUM_BUFFERS, 0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for stage in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        start_n = stage * BLOCK_N
        mask = (start_n + offs_n[:, None]) < hi
        tok_k = tlx.async_load(k_ptrs + start_n * stride_kn, tlx.local_view(k_buf, stage), mask=mask)
        tok_v = tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf, stage), mask=mask)
        tlx.async_load_commit_group([tok_k, tok_v])

    wait_tok = tlx.async_load_wait_group(NUM_BUFFERS - 1)
    kt_view0 = tlx.local_trans(tlx.local_view(k_buf, 0))
    kt_cur = tlx.local_load(kt_view0, token=wait_tok, relaxed=True)
    v_cur = tlx.local_load(tlx.local_view(v_buf, 0), token=wait_tok, relaxed=True)

    for block_id in tl.range(0, n_main * BLOCK_N, BLOCK_N, num_stages=0):
        i = block_id // BLOCK_N
        slot_cur = i % NUM_BUFFERS
        slot_nxt = (i + 1) % NUM_BUFFERS
        future_n = block_id + NUM_BUFFERS * BLOCK_N

        with tlx.warp_pipeline_stage("dot1", priority=0):
            qk = tl.dot(q, kt_cur)

        with tlx.warp_pipeline_stage("mem", priority=1):
            future_mask = (future_n + offs_n[:, None]) < hi
            tok_k = tlx.async_load(k_ptrs + future_n * stride_kn, tlx.local_view(k_buf, slot_cur), mask=future_mask)
            tok_v = tlx.async_load(v_ptrs + future_n * stride_vn, tlx.local_view(v_buf, slot_cur), mask=future_mask)
            tlx.async_load_commit_group([tok_k, tok_v])

        with tlx.warp_pipeline_stage("softmax", priority=0):
            kn = block_id + offs_n
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
            p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        with tlx.warp_pipeline_stage("dot2", priority=0):
            acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

        wait_tok = tlx.async_load_wait_group(NUM_BUFFERS - 1)
        kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_nxt))
        kt_cur = tlx.local_load(kt_view, token=wait_tok, relaxed=True)
        v_cur = tlx.local_load(tlx.local_view(v_buf, slot_nxt), token=wait_tok, relaxed=True)

    for tail in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        block_id = (n_main + tail) * BLOCK_N
        kn = block_id + offs_n

        with tlx.warp_pipeline_stage("tail_dot1", priority=0):
            qk = tl.dot(q, kt_cur)

        with tlx.warp_pipeline_stage("tail_softmax", priority=0):
            qk = tl.where(kn[None, :] < hi, qk, float("-inf"))
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
            p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        with tlx.warp_pipeline_stage("tail_dot2", priority=0):
            acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

        if tail == 0:
            wait_tok = tlx.async_load_wait_group(2)
            kt_view_tail1 = tlx.local_trans(tlx.local_view(k_buf, (n_main + 1) % NUM_BUFFERS))
            kt_cur = tlx.local_load(kt_view_tail1, token=wait_tok, relaxed=True)
            v_cur = tlx.local_load(tlx.local_view(v_buf, (n_main + 1) % NUM_BUFFERS), token=wait_tok, relaxed=True)
        if tail == 1:
            wait_tok = tlx.async_load_wait_group(1)
            kt_view_tail2 = tlx.local_trans(tlx.local_view(k_buf, (n_main + 2) % NUM_BUFFERS))
            kt_cur = tlx.local_load(kt_view_tail2, token=wait_tok, relaxed=True)
            v_cur = tlx.local_load(tlx.local_view(v_buf, (n_main + 2) % NUM_BUFFERS), token=wait_tok, relaxed=True)
        if tail == 2:
            wait_tok = tlx.async_load_wait_group(0)
            kt_view_tail3 = tlx.local_trans(tlx.local_view(k_buf, (n_main + 3) % NUM_BUFFERS))
            kt_cur = tlx.local_load(kt_view_tail3, token=wait_tok, relaxed=True)
            v_cur = tlx.local_load(tlx.local_view(v_buf, (n_main + 3) % NUM_BUFFERS), token=wait_tok, relaxed=True)

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


@triton.jit
def _attn_fwd_async_fav3_pipeline(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    REMAP_XCD: tl.constexpr,
):
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    if REMAP_XCD:
        pid_hz = _remap_xcd(pid_hz, Z * H)
    off_z = pid_hz // H
    off_h = pid_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX,
                other=0.0)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    NUM_BUFFERS: tl.constexpr = 2
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_BUFFERS)

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = (hi + BLOCK_N - 1) // BLOCK_N
    n_hot = tl.maximum(n_blocks - 2, 1)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Prologue matching the FAv3 sketch:
    # ACK0, ACK1, ACV0 -> LRK0 -> DOT1/VEC1_0 -> ACK2, ACV1 -> LRK1.
    tok_k0 = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0), mask=offs_n[:, None] < hi)
    tlx.async_load_commit_group([tok_k0])
    tok_k1 = tlx.async_load(k_ptrs + BLOCK_N * stride_kn, tlx.local_view(k_buf, 1),
                            mask=(BLOCK_N + offs_n[:, None]) < hi)
    tlx.async_load_commit_group([tok_k1])
    tok_v0 = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0), mask=offs_n[:, None] < hi)
    tlx.async_load_commit_group([tok_v0])

    wait_tok = tlx.async_load_wait_group(2)
    kt_view_p0 = tlx.local_trans(tlx.local_view(k_buf, 0))
    k_cur = tlx.local_load(kt_view_p0, token=wait_tok, relaxed=True)

    qk0 = tl.dot(q, k_cur)
    qk0 = tl.where(offs_n[None, :] < hi, qk0, float("-inf"))
    if IS_CAUSAL:
        qk0 = tl.where(offs_m[:, None] >= offs_n[None, :], qk0, float("-inf"))
    else:
        # Keep the 8-warp non-causal lowering on the predicated QK path; the
        # unpredicated form miscomputes long sequence tiles.
        qk0 = tl.where(offs_m[:, None] >= 0, qk0, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(qk0, 1) * QK_SCALE)
    p_prev = tl.math.exp2(qk0 * QK_SCALE - m_new[:, None])
    alpha_prev = tl.math.exp2(m_i - m_new)
    m_i = m_new

    tok_k2 = tlx.async_load(k_ptrs + (2 * BLOCK_N) * stride_kn, tlx.local_view(k_buf, 0),
                            mask=(2 * BLOCK_N + offs_n[:, None]) < hi)
    tlx.async_load_commit_group([tok_k2])
    tok_v1 = tlx.async_load(v_ptrs + BLOCK_N * stride_vn, tlx.local_view(v_buf, 1),
                            mask=(BLOCK_N + offs_n[:, None]) < hi)
    tlx.async_load_commit_group([tok_v1])

    wait_tok = tlx.async_load_wait_group(3)
    kt_view_p1 = tlx.local_trans(tlx.local_view(k_buf, 1))
    k_cur = tlx.local_load(kt_view_p1, token=wait_tok, relaxed=True)

    # Hot loop. Iteration t computes DOT1/VEC1[t] while retiring
    # VEC2/LRV/DOT2[t-1], then reads K[t+1] and issues V[t+1].
    for block_id in tl.range(BLOCK_N, n_hot * BLOCK_N, BLOCK_N, num_stages=0):
        t = block_id // BLOCK_N
        future_k = block_id + 2 * BLOCK_N
        future_v = block_id + BLOCK_N
        v_prev_slot = (t - 1) % 2
        k_next_slot = (t + 1) % 2
        k_overwrite_slot = t % 2

        with tlx.warp_pipeline_stage("vec2_dot1", priority=0):
            l_ij = tl.sum(p_prev, 1)
            acc_scaled = acc * alpha_prev[:, None]
            l_i = l_i * alpha_prev + l_ij
            kn = block_id + offs_n
            qk = tl.dot(q, k_cur)
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
            else:
                # See qk0: this is logically a no-op for valid program ids.
                qk = tl.where(offs_m[:, None] >= 0, qk, float("-inf"))

        wait_tok_v = tlx.async_load_wait_group(2)
        with tlx.warp_pipeline_stage("lrv_ack", priority=1):
            v_prev = tlx.local_load(tlx.local_view(v_buf, v_prev_slot), token=wait_tok_v, relaxed=True)
            tok_k = tlx.async_load(k_ptrs + future_k * stride_kn, tlx.local_view(k_buf, k_overwrite_slot),
                                   mask=(future_k + offs_n[:, None]) < hi)
            tlx.async_load_commit_group([tok_k])

        with tlx.warp_pipeline_stage("dot2_vec1", priority=0):
            acc = tl.dot(p_prev.to(v_prev.dtype), v_prev, acc_scaled)
            m_new = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
            p_cur = tl.math.exp2(qk * QK_SCALE - m_new[:, None])
            alpha_cur = tl.math.exp2(m_i - m_new)
            m_i = m_new

        wait_tok_k = tlx.async_load_wait_group(2)
        with tlx.warp_pipeline_stage("lrk_acv", priority=1):
            kt_view_lrk = tlx.local_trans(tlx.local_view(k_buf, k_next_slot))
            k_cur = tlx.local_load(kt_view_lrk, token=wait_tok_k, relaxed=True)
            tok_v = tlx.async_load(v_ptrs + future_v * stride_vn, tlx.local_view(v_buf, k_next_slot),
                                   mask=(future_v + offs_n[:, None]) < hi)
            tlx.async_load_commit_group([tok_v])
            p_prev = p_cur
            alpha_prev = alpha_cur

    last_t = n_hot
    # Epilogue 1: retire tile last_t-1.
    wait_tok_v = tlx.async_load_wait_group(2)
    v_prev = tlx.local_load(tlx.local_view(v_buf, (last_t - 1) % 2), token=wait_tok_v, relaxed=True)
    l_ij = tl.sum(p_prev, 1)
    acc = acc * alpha_prev[:, None]
    l_i = l_i * alpha_prev + l_ij
    acc = tl.dot(p_prev.to(v_prev.dtype), v_prev, acc)

    # Epilogue 2: DOT1/VEC1 for tile last_t.
    start_n = last_t * BLOCK_N
    kn = start_n + offs_n
    qk = tl.dot(q, k_cur)
    qk = tl.where(kn[None, :] < hi, qk, float("-inf"))
    if IS_CAUSAL:
        qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
    else:
        # See qk0: this is logically a no-op for valid program ids.
        qk = tl.where(offs_m[:, None] >= 0, qk, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
    p_prev = tl.math.exp2(qk * QK_SCALE - m_new[:, None])
    alpha_prev = tl.math.exp2(m_i - m_new)
    m_i = m_new

    wait_tok_k = tlx.async_load_wait_group(1)
    kt_view_tail = tlx.local_trans(tlx.local_view(k_buf, (last_t + 1) % 2))
    k_tail = tlx.local_load(kt_view_tail, token=wait_tok_k, relaxed=True)
    tok_v = tlx.async_load(v_ptrs + (last_t + 1) * BLOCK_N * stride_vn, tlx.local_view(v_buf, (last_t + 1) % 2),
                           mask=((last_t + 1) * BLOCK_N + offs_n[:, None]) < hi)
    tlx.async_load_commit_group([tok_v])

    # Epilogue 3: retire tile last_t while DOT1/VEC1 for tile last_t+1.
    wait_tok_v = tlx.async_load_wait_group(1)
    v_prev = tlx.local_load(tlx.local_view(v_buf, last_t % 2), token=wait_tok_v, relaxed=True)
    l_ij = tl.sum(p_prev, 1)
    acc = acc * alpha_prev[:, None]
    l_i = l_i * alpha_prev + l_ij
    acc = tl.dot(p_prev.to(v_prev.dtype), v_prev, acc)

    start_n = (last_t + 1) * BLOCK_N
    kn = start_n + offs_n
    qk = tl.dot(q, k_tail)
    qk = tl.where(kn[None, :] < hi, qk, float("-inf"))
    if IS_CAUSAL:
        qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
    else:
        # See qk0: this is logically a no-op for valid program ids.
        qk = tl.where(offs_m[:, None] >= 0, qk, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
    p_prev = tl.math.exp2(qk * QK_SCALE - m_new[:, None])
    alpha_prev = tl.math.exp2(m_i - m_new)
    m_i = m_new

    wait_tok_v = tlx.async_load_wait_group(0)
    v_prev = tlx.local_load(tlx.local_view(v_buf, (last_t + 1) % 2), token=wait_tok_v, relaxed=True)
    l_ij = tl.sum(p_prev, 1)
    acc = acc * alpha_prev[:, None]
    l_i = l_i * alpha_prev + l_ij
    acc = tl.dot(p_prev.to(v_prev.dtype), v_prev, acc)

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


@triton.jit
def _attn_fwd_async_prefetch(
    Q,
    K,
    V,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    REMAP_XCD: tl.constexpr,
    CAUSAL_SPLIT: tl.constexpr,
):
    """
    Prefetch flash attention with explicit modulo-scheduled prologue,
    hot loop (steady state), and epilogue.

    Design notes:
      * K and V are both double-buffered (2 LDS slots, ping-pong index
        i%2).
      * `local_trans` is applied to K so `local_load` lands directly in
        dot-operand layout 1, skipping the per-iter ds_write+barrier+
        ds_read shuffle that `tl.dot(q, k_cur.T)` would emit.
    Prologue:
    t = 0
    [GLDS_KV]

    Steady State (Hot Loop):
    t = i               t = i+1
    [LR_KV]
    [QK, SM0, SM1, PV]  [GLDS_KV],
    Epilogue:
                        t = i+1
                        [LR_KV]
                        [QK (masked), SM0, SM1, PV]
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    if REMAP_XCD:
        pid_hz = _remap_xcd(pid_hz, Z * H)
    off_z = pid_hz // H
    off_h = pid_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX,
                other=0.0)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    # K and V: 2 LDS slots each (ping-pong) -- avoids both the memdesc_trans
    # alias race for K and any single-buf RAW hazards.
    NUM_BUFFERS: tl.constexpr = 2
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_BUFFERS)

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = (hi + BLOCK_N - 1) // BLOCK_N
    n_main = tl.maximum(n_blocks - 1, 0)
    if IS_CAUSAL and CAUSAL_SPLIT:
        n_unmasked_blocks = tl.maximum(n_blocks - (BLOCK_M + BLOCK_N - 1) // BLOCK_N, 0)
    else:
        n_unmasked_blocks = 0

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    """
    Prologue:
    t = 0
    [GLDS_KV]
    """
    tok_k0 = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0), mask=offs_n[:, None] < N_CTX)
    tok_v0 = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0), mask=offs_n[:, None] < N_CTX)
    tlx.async_load_commit_group([tok_k0, tok_v0])
    """
    Steady State (Hot Loop):
    t = i               t = i+1
    [LR_KV]
    [QK, SM0, SM1, PV]  [GLDS_KV],
    """
    for block_id in tl.range(0, n_main * BLOCK_N, BLOCK_N, num_stages=0):
        next_off = block_id + BLOCK_N
        kn = block_id + offs_n
        next_mask = (next_off + offs_n[:, None]) < N_CTX

        i = block_id // BLOCK_N
        slot_cur = i % 2
        slot_nxt = (i + 1) % 2

        # LR_KV_ti
        wait_tok = tlx.async_load_wait_group(0)
        kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_cur))
        kt_cur = tlx.local_load(kt_view, token=wait_tok, relaxed=True)
        v_cur = tlx.local_load(tlx.local_view(v_buf, slot_cur), token=wait_tok, relaxed=True)

        # GLDS_KV_t(i+1), prefetch tile i+1 into the *other* slots.
        tok_k = tlx.async_load(k_ptrs + next_off * stride_kn, tlx.local_view(k_buf, slot_nxt), mask=next_mask)
        tok_v = tlx.async_load(v_ptrs + next_off * stride_vn, tlx.local_view(v_buf, slot_nxt), mask=next_mask)
        tlx.async_load_commit_group([tok_k, tok_v])

        # QK_ti
        qk = tl.dot(q, kt_cur)
        if IS_CAUSAL:
            if CAUSAL_SPLIT:
                if i >= n_unmasked_blocks:
                    qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
            else:
                qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))

        # SM_ti
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # PV_ti
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)
    """
    Epilogue:
    t = i+1
    [LR_KV]
    [QK (masked), SM0, SM1, PV]
    """
    # Consume tile n_main from slot (n_main % 2).
    wait_tok = tlx.async_load_wait_group(0)
    slot_last = n_main % 2
    kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_last))
    kt_cur = tlx.local_load(kt_view, token=wait_tok, relaxed=True)
    v_cur = tlx.local_load(tlx.local_view(v_buf, slot_last), token=wait_tok, relaxed=True)

    # QK_t(i+1) — with boundary + causal masking
    kn_last = n_main * BLOCK_N + offs_n
    qk = tl.dot(q, kt_cur)
    qk = tl.where(kn_last[None, :] < N_CTX, qk, float("-inf"))
    if IS_CAUSAL:
        if CAUSAL_SPLIT:
            if n_main >= n_unmasked_blocks:
                qk = tl.where(offs_m[:, None] >= kn_last[None, :], qk, float("-inf"))
        else:
            qk = tl.where(offs_m[:, None] >= kn_last[None, :], qk, float("-inf"))

    # SM0_t(i+1)
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
    p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
    l_ij = tl.sum(p, 1)

    # SM1_t(i+1)
    alpha = tl.math.exp2(m_i - m_ij)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    m_i = m_ij

    # PV_t(i+1)
    acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

    # Store output
    acc = acc / l_i[:, None]
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


# ═══════════════════════════════════════════════════════════════════════════
# Host wrapper
# ═══════════════════════════════════════════════════════════════════════════


def flash_attn_async_simple(q, k, v, sm_scale, causal=False, **kw):
    """Launch with K in original BHND layout — stride_kk=1 avoids alignment issues."""
    B, H, N_CTX, D = q.shape
    o = torch.empty_like(q)

    BLOCK_M = kw.pop("BLOCK_M", 256)
    BLOCK_N = kw.pop("BLOCK_N", 64)
    num_warps = kw.pop("num_warps", 4)

    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)
    _attn_fwd_async_simple[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        B,
        H,
        N_CTX,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
        REMAP_XCD=False,
        num_warps=num_warps,
        **kw,
    )
    return o


def flash_attn_async_prefetch(q, k, v, sm_scale, causal=False, **kw):
    """Prefetch FA with modulo-scheduled prologue/hot-loop/epilogue."""
    B, H, N_CTX, D = q.shape
    o = torch.empty_like(q)

    BLOCK_M = kw.pop("BLOCK_M", 256)
    # BLOCK_N=128 wins at D=64 nocausal (more compute per barrier),
    # but the diagonal masking cost overwhelms that for causal, and at
    # D=128 it blows the 64KB LDS budget for double-buffered K+V.
    BLOCK_N = kw.pop("BLOCK_N", 128 if (D <= 64 and not causal) else 64)
    num_warps = kw.pop("num_warps", 4)

    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)
    _attn_fwd_async_prefetch[grid](
        q,
        k,
        v,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        B,
        H,
        N_CTX,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        IS_CAUSAL=causal,
        REMAP_XCD=kw.pop("REMAP_XCD", False),
        CAUSAL_SPLIT=kw.pop("CAUSAL_SPLIT", False),
        num_warps=num_warps,
        **kw,
    )
    return o


def flash_attn_async_fav3(q, k, v, sm_scale, causal=False, **kw):
    """MI350x FAv3 entry point with shape-selected TLX schedules."""
    B, H, N_CTX, D = q.shape
    use_fav3_pipeline = kw.pop("USE_FAV3_PIPELINE", False)

    if causal and N_CTX >= 4096:
        kw.setdefault("BLOCK_M", 128)
        if D <= 64:
            kw.setdefault("BLOCK_N", 128)
            kw.setdefault("num_warps", 4)
        else:
            kw.setdefault("BLOCK_N", 64)
            kw.setdefault("num_warps", 2)
    elif D <= 64 and not causal:
        kw.setdefault("num_warps", 8)
    else:
        kw.setdefault("num_warps", 4)

    if use_fav3_pipeline and N_CTX >= 256:
        o = torch.empty_like(q)

        BLOCK_M = kw.pop("BLOCK_M", 256)
        BLOCK_N = kw.pop("BLOCK_N", 64 if D <= 64 else 32)
        num_warps = kw.pop("num_warps", 8)

        grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)
        _attn_fwd_async_fav3_pipeline[grid](
            q,
            k,
            v,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            B,
            H,
            N_CTX,
            sm_scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=D,
            IS_CAUSAL=causal,
            REMAP_XCD=kw.pop("REMAP_XCD", False),
            num_warps=num_warps,
            **kw,
        )
        return o

    # Older 4-stage staging prototype. Kept as an explicit experiment; the
    # FAv3-like ACK/ACV/LRK/DOT1/VEC1/VEC2/LRV/DOT2 pipeline is above.
    if kw.pop("USE_FAV3_BODY", False) and N_CTX >= 256:
        o = torch.empty_like(q)

        BLOCK_M = kw.pop("BLOCK_M", 256)
        BLOCK_N = kw.pop("BLOCK_N", 64)
        num_warps = kw.pop("num_warps")

        grid = (triton.cdiv(N_CTX, BLOCK_M), B * H)
        _attn_fwd_async_fav3[grid](
            q,
            k,
            v,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            B,
            H,
            N_CTX,
            sm_scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=D,
            IS_CAUSAL=causal,
            REMAP_XCD=kw.pop("REMAP_XCD", False),
            num_warps=num_warps,
            **kw,
        )
        return o

    return flash_attn_async_prefetch(q, k, v, sm_scale, causal, **kw)


def flash_attn_async_fav3_pipeline(q, k, v, sm_scale, causal=False, **kw):
    """Explicit 8-warp FAv3 ping-pong pipeline experiment."""
    kw.setdefault("num_warps", 8)
    kw.setdefault("USE_FAV3_PIPELINE", True)
    return flash_attn_async_fav3(q, k, v, sm_scale, causal, **kw)


# ═══════════════════════════════════════════════════════════════════════════
# Kernel registry — add new kernel wrappers here
# ═══════════════════════════════════════════════════════════════════════════

KERNEL_REGISTRY = {
    "async_simple": flash_attn_async_simple,
    "async_prefetch": flash_attn_async_prefetch,
    "async_fav3": flash_attn_async_fav3,
    "async_fav3_pipeline": flash_attn_async_fav3_pipeline,
}


def get_kernel(name):
    if name not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel: {name!r}. "
                         f"Available: {list(KERNEL_REGISTRY.keys())}")
    return KERNEL_REGISTRY[name]


# ═══════════════════════════════════════════════════════════════════════════
# Reference, verification, and Misc Utils
# ═══════════════════════════════════════════════════════════════════════════


def print_summary_table(results, kernel_names):
    """Print a markdown-style summary table of benchmark results."""
    providers = ["Torch SDPA"] + list(kernel_names)

    rows = []
    for key in sorted(results.keys()):
        B, H, D, N, causal = key
        rows.append((f"B={B}, H={H}, D={D}, N={N}, causal={causal}", results[key]))

    cfg_w = max(len("Config"), *(len(lbl) for lbl, _ in rows)) if rows else len("Config")
    col_w = max(14, *(len(p) for p in providers))

    hdr = f"| {'Config':<{cfg_w}} |" + "".join(f" {p:>{col_w}} |" for p in providers)
    sep = f"|{'-' * (cfg_w + 2)}|" + "".join(f"{'-' * (col_w + 2)}|" for _ in providers)

    print(f"\n{'=' * len(sep)}")
    print("Summary (TFLOPS)")
    print(f"{'=' * len(sep)}")
    print(hdr)
    print(sep)

    for label, prov in rows:
        vals = (f"{prov[p]['tflops']:>{col_w}.1f}" if p in prov else f"{'—':>{col_w}}" for p in providers)
        print(f"| {label:<{cfg_w}} |" + "".join(f" {v} |" for v in vals))

    print(f"{'=' * len(sep)}\n")


def ref_sdpa(q, k, v, sm_scale, causal=False):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)


def verify(name, got, ref, atol=2e-2, rtol=2e-2, log=True):
    diff = (got.float() - ref.float()).abs()
    ok = torch.allclose(ref, got, atol=atol, rtol=rtol)
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    status = "PASS" if ok else "FAIL"
    if log:
        print(f"  {name:<28} {status}  max={max_err:.6f}  mean={mean_err:.6f}")
    return ok


def run_correctness_check(kernel_fn, dtype, causal, B=2, H=4, N=512, D=128):
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
    sm = 1.0 / math.sqrt(D)
    ref = ref_sdpa(q, k, v, sm, causal)
    tag = f"causal={causal} N={N}"
    out = kernel_fn(q, k, v, sm, causal)
    return verify(f"{kernel_fn.__name__} [{tag}]", out, ref)


@pytest.mark.parametrize("causal", [False, True], ids=["nocausal", "causal"])
@pytest.mark.parametrize("n_test", [128, 192, 256, 500, 512, 1024])
@pytest.mark.parametrize("kernel_name", list(KERNEL_REGISTRY.keys()))
def test_fa_correctness(kernel_name, causal, n_test, dtype=torch.bfloat16, D=128):
    kernel_fn = get_kernel(kernel_name)
    ok = run_correctness_check(kernel_fn, dtype, causal, B=1, H=4, N=n_test, D=D)
    assert ok, f"Correctness failed: kernel={kernel_name} causal={causal} N={n_test}"


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════


def run_benchmark(args):
    causal_modes = [s.lower() in ("true", "1", "yes") for s in args.causal]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    results = {}

    for kernel_name in args.kernel:
        kernel_fn = get_kernel(kernel_name)
        for B in args.b:
            for H in args.hq:
                for D in args.d:
                    for N in args.sq:
                        for causal in causal_modes:
                            torch.manual_seed(42)
                            q = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
                            k = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
                            v = torch.randn(B, H, N, D, device=DEVICE, dtype=dtype)
                            sm = 1.0 / math.sqrt(D)

                            if causal:
                                valid_el = N * (N + 1) // 2
                            else:
                                valid_el = N * N
                            total_flops = 2 * 2.0 * B * H * valid_el * D

                            causal_str = "causal" if causal else "nc"
                            ref_sdpa_lambda = lambda: F.scaled_dot_product_attention(
                                q, k, v, is_causal=causal, scale=sm)

                            try:
                                tlx_sdpa_lambda = lambda: kernel_fn(q, k, v, sm, causal)
                                ref_out = ref_sdpa_lambda()
                                tlx_out = tlx_sdpa_lambda()
                                assert verify("", tlx_out, ref_out, log=False)
                            except Exception as e:
                                print(f"  {kernel_name:20s} D={D} N={N:5d} {causal_str:6s} -> SKIPPED ({e})")
                                continue

                            key = (B, H, D, N, causal)
                            if key not in results:
                                results[key] = {}

                            if "Torch SDPA" not in results[key]:
                                ms = triton.testing.do_bench(ref_sdpa_lambda, warmup=25, rep=100)
                                tflops = total_flops / ms * 1e-9
                                results[key]["Torch SDPA"] = {"ms": ms, "tflops": tflops}

                            ms = triton.testing.do_bench(tlx_sdpa_lambda, warmup=25, rep=100)
                            tflops = total_flops / ms * 1e-9
                            results[key][kernel_name] = {"ms": ms, "tflops": tflops}

    print_summary_table(results, args.kernel)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(prog="AMD TLX FA Pipelined")
    p.add_argument("-b", type=int, nargs="+", default=[1])
    p.add_argument("-hq", type=int, nargs="+", default=[64])
    p.add_argument("-sq", type=int, nargs="+", default=[1024, 8192, 16384])
    p.add_argument("-d", type=int, nargs="+", default=[64, 128])
    p.add_argument("-causal", type=str, nargs="+", default=["false"],
                   help="Causal modes to benchmark (e.g. -causal true false)")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--kernel", type=str, nargs="+", default=["async_simple", "async_prefetch", "async_fav3"],
                   help="Kernel variants to benchmark")
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
