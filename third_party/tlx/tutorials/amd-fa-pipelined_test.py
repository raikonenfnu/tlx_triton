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
def _remap_xcd(pid, PIDS_PER_XCD: tl.constexpr, TALL_XCDS: tl.constexpr, NUM_XCDS: tl.constexpr = 8):
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    return tl.where(xcd < TALL_XCDS, xcd * PIDS_PER_XCD + local_pid,
                    TALL_XCDS * PIDS_PER_XCD + (xcd - TALL_XCDS) * (PIDS_PER_XCD - 1) + local_pid)


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
):
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
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
        k_cur = tlx.local_load(tlx.local_view(k_buf, 0), token=wait_tok, relaxed=True)
        v_cur = tlx.local_load(tlx.local_view(v_buf, 0), token=wait_tok, relaxed=True)

        qk = tl.dot(q, k_cur.T)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        if start_n + BLOCK_N > N_CTX:
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
):
    """
    Prefetch flash attention with explicit modulo-scheduled prologue,
    hot loop (steady state), and epilogue

    Single-buffered K/V shared memory.
    Each commit group bundles K and V for the same tile.

    Prologue:
    t = 0
    [GLDS_KV]
    [LR_KV]

    Steady State (Hot Loop):
    t = i               t = i+1
    [QK, SM0, SM1, PV]  [GLDS_KV],
                        [LR_KV]

    Epilogue:
                        t = i+1
                        [QK (masked), SM0, SM1, PV]
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
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

    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, 1)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, 1)

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = (hi + BLOCK_N - 1) // BLOCK_N
    n_main = tl.maximum(n_blocks - 1, 0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    """
    Prologue:
    t = 0
    [GLDS_KV]
    [LR_KV]
    """
    # GLDS_KV_t0
    tok_k = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0), mask=offs_n[:, None] < N_CTX)
    tok_v = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0), mask=offs_n[:, None] < N_CTX)
    tlx.async_load_commit_group([tok_k, tok_v])

    # LR_KV_t0
    wait_tok = tlx.async_load_wait_group(0)
    k_cur = tlx.local_load(tlx.local_view(k_buf, 0), token=wait_tok, relaxed=True)
    v_cur = tlx.local_load(tlx.local_view(v_buf, 0), token=wait_tok, relaxed=True)
    """
    Steady State (Hot Loop):
    t = i               t = i+1
    [QK, SM0, SM1, PV]  [GLDS_KV],
                        [LR_KV]
    """
    for block_id in tl.range(0, n_main * BLOCK_N, BLOCK_N, num_stages=0):
        next_off = block_id + BLOCK_N
        kn = block_id + offs_n
        next_mask = (next_off + offs_n[:, None]) < N_CTX

        # QK_ti
        qk = tl.dot(q, k_cur.T)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))

        # SM0_ti
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # GLDS after SM0: overlaps with SM1 + PV
        tok_k = tlx.async_load(k_ptrs + next_off * stride_kn, tlx.local_view(k_buf, 0), mask=next_mask)
        tok_v = tlx.async_load(v_ptrs + next_off * stride_vn, tlx.local_view(v_buf, 0), mask=next_mask)
        tlx.async_load_commit_group([tok_k, tok_v])

        # SM1_ti
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # PV_ti
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

        # LR_KV_t(i+1)
        wait_tok = tlx.async_load_wait_group(0)
        k_cur = tlx.local_load(tlx.local_view(k_buf, 0), token=wait_tok, relaxed=True)
        v_cur = tlx.local_load(tlx.local_view(v_buf, 0), token=wait_tok, relaxed=True)
    """
    Epilogue:
    t = i+1
    [QK (masked), SM0, SM1, PV]
    """
    kn_last = n_main * BLOCK_N + offs_n

    # QK_t(i+1) — with boundary + causal masking
    qk = tl.dot(q, k_cur.T)
    qk = tl.where(kn_last[None, :] < N_CTX, qk, float("-inf"))
    if IS_CAUSAL:
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


@triton.jit
def _fa_softmax_part0(
    qk,
    m_i,
    QK_SCALE: tl.constexpr,
):
    m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
    p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    m_i = m_ij
    return p, l_ij, alpha, m_i


@triton.jit
def _fa_softmax_part1(
    acc,
    l_i,
    p,
    l_ij,
    alpha,
):
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    return acc, l_i, p


@triton.jit
def _fa_max_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _fa_apply_softmax(
    acc,
    l_i,
    m_i,
    qk,
    offs_m,
    kn,
    N_CTX,
    QK_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MASK_STEPS: tl.constexpr,
):
    if MASK_STEPS:
        qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))
    if IS_CAUSAL:
        qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))

    qk_max = tl.reduce(qk, 1, _fa_max_nan)
    m_ij = tl.maximum(m_i, qk_max * QK_SCALE, propagate_nan=tl.PropagateNan.ALL)
    p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    return acc, l_i, m_i, p


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
    N_CTX: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    USE_XCD_REMAP: tl.constexpr,
    XCD_PIDS_PER_XCD: tl.constexpr,
    XCD_TALL_XCDS: tl.constexpr,
):
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    off_z = pid_hz // H
    off_h_raw = pid_hz % H
    if USE_XCD_REMAP:
        off_h = _remap_xcd(off_h_raw, XCD_PIDS_PER_XCD, XCD_TALL_XCDS)
    else:
        off_h = off_h_raw

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX,
                other=0.0)
    q_smem = tlx.local_alloc((BLOCK_M, HEAD_DIM), Q.dtype.element_ty, 1,
                             layout=tlx.swizzled_shared_layout_encoding(8, 1, 16, [1, 0], [1, 1], [1, 1], [1, 1],
                                                                        [1, 1]))
    q_smem_view = tlx.local_view(q_smem, 0)
    tlx.local_store(q_smem_view, q)
    q = tlx.local_load(q_smem_view)
    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    kt_ptrs = K + k_off + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = tl.cdiv(hi, BLOCK_N)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if IS_CAUSAL:
        prefix_blocks = tl.maximum(n_blocks - (BLOCK_M // BLOCK_N), 0)
        k_buf_fast = tlx.local_alloc(
            (HEAD_DIM, BLOCK_N), K.dtype.element_ty, NUM_STAGES,
            layout=tlx.padded_shared_layout_encoding.with_gfx950_fa_bases([(512, 8)], (HEAD_DIM, BLOCK_N), 0))
        v_buf_fast = tlx.local_alloc(
            (BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_STAGES,
            layout=tlx.padded_shared_layout_encoding.with_gfx950_fa_bases([(512, 32)], (BLOCK_N, HEAD_DIM), 1))

        if prefix_blocks > NUM_STAGES:
            for stage in tl.range(0, NUM_STAGES, loop_unroll_factor=NUM_STAGES):
                start_n = stage * BLOCK_N
                tlx.async_load(kt_ptrs + start_n * stride_kn, tlx.local_view(k_buf_fast, stage))
                tlx.async_load_commit_group()
                tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf_fast, stage))
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(2 * NUM_STAGES - 1)
            k_tile = tlx.local_load(tlx.local_view(k_buf_fast, 0), relaxed=True)

            main_loop_end = prefix_blocks - NUM_STAGES
            for block_id in tl.range(0, main_loop_end, loop_unroll_factor=4):
                stage_idx = block_id % NUM_STAGES
                next_stage_idx = (block_id + 1) % NUM_STAGES
                start_n = block_id * BLOCK_N
                future_start_n = (block_id + NUM_STAGES) * BLOCK_N
                kn = start_n + offs_n

                with tlx.warp_pipeline_stage("dot1", priority=0):
                    qk = tl.dot(q, k_tile)

                tlx.async_load_wait_group(2 * NUM_STAGES - 2)

                with tlx.warp_pipeline_stage("mem1", priority=1):
                    v_tile = tlx.local_load(tlx.local_view(v_buf_fast, stage_idx), relaxed=True)
                    tlx.async_load(kt_ptrs + future_start_n * stride_kn, tlx.local_view(k_buf_fast, stage_idx))
                    tlx.async_load_commit_group()

                with tlx.warp_pipeline_stage("dot2a", priority=0):
                    acc, l_i, m_i, p = _fa_apply_softmax(acc, l_i, m_i, qk, offs_m, kn, N_CTX, QK_SCALE, BLOCK_M,
                                                         BLOCK_N, False, False)
                    p = p.to(v_tile.dtype)

                with tlx.warp_pipeline_stage("dot2b", priority=0):
                    acc = tl.dot(p, v_tile, acc)

                tlx.async_load_wait_group(2 * NUM_STAGES - 2)

                with tlx.warp_pipeline_stage("mem2", priority=1):
                    k_tile = tlx.local_load(tlx.local_view(k_buf_fast, next_stage_idx), relaxed=True)
                    tlx.async_load(v_ptrs + future_start_n * stride_vn, tlx.local_view(v_buf_fast, stage_idx))
                    tlx.async_load_commit_group()

            for tail_i in tl.static_range(0, NUM_STAGES):
                stage_idx = (main_loop_end + tail_i) % NUM_STAGES
                start_n = (main_loop_end + tail_i) * BLOCK_N
                kn = start_n + offs_n

                tlx.async_load_wait_group(2 * (NUM_STAGES - tail_i) - 1)
                k_tail = tlx.local_load(tlx.local_view(k_buf_fast, stage_idx), relaxed=True)
                qk = tl.dot(q, k_tail)
                acc, l_i, m_i, p = _fa_apply_softmax(acc, l_i, m_i, qk, offs_m, kn, N_CTX, QK_SCALE, BLOCK_M, BLOCK_N,
                                                     False, False)

                tlx.async_load_wait_group(2 * (NUM_STAGES - tail_i) - 2)
                v_tail = tlx.local_load(tlx.local_view(v_buf_fast, stage_idx), relaxed=True)
                acc = tl.dot(p.to(v_tail.dtype), v_tail, acc)
        else:
            for block_id in tl.range(0, prefix_blocks, num_stages=0):
                start_n = block_id * BLOCK_N
                kn = start_n + offs_n

                tok_k = tlx.async_load(kt_ptrs + start_n * stride_kn, tlx.local_view(k_buf_fast, 0))
                tok_v = tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf_fast, 0))
                tlx.async_load_commit_group([tok_k, tok_v])
                wait_tok = tlx.async_load_wait_group(0)
                k_cur = tlx.local_load(tlx.local_view(k_buf_fast, 0), token=wait_tok, relaxed=True)
                v_cur = tlx.local_load(tlx.local_view(v_buf_fast, 0), token=wait_tok, relaxed=True)
                qk = tl.dot(q, k_cur)
                acc, l_i, m_i, p = _fa_apply_softmax(acc, l_i, m_i, qk, offs_m, kn, N_CTX, QK_SCALE, BLOCK_M, BLOCK_N,
                                                     False, False)
                acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

        if BLOCK_M // BLOCK_N <= NUM_STAGES:
            tail_blocks = n_blocks - prefix_blocks
            for tail_i in tl.range(0, tail_blocks, num_stages=0):
                start_n = (prefix_blocks + tail_i) * BLOCK_N
                kn = start_n + offs_n
                mask_n = kn < N_CTX

                tlx.async_load(kt_ptrs + start_n * stride_kn, tlx.local_view(k_buf_fast, tail_i), mask=mask_n[None, :])
                tlx.async_load_commit_group()
                tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf_fast, tail_i), mask=mask_n[:, None])
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(0)

            for tail_i in tl.range(0, tail_blocks, num_stages=0):
                start_n = (prefix_blocks + tail_i) * BLOCK_N
                kn = start_n + offs_n

                k_cur = tlx.local_load(tlx.local_view(k_buf_fast, tail_i), relaxed=True)
                v_cur = tlx.local_load(tlx.local_view(v_buf_fast, tail_i), relaxed=True)
                qk = tl.dot(q, k_cur)
                acc, l_i, m_i, p = _fa_apply_softmax(acc, l_i, m_i, qk, offs_m, kn, N_CTX, QK_SCALE, BLOCK_M, BLOCK_N,
                                                     True, True)
                acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)
        else:
            for block_id in tl.range(prefix_blocks, n_blocks, num_stages=0):
                start_n = block_id * BLOCK_N
                kn = start_n + offs_n
                mask_n = kn < N_CTX

                tok_k = tlx.async_load(kt_ptrs + start_n * stride_kn, tlx.local_view(k_buf_fast, 0),
                                       mask=mask_n[None, :])
                tok_v = tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf_fast, 0), mask=mask_n[:,
                                                                                                                None])
                tlx.async_load_commit_group([tok_k, tok_v])
                wait_tok = tlx.async_load_wait_group(0)
                k_cur = tlx.local_load(tlx.local_view(k_buf_fast, 0), token=wait_tok, relaxed=True)
                v_cur = tlx.local_load(tlx.local_view(v_buf_fast, 0), token=wait_tok, relaxed=True)
                qk = tl.dot(q, k_cur)
                acc, l_i, m_i, p = _fa_apply_softmax(acc, l_i, m_i, qk, offs_m, kn, N_CTX, QK_SCALE, BLOCK_M, BLOCK_N,
                                                     True, True)
                acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)
    elif (not MASK_STEPS) and n_blocks > NUM_STAGES:
        k_buf_fast = tlx.local_alloc(
            (HEAD_DIM, BLOCK_N), K.dtype.element_ty, NUM_STAGES,
            layout=tlx.padded_shared_layout_encoding.with_gfx950_fa_bases([(512, 8)], (HEAD_DIM, BLOCK_N), 0))
        v_buf_fast = tlx.local_alloc(
            (BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_STAGES,
            layout=tlx.padded_shared_layout_encoding.with_gfx950_fa_bases([(512, 32)], (BLOCK_N, HEAD_DIM), 1))

        for stage in tl.range(0, NUM_STAGES, loop_unroll_factor=NUM_STAGES):
            start_n = stage * BLOCK_N
            tlx.async_load(kt_ptrs + start_n * stride_kn, tlx.local_view(k_buf_fast, stage))
            tlx.async_load_commit_group()
            tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf_fast, stage))
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(2 * NUM_STAGES - 1)
        k_tile = tlx.local_load(tlx.local_view(k_buf_fast, 0), relaxed=True)

        main_loop_end = n_blocks - NUM_STAGES
        for block_id in tl.range(0, main_loop_end, loop_unroll_factor=4):
            stage_idx = block_id % NUM_STAGES
            next_stage_idx = (block_id + 1) % NUM_STAGES
            start_n = block_id * BLOCK_N
            future_start_n = (block_id + NUM_STAGES) * BLOCK_N
            kn = start_n + offs_n

            with tlx.warp_pipeline_stage("dot1", priority=0):
                qk = tl.dot(q, k_tile)

            tlx.async_load_wait_group(2 * NUM_STAGES - 2)

            with tlx.warp_pipeline_stage("mem1", priority=1):
                v_tile = tlx.local_load(tlx.local_view(v_buf_fast, stage_idx), relaxed=True)
                tlx.async_load(kt_ptrs + future_start_n * stride_kn, tlx.local_view(k_buf_fast, stage_idx))
                tlx.async_load_commit_group()

            with tlx.warp_pipeline_stage("dot2a", priority=0):
                acc, l_i, m_i, p = _fa_apply_softmax(acc, l_i, m_i, qk, offs_m, kn, N_CTX, QK_SCALE, BLOCK_M, BLOCK_N,
                                                     False, False)
                p = p.to(v_tile.dtype)

            with tlx.warp_pipeline_stage("dot2b", priority=0):
                acc = tl.dot(p, v_tile, acc)

            tlx.async_load_wait_group(2 * NUM_STAGES - 2)

            with tlx.warp_pipeline_stage("mem2", priority=1):
                k_tile = tlx.local_load(tlx.local_view(k_buf_fast, next_stage_idx), relaxed=True)
                tlx.async_load(v_ptrs + future_start_n * stride_vn, tlx.local_view(v_buf_fast, stage_idx))
                tlx.async_load_commit_group()

        for tail_i in tl.static_range(0, NUM_STAGES):
            stage_idx = (main_loop_end + tail_i) % NUM_STAGES
            start_n = (main_loop_end + tail_i) * BLOCK_N
            kn = start_n + offs_n

            tlx.async_load_wait_group(2 * (NUM_STAGES - tail_i) - 1)
            k_tail = tlx.local_load(tlx.local_view(k_buf_fast, stage_idx), relaxed=True)
            qk = tl.dot(q, k_tail)
            acc, l_i, m_i, p = _fa_apply_softmax(acc, l_i, m_i, qk, offs_m, kn, N_CTX, QK_SCALE, BLOCK_M, BLOCK_N,
                                                 False, False)

            tlx.async_load_wait_group(2 * (NUM_STAGES - tail_i) - 2)
            v_tail = tlx.local_load(tlx.local_view(v_buf_fast, stage_idx), relaxed=True)
            acc = tl.dot(p.to(v_tail.dtype), v_tail, acc)
    elif n_blocks > NUM_STAGES - 1:
        k_buf_pipe = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_STAGES)
        v_buf_pipe = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_STAGES)

        for stage in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
            start_n = stage * BLOCK_N
            if MASK_STEPS:
                mask_n = (start_n + offs_n) < N_CTX
                mask = mask_n[:, None]
                tok_k = tlx.async_load(k_ptrs + start_n * stride_kn, tlx.local_view(k_buf_pipe, stage), mask=mask)
                tok_v = tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf_pipe, stage), mask=mask)
            else:
                tok_k = tlx.async_load(k_ptrs + start_n * stride_kn, tlx.local_view(k_buf_pipe, stage))
                tok_v = tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf_pipe, stage))
            tlx.async_load_commit_group([tok_k])
            tlx.async_load_commit_group([tok_v])

        for block_id in tl.range(NUM_STAGES - 1, n_blocks):
            tlx.async_load_wait_group((NUM_STAGES - 2) * 2)

            consumer = (block_id - (NUM_STAGES - 1)) % NUM_STAGES
            producer = block_id % NUM_STAGES
            start_n = (block_id - (NUM_STAGES - 1)) * BLOCK_N
            future_start_n = block_id * BLOCK_N
            kn = start_n + offs_n

            with tlx.warp_pipeline_stage("lds_k", priority=1):
                k_tile = tlx.local_load(tlx.local_view(k_buf_pipe, consumer), relaxed=True)

            with tlx.warp_pipeline_stage("dot1", priority=0):
                qk = tl.dot(q, k_tile.T)

            with tlx.warp_pipeline_stage("mem", priority=1):
                v_tile = tlx.local_load(tlx.local_view(v_buf_pipe, consumer), relaxed=True)
                if MASK_STEPS:
                    mask_n = (future_start_n + offs_n) < N_CTX
                    mask = mask_n[:, None]
                    tok_k = tlx.async_load(k_ptrs + future_start_n * stride_kn, tlx.local_view(k_buf_pipe, producer),
                                           mask=mask)
                    tok_v = tlx.async_load(v_ptrs + future_start_n * stride_vn, tlx.local_view(v_buf_pipe, producer),
                                           mask=mask)
                else:
                    tok_k = tlx.async_load(k_ptrs + future_start_n * stride_kn, tlx.local_view(k_buf_pipe, producer))
                    tok_v = tlx.async_load(v_ptrs + future_start_n * stride_vn, tlx.local_view(v_buf_pipe, producer))
                tlx.async_load_commit_group([tok_k])
                tlx.async_load_commit_group([tok_v])

            with tlx.warp_pipeline_stage("dot2a", priority=0):
                acc, l_i, m_i, p = _fa_apply_softmax(acc, l_i, m_i, qk, offs_m, kn, N_CTX, QK_SCALE, BLOCK_M, BLOCK_N,
                                                     IS_CAUSAL, MASK_STEPS)
                p = p.to(v_tile.dtype)

            with tlx.warp_pipeline_stage("dot2b", priority=0):
                acc = tl.dot(p, v_tile, acc)

        tlx.async_load_wait_group(0)
        for tail_i in tl.range(0, NUM_STAGES - 1, num_stages=0):
            stage_idx = (n_blocks - (NUM_STAGES - 1) + tail_i) % NUM_STAGES
            start_n = (n_blocks - (NUM_STAGES - 1) + tail_i) * BLOCK_N
            kn = start_n + offs_n

            k_tail = tlx.local_load(tlx.local_view(k_buf_pipe, stage_idx), relaxed=True)
            v_tail = tlx.local_load(tlx.local_view(v_buf_pipe, stage_idx), relaxed=True)
            qk = tl.dot(q, k_tail.T)
            acc, l_i, m_i, p = _fa_apply_softmax(acc, l_i, m_i, qk, offs_m, kn, N_CTX, QK_SCALE, BLOCK_M, BLOCK_N,
                                                 IS_CAUSAL, MASK_STEPS)
            acc = tl.dot(p.to(v_tail.dtype), v_tail, acc)
    else:
        k_buf_one = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, 1)
        v_buf_one = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, 1)

        for block_id in tl.range(0, n_blocks, num_stages=0):
            start_n = block_id * BLOCK_N
            kn = start_n + offs_n
            if MASK_STEPS:
                mask_n = (start_n + offs_n) < N_CTX
                mask = mask_n[:, None]
                tok_k = tlx.async_load(k_ptrs + start_n * stride_kn, tlx.local_view(k_buf_one, 0), mask=mask)
                tok_v = tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf_one, 0), mask=mask)
            else:
                tok_k = tlx.async_load(k_ptrs + start_n * stride_kn, tlx.local_view(k_buf_one, 0))
                tok_v = tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf_one, 0))
            tlx.async_load_commit_group([tok_k, tok_v])
            wait_tok = tlx.async_load_wait_group(0)
            k_cur = tlx.local_load(tlx.local_view(k_buf_one, 0), token=wait_tok, relaxed=True)
            v_cur = tlx.local_load(tlx.local_view(v_buf_one, 0), token=wait_tok, relaxed=True)
            qk = tl.dot(q, k_cur.T)
            acc, l_i, m_i, p = _fa_apply_softmax(acc, l_i, m_i, qk, offs_m, kn, N_CTX, QK_SCALE, BLOCK_M, BLOCK_N,
                                                 IS_CAUSAL, MASK_STEPS)
            acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]
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
        num_warps=num_warps,
        **kw,
    )
    return o


def flash_attn_async_prefetch(q, k, v, sm_scale, causal=False, **kw):
    """Prefetch FA with modulo-scheduled prologue/hot-loop/epilogue."""
    B, H, N_CTX, D = q.shape
    o = torch.empty_like(q)

    BLOCK_M = kw.pop("BLOCK_M", 256)
    BLOCK_N = kw.pop("BLOCK_N", 64)
    num_warps = kw.pop("num_warps", 8)

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
        num_warps=num_warps,
        **kw,
    )
    return o


def flash_attn_async_fav3(q, k, v, sm_scale, causal=False, **kw):
    """Four-stage 8-warp TLX FA schedule with explicit warp pipelining."""
    B, H, N_CTX, D = q.shape
    o = torch.empty_like(q)

    use_long_causal_d128 = causal and D == 128 and N_CTX >= 8192
    BLOCK_M = kw.pop("BLOCK_M", 128 if use_long_causal_d128 else 256)
    default_block_n = 32 if D == 128 and (N_CTX <= 1024 or use_long_causal_d128) else 64
    BLOCK_N = kw.pop("BLOCK_N", default_block_n)
    num_warps = kw.pop("num_warps", 4 if use_long_causal_d128 else 8)
    waves_per_eu = kw.pop("waves_per_eu", 2 if use_long_causal_d128 else 0)
    compiler_num_stages = kw.pop("num_stages", 3)
    use_xcd_remap = kw.pop("xcd_remap", False)
    mask_steps = causal or (N_CTX % BLOCK_N != 0)
    xcd_pids_per_xcd = triton.cdiv(H, 8)
    xcd_tall_xcds = H % 8
    if xcd_tall_xcds == 0:
        xcd_tall_xcds = 8

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
        NUM_STAGES=4,
        MASK_STEPS=mask_steps,
        USE_XCD_REMAP=use_xcd_remap,
        XCD_PIDS_PER_XCD=xcd_pids_per_xcd,
        XCD_TALL_XCDS=xcd_tall_xcds,
        num_warps=num_warps,
        num_stages=compiler_num_stages,
        waves_per_eu=waves_per_eu,
        **kw,
    )
    return o


# ═══════════════════════════════════════════════════════════════════════════
# Kernel registry — add new kernel wrappers here
# ═══════════════════════════════════════════════════════════════════════════

KERNEL_REGISTRY = {
    "async_simple": flash_attn_async_simple,
    "async_prefetch": flash_attn_async_prefetch,
    "async_fav3": flash_attn_async_fav3,
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
        kernel_kwargs = {}
        if kernel_name == "async_fav3":
            kernel_kwargs["xcd_remap"] = args.xcd_remap
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
                                tlx_sdpa_lambda = lambda: kernel_fn(q, k, v, sm, causal, **kernel_kwargs)
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
    p.add_argument("--kernel", type=str, nargs="+", default=["async_simple", "async_prefetch"],
                   help="Kernel variants to benchmark")
    p.add_argument("--xcd-remap", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable Gluon-style XCD head remap for async_fav3")
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
