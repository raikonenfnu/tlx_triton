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
def _remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = tl.where(tall_xcds == 0, NUM_XCDS, tall_xcds)
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    pid = tl.where(
        xcd < tall_xcds,
        xcd * pids_per_xcd + local_pid,
        tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid,
    )
    return pid


@triton.jit
def _attn_fwd_async_split_kv(
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
    GRID_MN,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    XCD_REMAP: tl.constexpr,
):
    """
    Split K/V commit groups so QK compute can overlap with V DMA.
    Optional XCD remapping for even workload distribution.
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid = tl.program_id(0)
    if XCD_REMAP:
        pid = _remap_xcd(pid, GRID_MN)
    NUM_M_BLOCKS = GRID_MN // (Z * H)
    pid_hz = pid // NUM_M_BLOCKS
    pid_m = pid % NUM_M_BLOCKS
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
    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

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
        mask = kn[:, None] < N_CTX

        tok_k = tlx.async_load(k_base + start_n * stride_kn, tlx.local_view(k_buf, 0), mask=mask)
        tlx.async_load_commit_group([tok_k])
        tok_v = tlx.async_load(v_base + start_n * stride_vn, tlx.local_view(v_buf, 0), mask=mask)
        tlx.async_load_commit_group([tok_v])

        tlx.async_load_wait_group(1)
        k_cur = tlx.local_load(tlx.local_view(k_buf, 0), relaxed=True)

        qk = tl.dot(q, k_cur.T)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        if start_n + BLOCK_N > N_CTX:
            qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))

        tlx.async_load_wait_group(0)
        v_cur = tlx.local_load(tlx.local_view(v_buf, 0), relaxed=True)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
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
def _attn_fwd_async_warp_pipe(
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
    GRID_MN,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    XCD_REMAP: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Warp-pipelined FA with 2-stage schedule (matching GEMM pattern).

    Prologue pre-fills NUM_STAGES buffers, hot loop has two pipeline stages:
      mfma (prio=0): QK dot + softmax + PV dot (all compute)
      [wait for DMA]
      mem  (prio=1): prefetch K+V for future tile, LDS load K+V for next iter
    Epilogue drains last NUM_STAGES tiles with a simple for loop.
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid = tl.program_id(0)
    if XCD_REMAP:
        pid = _remap_xcd(pid, GRID_MN)
    NUM_M_BLOCKS = GRID_MN // (Z * H)
    pid_hz = pid // NUM_M_BLOCKS
    pid_m = pid % NUM_M_BLOCKS
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
    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_STAGES)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_STAGES)

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = (hi + BLOCK_N - 1) // BLOCK_N

    # Prologue: issue K+V DMA for first NUM_STAGES blocks
    for i in tl.range(0, NUM_STAGES, loop_unroll_factor=NUM_STAGES):
        start_n = i * BLOCK_N
        mask = (start_n + offs_n[:, None]) < hi
        tok_k = tlx.async_load(k_ptrs + start_n * stride_kn, tlx.local_view(k_buf, i), mask=mask)
        tlx.async_load_commit_group([tok_k])
        tok_v = tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf, i), mask=mask)
        tlx.async_load_commit_group([tok_v])

    # Wait for buffer 0 K+V and load into regs
    tlx.async_load_wait_group((NUM_STAGES - 1) * 2)
    k_tile = tlx.local_load(tlx.local_view(k_buf, 0), relaxed=True)
    v_tile = tlx.local_load(tlx.local_view(v_buf, 0), relaxed=True)

    # Hot loop: 2-stage warp pipeline (mfma + mem), matching GEMM pattern.
    for tile_id in tl.range(0, n_blocks - NUM_STAGES, num_stages=0):
        prefetch_buf = tile_id % NUM_STAGES
        next_buf = (tile_id + 1) % NUM_STAGES
        start_n = tile_id * BLOCK_N
        k_prefetch = start_n + NUM_STAGES * BLOCK_N

        with tlx.warp_pipeline_stage("mfma", priority=0):
            qk = tl.dot(q, k_tile.T)
            if IS_CAUSAL:
                kn = start_n + offs_n
                qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
            p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            acc = tl.dot(p.to(v_tile.dtype), v_tile, acc)

        tlx.async_load_wait_group(2)

        with tlx.warp_pipeline_stage("mem", priority=1):
            prefetch_mask = (k_prefetch + offs_n[:, None]) < hi
            tok_k = tlx.async_load(k_ptrs + k_prefetch * stride_kn, tlx.local_view(k_buf, prefetch_buf),
                                   mask=prefetch_mask)
            tlx.async_load_commit_group([tok_k])
            tok_v = tlx.async_load(v_ptrs + k_prefetch * stride_vn, tlx.local_view(v_buf, prefetch_buf),
                                   mask=prefetch_mask)
            tlx.async_load_commit_group([tok_v])
            k_tile = tlx.local_load(tlx.local_view(k_buf, next_buf), relaxed=True)
            v_tile = tlx.local_load(tlx.local_view(v_buf, next_buf), relaxed=True)

    # Epilogue: drain the last NUM_STAGES tiles
    tail_start = tl.maximum(n_blocks - NUM_STAGES, 0)
    tlx.async_load_wait_group(0)
    for tail_i in tl.range(0, NUM_STAGES):
        block_id = tail_start + tail_i
        if block_id < n_blocks:
            stage = block_id % NUM_STAGES
            start_n = block_id * BLOCK_N
            kn = start_n + offs_n

            k_cur = tlx.local_load(tlx.local_view(k_buf, stage), relaxed=True)
            v_cur = tlx.local_load(tlx.local_view(v_buf, stage), relaxed=True)

            qk = tl.dot(q, k_cur.T)
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
    GRID_MN,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    XCD_REMAP: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Gluon-style pipelined FA for CDNA4 with software pipelining.

    Supports two modes based on NUM_STAGES:
      NUM_STAGES=1: Single-buffer with separate K/V commit groups.
                    Compatible with schedule_hint for best instruction scheduling.
      NUM_STAGES>1: Multi-buffer prologue/hot-loop/epilogue.
                    Better DMA-compute overlap but incompatible with schedule_hint
                    due to LLVM iota_range bug.

    Pipeline within each tile:
      issue K → issue V → wait K → QK dot → wait V → softmax → PV dot
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid = tl.program_id(0)
    if XCD_REMAP:
        pid = _remap_xcd(pid, GRID_MN)
    NUM_M_BLOCKS = GRID_MN // (Z * H)
    pid_hz = pid // NUM_M_BLOCKS
    pid_m = pid % NUM_M_BLOCKS
    off_z = pid_hz // H
    off_h = pid_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=offs_m[:, None] < N_CTX, other=0.0)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    if IS_CAUSAL:
        hi = min(N_CTX, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_STAGES)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_STAGES)

    k_base = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_base = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if NUM_STAGES == 1:
        # ── Single-buffer path: simple loop, schedule_hint-compatible ──
        for start_n in tl.range(0, hi, BLOCK_N, num_stages=0):
            kn = start_n + offs_n
            mask = kn[:, None] < N_CTX

            tok_k = tlx.async_load(k_base + start_n * stride_kn, tlx.local_view(k_buf, 0), mask=mask)
            tlx.async_load_commit_group([tok_k])
            tok_v = tlx.async_load(v_base + start_n * stride_vn, tlx.local_view(v_buf, 0), mask=mask)
            tlx.async_load_commit_group([tok_v])

            tlx.async_load_wait_group(1)
            k_cur = tlx.local_load(tlx.local_view(k_buf, 0), relaxed=True)

            qk = tl.dot(q, k_cur.T)
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
            if start_n + BLOCK_N > N_CTX:
                qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))

            tlx.async_load_wait_group(0)
            v_cur = tlx.local_load(tlx.local_view(v_buf, 0), relaxed=True)

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
            p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)
    else:
        # ── Multi-buffer path: prologue / hot-loop / epilogue ──
        n_blocks = (hi + BLOCK_N - 1) // BLOCK_N

        for i in tl.range(0, NUM_STAGES, loop_unroll_factor=NUM_STAGES):
            start_n = i * BLOCK_N
            mask = (start_n + offs_n[:, None]) < hi
            tok_k = tlx.async_load(k_base + start_n * stride_kn, tlx.local_view(k_buf, i), mask=mask)
            tlx.async_load_commit_group([tok_k])
            tok_v = tlx.async_load(v_base + start_n * stride_vn, tlx.local_view(v_buf, i), mask=mask)
            tlx.async_load_commit_group([tok_v])

        for tile_id in tl.range(0, n_blocks - NUM_STAGES, num_stages=0):
            buf = tile_id % NUM_STAGES
            start_n = tile_id * BLOCK_N
            kn = start_n + offs_n

            tlx.async_load_wait_group(2 * NUM_STAGES - 1)
            k_cur = tlx.local_load(tlx.local_view(k_buf, buf), relaxed=True)

            qk = tl.dot(q, k_cur.T)
            if IS_CAUSAL:
                qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))

            tlx.async_load_wait_group(2 * NUM_STAGES - 2)
            v_cur = tlx.local_load(tlx.local_view(v_buf, buf), relaxed=True)

            prefetch_n = (tile_id + NUM_STAGES) * BLOCK_N
            prefetch_mask = (prefetch_n + offs_n[:, None]) < hi
            tok_k = tlx.async_load(k_base + prefetch_n * stride_kn, tlx.local_view(k_buf, buf), mask=prefetch_mask)
            tlx.async_load_commit_group([tok_k])
            tok_v = tlx.async_load(v_base + prefetch_n * stride_vn, tlx.local_view(v_buf, buf), mask=prefetch_mask)
            tlx.async_load_commit_group([tok_v])

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
            p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

        tail_start = tl.maximum(n_blocks - NUM_STAGES, 0)
        tlx.async_load_wait_group(0)
        for tail_i in tl.range(0, NUM_STAGES, loop_unroll_factor=NUM_STAGES):
            block_id = tail_start + tail_i
            if block_id < n_blocks:
                buf = block_id % NUM_STAGES
                start_n = block_id * BLOCK_N
                kn = start_n + offs_n

                k_cur = tlx.local_load(tlx.local_view(k_buf, buf), relaxed=True)
                v_cur = tlx.local_load(tlx.local_view(v_buf, buf), relaxed=True)

                qk = tl.dot(q, k_cur.T)
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
                acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

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


# ═══════════════════════════════════════════════════════════════════════════
# Kernel registry — add new kernel wrappers here
# ═══════════════════════════════════════════════════════════════════════════

def flash_attn_async_fav3(q, k, v, sm_scale, causal=False, **kw):
    """Gluon-style pipelined FA with split K/V commit groups, XCD remap, schedule hint."""
    B, H, N_CTX, D = q.shape
    o = torch.empty_like(q)

    BLOCK_M = kw.pop("BLOCK_M", 256)
    BLOCK_N = kw.pop("BLOCK_N", 64)
    NUM_STAGES = kw.pop("NUM_STAGES", 1)
    num_warps = kw.pop("num_warps", 8)
    xcd_remap = kw.pop("xcd_remap", True)
    waves_per_eu = kw.pop("waves_per_eu", 2)
    # schedule_hint='memory-bound-attention' triggers LLVM iota_range crash
    # with multi-buffer prologue/epilogue loops, so only use for single-buffer.
    default_hint = "memory-bound-attention" if NUM_STAGES == 1 else None
    schedule_hint = kw.pop("schedule_hint", default_hint)

    num_m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    GRID_MN = num_m_blocks * B * H
    grid = (GRID_MN,)
    _attn_fwd_async_fav3[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, N_CTX, GRID_MN, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL=causal, XCD_REMAP=xcd_remap, NUM_STAGES=NUM_STAGES,
        num_warps=num_warps, waves_per_eu=waves_per_eu,
        schedule_hint=schedule_hint,
        **kw,
    )
    return o


def flash_attn_async_split_kv(q, k, v, sm_scale, causal=False, **kw):
    """Split K/V commit groups for better compute-memory overlap."""
    B, H, N_CTX, D = q.shape
    o = torch.empty_like(q)

    BLOCK_M = kw.pop("BLOCK_M", 256)
    BLOCK_N = kw.pop("BLOCK_N", 64)
    num_warps = kw.pop("num_warps", 8)
    xcd_remap = kw.pop("xcd_remap", True)
    schedule_hint = kw.pop("schedule_hint", "memory-bound-attention")

    num_m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    GRID_MN = num_m_blocks * B * H
    grid = (GRID_MN,)
    _attn_fwd_async_split_kv[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, N_CTX, GRID_MN, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL=causal, XCD_REMAP=xcd_remap,
        num_warps=num_warps, schedule_hint=schedule_hint,
        **kw,
    )
    return o


def flash_attn_async_warp_pipe(q, k, v, sm_scale, causal=False, **kw):
    """Warp-pipelined FA with single-loop warp pipeline."""
    B, H, N_CTX, D = q.shape
    o = torch.empty_like(q)

    BLOCK_M = kw.pop("BLOCK_M", 256)
    BLOCK_N = kw.pop("BLOCK_N", 64)
    NUM_STAGES = kw.pop("NUM_STAGES", 2)
    num_warps = kw.pop("num_warps", 8)
    xcd_remap = kw.pop("xcd_remap", True)
    schedule_hint = kw.pop("schedule_hint", "memory-bound-attention")

    num_m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    GRID_MN = num_m_blocks * B * H
    grid = (GRID_MN,)
    _attn_fwd_async_warp_pipe[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, N_CTX, GRID_MN, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL=causal, XCD_REMAP=xcd_remap, NUM_STAGES=NUM_STAGES,
        num_warps=num_warps, schedule_hint=schedule_hint,
        **kw,
    )
    return o


KERNEL_REGISTRY = {
    "async_simple": flash_attn_async_simple,
    "async_prefetch": flash_attn_async_prefetch,
    "async_fav3": flash_attn_async_fav3,
    "async_split_kv": flash_attn_async_split_kv,
    "async_warp_pipe": flash_attn_async_warp_pipe,
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
    p.add_argument("--kernel", type=str, nargs="+", default=["async_simple", "async_prefetch"],
                   help="Kernel variants to benchmark")
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
