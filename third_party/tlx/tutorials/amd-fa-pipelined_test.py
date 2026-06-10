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
def _remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr):
    """Remap a flat program id so contiguous pids land on the same XCD,
    improving L2 reuse for tiles that share K/V (same head)."""
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = tl.where(tall_xcds == 0, NUM_XCDS, tall_xcds)
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    return tl.where(
        xcd < tall_xcds,
        xcd * pids_per_xcd + local_pid,
        tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid,
    )


@triton.jit
def _attn_causal_tile(
    pid_m,
    q_off,
    k_off,
    v_off,
    o_off,
    Q,
    K,
    V,
    Out,
    k_buf,
    v_buf,
    stride_qm,
    stride_qk,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    stride_om,
    stride_ok,
    N_CTX,
    QK_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Compute one causal output m-tile (peeled mask).

    For query block `pid_m`, K blocks fully below the diagonal
    (start_n + BLOCK_N <= pid_m*BLOCK_M) need *no* mask; only the
    `BLOCK_M//BLOCK_N` diagonal blocks do. We run an unmasked steady-state
    loop (FMA-friendly softmax, no `tl.where`) then a short masked diagonal
    tail. The async double-buffered prefetch chain is continuous across both
    loops (slot = global_block_idx % 2), so there is no pipeline bubble.
    """
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < N_CTX,
                other=0.0)

    hi = min(N_CTX, (pid_m + 1) * BLOCK_M)

    k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    n_blocks = (hi + BLOCK_N - 1) // BLOCK_N
    # diag_start = pid_m*BLOCK_M is a multiple of BLOCK_N -> exact split.
    n_unmasked = (pid_m * BLOCK_M) // BLOCK_N
    n_unmasked = tl.minimum(n_unmasked, n_blocks)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Prologue: prefetch block 0.
    tok_k0 = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0), mask=offs_n[:, None] < N_CTX)
    tok_v0 = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0), mask=offs_n[:, None] < N_CTX)
    tlx.async_load_commit_group([tok_k0, tok_v0])

    # ── Unmasked steady-state loop (below-diagonal blocks) ──────────────────
    for block_id in tl.range(0, n_unmasked * BLOCK_N, BLOCK_N, num_stages=0):
        next_off = block_id + BLOCK_N
        i = block_id // BLOCK_N
        slot_cur = i % 2
        slot_nxt = (i + 1) % 2

        wait_tok = tlx.async_load_wait_group(0)
        kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_cur))
        kt_cur = tlx.local_load(kt_view, token=wait_tok, relaxed=True)
        v_cur = tlx.local_load(tlx.local_view(v_buf, slot_cur), token=wait_tok, relaxed=True)

        next_mask = (next_off + offs_n[:, None]) < N_CTX
        tok_k = tlx.async_load(k_ptrs + next_off * stride_kn, tlx.local_view(k_buf, slot_nxt), mask=next_mask)
        tok_v = tlx.async_load(v_ptrs + next_off * stride_vn, tlx.local_view(v_buf, slot_nxt), mask=next_mask)
        tlx.async_load_commit_group([tok_k, tok_v])

        qk = tl.dot(q, kt_cur)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * QK_SCALE)
        p = tl.math.exp2(qk * QK_SCALE - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc)

    # ── Masked diagonal loop ────────────────────────────────────────────────
    for block_id in tl.range(n_unmasked * BLOCK_N, n_blocks * BLOCK_N, BLOCK_N, num_stages=0):
        next_off = block_id + BLOCK_N
        kn = block_id + offs_n
        i = block_id // BLOCK_N
        slot_cur = i % 2
        slot_nxt = (i + 1) % 2

        wait_tok = tlx.async_load_wait_group(0)
        kt_view = tlx.local_trans(tlx.local_view(k_buf, slot_cur))
        kt_cur = tlx.local_load(kt_view, token=wait_tok, relaxed=True)
        v_cur = tlx.local_load(tlx.local_view(v_buf, slot_cur), token=wait_tok, relaxed=True)

        if next_off < hi:
            next_mask = (next_off + offs_n[:, None]) < N_CTX
            tok_k = tlx.async_load(k_ptrs + next_off * stride_kn, tlx.local_view(k_buf, slot_nxt), mask=next_mask)
            tok_v = tlx.async_load(v_ptrs + next_off * stride_vn, tlx.local_view(v_buf, slot_nxt), mask=next_mask)
            tlx.async_load_commit_group([tok_k, tok_v])

        qk = tl.dot(q, kt_cur)
        qk = tl.where(offs_m[:, None] >= kn[None, :], qk, float("-inf"))
        if not EVEN_N:
            qk = tl.where(kn[None, :] < N_CTX, qk, float("-inf"))

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
def _attn_fwd_async_prefetch_causal(
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
    NUM_M_BLOCKS: tl.constexpr,
    GRID_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Causal FA with mirror-paired load balancing + XCD L2 remap.

    Each program processes a *light* tile `pid_m` and its *heavy* mirror
    `NUM_M_BLOCKS-1-pid_m`. The combined K-block count is ~NUM_M_BLOCKS+1 for
    every program, so per-program work (and fixed overhead amortization) is
    uniform — eliminating the causal tail-wave imbalance.

    Launched with a flat 1D grid (GRID_M * Z * H). The flat pid is XCD-remapped
    and decoded as `pid_hz = pid // GRID_M`, `pid_m = pid % GRID_M`, so each XCD
    receives contiguous m-pairs of the same head (K/V L2 reuse).
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    if NUM_XCDS == 1:
        pid_m = tl.program_id(0)
        pid_hz = tl.program_id(1)
    elif GRID_M == 1:
        # Single m-pair: modulo decode folds badly in the loop-range analysis,
        # so special-case it (pid_m is always 0).
        pid = _remap_xcd(tl.program_id(0), GRID_M * Z * H, NUM_XCDS)
        pid_m = 0
        pid_hz = pid
    else:
        pid = _remap_xcd(tl.program_id(0), GRID_M * Z * H, NUM_XCDS)
        pid_m = pid % GRID_M
        pid_hz = pid // GRID_M
    off_z = pid_hz // H
    off_h = pid_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    # Allocate K/V LDS double-buffers once; reused sequentially across both
    # mirror tiles (tile A fully drains its async loads before tile B starts).
    NUM_BUFFERS: tl.constexpr = 2
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_BUFFERS)

    _attn_causal_tile(pid_m, q_off, k_off, v_off, o_off, Q, K, V, Out, k_buf, v_buf, stride_qm, stride_qk, stride_kn,
                      stride_kk, stride_vn, stride_vk, stride_om, stride_ok, N_CTX,
                      QK_SCALE=sm_scale * 1.44269504089, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
                      EVEN_N=EVEN_N)

    pid_mirror = NUM_M_BLOCKS - 1 - pid_m
    if pid_mirror > pid_m:
        _attn_causal_tile(pid_mirror, q_off, k_off, v_off, o_off, Q, K, V, Out, k_buf, v_buf, stride_qm, stride_qk,
                          stride_kn, stride_kk, stride_vn, stride_vk, stride_om, stride_ok, N_CTX,
                          QK_SCALE=sm_scale * 1.44269504089, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
                          EVEN_N=EVEN_N)


@triton.jit
def _attn_fwd_persistent_causal(
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
    NUM_M_BLOCKS: tl.constexpr,
    GRID_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Persistent causal FA — XCD-grouped + mirror-balanced.

    Launches exactly NUM_SMS resident programs. The HW pins program `pid` to
    XCD `pid % NUM_XCDS`, so we assign head `hz` to XCD `hz % NUM_XCDS`: every
    program loops over the heads owned by its XCD (`for hz in range(xcd, Z*H,
    NUM_XCDS)`), keeping each head's K/V resident in that XCD's L2 slice.

    Within an XCD the `NUM_LOCAL = NUM_SMS//NUM_XCDS` local programs split the
    head's m-tiles by *mirror pair* (`p` and `NUM_M_BLOCKS-1-p`) so each unit
    is balanced (combined ~NUM_M_BLOCKS+1 K-blocks), eliminating the causal
    tail imbalance. Q/K/V LDS buffers are allocated once and reused across all
    tiles a program processes.
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid = tl.program_id(0)
    xcd = pid % NUM_XCDS
    local = pid // NUM_XCDS
    NUM_LOCAL: tl.constexpr = NUM_SMS // NUM_XCDS

    NUM_BUFFERS: tl.constexpr = 2
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_BUFFERS)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    # Flatten (head_on_xcd, mirror_pair) work units and stride across the
    # NUM_LOCAL programs of this XCD, so every program stays busy even when
    # GRID_M < NUM_LOCAL (e.g. N=8192). heads_per_xcd rounds up; guard hz range.
    heads_per_xcd = (Z * H + NUM_XCDS - 1) // NUM_XCDS
    units = heads_per_xcd * GRID_M
    for unit in tl.range(local, units, NUM_LOCAL, num_stages=0):
        local_head = unit // GRID_M
        p = unit - local_head * GRID_M
        pid_hz = xcd + local_head * NUM_XCDS
        if pid_hz < Z * H:
            off_z = pid_hz // H
            off_h = pid_hz % H
            q_off = off_z * stride_qz + off_h * stride_qh
            k_off = off_z * stride_kz + off_h * stride_kh
            v_off = off_z * stride_vz + off_h * stride_vh
            o_off = off_z * stride_oz + off_h * stride_oh

            _attn_causal_tile(p, q_off, k_off, v_off, o_off, Q, K, V, Out, k_buf, v_buf, stride_qm, stride_qk,
                              stride_kn, stride_kk, stride_vn, stride_vk, stride_om, stride_ok, N_CTX,
                              QK_SCALE=QK_SCALE, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, EVEN_N=EVEN_N)
            p_mirror = NUM_M_BLOCKS - 1 - p
            if p_mirror > p:
                _attn_causal_tile(p_mirror, q_off, k_off, v_off, o_off, Q, K, V, Out, k_buf, v_buf, stride_qm,
                                  stride_qk, stride_kn, stride_kk, stride_vn, stride_vk, stride_om, stride_ok, N_CTX,
                                  QK_SCALE=QK_SCALE, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, EVEN_N=EVEN_N)


@triton.jit
def _attn_fwd_persistent_nomirror_causal(
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
    NUM_M_BLOCKS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_N: tl.constexpr,
    REVERSE: tl.constexpr,
):
    """Persistent causal FA WITHOUT mirror pairing — balance by averaging only.

    Same XCD-grouping as the mirror version (heads pinned to XCDs for L2), but
    each work unit is a *single* m-tile, not a balanced pair. To still balance,
    tiles are flattened **m-major** (`unit = m*heads_per_xcd + local_head`) so a
    program's round-robin set `{local, local+NUM_LOCAL, ...}` spreads its
    m-values across the full 0..NUM_M_BLOCKS range (step NUM_LOCAL//heads_per_xcd)
    rather than locking onto a fixed light/heavy band. This relies purely on the
    law-of-large-numbers averaging — no explicit pairing. Experiment to compare
    against the mirror-balanced persistent kernel.
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid = tl.program_id(0)
    xcd = pid % NUM_XCDS
    local = pid // NUM_XCDS
    NUM_LOCAL: tl.constexpr = NUM_SMS // NUM_XCDS

    NUM_BUFFERS: tl.constexpr = 2
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_BUFFERS)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    heads_per_xcd = (Z * H + NUM_XCDS - 1) // NUM_XCDS
    units = heads_per_xcd * NUM_M_BLOCKS
    for unit in tl.range(local, units, NUM_LOCAL, num_stages=0):
        m_idx = unit // heads_per_xcd          # m-major: slow index
        local_head = unit - m_idx * heads_per_xcd
        # REVERSE: schedule heaviest (bottom-of-mask) m-tiles first.
        pid_m = (NUM_M_BLOCKS - 1 - m_idx) if REVERSE else m_idx
        pid_hz = xcd + local_head * NUM_XCDS
        if pid_hz < Z * H:
            off_z = pid_hz // H
            off_h = pid_hz % H
            q_off = off_z * stride_qz + off_h * stride_qh
            k_off = off_z * stride_kz + off_h * stride_kh
            v_off = off_z * stride_vz + off_h * stride_vh
            o_off = off_z * stride_oz + off_h * stride_oh

            _attn_causal_tile(pid_m, q_off, k_off, v_off, o_off, Q, K, V, Out, k_buf, v_buf, stride_qm, stride_qk,
                              stride_kn, stride_kk, stride_vn, stride_vk, stride_om, stride_ok, N_CTX,
                              QK_SCALE=QK_SCALE, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, EVEN_N=EVEN_N)


@triton.jit
def _attn_fwd_persistent_balanced_causal(
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
    NUM_M_BLOCKS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Persistent causal FA with **constant-cost fold bundling** — as general as
    the no-mirror persistent kernel, as fast as static mirror.

    This generalizes mirror pairing rather than hardcoding it. The lesson from
    the snake / no-mirror experiments: per-program *total* balance is necessary
    but NOT sufficient. If a program runs all its heavy tiles first and its light
    tiles last, the kernel ends in an overhead-bound "all-light" tail (full
    prologue/epilogue per tile for only 1-4 K-blocks) and MFMA utilisation sags
    over time. Equivalently: **constant work *per iteration*** is what static
    mirror buys, and that requires bundling a heavy tile with a light one.

    So we bundle by cost: each iteration processes the heaviest-remaining and
    lightest-remaining m-tile **together** (`p` and `NUM_M_BLOCKS-1-p`), so every
    iteration costs ≈ `NUM_M_BLOCKS+1` K-blocks and fixed overheads stay
    amortised throughout — no light tail, uniform utilisation. This is a *fold*
    of the cost-sorted tile list; for the linear causal cost it reduces to the
    mirror pairing, but the construction only needs a monotone cost (a
    greedy-bundle — accumulate light tiles until the bundle hits the target cost
    — generalises it to arbitrary/non-linear profiles, which the *fixed* mirror
    `p<->N-1-p` rule cannot handle).

    Persistent + XCD-grouped (heads pinned to XCDs for L2). Bundles are flattened
    `(head_on_xcd, fold-pair)` and strided across the NUM_LOCAL programs of the
    XCD; because every bundle is constant-cost, plain round-robin striding
    balances — no snake/averaging needed.
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid = tl.program_id(0)
    xcd = pid % NUM_XCDS
    local = pid // NUM_XCDS
    NUM_LOCAL: tl.constexpr = NUM_SMS // NUM_XCDS

    NUM_BUFFERS: tl.constexpr = 2
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_BUFFERS)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    num_pairs = (NUM_M_BLOCKS + 1) // 2
    heads_per_xcd = (Z * H + NUM_XCDS - 1) // NUM_XCDS
    units = heads_per_xcd * num_pairs
    for unit in tl.range(local, units, NUM_LOCAL, num_stages=0):
        local_head = unit // num_pairs
        p = unit - local_head * num_pairs
        pid_hz = xcd + local_head * NUM_XCDS
        if pid_hz < Z * H:
            off_z = pid_hz // H
            off_h = pid_hz % H
            q_off = off_z * stride_qz + off_h * stride_qh
            k_off = off_z * stride_kz + off_h * stride_kh
            v_off = off_z * stride_vz + off_h * stride_vh
            o_off = off_z * stride_oz + off_h * stride_oh

            # Constant-cost bundle: light tile p + heavy tile NUM_M_BLOCKS-1-p.
            _attn_causal_tile(p, q_off, k_off, v_off, o_off, Q, K, V, Out, k_buf, v_buf, stride_qm, stride_qk,
                              stride_kn, stride_kk, stride_vn, stride_vk, stride_om, stride_ok, N_CTX,
                              QK_SCALE=QK_SCALE, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, EVEN_N=EVEN_N)
            p_mirror = NUM_M_BLOCKS - 1 - p
            if p_mirror > p:
                _attn_causal_tile(p_mirror, q_off, k_off, v_off, o_off, Q, K, V, Out, k_buf, v_buf, stride_qm,
                                  stride_qk, stride_kn, stride_kk, stride_vn, stride_vk, stride_om, stride_ok, N_CTX,
                                  QK_SCALE=QK_SCALE, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, EVEN_N=EVEN_N)


@triton.jit
def _attn_fwd_dynamic_causal(
    Q,
    K,
    V,
    Out,
    Counter,
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
    NUM_M_BLOCKS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    ITERS_CAP: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Dynamic (first-come-first-serve) causal FA via per-XCD atomic queues.

    Instead of a fixed tile->workgroup mapping, each program repeatedly grabs
    the next available tile from its XCD's work queue with an atomic counter:
    a program that drew a heavy tile finishes later and pulls fewer subsequent
    tiles, so load balances dynamically — the tail shrinks to ≤1 tile, no mirror
    pairing needed.

    L2 locality is preserved by keeping *one queue per XCD* (Counter[xcd]):
    head `hz` is owned by XCD `hz % NUM_XCDS`, so a program only ever touches
    its XCD's heads, keeping their K/V resident in that XCD's L2 slice. Tiles
    are ordered heavy-first (bottom of causal mask) so the long tiles are
    claimed first (LPT scheduling).
    """
    _assume_strides(stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz,
                    stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok)

    pid = tl.program_id(0)
    xcd = pid % NUM_XCDS

    NUM_BUFFERS: tl.constexpr = 2
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), K.dtype.element_ty, NUM_BUFFERS)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), V.dtype.element_ty, NUM_BUFFERS)

    QK_SCALE: tl.constexpr = sm_scale * 1.44269504089

    heads_per_xcd = (Z * H + NUM_XCDS - 1) // NUM_XCDS
    units_per_xcd = heads_per_xcd * NUM_M_BLOCKS

    # FCFS dynamic claim, but expressed as a *bounded* tl.range (scf.for) loop
    # rather than scf.while: TLX only lowers the async-prefetch pipeline over
    # scf.for. Upper bound = units_per_xcd guarantees coverage (no dropped
    # tiles); programs that have claimed their share simply spin the atomic and
    # skip via the guard.
    idx = tl.atomic_add(Counter + xcd, 1, sem="relaxed", scope="gpu")
    for _ in tl.range(0, ITERS_CAP, 1, num_stages=0):
        if idx < units_per_xcd:
            m_idx = idx // heads_per_xcd
            local_head = idx - m_idx * heads_per_xcd
            pid_m = NUM_M_BLOCKS - 1 - m_idx       # heavy (bottom-of-mask) first
            pid_hz = xcd + local_head * NUM_XCDS
            if pid_hz < Z * H:
                off_z = pid_hz // H
                off_h = pid_hz % H
                q_off = off_z * stride_qz + off_h * stride_qh
                k_off = off_z * stride_kz + off_h * stride_kh
                v_off = off_z * stride_vz + off_h * stride_vh
                o_off = off_z * stride_oz + off_h * stride_oh

                _attn_causal_tile(pid_m, q_off, k_off, v_off, o_off, Q, K, V, Out, k_buf, v_buf, stride_qm, stride_qk,
                                  stride_kn, stride_kk, stride_vn, stride_vk, stride_om, stride_ok, N_CTX,
                                  QK_SCALE=QK_SCALE, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM, EVEN_N=EVEN_N)
            idx = tl.atomic_add(Counter + xcd, 1, sem="relaxed", scope="gpu")


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
        num_warps=num_warps,
        **kw,
    )
    return o


def flash_attn_async_prefetch_causal(q, k, v, sm_scale, causal=False, **kw):
    """Causal-optimized prefetch FA (peeled diagonal mask).

    For causal=False this falls back to the generic prefetch kernel so the
    shared correctness suite still exercises both paths.
    """
    if not causal:
        return flash_attn_async_prefetch(q, k, v, sm_scale, causal=False, **kw)

    B, H, N_CTX, D = q.shape
    o = torch.empty_like(q)

    BLOCK_M = kw.pop("BLOCK_M", 256)
    # D<=64: BLOCK_N=128 wins (more compute per barrier; diagonal masking is
    # cheap now that it's peeled). D=128 must stay at 64 (LDS budget for
    # double-buffered K+V).
    BLOCK_N = kw.pop("BLOCK_N", 128 if D <= 64 else 64)
    num_warps = kw.pop("num_warps", 4)

    num_m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    # Mirror-pair tiles: program p handles tiles p and (num_m-1-p).
    grid_m = (num_m_blocks + 1) // 2
    num_xcds = kw.pop("NUM_XCDS", 8)
    # XCD remap needs a flat 1D grid so contiguous pids land on the same XCD.
    grid = (grid_m * B * H, ) if num_xcds != 1 else (grid_m, B * H)
    _attn_fwd_async_prefetch_causal[grid](
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
        NUM_M_BLOCKS=num_m_blocks,
        GRID_M=grid_m,
        NUM_XCDS=num_xcds,
        EVEN_N=(N_CTX % BLOCK_N == 0),
        num_warps=num_warps,
        **kw,
    )
    return o


def flash_attn_async_prefetch_persistent_causal(q, k, v, sm_scale, causal=False, **kw):
    """Persistent-style causal FA (fixed NUM_SMS resident programs).

    For causal=False this falls back to the generic prefetch kernel so the
    shared correctness suite still exercises both paths.
    """
    if not causal:
        return flash_attn_async_prefetch(q, k, v, sm_scale, causal=False, **kw)

    B, H, N_CTX, D = q.shape

    BLOCK_M = kw.pop("BLOCK_M", 256)
    BLOCK_N = kw.pop("BLOCK_N", 128 if D <= 64 else 64)
    num_warps = kw.pop("num_warps", 4)

    # Partial-block edge cases (N not a multiple of BLOCK_N) hit a compiler
    # iota_range crash with the persistent index decode; fall back to the
    # robust mirror-paired kernel (these tiny shapes are not perf-critical).
    if N_CTX % BLOCK_N != 0:
        return flash_attn_async_prefetch_causal(q, k, v, sm_scale, causal=True, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                                num_warps=num_warps, **kw)

    o = torch.empty_like(q)
    num_m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    grid_m = (num_m_blocks + 1) // 2
    num_xcds = kw.pop("NUM_XCDS", 8)
    cu_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    # Keep NUM_SMS a multiple of NUM_XCDS so each XCD gets equal local programs.
    num_sms = kw.pop("NUM_SMS", (cu_count // num_xcds) * num_xcds)
    grid = (num_sms, )
    _attn_fwd_persistent_causal[grid](
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
        NUM_M_BLOCKS=num_m_blocks,
        GRID_M=grid_m,
        NUM_SMS=num_sms,
        NUM_XCDS=num_xcds,
        EVEN_N=(N_CTX % BLOCK_N == 0),
        num_warps=num_warps,
        **kw,
    )
    return o


def flash_attn_async_prefetch_persistent_nomirror_causal(q, k, v, sm_scale, causal=False, **kw):
    """Persistent causal FA without mirror pairing (balance by averaging only)."""
    if not causal:
        return flash_attn_async_prefetch(q, k, v, sm_scale, causal=False, **kw)

    B, H, N_CTX, D = q.shape

    BLOCK_M = kw.pop("BLOCK_M", 256)
    BLOCK_N = kw.pop("BLOCK_N", 128 if D <= 64 else 64)
    num_warps = kw.pop("num_warps", 4)

    if N_CTX % BLOCK_N != 0:
        return flash_attn_async_prefetch_causal(q, k, v, sm_scale, causal=True, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                                num_warps=num_warps, **kw)

    o = torch.empty_like(q)
    num_m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    num_xcds = kw.pop("NUM_XCDS", 8)
    cu_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    num_sms = kw.pop("NUM_SMS", (cu_count // num_xcds) * num_xcds)
    grid = (num_sms, )
    _attn_fwd_persistent_nomirror_causal[grid](
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
        NUM_M_BLOCKS=num_m_blocks,
        NUM_SMS=num_sms,
        NUM_XCDS=num_xcds,
        EVEN_N=(N_CTX % BLOCK_N == 0),
        REVERSE=kw.pop("REVERSE", True),
        num_warps=num_warps,
        **kw,
    )
    return o


def flash_attn_async_prefetch_persistent_balanced_causal(q, k, v, sm_scale, causal=False, **kw):
    """Persistent causal FA with constant-cost fold bundling.

    As general as the no-mirror persistent kernel (persistent, cost-driven; the
    bundling generalises to any monotone cost via greedy-bundle), and as fast as
    the static mirror kernel.
    """
    if not causal:
        return flash_attn_async_prefetch(q, k, v, sm_scale, causal=False, **kw)

    B, H, N_CTX, D = q.shape

    BLOCK_M = kw.pop("BLOCK_M", 256)
    BLOCK_N = kw.pop("BLOCK_N", 128 if D <= 64 else 64)
    num_warps = kw.pop("num_warps", 4)

    if N_CTX % BLOCK_N != 0:
        return flash_attn_async_prefetch_causal(q, k, v, sm_scale, causal=True, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                                num_warps=num_warps, **kw)

    o = torch.empty_like(q)
    num_m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    num_xcds = kw.pop("NUM_XCDS", 8)
    cu_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    num_sms = kw.pop("NUM_SMS", (cu_count // num_xcds) * num_xcds)
    grid = (num_sms, )
    _attn_fwd_persistent_balanced_causal[grid](
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
        NUM_M_BLOCKS=num_m_blocks,
        NUM_SMS=num_sms,
        NUM_XCDS=num_xcds,
        EVEN_N=(N_CTX % BLOCK_N == 0),
        num_warps=num_warps,
        **kw,
    )
    return o


def flash_attn_async_prefetch_dynamic_causal(q, k, v, sm_scale, causal=False, **kw):
    """Dynamic FCFS causal FA via per-XCD atomic work queues."""
    if not causal:
        return flash_attn_async_prefetch(q, k, v, sm_scale, causal=False, **kw)

    B, H, N_CTX, D = q.shape

    BLOCK_M = kw.pop("BLOCK_M", 256)
    BLOCK_N = kw.pop("BLOCK_N", 128 if D <= 64 else 64)
    num_warps = kw.pop("num_warps", 4)

    if N_CTX % BLOCK_N != 0:
        return flash_attn_async_prefetch_causal(q, k, v, sm_scale, causal=True, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                                                num_warps=num_warps, **kw)

    o = torch.empty_like(q)
    num_m_blocks = triton.cdiv(N_CTX, BLOCK_M)
    num_xcds = kw.pop("NUM_XCDS", 8)
    cu_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    num_sms = kw.pop("NUM_SMS", (cu_count // num_xcds) * num_xcds)
    # Per-XCD atomic work-queue counters (fresh zeros each launch).
    counter = torch.zeros(num_xcds, dtype=torch.int32, device=q.device)
    heads_per_xcd = (B * H + num_xcds - 1) // num_xcds
    units_per_xcd = heads_per_xcd * num_m_blocks
    programs_per_xcd = max(1, num_sms // num_xcds)
    avg = triton.cdiv(units_per_xcd, programs_per_xcd)
    # Generous slack (3x avg) so an unlucky/slow program can still claim extra
    # tiles without dropping any; capped at the total queue length.
    iters_cap = kw.pop("ITERS_CAP", min(units_per_xcd, avg * 3))
    grid = (num_sms, )
    _attn_fwd_dynamic_causal[grid](
        q,
        k,
        v,
        o,
        counter,
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
        NUM_M_BLOCKS=num_m_blocks,
        NUM_XCDS=num_xcds,
        ITERS_CAP=iters_cap,
        EVEN_N=(N_CTX % BLOCK_N == 0),
        num_warps=num_warps,
        **kw,
    )
    return o


# ═══════════════════════════════════════════════════════════════════════════
# Kernel registry — add new kernel wrappers here
# ═══════════════════════════════════════════════════════════════════════════

KERNEL_REGISTRY = {
    "async_simple": flash_attn_async_simple,
    "async_prefetch": flash_attn_async_prefetch,
    "async_prefetch_causal": flash_attn_async_prefetch_causal,
    "async_prefetch_persistent_causal": flash_attn_async_prefetch_persistent_causal,
    "async_prefetch_persistent_nomirror_causal": flash_attn_async_prefetch_persistent_nomirror_causal,
    "async_prefetch_persistent_balanced_causal": flash_attn_async_prefetch_persistent_balanced_causal,
    "async_prefetch_dynamic_causal": flash_attn_async_prefetch_dynamic_causal,
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
