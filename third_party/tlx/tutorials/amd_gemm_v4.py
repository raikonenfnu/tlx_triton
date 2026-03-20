"""
AMD GEMM V4 - Global Prefetch Pipeline (TLX)

Ported from v4_global_prefetch in amd-gemm-pipelined-gfx950-gluon.py.

Double-buffered register prefetch pipeline with L2 workgroup swizzle:
  - 2 SMEM buffers for ping-pong
  - Global → register → SMEM path (tl.load + local_store) avoids ds_bpermute
    overhead vs direct buffer_load_to_local (async_load)
  - No masks on global loads (assumes K evenly divides BLOCK_K)
  - GROUP_SIZE_M workgroup swizzle for L2 cache locality

  Prologue:
      tl.load A0, B0 → regs
      local_store regs → buffer 0

  Main Loop (k = 0 .. iterMax-2):
      tl.load A_{k+1}, B_{k+1} → regs  (prefetch)
      local_load A_k, B_k ← buffer l_idx
      DOT(A_k, B_k)
      local_store regs → buffer g_idx

  Epilogue:
      local_load A_last, B_last ← buffer l_idx
      DOT(A_last, B_last)
      store(acc)
"""
import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_cuda, is_hip

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def matmul_kernel_v4_global_prefetch(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 2)
    buffers_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), 2)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    iterMax = tl.cdiv(K, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # --- Prologue: async copy iteration 0 → buffer 0 ---
    smem_a0 = tlx.local_view(buffers_A, 0)
    smem_b0 = tlx.local_view(buffers_B, 0)
    tok_a = tlx.async_load(a_ptrs, smem_a0, mask=offs_k[None, :] < K)
    tok_b = tlx.async_load(b_ptrs, smem_b0, mask=offs_k[:, None] < K)
    tlx.async_load_commit_group([tok_a, tok_b])
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk

    # --- Main Loop: double-buffered async copy + compute ---
    for k in tl.range(0, iterMax - 1, num_stages=0):
        l_idx = k % 2
        g_idx = 1 - l_idx

        smem_ag = tlx.local_view(buffers_A, g_idx)
        smem_bg = tlx.local_view(buffers_B, g_idx)
        tok_a = tlx.async_load(a_ptrs, smem_ag, mask=offs_k[None, :] < K - (k + 1) * BLOCK_K)
        tok_b = tlx.async_load(b_ptrs, smem_bg, mask=offs_k[:, None] < K - (k + 1) * BLOCK_K)
        tlx.async_load_commit_group([tok_a, tok_b])

        tok_wait = tlx.async_load_wait_group(1)

        smem_al = tlx.local_view(buffers_A, l_idx)
        smem_bl = tlx.local_view(buffers_B, l_idx)
        a = tlx.local_load(smem_al, token=tok_wait)
        b = tlx.local_load(smem_bl, token=tok_wait)
        acc = tl.dot(a, b, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # --- Epilogue: drain last buffer ---
    tok_wait = tlx.async_load_wait_group(0)
    l_idx = (iterMax - 1) % 2
    smem_al = tlx.local_view(buffers_A, l_idx)
    smem_bl = tlx.local_view(buffers_B, l_idx)
    a = tlx.local_load(smem_al, token=tok_wait)
    b = tlx.local_load(smem_bl, token=tok_wait)
    acc = tl.dot(a, b, acc)

    c = acc.to(tlx.dtype_of(c_ptr))

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape

    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    num_warps = 8

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

    matmul_kernel_v4_global_prefetch[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=4,
        num_warps=num_warps,
        waves_per_eu=2,
        matrix_instr_nonkdim=16,
    )
    return c


@pytest.mark.skipif(
    not is_hip(),
    reason="Requires AMD GPU",
)
def test_op():
    torch.manual_seed(0)
    a = torch.randn((8192, 8192), device=DEVICE, dtype=torch.float16)
    b = torch.randn((8192, 8192), device=DEVICE, dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    rtol = 1e-2
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol)
    print("Correctness test passed!")


# Benchmark
ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[4096],
        line_arg="provider",
        line_vals=["triton", ref_lib.lower()],
        line_names=["Triton V4 Global Prefetch", ref_lib],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-v4-global-prefetch-performance-fp16",
        args={},
    ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    # b = b.T.contiguous()
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles, rep=1000)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles, rep=1000)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    print("Running benchmarks...")
    benchmark.run(print_data=True)
