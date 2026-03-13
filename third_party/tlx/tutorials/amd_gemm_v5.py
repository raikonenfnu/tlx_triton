import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_hip

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def matmul_kernel_v5_local_prefetch(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    V5 Local prefetch pipeline - TLX conversion from Gluon.

    3-stage overlapped pipeline:
    - Stage 1: Async copy global → shared (iteration k+2)
    - Stage 2: Local load shared → registers (iteration k+1)
    - Stage 3: Compute/DOT (iteration k)

    Gluon pattern:
        Prologue:
            AC A0, B0 --> buffer 0
            AC A1, B1 --> buffer 1
            async_wait buffer 0
            local_load A0, B0 <-- buffer 0

        Main Loop (k in [0, iterMax-2]):
            DOT(A, B)                           # iteration k
            async_wait all
            AC A_{k+2}, B_{k+2} --> buffer g_idx    # iteration k+2
            local_load A_{k+1}, B_{k+1} <-- buffer l_idx  # iteration k+1

        Epilogue:
            DOT(final)
            store(acc)
    """

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Allocate shared memory (2 buffers for ping-pong)
    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 2)
    buffers_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), 2)

    # Compute offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    iterMax = tl.cdiv(K, BLOCK_K)

    ## Prologue: Issue async loads for first 2 iterations
    ## AC A0, B0 --> buffer 0
    a_smem_0 = tlx.local_view(buffers_A, 0)
    b_smem_0 = tlx.local_view(buffers_B, 0)
    token_a0 = tlx.async_load(a_ptrs, a_smem_0, mask=offs_k[None, :] < K)
    token_b0 = tlx.async_load(b_ptrs, b_smem_0, mask=offs_k[:, None] < K)
    tlx.async_load_commit_group([token_a0, token_b0])
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk

    ## AC A1, B1 --> buffer 1
    a_smem_1 = tlx.local_view(buffers_A, 1)
    b_smem_1 = tlx.local_view(buffers_B, 1)
    token_a1 = tlx.async_load(a_ptrs, a_smem_1, mask=offs_k[None, :] < K - BLOCK_K)
    token_b1 = tlx.async_load(b_ptrs, b_smem_1, mask=offs_k[:, None] < K - BLOCK_K)
    tlx.async_load_commit_group([token_a1, token_b1])
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk

    tlx.async_load_wait_group(1)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a = tlx.local_load(a_smem_1)
    b = tlx.local_load(b_smem_1)

    ## Main Loop: True 3-stage pipeline matching Gluon v5
    ## At iteration k:
    ##   - Local load iteration k from shared to registers (iteration k was prefetched earlier)
    ##   - DOT computes iteration k
    ##   - Async copy prefetches iteration k+2
    # Disable auto-pipelining with num_stages=0
    for k in tl.range(0, iterMax, num_stages=0):
        g_idx = k % 2       # Buffer for async copy (iteration k+2)
        l_idx = 1 - g_idx   # Buffer for local load (iteration k+1)

        acc = tl.dot(a, b, acc)
        tlx.async_load_wait_group(0)

        a_next_smem = tlx.local_view(buffers_A, g_idx)
        b_next_smem = tlx.local_view(buffers_B, g_idx)
        token_a = tlx.async_load(a_ptrs, a_next_smem, mask=offs_k[None, :] < K - (k + 2) * BLOCK_K)
        token_b = tlx.async_load(b_ptrs, b_next_smem, mask=offs_k[:, None] < K - (k + 2) * BLOCK_K)
        tlx.async_load_commit_group([token_a, token_b])

        a_load_shmem = tlx.local_view(buffers_A, l_idx)
        b_load_shmem = tlx.local_view(buffers_B, l_idx)
        a_next = tlx.local_load(a_load_shmem)
        b_next = tlx.local_load(b_load_shmem)

        a = a_next
        b = b_next

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    ## Epilogue
    ## iterMax - 1
    acc = tl.dot(a, b, acc)
    c = acc.to(tlx.dtype_of(c_ptr))

    # Store result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape

    # Use same block sizes as original Gluon kernel
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    num_warps = 4

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Launch grid
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)

    matmul_kernel_v5_local_prefetch[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
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
    print("✓ Correctness test passed!")


# Benchmark
ref_lib = 'rocBLAS'

configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[2048, 4096, 8192],
        line_arg="provider",
        line_vals=[ref_lib.lower(), "triton"],
        line_names=[ref_lib, "Triton V5 Local Prefetch"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-v5-local-prefetch-performance-fp16",
        args={},
    ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
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
