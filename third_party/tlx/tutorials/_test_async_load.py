"""Minimal reproducer for async_load on AMD."""
import torch
import traceback
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def test_no_loop_kernel(
    a_ptr, c_ptr,
    M, K,
    stride_am, stride_ak,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak

    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 2)
    smem_a0 = tlx.local_view(buffers_A, 0)
    tok_a = tlx.async_load(a_ptrs, smem_a0, mask=offs_k[None, :] < K)
    tlx.async_load_commit_group([tok_a])
    tlx.async_load_wait_group(0)
    a = tlx.local_load(smem_a0)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = c_ptr + offs_cm * stride_cm
    tl.store(c_ptrs, tl.sum(a, axis=1))


@triton.jit
def test_with_loop_kernel(
    a_ptr, c_ptr,
    M, K,
    stride_am, stride_ak,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak

    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 2)

    smem_a0 = tlx.local_view(buffers_A, 0)
    tok_a = tlx.async_load(a_ptrs, smem_a0, mask=offs_k[None, :] < K)
    tlx.async_load_commit_group([tok_a])
    a_ptrs += BLOCK_K * stride_ak

    iterMax = tl.cdiv(K, BLOCK_K)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in tl.range(0, iterMax - 1, num_stages=0):
        l_idx = k % 2
        g_idx = 1 - l_idx

        smem_ag = tlx.local_view(buffers_A, g_idx)
        tok_a = tlx.async_load(a_ptrs, smem_ag, mask=offs_k[None, :] < K - (k + 1) * BLOCK_K)
        tlx.async_load_commit_group([tok_a])

        tlx.async_load_wait_group(1)

        smem_al = tlx.local_view(buffers_A, l_idx)
        a = tlx.local_load(smem_al)
        acc += tl.sum(a, axis=1)

        a_ptrs += BLOCK_K * stride_ak

    tlx.async_load_wait_group(0)
    l_idx = (iterMax - 1) % 2
    smem_al = tlx.local_view(buffers_A, l_idx)
    a = tlx.local_load(smem_al)
    acc += tl.sum(a, axis=1)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_ptrs = c_ptr + offs_cm * stride_cm
    tl.store(c_ptrs, acc.to(tl.float16))


if __name__ == "__main__":
    M, K = 64, 256
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)

    print("=== Test 1: async_load WITHOUT loop ===")
    try:
        c = torch.empty((M,), device=DEVICE, dtype=torch.float16)
        test_no_loop_kernel[(1,)](a, c, M, K, a.stride(0), a.stride(1), c.stride(0), BLOCK_M=64, BLOCK_K=64, num_warps=4)
        print(f"SUCCESS: c={c[:5]}")
    except Exception as e:
        traceback.print_exc()

    print("\n=== Test 2: async_load WITH loop ===")
    try:
        c = torch.empty((M,), device=DEVICE, dtype=torch.float16)
        test_with_loop_kernel[(1,)](a, c, M, K, a.stride(0), a.stride(1), c.stride(0), BLOCK_M=64, BLOCK_K=64, num_warps=4)
        print(f"SUCCESS: c={c[:5]}")
    except Exception as e:
        traceback.print_exc()
