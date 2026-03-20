"""Sweep waves_per_eu for TLX GEMM."""
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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

    smem_a0 = tlx.local_view(buffers_A, 0)
    smem_b0 = tlx.local_view(buffers_B, 0)
    tok_a = tlx.async_load(a_ptrs, smem_a0, mask=offs_k[None, :] < K)
    tok_b = tlx.async_load(b_ptrs, smem_b0, mask=offs_k[:, None] < K)
    tlx.async_load_commit_group([tok_a, tok_b])
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk

    for k in tl.range(0, iterMax - 1, num_stages=0):
        l_idx = k % 2
        g_idx = 1 - l_idx
        smem_ag = tlx.local_view(buffers_A, g_idx)
        smem_bg = tlx.local_view(buffers_B, g_idx)
        tok_a = tlx.async_load(a_ptrs, smem_ag, mask=offs_k[None, :] < K - (k + 1) * BLOCK_K)
        tok_b = tlx.async_load(b_ptrs, smem_bg, mask=offs_k[:, None] < K - (k + 1) * BLOCK_K)
        tlx.async_load_commit_group([tok_a, tok_b])
        tlx.async_load_wait_group(1)
        smem_al = tlx.local_view(buffers_A, l_idx)
        smem_bl = tlx.local_view(buffers_B, l_idx)
        a = tlx.local_load(smem_al)
        b = tlx.local_load(smem_bl)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    tlx.async_load_wait_group(0)
    l_idx = (iterMax - 1) % 2
    smem_al = tlx.local_view(buffers_A, l_idx)
    smem_bl = tlx.local_view(buffers_B, l_idx)
    a = tlx.local_load(smem_al)
    b = tlx.local_load(smem_bl)
    acc = tl.dot(a, b, acc)

    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 4096, 4096, 4096
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    configs = [
        (8, 0),   # default
        (8, 1),
        (8, 2),
        (8, 3),
        (4, 0),
        (4, 2),
    ]

    print(f"{'warps':>5} {'waves_per_eu':>12}  {'TFLOPS':>8}")
    for nw, wpe in configs:
        try:
            c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
            grid = (triton.cdiv(M, 256) * triton.cdiv(N, 256), 1)
            kwargs = dict(
                BLOCK_M=256, BLOCK_N=256, BLOCK_K=64,
                GROUP_SIZE_M=4, num_warps=nw,
                matrix_instr_nonkdim=16,
            )
            if wpe > 0:
                kwargs['waves_per_eu'] = wpe
            ms = triton.testing.do_bench(lambda: matmul_kernel[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                **kwargs,
            ), rep=100)
            tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
            print(f"{nw:5d} {wpe:12d}  {tflops:8.1f}")
        except Exception as e:
            print(f"{nw:5d} {wpe:12d}  {'FAIL':>8}  {str(e)[:50]}")
