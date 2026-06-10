"""Same-run GLU TFLOPS comparison on identical data (the trustworthy way to
compare -- cold vs warm GPU clocks can skew cross-run numbers ~2x).

Measures, for M=1024, N=21568, K in {256,512,1024}:
  - rocBLAS pure GEMM (no GLU)        : torch.matmul    -> floor (not apples-to-apples)
  - PyTorch baseline (addmm + GLU)    : 2 eager ops
  - test_gemm_glu.py (old TLX kernel) : autotuned small-tile fused
  - v9-quad fused (ours)              : amd-gemm-glu-v9-tlx_test.py

Usage:  PYTHONPATH=<repo>/python python glu_compare.py
"""
import importlib.util, os, torch, triton

HERE = os.path.dirname(__file__)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ours = load("ours", "amd-gemm-glu-v9-tlx_test.py")
old = load("old", "test_gemm_glu.py")  # defines tlx_fused_addmm_glu + pytorch_baseline

DEVICE = ours.DEVICE
tflops = lambda ms, M, N, K: 2 * M * N * K * 1e-12 / (ms * 1e-3)

M, N = 1024, 21568
for K in [256, 512, 1024]:
    torch.manual_seed(0)
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(N, device=DEVICE, dtype=torch.float16)
    y = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
    ref = ours.pytorch_baseline(bias, a, b, y)

    print(f"\n{'='*74}\n  M={M}  N={N}  K={K}  fp16\n{'='*74}")
    print(f"  {'kernel':<34s}{'TFLOPS':>9s}{'time(us)':>11s}  {'note':<12s}")

    ms = triton.testing.do_bench(lambda: torch.matmul(a, b), rep=200)
    rb = tflops(ms, M, N, K)
    print(f"  {'rocBLAS pure GEMM (no GLU)':<34s}{rb:9.1f}{ms*1000:11.2f}  {'floor':<12s}")

    ms = triton.testing.do_bench(lambda: ours.pytorch_baseline(bias, a, b, y), rep=200)
    print(f"  {'PyTorch baseline (addmm+GLU)':<34s}{tflops(ms,M,N,K):9.1f}{ms*1000:11.2f}  {tflops(ms,M,N,K)/rb*100:.0f}% of rocBLAS")

    o = old.tlx_fused_addmm_glu(bias, a, b, y)
    ok = "OK" if torch.allclose(o, ref, atol=2e-2, rtol=2e-2) else "MISMATCH"
    ms = triton.testing.do_bench(lambda: old.tlx_fused_addmm_glu(bias, a, b, y), rep=200)
    print(f"  {'test_gemm_glu.py (old TLX)':<34s}{tflops(ms,M,N,K):9.1f}{ms*1000:11.2f}  {tflops(ms,M,N,K)/rb*100:.0f}% of rocBLAS [{ok}]")

    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    ours.tlx_fused_addmm_glu(bias, a, b, y, c)
    ok = "OK" if torch.allclose(c, ref, atol=2e-2, rtol=2e-2) else "MISMATCH"
    ms = triton.testing.do_bench(lambda: ours.tlx_fused_addmm_glu(bias, a, b, y, c), rep=200)
    print(f"  {'v9-quad fused (OURS)':<34s}{tflops(ms,M,N,K):9.1f}{ms*1000:11.2f}  {tflops(ms,M,N,K)/rb*100:.0f}% of rocBLAS [{ok}]")
