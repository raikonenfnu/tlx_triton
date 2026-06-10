"""Single-config runner for ATT profiling of the v9 fused GLU GEMM.

Usage: python _prof_run_glu.py BM BN BK NW GM [MxNxK]
"""
import sys
import torch
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "m", os.path.join(os.path.dirname(__file__), "amd-gemm-glu-v9-tlx_test.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

bm, bn, bk, nw, gm = (int(x) for x in sys.argv[1:6])
size = sys.argv[6] if len(sys.argv) > 6 else "1024x21568x256"
M, N, K = (int(x) for x in size.lower().split("x"))

torch.manual_seed(0)
a = torch.randn((M, K), device=m.DEVICE, dtype=torch.float16)
b = torch.randn((K, N), device=m.DEVICE, dtype=torch.float16)
bias = torch.randn(N, device=m.DEVICE, dtype=torch.float16)
y = torch.randn((M, N), device=m.DEVICE, dtype=torch.float16)
c = torch.empty((M, N), device=m.DEVICE, dtype=torch.float16)

for _ in range(4):
    m.tlx_fused_addmm_glu(bias, a, b, y, c, bm, bn, bk, nw, gm)
torch.cuda.synchronize()
print(f"done glu {bm}x{bn}x{bk} nw{nw} gm{gm} size{size}")
