"""Minimal single-config runner for ATT profiling of the v9 TLX GEMM.

Usage: python _prof_run.py BM BN BK NB NW GM [SIZE]
Runs a few invocations so rocprofv3 kernel_iteration_range can capture one.
"""
import sys
import torch
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "m", os.path.join(os.path.dirname(__file__), "amd-gemm-v9-tlx_test.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Usage: _prof_run.py BM BN BK NB NW GM [SIZE] [kind]
# kind: "quad" (default) -> gemm_v9_quad, "mono" -> gemm_v9
bm, bn, bk, nb, nw, gm = (int(x) for x in sys.argv[1:7])
# size may be a single int "8192" (square) or "MxNxK" (rectangular).
size = sys.argv[7] if len(sys.argv) > 7 else "8192"
kind = sys.argv[8] if len(sys.argv) > 8 else "quad"
if "x" in size:
    M, N, K = (int(x) for x in size.lower().split("x"))
else:
    M = N = K = int(size)

torch.manual_seed(42)
a = torch.randn((M, K), device=m.DEVICE, dtype=torch.float16)
b = torch.randn((K, N), device=m.DEVICE, dtype=torch.float16)
c = torch.empty((M, N), device=m.DEVICE, dtype=torch.float16)

# A handful of launches; ATT captures iteration [1].
for _ in range(4):
    if kind == "quad":
        m.run_quad(a, b, c, bm, bn, bk, nw, gm, wpeu=2 if nw >= 8 else 0)
    else:
        m.run(a, b, c, bm, bn, bk, nb, nw, gm)
torch.cuda.synchronize()
print(f"done {kind} {bm}x{bn}x{bk} nb{nb} nw{nw} gm{gm} size{size}")
