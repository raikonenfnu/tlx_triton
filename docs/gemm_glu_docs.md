# Fused GEMM + GLU on gfx950: `amd-gemm-glu-v9-tlx_test.py`

This documents the optimizations in the TLX v9-quad fused GLU kernel
(`third_party/tlx/tutorials/amd-gemm-glu-v9-tlx_test.py`) relative to the original
`third_party/tlx/tutorials/test_gemm_glu.py`, and what bottleneck each one solves.

## Workload

Both kernels compute the same fused op (one kernel launch, no intermediate
round-trip to HBM):

```
x   = A @ B + bias       # bias: (N,) broadcast over the M rows
out = x + x * Y          # GLU gate, Y: (M, N)
```

with `A: (M, K)`, `B: (K, N)`, `out/Y: (M, N)`, all fp16, fp32 accumulation.
Target shapes are skinny / large-N: `M=1024, N=21568, K in {256, 512, 1024}`.

The fusion itself is **not** the differentiator — `test_gemm_glu.py` already fuses
addmm+GLU into one kernel. Every optimization below is about *how* the GEMM and
the epilogue are executed.

## Headline results (same warmed-up run, identical data)

Measured with `glu_compare.py` (all four in one process). TFLOPS counts GEMM work
only (`2*M*N*K`); % is throughput relative to the rocBLAS **pure-GEMM** floor
(`torch.matmul`, no bias/GLU and no 44 MB `Y` read — a floor, not an
apples-to-apples target).

| K | rocBLAS GEMM (floor) | PyTorch addmm+GLU | test_gemm_glu.py (old) | **v9-quad fused (this)** |
|---|---|---|---|---|
| 256 | 281.2 TF (40.2 us) | 50.5 TF (18%) | 232.1 TF (83%) | **244.8 TF (87%)** |
| 512 | 448.0 TF (50.5 us) | 97.3 TF (22%) | 317.8 TF (71%) | **379.3 TF (85%)** |
| 1024 | 617.0 TF (73.3 us) | 180.7 TF (29%) | 467.5 TF (76%) | **538.7 TF (87%)** |

Net vs the old kernel: **+5% / +19% / +15%** (K=256/512/1024). Net vs PyTorch
eager: **4.8–6.4x**. Both fused kernels match the PyTorch reference within
`atol=rtol=2e-2`.

> Benchmarking note: always compare kernels *within the same process/run*. Cold
> vs warm GPU clock state can change absolute TFLOPS by ~2x and produce
> misleading cross-run comparisons — this is exactly why `glu_compare.py` exists.

## The old kernel (`test_gemm_glu.py`) and its bottlenecks

Design: `triton.autotune` over small tiles (64x64x64, 128x64, 128x128) with a
hand-written TLX SMEM pipeline (`NUM_STAGES=2`, i.e. single in-flight buffer) and
`GROUP_SIZE_M=1`.

Bottlenecks on the skinny large-N shapes:

1. **Low MFMA utilization / exposed steady-state memory latency.** Small tiles +
   single-buffer pipeline give little compute per memory wait, so MFMAs stall on
   operands and the math units idle.
2. **Plain row-major PID order, no L2 locality.** `GROUP_SIZE_M=1` walks output
   tiles in raw `pid` order; consecutive workgroups don't share an XCD's L2, so
   the per-XCD A/B footprint is large.
3. **Exposed epilogue.** Bias add, `Y` load, and the GLU are all issued
   back-to-back at the very end with the `Y` load consumed immediately
   (load-use), so the gate-read HBM latency is fully exposed.

## Optimizations in the v9-quad fused kernel

Each row: the optimization, the bottleneck it targets, and the evidence.

### 1. v9 quadrant-sliced, warp-pipelined GEMM hot loop
- **What:** 128/256-wide tiles split into four `HM x HN` quadrant accumulators
  (`acc_tl/bl/tr/br`). Operands are streamed global->shared with AMD async buffer
  ops (`buffer_load_to_local`), double-buffered (`nBuffers=2`) and unrolled by 2,
  with LDS reads / global prefetches interleaved between the MFMAs.
- **Bottleneck solved:** low MFMA utilization and exposed steady-state load
  latency (old bottleneck #1). The deeper pipeline keeps four independent MFMA
  chains busy while the next K-tiles are in flight.
- **Evidence:** steady-state stalls drop; effective GEMM throughput rises toward
  the rocBLAS floor (e.g. K=1024: 468 -> 539 TF).

### 2. XCD-aware PID remap + `GROUP_SIZE_M` swizzle (L2 locality — the v9 headline)
- **What:** `get_pids()` first remaps `pid` so consecutive tiles land on the same
  XCD (gfx950 has 8 XCDs), then applies a `GROUP_SIZE_M` super-grouping. This
  shrinks the per-XCD input footprint to ~`GM + ceil(P/GM)` (optimal `GM ~
  sqrt(P)`, `P` = workgroups/XCD).
- **Bottleneck solved:** L2 thrash / oversized per-XCD A/B footprint (old
  bottleneck #2). More A/B reuse hits in L2 instead of going to HBM.

### 3. Tile size `128x256` default (occupancy / memory-level parallelism)
- **What:** default tile changed from `256x256` to `128x256`.
- **Bottleneck solved:** **occupancy starvation** on these memory-bound shapes.
  At `256x256` the grid is only `cdiv(1024,256)*cdiv(21568,256) = 340` workgroups
  for 256 CUs (~1.3 per CU) — far too few in-flight memory requests to saturate
  HBM. `128x256` doubles the grid to 680, lifting MLP and bandwidth utilization.
  A smaller side benefit: `HM=64` halves the accumulator VGPR footprint (128 vs
  256 VGPR), leaving register headroom for the epilogue `Y` prefetch (#5).
- **Evidence (tile sweep, fused GLU, % of rocBLAS):**

  | tile | grid | K=256 | K=512 | K=1024 |
  |---|---|---|---|---|
  | 256x256 | 340 | 81% | 81% | 78% |
  | **128x256** | 680 | 87% | 89% | 86% |
  | 128x128 | 1352 | 90% | 89% | 84% |

  `128x256` is the most robust across the K range, so it is the default.

### 4. Interleaved epilogue stores
- **What:** each quadrant's `C` is stored the instant its accumulator is final,
  rather than computing all four then storing all four.
- **Bottleneck solved:** a fully-exposed serial store tail. ATT on the plain GEMM
  showed bunched-at-end stores were ~63% of stalls on the K=256 skinny case;
  interleaving overlaps each store's HBM-write latency with the remaining MFMAs.

### 5. Peeled final steady iteration + 2-ahead `Y` gate prefetch
- **What:** the epilogue is restructured so all address/mask/bias math is hoisted
  up front and the **first two `Y` gate quadrants are loaded before the
  penultimate (`iterMax-2`) dot block**; the other two are fetched one-ahead in
  the final block. This "peels" the last steady iteration into the epilogue so
  the gate reads overlap ~8 MFMAs instead of only the final 4.
- **Bottleneck solved:** exposed `Y` load-use latency in the epilogue (old
  bottleneck #3). ATT on the fused kernel showed the `Y` loads
  (`buffer_load_dwordx4`) plus the dependent `acc+bias`/store were the top stall
  sources; prefetching breaks the load-use chain.
- **Why only 2 (not 4):** prefetching all four `Y` tiles was tried and *regressed*
  the compute-bound K=1024 case (539 -> 473 TF) because holding 4 fp16 tiles cut
  occupancy. Peeling 2 keeps peak live `Y` tiles at 2 — no occupancy loss — and
  was the best variant on every shape.
- **Evidence (fused GLU, TFLOPS):**

  | variant | K=256 | K=512 | K=1024 |
  |---|---|---|---|
  | load-use (no prefetch) | 207 | 335 | 485 |
  | 1-ahead prefetch | 235 | 367 | 528 |
  | peel + 4 prefetch | 232 | 367 | 473 |
  | **peel + 2 prefetch** | **245** | **381** | **539** |

### 6. Edge masking for ragged M/N (correctness, no interior-tile cost)
- **What:** loads keep clean affine offsets and rely on AMD buffer ops being
  hardware out-of-bounds-safe (OOB lanes return 0, no fault); only the stores are
  masked (`m_top/m_bot/n_left/n_right` from `rem_m/rem_n`).
- **Bottleneck solved:** not a perf bottleneck but a correctness constraint —
  arbitrary `M, N` work without adding per-iteration masking cost to interior
  tiles. (K is still assumed divisible by `2*BLOCK_K`, the unroll-by-2 pipeline
  has no K-remainder path.)

## Compiler-side note: transposed MFMA layout (shared by both kernels)

The AMD backend (`third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp`)
already selects an `isTransposed` MFMA encoding "to enable larger vectorization
for global store instructions" — confirmed in our TTGIR
(`#ttg.amd_mfma<{... instrShape=[16,16,32], isTransposed=true}>`). This is why the
output stores vectorize to `buffer_store_dwordx2`. Going wider (`dwordx4`) would
require `tilesPerWarp={2,1}`, which the backend only enables for chain-dot
(attention-style) patterns, not a standalone GEMM. So this is *not* a lever we are
leaving unused — both kernels already benefit, and the wider path is unreachable
for this op shape.

## Remaining bottleneck

After all of the above the fused kernel is **memory-movement bound, not
hotspot-bound** (ATT stall fraction ~54% at K=256). The fusion must read a
C-sized `Y` (44 MB) *and* write `C` (44 MB) — roughly 2x the HBM traffic of a
pure GEMM — and at K=256 there is little compute to hide it behind. Measured
aggregate bandwidth at K=256 is ~2.07 TB/s, already *higher* than rocBLAS's pure
GEMM on the same shape (~1.37 TB/s), so we are close to the practical ceiling for
this op. The main residual lever is MLP/occupancy (smaller tiles help the lowest-K
case), not instruction-level changes.

## Reproducing — measuring TFLOPS each way

All commands run from `third_party/tlx/tutorials` with the repo's Python on the
path. TFLOPS in every harness below is `2*M*N*K / time` over
`M=1024, N=21568, K in {256,512,1024}`.

```bash
cd third_party/tlx/tutorials
export PYTHONPATH=<repo>/python      # e.g. /home/stwinata/nod/tlx/tlx_triton/python
```

### 1. All four side-by-side (recommended — same-run, apples-to-apples)

Prints rocBLAS pure GEMM, PyTorch baseline, the old `test_gemm_glu.py` kernel, and
our v9-quad fused kernel, each with TFLOPS / time / % of rocBLAS, plus a
correctness check for the two fused kernels:

```bash
python glu_compare.py
```

### 2. Our kernel only (v9-quad fused) + rocBLAS + PyTorch baseline

`amd-gemm-glu-v9-tlx_test.py`'s `__main__` benchmarks ours against the rocBLAS
GEMM floor and the PyTorch addmm+GLU baseline:

```bash
python amd-gemm-glu-v9-tlx_test.py
# rocBLAS pure GEMM (no GLU)   : ... TFLOPS
# Baseline (PyTorch addmm+GLU) : ... TFLOPS
# TLX v9-quad fused (1 kernel) : ... TFLOPS   <- ours
```

### 3. Old kernel only (`test_gemm_glu.py`) + PyTorch baseline

```bash
python test_gemm_glu.py
# Baseline (PyTorch addmm + GLU) : ... TFLOPS
# TLX fused (1 kernel)           : ... TFLOPS   <- old autotuned kernel
```

### 4. rocBLAS pure GEMM floor on its own

```bash
python - <<'PY'
import torch, triton
M, N = 1024, 21568
for K in (256, 512, 1024):
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ms = triton.testing.do_bench(lambda: torch.matmul(a, b), rep=200)
    print(f"K={K:5d}  rocBLAS GEMM {2*M*N*K*1e-12/(ms*1e-3):7.1f} TFLOPS ({ms*1e3:.1f} us)")
PY
```

> Caveat: running #2/#3/#4 in separate processes can give inconsistent absolute
> TFLOPS because of GPU clock/thermal state. For any cross-kernel claim, trust
> the single-process numbers from `glu_compare.py` (#1).

## Reproducing — ATT profiling (gfx950 / ROCm)

```bash
cd third_party/tlx/tutorials
PYTHONPATH=<repo>/python bash _att_profile_glu.sh 128 256 64 4 8 1024x21568x256
python <analyze_att.py> /home/nod/agent_programs/v9/att_glu_results
```
