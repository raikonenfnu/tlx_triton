# TLX / AMD GPU Optimization — Overview

**Purpose**: Reference for optimizing TLX (Triton fork) kernels and compiler paths for **AMD MI300 / gfx950-class** GPUs.

**Scope**: Shared-memory encodings, async DMA ↔ layout alignment, async token IR plumbing, barrier elimination, warp/occupancy tuning, assembly triage.

---

## Performance anchor (4096×4096 FP16 GEMM, MI300)

| Milestone | TFLOPS | Notes |
|-----------|--------|--------|
| Baseline | 626 | Initial async_load kernel |
| AMD swizzle | ~700 | Correct `SwizzledSharedEncoding` params (+~12%) |
| PaddedShared detour | 657 | Reverted — needs MFMA-matched bases to win |
| Pipeline (8 warps, wpe=2) | ~700 | Occupancy tuning |
| `local_load(..., token=wait)` | ~785 | Skip redundant LDS barriers (+~12%) |
| AB^T layout + compiler fix | ~795 | Order-matched shared layout |

**External references (same problem size)**: rocBLAS ~1101; Gluon ~804 TFLOPS.

---

## Key codebase locations

| Area | Path |
|------|------|
| TLX memory ops (`local_alloc`, async load, `local_load`) | `third_party/tlx/language/tlx/mem_ops.py` |
| TLX types (`async_token`, layouts) | `third_party/tlx/language/tlx/types.py` |
| Shared layout assignment pass | `third_party/tlx/dialect/lib/Transforms/InsertRequireLayout.cpp` |
| TLX dialect Python bindings | `third_party/tlx/dialect/triton_tlx.cc` |
| AMD: direct-to-LDS eligibility, async-wait barrier skip | `third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp` |
| Reference GEMM | `third_party/tlx/tutorials/amd_gemm_v4.py` |

---

## Doc map

| File | Use when |
|------|----------|
| `shared_memory_layouts.md` | LLVM cast failures, swizzle params, DMA order vs shared order |
| `async_operations.md` | Pipelining, tokens, `syncedViaAsyncWait`, loop-carried `async_token` |
| `kernel_tuning.md` | Warps, `waves_per_eu`, block shapes, reading AMDGCN |
| `common_errors.md` | Symptom → root cause → fix |
| `optimization_roadmap.md` | What to try next, expected impact |

---

## Quick wins checklist

1. **gfx***: use `SwizzledSharedEncodingAttr` (not `NVMMASharedEncodingAttr`) with MFMA-derived vec/perPhase/maxPhase — see `make_amd_swizzled_layout()` in `mem_ops.py`.
2. **Async global → shared**: shared encoding **order** must match producer tensor memory order (compiler: `adjustEncodingForProducerOrder` in `InsertRequireLayout.cpp`).
3. After `async_load_wait_group`, pass the wait **token** into `tlx.local_load(..., token=...)` to avoid redundant `s_barrier` before each load.
