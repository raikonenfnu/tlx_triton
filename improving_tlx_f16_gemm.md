# Improving TLX F16 GEMM on AMD (GFX950/MI300)

## Baseline Performance (4096x4096 @ fp16)

| Kernel | TFLOPS | % of rocBLAS |
|---|---|---|
| rocBLAS | ~1000 | 100% |
| Gluon v4_global_prefetch | 804 | ~80% |
| TLX amd_gemm_v4 (async_load) | 626 | ~63% |

## Root Cause Analysis

### Compiler Fixes Required (Done)
1. **async_token missing _flatten_ir** — `async_token` could not be a loop-carried
   variable in scf.for. Fixed by adding `async_token_type` and `_flatten_ir`.
2. **NVMMASharedEncodingAttr on AMD** — `local_alloc` unconditionally used
   NVIDIA-specific shared encoding, which AMD's direct-to-LDS lowering rejects.
   Fixed by selecting `SwizzledSharedEncodingAttr` on gfx* targets.

### Performance Gap: Gluon (804) vs TLX (626) — Key Differences

| Aspect | Gluon | TLX | Impact |
|---|---|---|---|
| Shared layout | PaddedSharedLayout (bank-conflict-free) | SwizzledShared (vec=1, no swizzle) | HIGH — bank conflicts on LDS reads |
| Load strategy | buffer_load_to_shared (scalar ptr + i32 offsets) | async_load via tensor pointers (i64 per thread) | MEDIUM — register pressure |
| Load layout | Explicit DistributedLinearLayout | Compiler-chosen BlockedLayout | MEDIUM — suboptimal coalescing |
| Shared load | load_shared_relaxed (no redundant barriers) | local_load (inserts barriers) | LOW-MEDIUM |
| Dot op | Explicit mfma() with layout | tl.dot() | LOW — same underlying op |

## Optimization Plan (ordered by expected impact)

### Phase 1: Shared Memory Layout
- [ ] Add proper swizzling params to SwizzledSharedEncoding (vec=8, perPhase=2, maxPhase=8)
- [ ] This directly reduces LDS bank conflicts during local_load

### Phase 2: Tile Configuration Tuning
- [ ] Test different BLOCK_M/N/K and num_warps combinations
- [ ] The gluon version uses 256x256x64 with 4 warps

### Phase 3: Pipeline Structure
- [ ] Evaluate wait_group(1) vs wait_group(2) in main loop
- [ ] Consider removing masks on loads when K % BLOCK_K == 0

## Changelog

- **v0**: Baseline async_load kernel — 626 TFLOPS
