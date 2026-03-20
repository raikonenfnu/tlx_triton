# Improving TLX F16 GEMM on AMD (GFX950/MI300)

## Current Performance (4096x4096 @ fp16)

| Kernel | TFLOPS | % of rocBLAS |
|---|---|---|
| rocBLAS | ~1101 | 100% |
| Gluon v4_global_prefetch | 804 | ~73% |
| **TLX amd_gemm_v4 (current, AB_T)** | **~795** | **~72%** |
| TLX amd_gemm_v4 (v2) | ~785 | ~71% |
| TLX amd_gemm_v4 (initial) | 626 | ~57% |

## Compiler Fixes (Done)

1. **async_token loop-carry support** — `async_token` lacked `_flatten_ir`
   and a proper `type`, preventing it from being a loop-carried variable
   in `scf.for`. Fixed by adding `async_token_type` with `_flatten_ir` /
   `_unflatten_ir` / `_flatten_ir_types`, and a `get_async_token_type`
   C++ binding.

2. **NVMMASharedEncoding on AMD** — `local_alloc` unconditionally used
   NVIDIA-specific `NVMMASharedEncodingAttr`, which AMD's `canLoadDirectToLDS`
   rejects (the `BufferLoadToLocalOpConversion` returns `failure()`, producing
   `builtin.unrealized_conversion_cast` during LLVM translation). Fixed by
   selecting `SwizzledSharedEncodingAttr` on `gfx*` targets.

3. **Proper swizzle parameters** — Default `SwizzledSharedEncoding(vec=1,
   perPhase=1, maxPhase=1)` provides zero bank-conflict avoidance. Replaced
   with AMD MFMA-derived formula from `AMDMfmaEncodingAttr::composeSharedLayoutForOperand`:
   - `vec = 128 / elemBitWidth` (8 for f16)
   - `perPhase = max(1, elemsPerBanksRow / innerDimLen)` (2 for 64-wide K)
   - `maxPhase = max(min(simdWidth/perPhase, innerDim/vec), 1)` (8 for f16)

   Result: 626 → ~700 TFLOPS (+12%).

## Padding vs Swizzling Analysis

### What Gluon uses: PaddedSharedLayout with custom linear component

```
PaddedSharedLayout(
    [[512, 16]],          # 16 padding elements every 512 elements
    [[0,1],[0,2],...],    # custom row-permuting linear layout
    [], [BLOCK_M, BLOCK_K])
```

The linear component reorders rows (interleaved M-dim bases) to match MFMA
wavefront access patterns. This combination of padding + row permutation
achieves bank-conflict-free access at the cost of ~3% shared memory overhead.

### What TLX uses: SwizzledSharedEncoding

```
swizzled_shared<{vec=8, perPhase=2, maxPhase=8, order=[1,0]}>
```

XOR-based address remapping; no memory waste, no explicit row permutation.
The AMD LLVM backend transfers swizzling to source pointers via lane permute.

### Experiment result

Padded with identity mapping (no custom row permutation): 657 TFLOPS.
Swizzled with proper params: ~700 TFLOPS.

**Conclusion**: PaddedShared only wins with carefully crafted offset bases
matching the MFMA access pattern. Without those, swizzling is strictly better.
Adding PaddedShared support to TLX for future use, but default remains swizzled.

## TTGIR Comparison

### TLX kernel IR (key attributes)
```
#blocked  = sizePerThread=[1,8] threadsPerWarp=[8,8] warpsPerCTA=[4,1]   -- A loads
#blocked1 = sizePerThread=[1,8] threadsPerWarp=[2,32] warpsPerCTA=[4,1]  -- B loads
#shared   = swizzled<{vec=8, perPhase=2, maxPhase=8, order=[1,0]}>       -- A smem
#shared1  = swizzled<{vec=8, perPhase=1, maxPhase=16, order=[1,0]}>      -- B smem
#mma      = amd_mfma<{v=4, warps=[2,2], instr=[16,16,32], transposed}>
```

### Gluon kernel IR (key attributes)
```
DistributedLinearLayout with custom reg/lane/warp bases                  -- A/B loads
PaddedShared with custom linear component                                -- A/B smem
amd_mfma<{v=4, warps=[2,2], instr=[16,16,32], transposed}>              -- same
```

### Key differences
| Aspect | Gluon | TLX | Notes |
|---|---|---|---|
| Load layout | DistributedLinearLayout | BlockedLayout | DLL gives optimal coalescing |
| B shared order | [0,1] (K inner) | [1,0] (N inner) | Affects swizzle effectiveness |
| Shared encoding | PaddedShared + row permute | SwizzledShared | See analysis above |
| Store | buffer_store via MFMA layout | buffer_store via MFMA layout | Same |

## AMDGCN Assembly Comparison (TLX 8w vs Gluon 4w)

| Metric | TLX (8w) | Gluon (4w) | Notes |
|---|---|---|---|
| VGPRs | 252 | 448 | TLX lower → 2 waves/EU |
| AGPRs | 0 | 256 | Gluon accumulates in AGPRs |
| MFMAs | 128 | 256 | TLX: smaller sub-tile per warp |
| s_waitcnt | 25 | 19 | TLX still has more waits |
| s_barrier | 3 | 3 | Same |
| Scratch | 0 | 0 | No spilling in either |

## Remaining Optimization Ideas

### High Impact
- [ ] Expose DistributedLinearLayout or equivalent in TLX for load coalescing
- [x] Allow per-operand shared layout order in TLX (B should use [0,1])
  - Solved at compiler level: `adjustEncodingForProducerOrder` in `InsertRequireLayout`
- [ ] Add PaddedShared with MFMA-matched offset bases as a built-in option

### Medium Impact
- [ ] Tune num_stages / pipeline depth
- [ ] Profile LDS bank conflicts with rocprof to quantify remaining conflicts

### Low Impact
- [ ] Remove redundant masks when K % BLOCK_K == 0

## Changelog

- **v0**: Baseline async_load kernel — 626 TFLOPS
- **v1**: Proper swizzle params (vec=8, perPhase=2, maxPhase=8) — ~700 TFLOPS (+12%)
- **v1.1**: Tested PaddedShared with identity mapping — 657 TFLOPS (reverted, swizzled better)
- **v1.2**: Added PaddedSharedEncoding bindings for future use; tuned pipeline
  (wait_group(1), 8 warps, matrix_instr_nonkdim=16)
- **v2**: Pass async wait token to local_load — ~785 TFLOPS (+12% from v1.2)
  The `syncedViaAsyncWait` annotation eliminates redundant LDS barriers.
- **v3**: AB_T layout support — ~795 TFLOPS with K-contiguous B matrix
  Added `adjustEncodingForProducerOrder` to `InsertRequireLayout.cpp`: when
  `async_copy_global_to_local` fills shared memory, the shared encoding's order
  must match the global source's order (DMA hardware cannot transpose). The pass
  traces from `local_load` → allocation → `async_copy` producer, compares orders,
  and recomputes `SwizzledSharedEncodingAttr` via `DotOperandEncodingAttr` if they
  differ. This is ported from commit c622ce59d (origin).
