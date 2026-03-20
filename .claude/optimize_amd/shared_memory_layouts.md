# Shared Memory Layouts (AMD) — CRITICAL

## Problem: NVMMA shared on AMD

`local_alloc` historically used NVIDIA-specific **`NVMMASharedEncodingAttr`** unconditionally.

AMD’s `canLoadDirectToLDS` (`third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp`) **rejects** that encoding → `BufferLoadToLocalOpConversion` can fail at LLVM lowering with **`builtin.unrealized_conversion_cast`**.

**Action**: On AMD (`arch.startswith("gfx")`), select **`SwizzledSharedEncodingAttr`** (see `make_amd_swizzled_layout()` in `third_party/tlx/language/tlx/mem_ops.py`).

---

## Swizzle parameters (MFMA-aligned)

Defaults `vec=1, perPhase=1, maxPhase=1` **do not** avoid bank conflicts meaningfully.

Derive from the same logic as `AMDMfmaEncodingAttr::composeSharedLayoutForOperand`:

```text
num_banks        = 64 for gfx95x, else 32
bank_bit_width   = 32
simd_width       = 16
vec              = min(128 // elem_bits, inner_dim)
elems_per_banks_row = (num_banks * bank_bit_width) // elem_bits
per_phase        = max(1, elems_per_banks_row // inner_dim)
max_phase        = max(min(simd_width // per_phase, inner_dim // vec), 1)
```

**Example**: FP16, `BLOCK_K=64` → `vec=8, perPhase=2, maxPhase=8` (~+12% vs naive swizzle in reported runs: 626 → ~700 TFLOPS).

---

## Swizzled vs Padded (Gluon-style)

| Approach | Mechanism | Overhead | When it wins |
|----------|-----------|----------|--------------|
| **Swizzled** (TLX default on AMD) | XOR-style bank remap | None | **Default choice** with correct params |
| **Padded** | Explicit padding + linear layout reordering rows for MFMA waves | ~3% memory | Only with **MFMA-matched offset bases / row permutation** |

**Empirical**: Padded + **identity** mapping (no row perm) ≈ **657 TFLOPS** vs swizzled with proper params ≈ **700 TFLOPS**. Padded only beats swizzle when the layout matches MFMA access patterns carefully.

---

## CRITICAL: Async DMA order must match shared order

**Symptom**: `canLoadDirectToLDS` fails even with SwizzledShared — LLVM cast / conversion issues.

**Cause**: Global load uses a layout whose **memory order** differs from the **shared encoding order**. Example: B is K-contiguous (`strides [1, K]`, e.g. `b.T.contiguous().T`) → `BlockedLayout` with **`order=[0,1]`**, while default swizzled shared may use **`order=[1,0]`**. Hardware DMA **cannot transpose** to reconcile.

**Compiler fix** (`InsertRequireLayout.cpp`): `adjustEncodingForProducerOrder`

1. Trace `local_load` → underlying allocation (peel `MemDescIndexOp`, `MemDescSubsliceOp`, `MemDescTransOp`).
2. Find `AsyncCopyGlobalToLocalOp` users of that allocation.
3. Get source tensor order via `getOrderForMemory()`.
4. If it differs from shared encoding order, recompute:
   `SwizzledSharedEncodingAttr::get(ctx, dotOpEnc, shape, producerOrder, cgaLayout, bitWidth, false)`.

**Safety**: Check **`DistributedEncodingTrait`** before `getOrderForMemory()` to avoid assertion failures on unsupported encodings.

---

## Agent checklist

- [ ] AMD path uses SwizzledShared, not NVMMAShared.
- [ ] Swizzle vec/perPhase/maxPhase computed from `elem_bits` and inner K tile (gfx95x uses `num_banks=64`).
- [ ] If async global→LDS is used, verify producer order vs shared order (compiler pass or IR inspection).
- [ ] If conversion still fails at LLVM, first suspect encoding + order mismatch before unrelated codegen bugs.
