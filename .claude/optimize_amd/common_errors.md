# Common Errors — Symptom → Cause → Fix

Quick lookup for LLVM / frontend / perf failures when working on TLX + AMD.

---

## LLVM / lowering

| Error / symptom | Root cause | Fix |
|-----------------|------------|-----|
| `builtin.unrealized_conversion_cast` during LLVM translation (often around `BufferLoadToLocal` / shared loads) | Shared memory encoding **incompatible** with AMD **`canLoadDirectToLDS`** (e.g. **NVMMAShared** on gfx) | Use **`SwizzledSharedEncodingAttr`** on AMD; ensure **shared order matches async copy source order** (see `InsertRequireLayout.cpp` / `shared_memory_layouts.md`) |
| Assertion / crash in layout order helpers | Calling **`getOrderForMemory()`** on encoding without **`DistributedEncodingTrait`** | Guard with trait check before querying order |

---

## Python / frontend (`ast_to_ttir`)

| Error / symptom | Root cause | Fix |
|-----------------|------------|-----|
| `NotImplementedError` in `ast_to_ttir` at **`for`** loop | Loop-carried value (e.g. **`async_token`**) missing IR protocol | Implement **`_flatten_ir`**, **`_unflatten_ir`**, **`_flatten_ir_types`**, proper **`type`** on the token type; bind in `triton_tlx.cc` (`get_async_token_type`) |

---

## Performance (correct results, low TFLOPS)

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Far below roofline; high `s_barrier` in ASM | Redundant LDS barriers after async wait | Pass **`async_load_wait_group`** token to **`tlx.local_load(..., token=...)`** → **`syncedViaAsyncWait`** |
| Suboptimal but legal code | **Bank conflicts** from default swizzle | Apply MFMA-derived **vec / perPhase / maxPhase** (`make_amd_swizzled_layout` in `mem_ops.py`) |
| Regression after layout experiment | **PaddedShared** without MFMA-matched permutation | Prefer **swizzled** unless replicating Gluon’s full padded + base-offset story |

---

## Debugging order

1. Confirm arch is **gfx*** and shared encoding is **not** NVMMA.
2. If async global→LDS: verify **producer order** vs **shared encoding order**.
3. Verify **token** on **`local_load`** after waits.
4. Read ASM: **scratch**, **barriers**, **waitcnt**, **VGPR/AGPR**.
