# Optimization Roadmap (TLX on AMD)

Ranked ideas from a multi-day MI300 / gfx950 GEMM push. Use as a backlog for compiler + kernel work.

---

## Tier A — High potential

### 1. DistributedLinearLayout (DLL) for global loads

- **Idea**: Gluon uses custom **DLL** for global memory coalescing; TLX often ends up in **BlockedLayout** for loads.
- **Hypothesis**: Exposing DLL (or equivalent) in TLX could improve load efficiency and feed better async DMA patterns.
- **Hints**: Study Gluon load lowering; identify where TLX pins layouts; extend dialect / ops or passes to allow DLL on load paths without breaking `canLoadDirectToLDS` assumptions.

### 2. AGPR accumulation for `dot`

- **Observation**: Reference TLX path showed **AGPR=0** (accumulation in **VGPR**); Gluon used **AGPRs** heavily, freeing VGPR for larger tiles.
- **Hypothesis**: Enabling MFMA accumulator register usage could unlock larger tiles / lower VGPR pressure.
- **Hints**: Trace `dot` → MFMA lowering on AMD; compare to Gluon’s register class assignment; may need backend or LLVM intrinsic path changes.

---

## Tier B — Conditional win

### 3. PaddedShared with MFMA-matched bases

- **Finding**: Padded only beat swizzle in experiments when row permutation / offset bases matched MFMA wavefront access.
- **Action**: If pursuing padded layouts, replicate **full** Gluon-style linear layout + bases, not padding alone.

---

## Tier C — Maintenance / verification

- Keep **order-matching** (`adjustEncodingForProducerOrder`) in mind when adding new memdesc transforms or async copy variants.
- Extend **error catalog** (`common_errors.md`) when new assert sites or LLVM casts appear.
- Re-benchmark against **rocBLAS** / **Gluon** when changing default layouts or barrier behavior.

---

## Reference numbers (4096×4096 FP16, MI300)

| Stack | ~TFLOPS |
|-------|---------|
| TLX (tuned) | ~795 |
| Gluon | ~804 |
| rocBLAS | ~1101 |

Gap to rocBLAS suggests both **library-specific** and **compiler** headroom remain.
