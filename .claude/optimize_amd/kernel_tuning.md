# Kernel Tuning & AMDGCN Diagnostics

## Warp configuration and occupancy (MI300 / gfx950 class)

**Tested axes**: 4 vs 8 warps; `waves_per_eu` 1 vs 2.

**Best reported combo** for reference TLX GEMM:

- **8 warps**
- **`waves_per_eu=2`**
- **`matrix_instr_nonkdim=16`**

**Tradeoff intuition**:

| Config | Typical observation |
|--------|---------------------|
| **8 warps** | Lower VGPR pressure example: **VGPR=252, AGPR=0** → fits **2 waves/EU**; smaller tile per warp but higher occupancy |
| **4 warps** (e.g. Gluon-style) | **VGPR=448, AGPR=256**, **1 wave/EU**; larger per-warp tile |

Occupancy from **8 warps + 2 waves/EU** offset smaller per-warp tiles in the documented experiment.

---

## Pipeline depth / block shapes

- Tune **K-tile** (`BLOCK_K`) together with **swizzle** inner dimension — see `shared_memory_layouts.md`.
- Deeper async pipelines: watch **token** correctness across stages (see `async_operations.md`).
- Reference kernel: `third_party/tlx/tutorials/amd_gemm_v4.py`.

---

## AMDGCN assembly checklist

When comparing kernels or after compiler changes, inspect:

| Metric | What to look for |
|--------|------------------|
| **VGPR vs AGPR** | **AGPR** = dedicated accumulators. **AGPR=0** often means accumulators in **VGPR** (higher pressure). Search for `v_accvgpr_write` / `v_accvgpr_read`. |
| **`s_waitcnt`** | More waits → more stalls. Compare to a known-good kernel. |
| **`s_barrier`** | Should be **minimal**. Extra barriers from non-synced `local_load` paths hurt (fix with wait **token** on `local_load`). |
| **Scratch** | **Must be 0**. Any scratch ⇒ spilling ⇒ large regression. |
| **MFMA count** | Proxy for tile size per warp; more MFMAs ↔ larger tiles ↔ higher register pressure. |

---

## Agent workflow

1. Establish baseline TFLOPS + correct numerics.
2. Dump / disassemble; verify **scratch==0**, barrier count, waitcnt profile.
3. If VGPR-bound, consider fewer live values, smaller tiles, or (future) **AGPR** accumulation — see `optimization_roadmap.md`.
4. If barriers dominate after async waits, verify **`syncedViaAsyncWait`** path.
