# Async Load Pipeline & Tokens (TLX / AMD)

## Pattern

Typical software pipeline:

1. `tlx.async_load(...)` → returns **`async_token`**
2. `tlx.async_commit_group()` / grouping as modeled by your kernel
3. `tok_wait = tlx.async_load_wait_group(...)`
4. `tlx.local_load(memdesc, token=tok_wait)` — **pass the wait token**

---

## CRITICAL: `async_token` as loop-carried variable

**Symptom**: `NotImplementedError` in `ast_to_ttir` when using `async_token` in `scf.for` iter_args / loop-carried state.

**Cause**: Type lacked the Triton IR flattening protocol: `_flatten_ir`, `_unflatten_ir`, `_flatten_ir_types`, and a correct `type` property.

**Fix** (reference implementation):

- `third_party/tlx/language/tlx/types.py` — `async_token_type` with full flatten/unflatten support
- `third_party/tlx/dialect/triton_tlx.cc` — `get_async_token_type` binding

**Action for new types**: Any value carried across `scf.for` must implement the same protocol as other Triton-carriable types.

---

## HIGH IMPACT: Redundant LDS barriers (~+12% in reported GEMM)

**Default behavior**: After `async_load_wait_group`, AMD backend may insert **`s_barrier`** before **every** `local_load`.

**When redundant**: Data is already synchronized by the async wait.

**Fix**: Pass the wait-group return token into `local_load`:

```python
# Conceptual — API names match TLX tutorials / mem_ops
tok = tlx.async_load_wait_group(n)
tlx.local_load(src, token=tok)
```

**Mechanism**:

- Frontend annotates `local_load` with **`ttg.amdg.syncedViaAsyncWait`**
- `annotateLocalLoadsSyncedViaAsyncWait` in `Utility.cpp` uses that to **skip** the extra barrier

**Observed**: ~700 → ~785 TFLOPS in the documented tuning arc.

---

## Design notes for agents

- Treat **token plumbing** as part of the perf model, not optional hygiene.
- If assembly shows many **`s_barrier`** around shared loads, verify whether wait tokens are wired through to `local_load`.
- Loop pipelining: stage **tokens** like any other carried dependency; missing flatten support blocks the optimizer from forming legal `scf.for`.

---

## Related files

| Concern | File |
|---------|------|
| Mem ops surface | `third_party/tlx/language/tlx/mem_ops.py` |
| Token type | `third_party/tlx/language/tlx/types.py` |
| Bindings | `third_party/tlx/dialect/triton_tlx.cc` |
| Barrier elision | `third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp` |
