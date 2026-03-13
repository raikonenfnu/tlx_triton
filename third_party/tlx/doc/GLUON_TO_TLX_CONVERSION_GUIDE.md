# Gluon to TLX Conversion Guide for Claude Agents

## Overview

This guide provides a systematic approach to converting Gluon kernels to TLX kernels for AMD GPU targets. TLX is a higher-level abstraction that automatically handles layout inference, while Gluon requires explicit layout specifications.

**Key Philosophy Difference:**
- **Gluon**: Explicit layout control at every stage (load, store, shared memory)
- **TLX**: Implicit layout inference by the compiler; developer focuses on algorithm

---

## Core API Mappings

### 1. Async Copy Operations

#### `buffer_load_to_shared` → `async_load`

**Gluon:**
```python
gl.amd.cdna4.async_copy.buffer_load_to_shared(
    smemA.index(g_idx),      # destination (shared memory view)
    a_base,                   # source base pointer
    a_offsets                 # offset tensor
)
```

**TLX:**
```python
tlx.async_load(
    a_base + a_offsets,      # source pointers (combined base + offsets)
    tlx.local_view(smemA, g_idx),  # destination (shared memory view)
    mask=offs_k[None, :] < K - g_idx * BLOCK_K  # optional mask
)
```

**Key Differences:**
- Gluon: destination first, source pointer and offsets separate
- TLX: source first (pointer arithmetic), destination second
- TLX: explicit mask parameter for bounds checking
- TLX: returns a token (though often unused in simple cases)

---

#### `commit_group` → `async_load_commit_group`

**Gluon:**
```python
gl.amd.cdna4.async_copy.commit_group()
```

**TLX:**
```python
token_a = tlx.async_load(...)
token_b = tlx.async_load(...)
tlx.async_load_commit_group([token_a, token_b])
```

**Key Differences:**
- Gluon: Implicit group (all pending async ops)
- TLX: Explicit token list to commit specific operations together

---

#### `wait_group` → `async_load_wait_group`

**Gluon:**
```python
gl.amd.cdna4.async_copy.wait_group(1)  # wait until only 1 group remains pending
```

**TLX:**
```python
tlx.async_load_wait_group(1)  # wait until only 1 group remains pending
```

**Identical semantics:** Both wait until N groups remain in flight.

---

### 2. Shared Memory Operations

#### `allocate_shared_memory` → `local_alloc`

**Gluon:**
```python
smemA = gl.allocate_shared_memory(
    a_ptr.dtype.element_ty,              # element type
    [nBuffers, BLOCK_M, BLOCK_K],        # shape (3D for double buffering)
    sharedLayoutA                         # explicit PaddedSharedLayout
)
```

**TLX:**
```python
buffers_A = tlx.local_alloc(
    (BLOCK_M, BLOCK_K),                  # shape (2D tile)
    tlx.dtype_of(a_ptr),                 # element type via dtype_of
    NUM_BUFFERS                           # number of buffers (3rd argument)
)
```

**Key Differences:**
- Gluon: Shape includes buffer dimension `[nBuffers, ...]`, explicit layout
- TLX: Shape is just the tile `(M, K)`, buffer count is separate parameter
- TLX: No layout specification - compiler infers optimal padding and swizzle

---

#### `smem.index(idx)` → `local_view`

**Gluon:**
```python
a_k_view = smemA.index(k % 2)  # method call on smem object
```

**TLX:**
```python
a_k_view = tlx.local_view(buffers_A, k % NUM_BUFFERS)  # function call
```

**Identical semantics:** Both select the k-th buffer from double/triple buffered shared memory.

---

#### `load_shared_relaxed` → `local_load`

**Gluon:**
```python
a = gl.amd.cdna4.async_copy.load_shared_relaxed(
    smemA.index(l_idx),      # source (shared memory view)
    dotOpLayoutA             # target layout for dot operand
)
```

**TLX:**
```python
a_k_prev_shmem = tlx.local_view(buffers_A, buf)
a_k_prev_reg = tlx.local_load(a_k_prev_shmem)  # no layout argument
```

**Key Differences:**
- Gluon: Explicit `dotOpLayoutA` specifies register layout for MMA
- TLX: Layout is inferred from subsequent `tl.dot` usage
- Gluon: Single call combines view + load
- TLX: Separate `local_view` then `local_load`

---

### 3. Layout Specifications

#### Gluon: Explicit Layouts

Gluon requires explicit layout definitions for all data movement:

```python
# Global load layout
gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [128, 0]],
    lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
    warp_bases=[[1, 0], [2, 0]],
    block_bases=[],
    shape=[BLOCK_M, BLOCK_K],
)

# Shared memory layout
sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
    [[512, 16]],  # padding configuration
    [[0, 1], [0, 2], [0, 4], [0, 8], ..., [128, 0]],  # swizzle bases
    [],
    [BLOCK_M, BLOCK_K],
)

# MMA layout
mfmaLayout: gl.constexpr = gl.AMDMFMALayout(
    version=4,
    instr_shape=[16, 16, 32],
    transposed=True,
    warps_per_cta=[2, 2]
)

# Dot operand layouts
dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
    operand_index=0,
    parent=mfmaLayout,
    k_width=8
)
```

#### TLX: Layout Inference

**TLX completely eliminates explicit layouts:**
```python
# No layout definitions needed!
# Compiler infers:
# - Shared memory swizzle and padding from access patterns
# - Register layouts from tl.dot() usage
# - Global load distribution from thread block configuration
```

**What TLX still requires:**
- `num_warps` in kernel launch (e.g., `num_warps=4`)
- Block size constants (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`)
- Matrix instruction hints via autotune configs (`matrix_instr_nonkdim`)

---

### 4. Store Operations

#### `buffer_store` → `tl.store`

**Gluon:**
```python
gl.amd.cdna3.buffer_store(
    ptr=c_base,
    offsets=c_offsets,
    stored_value=c,
    mask=c_mask
)
```

**TLX:**
```python
c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

**Key Differences:**
- Gluon: Separate base pointer and offsets
- TLX: Combined pointer arithmetic before store
- Identical mask semantics

---

### 5. MMA Operations

**Gluon:**
```python
acc = gl.amd.cdna3.mfma(a, b, acc)
```

**TLX:**
```python
acc = tl.dot(a_k_prev_reg, b_k_prev_reg, acc)
```

**Identical semantics:** Both perform matrix multiply-accumulate. TLX uses Triton's generic `tl.dot` which lowers to MFMA on AMD GPUs.

---

### 6. Type Conversion

**Gluon:**
```python
a_ptr.dtype.element_ty  # access underlying element type
c = acc.to(a_ptr.dtype.element_ty)
```

**TLX:**
```python
tlx.dtype_of(a_ptr)     # extract dtype from pointer
c = acc.to(tlx.dtype_of(c_ptr))
```

---

### 7. Range Operations

**Gluon:**
```python
for k in range(0, iterMax - 1):  # Python range
    ...
```

**TLX:**
```python
for k in tl.range(NUM_STAGES - 1, K_ITERS, num_stages=0):  # tl.range for pipelining control
    ...
```

**Key Differences:**
- TLX: `tl.range` allows `num_stages` parameter to disable auto-pipelining
- TLX: `loop_unroll_factor` for explicit unrolling (e.g., epilogue)

---

## Complete Example: Pipelined GEMM Prologue

### Gluon Implementation

```python
@gluon.jit
def v4_global_prefetch(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    # 1. Define layouts explicitly
    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(...)
    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(...)
    mfmaLayout: gl.constexpr = gl.AMDMFMALayout(...)
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(...)

    # 2. Allocate shared memory with layout
    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty,
        [nBuffers, BLOCK_M, BLOCK_K],
        sharedLayoutA
    )

    # 3. Compute offsets with layout hints
    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak

    # 4. Prologue: async load into buffer 0
    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA.index(g_idx), a_base, a_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    # 5. Main loop
    for k in range(0, iterMax - 1):
        l_idx = k % 2
        g_idx = 1 - l_idx

        # Load next iteration
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA.index(g_idx), a_base, a_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        # Wait for previous iteration
        gl.amd.cdna4.async_copy.wait_group(1)

        # Load from shared to registers with layout
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smemA.index(l_idx), dotOpLayoutA
        )
        acc = gl.amd.cdna3.mfma(a, b, acc)
```

### TLX Implementation

```python
@triton.jit
def matmul_kernel_pipelined_mi300(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    # 1. No layout definitions needed!

    # 2. Allocate shared memory (layout inferred)
    NUM_BUFFERS = NUM_STAGES - 1
    buffers_A = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_K),
        tlx.dtype_of(a_ptr),
        NUM_BUFFERS
    )

    # 3. Compute offsets (no layout hints)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # 4. Prologue: async load with token tracking
    for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        a_smem_view = tlx.local_view(buffers_A, i)
        token_a = tlx.async_load(
            a_ptrs,
            a_smem_view,
            mask=offs_k[None, :] < K - i * BLOCK_SIZE_K
        )
        a_ptrs += BLOCK_SIZE_K * stride_ak
        tlx.async_load_commit_group([token_a, token_b])

    # 5. Main loop (disable auto-pipelining)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(NUM_STAGES - 1, K_ITERS, num_stages=0):
        # Prefetch next iteration
        a_k_smem_view = tlx.local_view(buffers_A, k % NUM_BUFFERS)
        token_a = tlx.async_load(
            a_ptrs,
            a_k_smem_view,
            mask=offs_k[None, :] < K - k * BLOCK_SIZE_K
        )
        tlx.async_load_commit_group([token_a, token_b])

        # Compute on previous iteration (no layout needed)
        buf = (k - NUM_STAGES - 1) % NUM_BUFFERS
        a_k_prev_shmem = tlx.local_view(buffers_A, buf)
        tlx.async_load_wait_group(2)
        a_k_prev_reg = tlx.local_load(a_k_prev_shmem)
        acc = tl.dot(a_k_prev_reg, b_k_prev_reg, acc)

        a_ptrs += BLOCK_SIZE_K * stride_ak
```

---

## Conversion Checklist

When converting a Gluon kernel to TLX, follow this systematic process:

### Step 1: Remove Layout Definitions
- [ ] Delete all `gl.DistributedLinearLayout` definitions
- [ ] Delete all `gl.PaddedSharedLayout` definitions
- [ ] Delete all `gl.AMDMFMALayout` definitions
- [ ] Delete all `gl.DotOperandLayout` definitions
- [ ] Delete all `gl.SliceLayout` uses in `gl.arange`

### Step 2: Update Memory Allocation
- [ ] Replace `gl.allocate_shared_memory(dtype, [nBuffers, ...], layout)` with `tlx.local_alloc((M, K), dtype, nBuffers)`
- [ ] Change dtype access from `ptr.dtype.element_ty` to `tlx.dtype_of(ptr)`

### Step 3: Update Offset Computation
- [ ] Replace `gl.arange(0, SIZE, gl.SliceLayout(...))` with `tl.arange(0, SIZE)`
- [ ] Combine base pointers and offsets: `ptr + (offsets)` instead of separate arguments

### Step 4: Update Async Copy Operations
- [ ] Replace `gl.amd.cdna4.async_copy.buffer_load_to_shared(dest.index(i), src, offsets)` with `tlx.async_load(src + offsets, tlx.local_view(dest, i), mask=...)`
- [ ] Replace `gl.amd.cdna4.async_copy.commit_group()` with `tlx.async_load_commit_group([token_a, token_b])`
- [ ] Replace `gl.amd.cdna4.async_copy.wait_group(N)` with `tlx.async_load_wait_group(N)`
- [ ] Capture and pass tokens from `async_load` to `commit_group`

### Step 5: Update Shared Memory Access
- [ ] Replace `smem.index(i)` with `tlx.local_view(smem, i)`
- [ ] Replace `gl.amd.cdna4.async_copy.load_shared_relaxed(smem.index(i), layout)` with `tlx.local_load(tlx.local_view(smem, i))`

### Step 6: Update MMA Operations
- [ ] Replace `gl.amd.cdna3.mfma(a, b, acc)` with `tl.dot(a, b, acc)`

### Step 7: Update Store Operations
- [ ] Replace `gl.amd.cdna3.buffer_store(ptr=base, offsets=offs, stored_value=val, mask=m)` with `tl.store(base + offs, val, mask=m)`

### Step 8: Update Control Flow
- [ ] Replace Python `range()` with `tl.range()` where pipelining control is needed
- [ ] Add `num_stages=0` to disable auto-pipelining in main loops
- [ ] Add `loop_unroll_factor=N` for epilogue/prologue loops

### Step 9: Update Decorator and Launch
- [ ] Replace `@gluon.jit` with `@triton.jit` (with autotune if needed)
- [ ] Replace `@triton.autotune(...)` configs: remove layout-related keys
- [ ] Ensure `num_warps` is specified in kernel launch or autotune config

### Step 10: Verify Correctness
- [ ] Run: `pytest third_party/tlx/tutorials/testing/test_correctness.py`
- [ ] Check for layout-related compilation errors
- [ ] Validate output against reference (torch.matmul, etc.)

---

## Common Pitfalls

### 1. **Forgetting to combine pointer arithmetic**
❌ **Wrong:**
```python
tlx.async_load(a_base, a_offsets, tlx.local_view(...))  # separate base and offsets
```
✅ **Correct:**
```python
tlx.async_load(a_base + a_offsets, tlx.local_view(...))  # combined
```

### 2. **Not capturing async_load tokens**
❌ **Wrong:**
```python
tlx.async_load(a_ptrs, a_smem_view)
tlx.async_load_commit_group()  # nothing to commit!
```
✅ **Correct:**
```python
token_a = tlx.async_load(a_ptrs, a_smem_view)
token_b = tlx.async_load(b_ptrs, b_smem_view)
tlx.async_load_commit_group([token_a, token_b])
```

### 3. **Calling local_view + local_load in one step**
❌ **Wrong (Gluon style):**
```python
a = tlx.local_load(buffers_A.index(i))  # .index() doesn't exist
```
✅ **Correct:**
```python
a_view = tlx.local_view(buffers_A, i)
a = tlx.local_load(a_view)
```

### 4. **Providing layout arguments to TLX APIs**
❌ **Wrong:**
```python
buffers_A = tlx.local_alloc((M, K), dtype, NUM_BUFFERS, layout=sharedLayoutA)
```
✅ **Correct:**
```python
buffers_A = tlx.local_alloc((M, K), dtype, NUM_BUFFERS)  # no layout!
```

### 5. **Using Python range instead of tl.range for pipelined loops**
❌ **Wrong:**
```python
for k in range(NUM_STAGES - 1, K_ITERS):  # Python range can't control pipelining
```
✅ **Correct:**
```python
for k in tl.range(NUM_STAGES - 1, K_ITERS, num_stages=0):  # disable auto-pipeline
```

### 6. **Incorrect buffer indexing math**
Gluon and TLX both use modulo for ping-pong buffering, but ensure the logic matches:
```python
# Gluon style
l_idx = k % 2
g_idx = 1 - l_idx

# TLX style (same semantics)
buf = k % NUM_BUFFERS
next_buf = (k + 1) % NUM_BUFFERS
```

---

## Performance Considerations

While TLX infers layouts automatically, you can still influence performance:

1. **Block size selection**: Larger blocks (e.g., 256x256) improve occupancy but increase register pressure
   ```python
   BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 256, 256, 64  # tune via autotune
   ```

2. **Warp count**: Controls parallelism within a CTA
   ```python
   num_warps=8  # 4 or 8 typical for AMD
   ```

3. **Pipeline depth**: Balance latency hiding vs. shared memory usage
   ```python
   NUM_STAGES = 2  # 2-3 typical for CDNA architectures
   ```

4. **Matrix instruction shape**: Via autotune config
   ```python
   triton.Config({'matrix_instr_nonkdim': 16, ...}, num_warps=4)
   ```

5. **Waves per EU**: AMD-specific occupancy hint
   ```python
   triton.Config({'waves_per_eu': 2, ...}, num_warps=8)
   ```

---

## Reference Files

**Gluon example:**
`/home/stwinata/nod/tlx_work/gfx9-gluon-tutorials/kernels/gemm/a16w16/v4_global_prefetch/matmul_kernel.py`

**TLX example:**
`third_party/tlx/tutorials/amd-gemm-pipelined-gfx950.py`

**API documentation:**
- TLX primitives: Use `/tlx-api-reference` skill
- Full reference: `third_party/tlx/README.md`

---

## Summary Table

| **Operation**                  | **Gluon**                                                      | **TLX**                                                   |
|--------------------------------|----------------------------------------------------------------|-----------------------------------------------------------|
| Kernel decorator               | `@gluon.jit`                                                   | `@triton.jit` (with optional `@triton.autotune`)          |
| Async global→shared            | `gl.amd.cdna4.async_copy.buffer_load_to_shared(dst, src, off)` | `tlx.async_load(src+off, dst, mask=...)`                  |
| Commit async group             | `gl.amd.cdna4.async_copy.commit_group()`                       | `tlx.async_load_commit_group([tok_a, tok_b])`             |
| Wait for async group           | `gl.amd.cdna4.async_copy.wait_group(N)`                        | `tlx.async_load_wait_group(N)`                            |
| Allocate shared memory         | `gl.allocate_shared_memory(dtype, [n,M,K], layout)`            | `tlx.local_alloc((M,K), dtype, n)`                        |
| Index into buffer              | `smem.index(i)`                                                | `tlx.local_view(smem, i)`                                 |
| Load shared→registers          | `gl.amd.cdna4.async_copy.load_shared_relaxed(src, layout)`     | `tlx.local_load(src)`                                     |
| Matrix multiply-accumulate     | `gl.amd.cdna3.mfma(a, b, acc)`                                 | `tl.dot(a, b, acc)`                                       |
| Store to global                | `gl.amd.cdna3.buffer_store(ptr=p, offsets=o, stored_value=v)`  | `tl.store(p+o, v, mask=...)`                              |
| Layout specification           | Explicit (DistributedLinearLayout, PaddedSharedLayout, etc.)   | Implicit (compiler inferred)                              |
| Type extraction                | `ptr.dtype.element_ty`                                         | `tlx.dtype_of(ptr)`                                       |
| Range with pipeline control    | Python `range()`                                               | `tl.range(..., num_stages=0, loop_unroll_factor=N)`       |

---

## Quick Reference: Side-by-Side Prologue

### Gluon
```python
g_idx = 0
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
gl.amd.cdna4.async_copy.commit_group()
```

### TLX
```python
for i in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
    a_smem_view = tlx.local_view(buffers_A, i)
    b_smem_view = tlx.local_view(buffers_B, i)
    token_a = tlx.async_load(a_ptrs, a_smem_view, mask=offs_k[None, :] < K - i * BLOCK_K)
    token_b = tlx.async_load(b_ptrs, b_smem_view, mask=offs_k[:, None] < K - i * BLOCK_K)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
    tlx.async_load_commit_group([token_a, token_b])
```

---

## Quick Reference: Side-by-Side Main Loop

### Gluon
```python
for k in range(0, iterMax - 1):
    l_idx = k % 2
    g_idx = 1 - l_idx

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.wait_group(1)
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    acc = gl.amd.cdna3.mfma(a, b, acc)
```

### TLX
```python
for k in tl.range(NUM_STAGES - 1, K_ITERS, num_stages=0):
    # Prefetch k+NUM_STAGES-1
    a_k_smem_view = tlx.local_view(buffers_A, k % NUM_BUFFERS)
    token_a = tlx.async_load(a_ptrs, a_k_smem_view, mask=offs_k[None, :] < K - k * BLOCK_K)
    tlx.async_load_commit_group([token_a, token_b])

    # Compute on k
    buf = (k - NUM_STAGES - 1) % NUM_BUFFERS
    a_k_prev_shmem = tlx.local_view(buffers_A, buf)
    tlx.async_load_wait_group(2)
    a_k_prev_reg = tlx.local_load(a_k_prev_shmem)
    acc = tl.dot(a_k_prev_reg, b_k_prev_reg, acc)

    a_ptrs += BLOCK_K * stride_ak
```

---

This guide should enable you to systematically convert any Gluon kernel to TLX. Remember: **TLX's key advantage is eliminating layout boilerplate while maintaining performance through compiler inference.**
