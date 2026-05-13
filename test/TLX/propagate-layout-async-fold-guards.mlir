// RUN: triton-opt -split-input-file --tlx-propagate-layout %s | FileCheck %s

// Test that tt.trans → local_alloc → local_load(dot_op) is NOT folded into
// convert_layout: the shared-memory encoding order carries transposition
// semantics that convert_layout cannot reproduce.

#blocked_tr = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_tr_out = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma_tr = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared_tr = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem_tr = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @trans_local_alloc_load_not_folded
  tt.func public @trans_local_alloc_load_not_folded(%arg0: tensor<32x64xf16, #blocked_tr>) -> tensor<64x64xf32, #mma_tr> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_tr>
    %b_arg = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_tr, kWidth = 4}>>
    %trans = tt.trans %arg0 {order = array<i32: 1, 0>} : tensor<32x64xf16, #blocked_tr> -> tensor<64x32xf16, #blocked_tr_out>
    // The local_alloc + local_load through trans must survive — not be folded.
    // CHECK: ttg.local_alloc
    %alloc = ttg.local_alloc %trans : (tensor<64x32xf16, #blocked_tr_out>) -> !ttg.memdesc<64x32xf16, #shared_tr, #smem_tr>
    // CHECK: ttg.local_load
    %load = ttg.local_load %alloc : !ttg.memdesc<64x32xf16, #shared_tr, #smem_tr> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_tr, kWidth = 4}>>
    // CHECK-NOT: ttg.convert_layout
    %dot = tt.dot %load, %b_arg, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_tr, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_tr, kWidth = 4}>> -> tensor<64x64xf32, #mma_tr>
    tt.return %dot : tensor<64x64xf32, #mma_tr>
  }
}

// -----

// Test that a token-bearing local_load (async DMA buffer) feeding an scf.for
// iter_arg is NOT folded when the loop body does local_alloc → local_load(dot).
// The LDS round-trip uses ds_read_tr16.b64 which is ~2x faster than
// warp-shuffle convert_layout on AMDGPU.

#blocked_async = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_async = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared_async = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_async = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @async_token_local_load_not_folded
  tt.func public @async_token_local_load_not_folded() -> tensor<64x64xf32, #mma_async> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_async>
    %b_arg = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_async, kWidth = 4}>>
    %dma_buf = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #shared_async, #smem_async, mutable>
    %buf0 = ttg.memdesc_index %dma_buf[%c0_i32] : !ttg.memdesc<2x64x32xf16, #shared_async, #smem_async, mutable> -> !ttg.memdesc<64x32xf16, #shared_async, #smem_async, mutable>
    %wait = ttg.async_wait {num = 0 : i32}
    %a_init = ttg.local_load %buf0 token %wait : !ttg.memdesc<64x32xf16, #shared_async, #smem_async, mutable> -> tensor<64x32xf16, #blocked_async>
    // CHECK: scf.for
    %result:2 = scf.for %i = %c0_i32 to %c2_i32 step %c1_i32
        iter_args(%acc = %cst, %a_reg = %a_init)
        -> (tensor<64x64xf32, #mma_async>, tensor<64x32xf16, #blocked_async>) : i32 {
      // The local_alloc + local_load must survive — source traces to async DMA.
      // CHECK: ttg.local_alloc
      %a_tmp = ttg.local_alloc %a_reg : (tensor<64x32xf16, #blocked_async>) -> !ttg.memdesc<64x32xf16, #shared_async, #smem_async>
      // CHECK: ttg.local_load
      %a_dot = ttg.local_load %a_tmp : !ttg.memdesc<64x32xf16, #shared_async, #smem_async> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_async, kWidth = 4}>>
      %dot = tt.dot %a_dot, %b_arg, %acc : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_async, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_async, kWidth = 4}>> -> tensor<64x64xf32, #mma_async>
      %buf1 = ttg.memdesc_index %dma_buf[%c1_i32] : !ttg.memdesc<2x64x32xf16, #shared_async, #smem_async, mutable> -> !ttg.memdesc<64x32xf16, #shared_async, #smem_async, mutable>
      %wait2 = ttg.async_wait {num = 0 : i32}
      %a_next = ttg.local_load %buf1 token %wait2 : !ttg.memdesc<64x32xf16, #shared_async, #smem_async, mutable> -> tensor<64x32xf16, #blocked_async>
      scf.yield %dot, %a_next : tensor<64x64xf32, #mma_async>, tensor<64x32xf16, #blocked_async>
    }
    tt.return %result#0 : tensor<64x64xf32, #mma_async>
  }
}

// -----

// Positive test: a normal (non-trans, non-async) local_alloc → local_load(dot)
// SHOULD still be folded into convert_layout.

#blocked_pos = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma_pos = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
#shared_pos = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_pos = #ttg.shared_memory

module attributes {tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @normal_local_alloc_load_is_folded
  tt.func public @normal_local_alloc_load_is_folded(%arg0: tensor<64x32xf16, #blocked_pos>) -> tensor<64x64xf32, #mma_pos> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma_pos>
    %b_arg = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_pos, kWidth = 4}>>
    %alloc = ttg.local_alloc %arg0 : (tensor<64x32xf16, #blocked_pos>) -> !ttg.memdesc<64x32xf16, #shared_pos, #smem_pos>
    %load = ttg.local_load %alloc : !ttg.memdesc<64x32xf16, #shared_pos, #smem_pos> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_pos, kWidth = 4}>>
    // CHECK-NOT: ttg.local_alloc %
    // CHECK-NOT: ttg.local_load
    // CHECK: ttg.convert_layout
    %dot = tt.dot %load, %b_arg, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_pos, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_pos, kWidth = 4}>> -> tensor<64x64xf32, #mma_pos>
    tt.return %dot : tensor<64x64xf32, #mma_pos>
  }
}
