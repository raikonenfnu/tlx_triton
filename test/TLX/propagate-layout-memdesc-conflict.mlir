// RUN: triton-opt -split-input-file --tlx-propagate-layout %s | FileCheck %s

// When multiple local_loads from the same allocation require different swizzled
// shared encodings, backward propagation produces a conflict ("unknown").
// The surviving memdesc require_layout ops must be dropped (replaced with their
// source) so they don't leak into the LLVM lowering pipeline.

#shared_default = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_swiz_a  = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared_swiz_b  = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @memdesc_require_layout_conflict_is_dropped
  tt.func public @memdesc_require_layout_conflict_is_dropped() -> (tensor<64x64xbf16, #blocked>, tensor<64x64xbf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32

    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16, #shared_default, #smem, mutable>

    %view0 = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x64x64xbf16, #shared_default, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared_default, #smem, mutable>
    %view1 = ttg.memdesc_index %alloc[%c1] : !ttg.memdesc<2x64x64xbf16, #shared_default, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared_default, #smem, mutable>

    // Two require_layout ops requesting different swizzle encodings from the
    // same allocation — backward propagation cannot unify them.
    // CHECK-NOT: tlx.require_layout
    %req0 = tlx.require_layout %view0 : !ttg.memdesc<64x64xbf16, #shared_default, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared_swiz_a, #smem, mutable>
    %req1 = tlx.require_layout %view1 : !ttg.memdesc<64x64xbf16, #shared_default, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared_swiz_b, #smem, mutable>

    // CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<64x64xbf16,
    // CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<64x64xbf16,
    %val0 = ttg.local_load %req0 : !ttg.memdesc<64x64xbf16, #shared_swiz_a, #smem, mutable> -> tensor<64x64xbf16, #blocked>
    %val1 = ttg.local_load %req1 : !ttg.memdesc<64x64xbf16, #shared_swiz_b, #smem, mutable> -> tensor<64x64xbf16, #blocked>

    tt.return %val0, %val1 : tensor<64x64xbf16, #blocked>, tensor<64x64xbf16, #blocked>
  }
}

// -----

// When there is no conflict (single layout requirement), propagation resolves
// the encoding and require_layout folds away as a no-op.

#shared_unswiz = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_swiz   = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
// CHECK-DAG: #[[$RESOLVED:.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#smem2 = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @memdesc_require_layout_no_conflict_folds
  tt.func public @memdesc_require_layout_no_conflict_folds() -> tensor<64x64xbf16, #blocked2> {
    %c0 = arith.constant 0 : i32

    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16, #[[$RESOLVED]], #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16, #shared_unswiz, #smem2, mutable>

    %view = ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<2x64x64xbf16, #shared_unswiz, #smem2, mutable> -> !ttg.memdesc<64x64xbf16, #shared_unswiz, #smem2, mutable>

    // CHECK-NOT: tlx.require_layout
    %req = tlx.require_layout %view : !ttg.memdesc<64x64xbf16, #shared_unswiz, #smem2, mutable> -> !ttg.memdesc<64x64xbf16, #shared_swiz, #smem2, mutable>

    // CHECK: ttg.local_load %{{.*}} : !ttg.memdesc<64x64xbf16, #[[$RESOLVED]], #smem, mutable>
    %val = ttg.local_load %req : !ttg.memdesc<64x64xbf16, #shared_swiz, #smem2, mutable> -> tensor<64x64xbf16, #blocked2>

    tt.return %val : tensor<64x64xbf16, #blocked2>
  }
}
