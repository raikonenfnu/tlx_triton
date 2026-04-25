// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx950 matrix-instruction-size=16" | FileCheck %s

// Verify that chain-dot detection works across loop iteration boundaries.
//
// In a pipelined FA kernel the QK dot produces `p` which is yielded as a
// loop-carried value.  On the *next* iteration, that yielded value becomes
// operand A of the PV dot.  Without cross-iteration awareness:
//   - isChainDotHead(QK) returns false  → wrong warpsPerCTA / tilesPerWarp
//   - isChainDotTail(PV) returns false  → kWidth=8 instead of kWidth=4
//
// With the fix both dots are recognized as a chain, giving consistent
// kWidth=4 on the PV dot (the chain-dot tail).

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

// CHECK-LABEL: cross_iter_chain_dot_fa
// The PV dot (tail) must have kWidth = 4 (not 8) because it consumes
// the output of the QK dot via a loop-carried value.  The QK dot (head)
// keeps kWidth = 8 since kWidth is only constrained on the tail.
// CHECK: tt.dot {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
// CHECK: tt.dot {{.*}} -> tensor<128x64xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @cross_iter_chain_dot_fa(
      %q: tensor<128x128xf16, #dotOp0>,
      %k: tensor<128x64xf16, #dotOp1>,
      %v: tensor<64x128xf16, #dotOp1>,
      %acc_init: tensor<128x128xf32, #blocked>,
      %p_init: tensor<128x64xf16, #dotOp0>,
      %lb: index, %ub: index, %step: index) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %result:2 = scf.for %iv = %lb to %ub step %step
        iter_args(%acc = %acc_init, %p_prev = %p_init) -> (tensor<128x128xf32, #blocked>, tensor<128x64xf16, #dotOp0>) {
      // PV dot: operand A (%p_prev) comes from the *previous* iteration's
      // QK dot result via the yield back-edge.  This is the chain-dot tail.
      %pv = tt.dot %p_prev, %v, %acc : tensor<128x64xf16, #dotOp0> * tensor<64x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>

      // QK dot: its result feeds into the *next* iteration's PV dot.
      // This is the chain-dot head (across iteration boundary).
      %qk = tt.dot %q, %k, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x64xf16, #dotOp1> -> tensor<128x64xf32, #blocked>
      %qk_f16 = arith.truncf %qk : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
      %p_next = ttg.convert_layout %qk_f16 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #dotOp0>

      scf.yield %pv, %p_next : tensor<128x128xf32, #blocked>, tensor<128x64xf16, #dotOp0>
    }
    tt.return %result#0 : tensor<128x128xf32, #blocked>
  }
}

// -----

// Same pattern but with bf16 operands.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>

// CHECK-LABEL: cross_iter_chain_dot_fa_bf16
// CHECK: tt.dot {{.*}} : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
// CHECK: tt.dot {{.*}} -> tensor<128x64xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @cross_iter_chain_dot_fa_bf16(
      %q: tensor<128x128xbf16, #dotOp0>,
      %k: tensor<128x64xbf16, #dotOp1>,
      %v: tensor<64x128xbf16, #dotOp1>,
      %acc_init: tensor<128x128xf32, #blocked>,
      %p_init: tensor<128x64xbf16, #dotOp0>,
      %lb: index, %ub: index, %step: index) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %result:2 = scf.for %iv = %lb to %ub step %step
        iter_args(%acc = %acc_init, %p_prev = %p_init) -> (tensor<128x128xf32, #blocked>, tensor<128x64xbf16, #dotOp0>) {
      %pv = tt.dot %p_prev, %v, %acc : tensor<128x64xbf16, #dotOp0> * tensor<64x128xbf16, #dotOp1> -> tensor<128x128xf32, #blocked>
      %qk = tt.dot %q, %k, %cst : tensor<128x128xbf16, #dotOp0> * tensor<128x64xbf16, #dotOp1> -> tensor<128x64xf32, #blocked>
      %qk_bf16 = arith.truncf %qk : tensor<128x64xf32, #blocked> to tensor<128x64xbf16, #blocked>
      %p_next = ttg.convert_layout %qk_bf16 : tensor<128x64xbf16, #blocked> -> tensor<128x64xbf16, #dotOp0>
      scf.yield %pv, %p_next : tensor<128x128xf32, #blocked>, tensor<128x64xbf16, #dotOp0>
    }
    tt.return %result#0 : tensor<128x128xf32, #blocked>
  }
}
