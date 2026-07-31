// RUN: triton-opt %s -split-input-file -triton-amdgpu-insert-instruction-sched-hints="variant=attention" | FileCheck %s -check-prefix=INSTR_HINT
// RUN: triton-opt %s -split-input-file -triton-amdgpu-insert-instruction-sched-hints="variant=attention" -triton-amdgpu-lower-insert-instruction-sched-hints -verify-diagnostics | FileCheck %s -check-prefix=LOWER_HINT
// RUN: triton-opt %s -split-input-file -triton-amdgpu-insert-instruction-sched-hints="variant=scaled_gemm" | FileCheck %s -check-prefix=SCALED_INSTR_HINT
// RUN: triton-opt %s -split-input-file -triton-amdgpu-insert-instruction-sched-hints="variant=scaled_gemm" -triton-amdgpu-lower-insert-instruction-sched-hints -verify-diagnostics | FileCheck %s -check-prefix=SCALED_LOWER_HINT

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [32, 32, 8], isTransposed = true}>
#dot_op_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dot_op_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
// INSTR_HINT-LABEL: @insert_schedule_hint
// LOWER_HINT-LABEL: @insert_schedule_hint
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @insert_schedule_hint(
    %lb : index, %ub : index, %step : index,
    %arg0: tensor<128x128xf32, #dot_op_a>,
    %arg1: tensor<128x128xf32, #dot_op_b>,
    %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // INSTR_HINT: scf.for
    // INSTR_HINT-NEXT: amdg.instruction_sched_hint
    // INSTR_HINT-SAME: variant = #amdg.SchedHintVariant<attention>

    // LOWER_HINT: scf.for
    // LOWER_HINT-NEXT: rocdl.sched.barrier 0
    // LOWER_HINT-COUNT-2: tt.dot
    // LOWER_HINT: rocdl.iglp.opt 2
    // LOWER_HINT-NEXT: rocdl.sched.barrier 0
    // LOWER_HINT-NEXT: scf.yield
    %loop = scf.for %iv = %lb to %ub step %step iter_args(%c = %cst) -> (tensor<128x128xf32, #mma>) {
      %4 = tt.dot %arg0, %arg1, %c : tensor<128x128xf32, #dot_op_a> * tensor<128x128xf32, #dot_op_b> -> tensor<128x128xf32, #mma>
      %5 = math.exp2 %4 : tensor<128x128xf32, #mma>
      %6 = ttg.convert_layout %5 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #dot_op_a>
      %7 = tt.dot %6, %arg1, %c : tensor<128x128xf32, #dot_op_a> * tensor<128x128xf32, #dot_op_b> -> tensor<128x128xf32, #mma>
      scf.yield %7 : tensor<128x128xf32, #mma>
    }
    %8 = ttg.convert_layout %loop : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %8 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 64], isTransposed = true}>
#dot_op_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#dot_op_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
// SCALED_INSTR_HINT-LABEL: @insert_scaled_accumulation_schedule_hint
// SCALED_LOWER_HINT-LABEL: @insert_scaled_accumulation_schedule_hint
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @insert_scaled_accumulation_schedule_hint(
    %lb : index, %ub : index, %step : index,
    %arg0: tensor<128x128xi8, #dot_op_a>,
    %arg1: tensor<128x128xi8, #dot_op_b>
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // SCALED_INSTR_HINT: scf.for
    // SCALED_INSTR_HINT-NEXT: amdg.instruction_sched_hint
    // SCALED_INSTR_HINT-SAME: variant = #amdg.SchedHintVariant<scaled_gemm>

    // SCALED_LOWER_HINT: scf.for
    // SCALED_LOWER_HINT-COUNT-2: tt.dot_scaled
    // SCALED_LOWER_HINT: rocdl.iglp.opt 2
    // SCALED_LOWER_HINT-NEXT: scf.yield
    // SCALED_LOWER_HINT-NOT: rocdl.sched.barrier
    %loop = scf.for %iv = %lb to %ub step %step iter_args(%c = %cst) -> (tensor<128x128xf32, #mma>) {
      %0 = tt.dot_scaled %arg0, %arg1, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #dot_op_a> * tensor<128x128xi8, #dot_op_b> -> tensor<128x128xf32, #mma>
      %1 = tt.dot_scaled %arg0, %arg1, %0 lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x128xi8, #dot_op_a> * tensor<128x128xi8, #dot_op_b> -> tensor<128x128xf32, #mma>
      scf.yield %1 : tensor<128x128xf32, #mma>
    }
    tt.return
  }
}
