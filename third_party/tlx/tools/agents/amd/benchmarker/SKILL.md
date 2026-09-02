---
name: gfx9-bf16-gemm-benchmarker
description: Search, correctness-check, and report the best TLX gfx9 BF16 GEMM or addmm implementation for maintained production shapes on AMD CDNA GPUs. Use for TLX-versus-PyTorch GEMM comparisons, per-shape kernel selection, or publishing reproducible benchmark issues.
---

# gfx9 BF16 GEMM Benchmarker

Use `scripts/benchmark_gfx9_bf16_gemm.py` as the source of truth for candidate
eligibility, correctness, timing, output schema, and issue formatting. Keep named
workloads in `targets/*.json`; do not hard-code production shapes into the script.
`problem_set1_bf16_gemm_shapes.json` names the original issue-20 suite;
`problem_set2_gfx950_priority_problems.json` is the runnable projection of the
priority-problem dataset, including its explicit operand-layout conventions.
Preserve the source inventory in `datasets/` when its layouts are unknown.

## Workflow

1. Work from a clean checkout of the requested TLX revision. Record the full Git
   commit and do not silently benchmark a different editable Triton install.
2. For native/compiler changes, run `make` before GPU tests. Confirm the imported
   `triton.__file__` belongs to the checkout.
3. Verify the target is the intended gfx architecture and clocks are not locked.
   Record GPU, ROCm, PyTorch, Triton, commit, clock policy, and benchmark policy.
4. Run the maintained suite with a fresh Triton cache. The harness must compile
   and verify each candidate before timing it. Unsupported, failed, and incorrect
   candidates are results, not reasons to abort the whole suite.
5. Select the lowest median latency among correct candidates. Compare it with the
   matching PyTorch operation: `torch.matmul` for `mm`, `torch.addmm` for `addmm`.
6. Preserve raw JSON and generated Markdown together. Repeat the final winners
   when a result is close or noisy; do not publish a winner whose advantage is
   smaller than observed run-to-run variance.
7. Before creating or editing a GitHub issue, inspect existing matching issues and
   obtain authorization for the external mutation if it was not already requested.
   Include the exact command, revision, hardware, timing policy, correctness
   tolerances, layouts/strides, complete result table, exclusions, and raw artifact.

Recommended command from the Triton repository root:

```bash
ROCR_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/tlx-gfx9-bf16-$(git rev-parse --short HEAD) \
python third_party/tlx/tools/agents/amd/benchmarker/scripts/benchmark_gfx9_bf16_gemm.py \
  --suite third_party/tlx/tools/agents/amd/benchmarker/targets/problem_set1_bf16_gemm_shapes.json \
  --output-dir /tmp/tlx-gfx9-bf16-results
```

Use `--quick` only to validate the harness. Publish results from the default
three-repeat timing policy or a stricter policy.
