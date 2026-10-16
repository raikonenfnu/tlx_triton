# AITER public ops and optional search (AMD only)

By default, Scout measures one public AITER operator per target case.
Attention uses the public dense or packed attention API.
GEMM uses `aiter.tuned_gemm.gemm_a16w16`, which owns implementation selection.
The report labels its dispatch, including Torch fallback.
This baseline needs no LLM search or manual tuning.

Enable `--aiter-search` only for a broader diagnostic comparison.
Select `--provider claude` or `codex` when that comparison also needs agent proposals.
The instructions below apply to this optional search.
Find the fastest observed existing AITER variant for the exact target contract.
The installed checkout takes precedence over online examples.
Primary source: https://github.com/ROCm/aiter .
This backend currently registers gfx950.
Other architectures need separate capability and correctness checks.

## Setup boundary and source map

An engineer or setup agent must complete [environment setup](setup.md) before discovery.
Discovery uses read-only source tools.
Do not install dependencies, replace Triton, edit kernels, or run benchmarks during discovery.
Report missing dependencies as unavailable.
Report inaccessible source or failed research tools as blocked.

The controller supplies the installed source path, Git SHA, runtime, and measured seed feedback.
Research starts in that AITER checkout, including when it is outside TLX.
Read these files for the requested operation:

| Workload | Public dispatch | Alternative implementations and evidence |
| --- | --- | --- |
| Packed or dense attention | `aiter/ops/mha.py` | `aiter/ops/triton/attention/mha.py`, its Triton/Gluon kernels, attention tests, and architecture-specific configs |
| FP16/BF16 GEMM | `aiter/tuned_gemm.py`, `aiter/ops/gemm_op_a16w16.py` | `aiter/ops/triton/gemm/basic/gemm_a16w16.py`, persistent kernels, ASM/CK/FlyDSL paths, tests, and tuning records |
| Dispatch and build policy | `aiter/__init__.py`, `aiter/jit/core.py` | `setup.py`, `aiter/jit/optCompilerConfig.json`, `aiter/configs/`, and Triton config resolution |

Check actual signatures, output types, hardware gates, and configuration keys.
Follow wrappers into the selected implementation.
Do not infer a callable signature from a kernel name.

## Bounded search

1. Inspect the public dispatch and every applicable seed result.
2. Identify source-supported implementations for the exact dtype, layout, shape, and mask.
3. Check architecture-specific defaults and exact-shape tuning records.
4. Propose a few distinct paths or justified configurations within the remaining budget.
5. Use measured feedback to refine later proposals.

Attention seeds compare public dispatch, direct Triton MHA, and direct Gluon MHA.
GEMM search seeds compare public dispatch with standard and persistent Triton wrappers.
A rejected seed remains evidence about that path, not the entire library.
Prioritize viable alternatives when one path fails.
The controller checks and measures every proposal, then independently remeasures its selection.

For attention, compare CK/ASM public dispatch with applicable direct Triton/Gluon paths.
Inspect FlyDSL dispatch when the current architecture and dtype support it.
Current gfx950 FlyDSL attention dispatch primarily serves FP8, so do not assume it serves FP16.
Inspect default configuration selection before changing block sizes, warp counts, or stage counts.
Use tuning records and supported configuration keys to justify each change.
Keep a configuration reusable across calls when a wrapper copies or mutates its dictionary.

For GEMM, examine exact-shape dispatch before selecting a generic kernel.
Consider persistent scheduling, split-K, supported direct configurations, and native tuned implementations.
Check reduction and temporary-buffer costs.
Architecture-specific tuning data belongs to its architecture.
A gfx942 record does not establish gfx950 performance.
Do not change a shared tuning database during discovery.
Pass supported configuration parameters through an adapter instead.

## Attention semantics

Packed inputs use THD layout with separate int32 cumulative Q/K lengths.
Pass actual maximum lengths, zero dropout, `softmax_scale=D**-0.5`, and the target's causal flag.
Check GQA, unequal Q/K lengths, bottom-right causal alignment, empty sequences, strides, and head-dimension limits.
Fully masked rows must produce the reference's zero output.
Use the supplied cumulative lengths without host reads inside the timed call.

Dense inputs use BHSD layout.
Public and direct MHA wrappers commonly expect BSHD.
A transpose view is allowed when the implementation supports its strides.
Required copies and conversion back to canonical BHSD belong inside `run`.
Return a new canonical tensor.
Extract the output explicitly if the selected API returns auxiliary tensors.

FP16 must remain FP16 throughout the measured operation.
Do not convert to BF16 or FP8 to access an assembly or FlyDSL path.
BF16 assembly measurements cannot serve as FP16 controls.
Do not measure backward, dropout generation, extra masks, or another causal convention.
An implementation's required auxiliary work remains part of its measured call.

The checked direct attention wrapper accepts `backend="triton"` or `backend="gluon"` and optional `config`.
These parameters belong to `aiter.ops.triton.attention.mha`, not necessarily the top-level public API.
Unsupported optional features can exclude Gluon even when Triton works.
Check the current capability helpers before proposing it.

## GEMM semantics and fallback detection

The target computes `A @ B` and returns the input dtype.
The Triton `gemm_a16w16` wrapper computes `X @ W.T` and defaults to BF16 output.
Pass `B.T.contiguous()` and explicitly select the target output dtype.
Keep any required packing inside `run`.
Column-major B can make that transpose a view.
Row-major B can require a measured copy.

Disable bias, activation, quantization, and scales for plain GEMM.
Allocate each returned output inside `run`.
An output-buffer API can use a newly allocated buffer per call.
Do not precompute a packed weight or reuse a returned output across calls.

`TunedGemm.mm` and related dispatch can select Torch, hipBLASLt, skinny, ASM, Triton, Opus, or FlyDSL.
Check the tuning database's dtype, layout, architecture, and selected `libtype`.
Label a Torch fallback explicitly.
Do not report it as an independent AITER kernel.
Native split-K paths can require semaphore/workspace handling.
Use their existing wrapper rather than inventing an execution protocol.

## Proposal evidence

Return adapters with `build(case, params) -> run(inputs) -> Tensor`.
Use existing AITER implementations only.
Do not add private timing code or substitute another backend.

Each rationale must identify:
- The source file, callable, and implementation path.
- Supported dtype/layout/shape and architecture.
- Configuration choice and source or tuning evidence.
- Conversion, allocation, and reduction work inside `run`.
- Known fallback or unsupported-feature behavior.

Return no proposals when no justified variant remains.
Never infer performance from a source comment or an unrelated published benchmark.
The shared harness owns correctness, input agreement, timing, and winner selection.

## Imported ADDMM/BMM and MXFP8 public-op comparisons

ADDMM uses `aiter.tuned_gemm.gemm_a16w16` with the supplied vector bias and explicit output dtype.
Set bias=true when inspecting its dispatch record. Preserve unit alpha/beta.
Required column-major-A conversion and B packing belong inside each call.
Shared-A BMM uses the public `batched_gemm_bf16_CK` wrapper on BF16 inputs.
Keep the zero A batch stride. Convert B to the wrapper's BNK layout inside timing.
Its CK instances use BF16 inputs. Keep FP16 unsupported rather than converting it to BF16.

For `flash_attn_mxfp8`, this backend supplies a labeled BF16 public attention reference.
Use the case's BF16 inputs, explicit softmax scale, and causal flag.
Do not claim this path implements TLX's block-scaled Q/K/V/P recipe.
Pre-quantized AITER code-object measurements use another input recipe and graph timing.
Do not import those numbers into the eager public-op target.
