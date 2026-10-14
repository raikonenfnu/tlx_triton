# AITER (AMD only)

Find the fastest observed AITER implementation that supports the target's EXACT dtype and semantics.
AITER is vendor-specific: do not activate this backend on NVIDIA. Inspect the installed checkout,
its version/commit, hardware dispatch, tests, tuning configuration, and public wrappers. Primary
source: https://github.com/ROCm/aiter . APIs change. Installed source takes precedence.
Do not install AITER or replace Triton during discovery. Use a separately configured runtime
when dependencies conflict. A missing dependency is unavailable, not a performance loss.

For packed attention, inspect `aiter.flash_attn_varlen_func` and the CK/assembly dispatch it calls.
Pass packed THD FP16 inputs, separate int32 cumulative Q/K lengths, actual maxima, zero dropout,
scale D**-0.5, and the target's causal flag. check return type and optional auxiliary outputs.
Some versions choose CK for FP16 and assembly for BF16. PR #3921 reports that distinction.
Its BF16 assembly numbers are not valid controls for `varlen_f16_fa.json`.
Do not convert FP16 to BF16 to access another path. Check GQA, rectangular causal alignment,
empty sequences, strides, head-dimension limits, and whether the chosen path returns zero for
fully masked rows. If an assembly path rejects the contract, retain the rejection and try a
source-supported same-dtype alternative. Record the actual dispatch path when discoverable.

For dense attention, inspect `flash_attn_func` and the appropriate Triton/CK/assembly entry points.
The common input is BHSD. Some AITER APIs use BSHD. A stride-only view is allowed if supported. Required copies and canonical BHSD output conversion remain inside run. Do not accidentally
measure backward, dropout state generation, or a different causal convention.
For GEMM, inspect `aiter.ops.triton.gemm.basic.gemm_a16w16.gemm_a16w16` and its actual B contract.
Check tuned_gemm.TunedGemm.mm and the installed tuning database only when its layout matches.
Source: https://github.com/ROCm/aiter/blob/main/aiter/tuned_gemm.py .
TunedGemm can dispatch to torch, hipblaslt, skinny, assembly, Triton, or other paths. A torch
fallback must be labeled as a fallback. Do not present it as a distinct faster AITER kernel.
Include transposition/packing and check that bias/scales are disabled for the plain GEMM target.

Search the public dispatch first, then supported same-dtype direct variants and tuning records.
Check architecture filtering in CSV/databases. Records for gfx942 are not gfx950 evidence.
Do not use an unverified launch signature. Agent proposals should explain the selected source,
algorithm, dtype constraints, tuning data, and fallback status. Return no proposals if installation
or compatibility prevents a justified candidate. The harness owns all correctness and timing.
