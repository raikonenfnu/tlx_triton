# TLX public operations

Find the fastest observed supported public TLX operation for the exact target case.
Read `third_party/tlx/ops/__init__.py`, `_catalog.py`, architecture dispatch, and the selected
kernel module before proposing a variant. The public API dispatches one implementation per
operation and architecture. It has no generic `variant=` argument. Internal kernels are a
research surface, not a public launch contract. Label a private adapter as experimental and
record the exact source and launch configuration. Never substitute a tutorial under this backend.

Use the public API seed first. For GEMM, inspect `ops/kernels/mm/gfx950.py` and its heuristic
selection. gfx950 currently supports the heuristic space. Do not request `space='full'` from
an unrelated NVIDIA path. Preserve A/B layout and output dtype. If using a private launch,
copy all launch metadata from a valid in-tree caller and account for allocation and required
packing. Check split-K/stream-K reduction costs, divisibility, tail masking, waves, and scratch
initialization. A small kernel with unsupported tails is not a candidate for a different shape.

For dense attention, use `tlx.ops.flash_attn` and its full supported tuning space.
The smoke space establishes functionality, not exhaustive performance. check head dimensions,
causal support, grouped heads, and supported strides in the dispatch spec. Count autotuning
against the worker deadline and exclude its one-time cost from steady-state samples.
Record the selected autotune configuration if it can be inspected through the supported object.

For packed attention, use `tlx.ops.flash_attn_varlen(q,k,v,cu_q,cu_k,max_q,max_k,causal=...)`.
PR #3921 supplies the original gfx950 operation, its focus shapes, and tests:
https://github.com/facebookexperimental/triton/pull/3921
Read the CURRENT implementation: main includes later changes, including split scheduling and
scratch reuse. Do not treat PR-head timing as current performance. Check D=128, MHA/GQA,
int32 offsets, bottom-right causal alignment, and zero-output rows. Use the actual maximum lengths. Oversized maxima change launch work. Do not synchronize GPU offsets to the host in the adapter.

Search a justified small set: public dispatch first, then source-supported launch/scheduling
alternatives. Use measured feedback to focus the remaining budget on the slow cases. If only
one fixed configuration exists, say so. Do not fabricate knobs or claim that a single observation
proves global optimality. Public seed failures must remain visible even if a private adapter wins.
The shared harness decides correctness and latency. Return an empty proposal list when there is
no credible existing variant. Record aliases and identical kernel bodies to avoid overstating coverage.

## MXFP8, bias GEMM, and batched GEMM targets

For `flash_attn_mxfp8`, use the public `tlx.ops.flash_attn_mxfp8` on BF16 BHSD tensors.
Pass the case's explicit softmax scale and causal flag. Quantization remains inside each timed call.
Do not substitute BF16 flash attention under the TLX MXFP8 column.
The gfx950 target requires D=128, equal heads, and sequence lengths divisible by 256.
Its non-causal exp2 approximation remains subject to the target's absolute and relative RMS error limits.

For ADDMM, call `tlx.ops.addmm(bias, a, b)` with the supplied length-N bias and unit alpha/beta.
Preserve both matrix layouts. Include packing and any additional allocation inside `run`.
For shared-A BMM, the existing catalog provider resides in `ops/kernels/bmm/gfx950_torch.py`.
Use `mode="force"` and disable `triton.cudagraphs`. A stock Inductor fallback cannot count as a TLX result.
Record this provider as TorchTLX. Missing TorchTLX compiler integration is a dependency limitation.
The A batch stride is zero. Do not replace it with distinct batch data or pre-materialize it outside timing.
