# PyTorch reference implementations

Find the fastest observed supported PyTorch implementation under the same inference contract.
Read the installed Torch version and GPU library availability. Use public Torch operations.
Do not import TLX/AITER or compiled custom kernels under this backend. The oracle is independent
FP32 math. It is not the performance baseline. TF32 is disabled by the shared harness.

For GEMM, begin with torch.mm on the provided A/B strides and FP16/BF16 dtype. Preserve dtype,
no bias, and returned output allocation. Compare supported public alternatives only when they
are materially different. Do not precompute B conversion outside run, silently enable reduced
precision beyond the target, or reuse an output tensor when the contract requires allocation.
Record the Torch source SHA and ROCm/CUDA runtime. The library may internally choose rocBLAS or
hipBLASLt. Avoid attributing a specific low-level kernel without evidence.

For dense attention, use torch.nn.functional.scaled_dot_product_attention with zero dropout,
scale D**-0.5, and canonical BHSD output. Compare automatic selection with source-supported
`torch.nn.attention.sdpa_kernel` choices for this installation and architecture. The seeds use
auto and math. Test flash/efficient choices only when enabled in the installed runtime. An
unsupported warning/error is evidence, not permission to fall back silently. For unequal Q/K
lengths, do not assume `is_causal=True` has bottom-right alignment. Build the explicit mask.
GQA must preserve the exact query-to-KV grouping. Required repetition remains inside run.

For packed attention, there may be no supported native public varlen API. The seed slices each
sequence, calls SDPA, and concatenates the packed THD output, all inside the timed callable.
A useful competing variant pads to the actual batch maximum, builds a padding/causal mask,
executes batched SDPA, and extracts the packed result. Include padding, mask construction,
and extraction in every timed call. Compare these strategies. Small batches and length variance
can change the winner. Do not measure pre-padded inputs against a TLX packed-input call.
Pre-padding outside timing changes the input contract. Graph replay changes the timing contract.
Neither measurement can serve as an eager packed-input control.

Consider public torch.compile only as an explicitly labeled variant with its compiler/runtime
pins and full semantics preserved. Compile and check it before timing, include graph breaks and
packing that remain in execution, and record any backend dependency. Do not use graph replay
under the eager target. Stop when remaining candidate budget has no justified alternative.
The shared harness checks correctness on changed seeds, controls timing, and reports the
best observed valid variant. Unsupported API paths remain visible in search logs.

## Imported bias/batched GEMM and MXFP8 reference targets

ADDMM uses `torch.addmm(bias,a,b)` with a length-N bias and unit alpha/beta.
BMM uses `torch.bmm(a,b)` with the supplied batch strides.
Keep shared A as a zero-stride view rather than distinct batch matrices.
Preserve both MM operand layouts, including column-major A.

For `flash_attn_mxfp8`, PyTorch supplies a labeled BF16 attention reference, not an MXFP8 kernel.
Use the explicit case scale and canonical BHSD tensors. Equal-length dense causal attention permits native `is_causal=True`.
The source benchmark uses scale 0.5 and input standard deviation 0.5.
The controller also checks relative RMS error against its independent FP32 oracle.
Research begins in the installed Torch package supplied by the runtime identity.
Source-read failures remain blocked discovery. Measured seeds can still supply valid observations.
