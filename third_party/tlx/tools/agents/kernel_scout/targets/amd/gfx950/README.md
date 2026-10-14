# gfx950 targets

All targets use the shared input generator, FP32 oracle, eager timer, and returned-output contract.
Select complete files or use `--case` to select named cases.
Changed target contracts require a fresh weekly baseline before nightly reuse.

| Target | Cases | Source and operation |
| --- | --- | --- |
| `varlen_f16_fa.json` | 6 | PR #3921 packed FP16 attention |
| `bf16_fa.json` | 2 | Dense BF16 attention smoke cases |
| `mxfp8_fa.json` | 8 | PR #3824 public MXFP8 forward attention |
| `a16w16_gemm.json` | 22 | Two smoke cases and 10 issue-sourced MM shapes in BF16 and FP16 |
| `a16w16_addmm.json` | 20 | Ten issue-sourced bias ADDMM shapes in BF16 and FP16 |
| `a16w16_bmm.json` | 2 | One issue-sourced shared-A BMM shape in BF16 and FP16 |

## MXFP8 attention

[PR #3824](https://github.com/facebookexperimental/triton/pull/3824) supplies the `gfx950_1` shapes.
They use batch 4, 32 query/KV heads, head dimension 128, and lengths 1024, 2048, 4096, and 8192.
Each length has causal and non-causal forward cases.
The source benchmark specifies softmax scale 0.5, normal input standard deviation 0.5, and absolute tolerance 0.2.
Relative tolerance is zero.
Scout also requires relative RMS output error at most 0.1 against the FP32 oracle.
This additional check rejects zero outputs that a loose absolute tolerance can otherwise accept.

The operation receives contiguous BF16 BHSD Q/K/V and returns BF16 BHSD output.
The public TLX op quantizes Q/K/V inside every call.
The non-causal gfx950 implementation also approximates exp2 within its declared numerical budget.
The FP32 oracle uses the original BF16 inputs and the explicit scale.
All backends receive identical CPU-seeded input bits.
The seeds differ from the PR's GPU-generated inputs.

AITER and PyTorch provide labeled BF16 attention references.
They do not claim the same MXFP8 quantization recipe as TLX.
The tutorial backend declares this operation unsupported because no independent gfx950 MXFP8 tutorial exists.
The PR's pre-quantized AITER code-object comparison uses a different recipe and HIP graph timing.
Scout does not import those timings or bypass quantization.
Useful causal FLOPs count the exact visible triangle rather than half of a square.

## GEMM issue corpus

The source issues are [#20](https://github.com/raikonenfnu/tlx_triton/issues/20),
[#21](https://github.com/raikonenfnu/tlx_triton/issues/21), and [#22](https://github.com/raikonenfnu/tlx_triton/issues/22).
The JSON records the inspected suite revision and a source issue for each imported case.
The original measurements use BF16.
FP16 counterparts preserve dimensions, layouts, and input scaling, with distinct case IDs.
Historical performance numbers and fork-specific tuning results are not current baseline measurements.

MM preserves both A and B layouts, including column-major A.
ADDMM computes `bias + A @ B`, with a length-N vector bias and unit alpha/beta.
Its A is row-major and B is column-major.
The shared-A BMM case uses batch 3072, M=448, N=160, and K=931.
One row-major A matrix serves every batch through a zero batch stride.
B is row-major with distinct data per batch.

Imported MM/ADDMM inputs use the source generator's inverse-square-root column scaling.
The GEMM targets also require relative RMS output error at most 0.02.
Shared-A BMM inputs use standard normal data.
The harness generates the values independently on CPU rather than reproducing old GPU RNG bits.
The conventional GEMM metric counts `2 * batch * M * N * K` FLOPs.
Logical bytes include bias and count shared A storage once.
Source weights remain metadata. Scout does not compute weighted aggregate speedups.

TLX ops uses the public ADDMM op and the existing TorchTLX BMM provider in force mode.
The latter disables CUDA graphs and cannot silently select a stock Inductor kernel.
Torch builds without the `triton.tlx_mode` integration report that provider unsupported.
Tutorials use the existing gfx950 ADDMM and shared-A BMM wrappers.
AITER ADDMM uses its public tuned GEMM with bias and the requested output dtype.
AITER BMM uses its public BF16 CK wrapper, with required B packing inside timing.
Its FP16 input case remains unsupported rather than converting to BF16.
