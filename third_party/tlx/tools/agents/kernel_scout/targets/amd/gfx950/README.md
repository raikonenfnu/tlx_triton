# gfx950 targets

These six files define 60 cases. Each JSON file contains the complete workload contract.
No external issue, pull request, or historical timing is needed to run a case.
All cases use shared CPU-seeded inputs, an independent FP32 oracle, and eager `triton.testing.do_bench` timing.
Each measured call returns a new output tensor and includes required allocation, packing, and conversion.

| Target | Cases | Operation |
| --- | --- | --- |
| [varlen_f16_fa.json](varlen_f16_fa.json) | 6 | Packed FP16 forward attention |
| [bf16_fa.json](bf16_fa.json) | 2 | Dense BF16 forward attention |
| [mxfp8_fa.json](mxfp8_fa.json) | 8 | MXFP8 forward attention with BF16 inputs and output |
| [a16w16_gemm.json](a16w16_gemm.json) | 22 | Matrix multiplication, including two small smoke cases |
| [a16w16_addmm.json](a16w16_addmm.json) | 20 | Matrix multiplication with vector bias |
| [a16w16_bmm.json](a16w16_bmm.json) | 2 | Batched matrix multiplication with shared A |

## Run the targets

Run these commands from `third_party/tlx/tools/agents/kernel_scout` with a source-built TLX environment.
Set `SCOUT_PYTHON` to that environment's Python executable.
Select an idle physical GPU with `--gpu`.

```bash
"$SCOUT_PYTHON" appctl.py validate targets/amd/gfx950/*.json
"$SCOUT_PYTHON" appctl.py run --mode weekly --targets targets/amd/gfx950/*.json \
  --state /path/to/scout-state --gpu 0 --execution persistent --screen-candidates --execute
"$SCOUT_PYTHON" appctl.py run --mode nightly --targets targets/amd/gfx950/*.json \
  --state /path/to/scout-state --gpu 0 --execution persistent --screen-candidates --execute
```

Use the same target selection, state directory, runtime overrides, and execution policy for both runs.
Use `--case CASE_ID` to select named cases before creating the weekly baseline.
Changed target contracts require a fresh weekly baseline.
The [application README](../../../README.md) describes backend setup, discovery, reports, and publication.

## FP16 and BF16 attention

Packed FP16 inputs use token-head-dimension layout and separate int32 cumulative Q/K lengths.
Cases cover batches of 1, 10, and 13 sequences, with uniform or ragged lengths and a causal control.
Q and KV lengths match within each sequence. Ragged cases use fixed lengths between 700 and 1200.
The JSON stores every realized sequence length. No random shape sampling occurs during a run.
Causal masking uses bottom-right alignment. Fully masked rows return zero.
Softmax scale defaults to the inverse square root of head dimension.

Dense BF16 cases use batch 1, four query/KV heads, head dimension 128, and sequence length 1024.
They cover causal and non-causal attention with batch-head-sequence-dimension (BHSD) layout.
Both targets use absolute and relative tolerances of 0.02.

## MXFP8 attention

Cases use batch 4, 32 query/KV heads, head dimension 128, and lengths 1024, 2048, 4096, and 8192.
Each length has causal and non-causal forward cases.
Softmax scale and normal input standard deviation are both 0.5.
Absolute tolerance is 0.2, relative tolerance is zero, and relative RMS error must not exceed 0.1.
The RMS check rejects trivial outputs that the absolute tolerance alone could accept.

Inputs and output are contiguous BF16 BHSD tensors.
The public TLX operation quantizes Q/K/V inside every measured call.
The FP32 oracle uses the original BF16 inputs and the explicit scale.
AITER and PyTorch supply labeled BF16 references. They do not implement the same quantization recipe.
Pre-quantized inputs and graph replay are outside this contract.
The tutorial backend reports unsupported when it has no independent implementation.
Useful causal FLOPs count the exact visible triangle.

## Matrix multiplication

Ten MM shapes and ten ADDMM shapes each have BF16 and FP16 cases.
The JSON files specify every M, N, K, layout, and input scale.
The MM file also includes two square BF16 smoke cases.
MM preserves both operand layouts, including column-major A where specified.
ADDMM computes `bias + A @ B` with a length-N bias and unit alpha/beta.
Its A is row-major and B is column-major.

The BMM cases use batch 3072, M=448, N=160, and K=931 in BF16 and FP16.
One row-major A matrix serves every batch through a zero batch stride.
B is row-major with distinct data per batch.

The larger MM/ADDMM cases scale normal inputs by `1/sqrt(K)` for A and `1/sqrt(N)` for B.
BMM uses standard normal inputs. JSON values define bias scaling and smoke-case distributions.
All GEMM targets require relative RMS error at most 0.02, in addition to their elementwise tolerances.
Compute metrics count `2 * batch * M * N * K` FLOPs.
Logical bytes include bias and count shared A storage once.
Source weights are descriptive metadata. Scout does not calculate weighted aggregate speedups.

Backend support depends on the checked source and installed runtime.
TorchTLX BMM requires `triton.tlx_mode` integration and force mode, with CUDA graphs disabled.
Missing integration produces an unsupported result.
AITER BMM uses its public BF16 CK wrapper and includes required B packing in timing.
Its FP16 case remains unsupported. The adapter must not convert FP16 input to BF16.
