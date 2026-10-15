# Create a target and benchmarker

Run commands from `third_party/tlx/tools/agents/kernel_scout`.
Use the source-built TLX Python environment for GPU validation.
The template and config checks need only Python's standard library.

## Add cases for an existing operation

Create a starter file:

```bash
python appctl.py target-init --id my_varlen_f16_fa \
  --operation flash_attn_varlen --vendor amd --arch gfx950 \
  --metric compute --output targets/amd/gfx950/my_varlen_f16_fa.json
python appctl.py validate targets/amd/gfx950/my_varlen_f16_fa.json
```

Supported operations are `mm`, `addmm`, `bmm`, `flash_attn`, `flash_attn_varlen`, and `flash_attn_mxfp8`.
The command refuses an existing output file.
Starter dimensions do not establish hardware or backend support.
Edit the cases to match the actual workload before interpreting performance.

| Field | Contract |
| --- | --- |
| `id` / case `id` | Distinct lowercase identifiers. Use letters, digits, and underscores. |
| `hardware` | Explicit vendor and architecture list. Examples are amd/gfx950 and nvidia/90. |
| `output_policy` | `returned_tensor`. Include required allocation, packing, conversion, and extraction in each call. |
| `metric` | `compute` reports useful TFLOP/s. `memory` reports logical TB/s. Neither value proves the hardware bottleneck. |
| `timing` | Warmup and repetition are milliseconds. Use at least three independent repeats. Define spread and tie thresholds. |
| `correctness` | At least two distinct seeds and nonnegative atol/rtol, with at least one positive tolerance. |
| `skills` | Keep `benchmark-fairness` with the target contract. |
| `cases` | Freeze every case's dtype, dimensions, strides, and operation semantics. |
| `source` | Optional evidence, source revision, shape suite, and notes. Avoid unbounded raw logs. |

Operation-specific fields:

| Operation | Case fields and semantics |
| --- | --- |
| `mm` | `m`, `n`, `k`, `dtype`, `a_layout`, `b_layout`. A defaults to row-major. Both matrices allow column-major layout. |
| `addmm` | MM fields plus `bias_layout="vector"`. Length-N bias, with alpha=beta=1. |
| `bmm` | MM fields plus `batch` and `shared_a`. Shared A has a zero batch stride. |
| `flash_attn` | `q_lengths`, `kv_lengths`, `heads`, `kv_heads`, `head_dim`, `causal`, `dtype`. Dense BHSD. All lengths must match. |
| `flash_attn_varlen` | The same attention fields. Packed THD and int32 cumulative lengths. Q/KV lengths can differ. |
| `flash_attn_mxfp8` | Dense BF16 BHSD, D=128, equal query/KV heads, and lengths divisible by 256. Quantization stays inside the op. |

Attention uses an explicit `sm_scale` when present, otherwise D**-0.5. Dropout is zero.
Optional `a_std`, `b_std`, `bias_std`, and `qkv_std` specify normal input standard deviations.
Their default is one. Input generation remains shared across backends.
Optional `correctness.max_relative_rms_error` bounds RMS output error relative to the FP32 reference's RMS.
Use it when small reference magnitudes make an absolute tolerance insufficient.
The query head count must be divisible by the KV head count.
Each KV head serves one consecutive group of query heads.
Causal packed attention uses bottom-right alignment.
A fully masked query row returns zero.
The harness owns offset creation and the canonical output layout.
Use separate cases for FP16 and BF16.
Add edge cases only when the selected implementations support their contract.
Examples include partial tiles, unequal Q/KV lengths, grouped heads, and empty sequences.

Check one small case first:

```bash
python appctl.py run --mode registration \
  --targets targets/amd/gfx950/my_varlen_f16_fa.json \
  --state /path/to/new-target-state --case ragged_small --gpu 2 \
  --provider builtin --rounds 0 --execution persistent --execute
```

Put source Triton before any wheel Triton on PYTHONPATH.
Select an idle physical GPU index.
The controller checks physical PCI selection, ownership, actual input fingerprints, correctness, and timing stability.
Inspect `registration.json`, `report.json`, candidate logs, and search manifests.
Registration saves checked coverage and returns a nonzero exit code when an applicable backend fails or is unavailable or unstable.
Each ready case needs a stable TLX measurement and a stable AITER or PyTorch measurement.
Unsupported backends remain visible, with their reasons.
The receipt covers only the selected cases and recorded source/runtime pins.
Remove `--case` to check the complete target after the initial smoke check.
Registration does not create an immutable weekly baseline or publish a GitHub report.
Run the report's pinned reproducer command.
An unsupported backend or missing dependency cannot supply a performance number.

Enable discovery after the seed path works:

```bash
python appctl.py run --mode registration \
  --targets targets/amd/gfx950/my_varlen_f16_fa.json \
  --state /path/to/discovery-state --gpu 2 --provider claude \
  --context /path/to/megabrain/context.md --max-candidates 6 --rounds 2 \
  --execution persistent --candidate-cache /path/to/candidates --execute
```

Resolve registration failures before the first weekly run.
Use `--mode weekly --execution persistent --screen-candidates --discovery off` for routine measurement.
Pass the same candidate-cache and context paths to reuse inspected adapters when all pins match.
Cache misses use seeds only. Check reported candidate counts before claiming search coverage.
Use a fresh state directory when replacing an existing weekly baseline.
Pass the same targets and case filter to nightly runs.
A new target contract selects a new campaign and needs a matching weekly baseline.

## Add an operation benchmarker

Implement the operation's shared contract before backend variants.
Use this extension map:

| File | Required change |
| --- | --- |
| `contracts.py` | Add the operation name, required case fields, shape checks, useful FLOPs, and logical byte counts. |
| `benchmarking/inputs.py` | Add canonical seeded inputs and an independent FP32 oracle. Preserve dtype/layout in input fingerprints. |
| `benchmarking/worker.py` | Reuse correctness and timing. Extend the output contract only if the operation requires multiple outputs. |
| `reporting.py` | Add a readable problem description. Keep latency, metric units, uncertainty, and source evidence visible. |
| `backends/*/backend.json` | Declare support only where an implementation exists for that vendor and architecture. |
| `backends/*/instructions.md` | Document checked entry points, numerical options, tuning choices, and unsupported cases. |
| `backends/seeds.py` | Add seed candidates and a callable over harness-owned inputs. Include required conversions inside the call. |
| `target_templates.py` / `appctl.py` | Add a starter case and expose the operation in `target-init` after the contract exists. |
| `tests/test_core.py` | Test schema failures, arithmetic, backend eligibility, and controller integration. Add meaningful GPU checks when supported. |

Implement seeded generation once in `make_inputs` and the FP32 oracle once in `reference`.
Test repeat generation with the same seed and changed values with a different seed.
Test declared layouts, including zero batch strides and packed offsets when applicable.
Check the oracle against a small independently calculated example.
Use `MeasurementSession.prepare` for per-case reuse across candidates.
Use `cached_host_inputs` for identical CPU values across separate backend runtimes.
Do not save oracle outputs, GPU tensors, or timing results in the adapter research cache.

After registration, inspect `search.phase_seconds` in the report.
Separate input preparation from first-call compilation and autotuning in correctness checks.
A slow first call does not imply a slow kernel.

Do not derive the oracle from the candidate kernel.
Do not place benchmark timing inside backend adapters.
Do not count padded or masked arithmetic as useful work without an explicit target convention.
Keep inference and backward workloads in separate contracts.
For memory operations, document which logical reads and writes the byte count includes.
A logical byte count does not measure HBM transactions.

Run the application tests:

```bash
python -m pytest tests -s --tb=short
```

Run affected operation cases through the worker on supported hardware.
Keep an unavailable GPU or dependency explicit in the validation report.
Recheck existing targets when a shared validator, oracle, timing contract, or metric rule changes.
