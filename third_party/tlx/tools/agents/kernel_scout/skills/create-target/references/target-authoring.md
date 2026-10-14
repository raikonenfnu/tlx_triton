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

Supported operations are `mm`, `flash_attn`, and `flash_attn_varlen`.
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
| `correctness` | At least two distinct seeds and positive atol/rtol. Choose tolerances from numerical evidence. |
| `skills` | Keep `benchmark-fairness` with the target contract. |
| `cases` | Freeze every case's dtype, dimensions, strides, and operation semantics. |
| `source` | Optional evidence, source revision, shape suite, and notes. Avoid unbounded raw logs. |

Operation-specific fields:

| Operation | Case fields and semantics |
| --- | --- |
| `mm` | `m`, `n`, `k`, `dtype`, `b_layout`. A is row-major. B is row-major or column-major. No bias. |
| `flash_attn` | `q_lengths`, `kv_lengths`, `heads`, `kv_heads`, `head_dim`, `causal`, `dtype`. Dense BHSD. All lengths must match. |
| `flash_attn_varlen` | The same attention fields. Packed THD and int32 cumulative lengths. Q/KV lengths can differ. |

Attention uses scale D**-0.5 and zero dropout.
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
python appctl.py run --mode weekly \
  --targets targets/amd/gfx950/my_varlen_f16_fa.json \
  --state /path/to/new-target-state --case ragged_small --gpu 2 \
  --provider builtin --rounds 0 --execute
```

Put source Triton before any wheel Triton on PYTHONPATH.
Select an idle physical GPU index.
The controller checks physical PCI selection, ownership, actual input fingerprints, correctness, and timing stability.
Inspect `report.json`, candidate logs, and search manifests.
Run the report's pinned reproducer command.
An unsupported backend or missing dependency cannot supply a performance number.

Enable discovery after the seed path works:

```bash
python appctl.py run --mode weekly \
  --targets targets/amd/gfx950/my_varlen_f16_fa.json \
  --state /path/to/discovery-state --gpu 2 --provider claude \
  --context /path/to/megabrain/context.md --max-candidates 6 --rounds 2 --execute
```

Use a fresh state directory for this second weekly run.
The controller preserves an existing weekly baseline.
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
