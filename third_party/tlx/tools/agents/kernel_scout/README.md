# TLX Kernel Scout

Scout finds the best observed existing kernel variants for a target and compares TLX ops,
TLX tutorials, AITER, and PyTorch. An `appctl.py` controller owns measurements, frozen weekly
controls, reports, and GitHub publication. Python Codex/Claude adapters own bounded source
research and variant proposals. This keeps measurement reproducible and limits agent cost.

The first target is `targets/amd/gfx950/varlen_f16_fa.json`, from
[PR #3921](https://github.com/facebookexperimental/triton/pull/3921). It freezes the five
`gfx950_1` focus shapes and their realized lengths, plus a causal control. The original op is
already present in the initial main checkout (`302d051fe`). Scout measures the current implementation.
The PR's graph timings and BF16 assembly results are not imported as FP16/eager controls.
Additional targets cover a16w16 GEMM and BF16 dense attention.

## Common benchmark contract

Every backend receives the same seeded data, dtype, shapes, strides, mask semantics, and output
layout. `benchmarking/inputs.py` owns generation and an independent FP32 oracle. At least two
seeds check outputs, input mutation, finiteness, and output aliasing. Each timed call returns a
new canonical output tensor. Required padding, packing, copies, and extraction are included.
This matters for packed attention: a pre-padded Torch call and a packed TLX call have different
input contracts. Torch seeds compare per-sequence SDPA and padded batched SDPA with all conversion
work included. Read [the fairness skill](skills/benchmark-fairness/SKILL.md) for the complete contract.

Timing uses the same eager `triton.testing.do_bench` warmup/rep milliseconds, cache-clearing
behavior, quantiles, and independent repeats. Candidates must be correct and stable. A selected
variant is checked in a new process. Winners use latency and uncertainty. Throughput reports
use useful TFLOP/s or logical TB/s. Neither metric is a hardware counter. Device admission checks,
clock/power/temperature snapshots, source commits, package/runtime versions, compiler hashes,
and candidate digests are saved in JSON. Clocks are not changed automatically.

This is a bounded search, not a proof that every possible variant was evaluated. Agents select
existing implementations and supported launch knobs. Kernel rewriting belongs to the existing
kernel optimizer. Generated adapters are trusted executable Python. Inspect them when needed.
Correctness tests do not establish that arbitrary code is honest or safe.

## Run locally

Use the source-built TLX environment. Put source Triton before any wheel Triton on PYTHONPATH:

```bash
cd third_party/tlx/tools/agents/kernel_scout
export PYTHONPATH=/path/to/triton/python:/path/to/torch/site-packages
/path/to/venv/tlx.venv/bin/python appctl.py validate targets/amd/gfx950/*.json
/path/to/venv/tlx.venv/bin/python appctl.py run --mode weekly \
  --targets targets/amd/gfx950/varlen_f16_fa.json --state /path/to/scout-state
```

The last command previews. Add `--execute` for measurements. The default `--provider builtin`
uses known seed recipes without an LLM. Set `--provider codex` or `claude` for agentic discovery:

```bash
/path/to/python appctl.py run --mode weekly --execute --provider codex \
  --context /path/to/megabrain/context.md --rounds 2 --max-candidates 6 \
  --targets targets/amd/gfx950/varlen_f16_fa.json --state /path/to/scout-state
/path/to/python appctl.py run --mode nightly --execute --provider codex \
  --context /path/to/megabrain/context.md \
  --targets targets/amd/gfx950/varlen_f16_fa.json --state /path/to/scout-state
```

`--case one_x_1000` selects a small PR smoke case. Use the same filter for weekly and nightly.
`--gpu` is the physical GPU index. The worker exposes that device as logical GPU 0 before import.
`--model`, `--agent-timeout`, and `--worker-timeout` bound each worker. `--max-candidates` includes
seeds, and `--rounds` bounds discovery iterations. Authentication/provider failures stop the run.
Unsupported or failed kernel candidates remain recorded. Missing AITER is unavailable.
There is no known independent packed-varlen tutorial seed at the initial source revision. Agent discovery can investigate it, and must report unsupported when none exists.

Use `--runtimes /path/to/runtimes.json` for per-backend environments:

```json
{"aiter": {"python": "/path/to/aiter/bin/python", "pythonpath": "/path/to/aiter-triton/python:/path/to/torch/site-packages"}}
```

Keep device and operation contracts consistent across runtimes. Scout does not install dependencies.
Source revision and native compiler pins show implementation differences. Vendor restrictions are
in `backends/*/backend.json`. Add backends with [the registration skill](skills/register-backend/SKILL.md).
The detailed backend instructions are injected into every discovery prompt.

## Artifacts and publication

Each run writes report.md/report.json, per-candidate logs, search manifests, proposals, and
reproducer requests under `STATE/runs/`. The Markdown has a performance table, winner column,
reproducer-command table, and provenance/coverage table. Commands reject changed source,
compiler, runtime, or physical-device pins. Keep old environments/checkouts for weekly reproduction.

Weekly freezes PyTorch, AITER, and the selected TLX ops baseline in an immutable baseline file.
Nightly measures current TLX ops/tutorials and labels reused controls with their original dates.
Target, timing, case selection, device/runtime, and registry mismatches reject reuse. These are
trend comparisons. For a simultaneous comparison rerun all backends in a fresh state/campaign.
Local artifacts are authoritative evidence. GitHub receives Markdown summaries only.

```bash
/path/to/python appctl.py publish --report /path/to/run/report.json --repo raikonenfnu/tlx_triton
/path/to/python appctl.py publish --report /path/to/run/report.json --repo raikonenfnu/tlx_triton --execute
```

Publication previews by default. Weekly creates or updates one owned issue, checks it, then
links and closes earlier weeks of that campaign. Nightly creates or updates one dated comment
on the owned open weekly issue. The repository is explicit and configurable. Publication tests
use a fake API. Running tests never creates or closes real issues.
See [deployment instructions](deploy/README.md) for the nightly runner and cron schedule.

## Tests

```bash
/path/to/venv/tlx.venv/bin/python -m pytest tests -s --tb=short
SCOUT_TEST_GPU=1 /path/to/venv/tlx.venv/bin/python -m pytest tests -s --tb=short
```

CPU tests cover contracts, arithmetic, selection, baseline checks, fake Codex/Claude protocols,
and weekly/nightly publication transitions. The opt-in GPU test runs source-built TLX through
the actual common harness and rejects a deliberately incorrect mutating adapter.
Live provider discovery is separate from fake CLI protocol validation.
