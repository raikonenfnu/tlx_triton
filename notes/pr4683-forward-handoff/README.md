# MXFP8 varlen forward optimization handoff

This is the continuation record for optimizing gfx950 packed-varlen MXFP8 attention forward. It records the implementation at **`c338ba12e04c256a591cd8cb451b74aaf90c9a78`**, measurements through **2026-11-13 UTC**, decisions, failed experiments, and work that remains. Continue from the validated implementation; do not infer that a reported empirical read reference is a hardware roofline or that the inherited accuracy failures were fixed.

## Start here

- Original production-shape reproducer: [facebookexperimental/triton#4683](https://github.com/facebookexperimental/triton/pull/4683), Lei Wang, head `d3774785447d0385c0027dcef799be5188d462ed`.
- Optimization draft: [facebookexperimental/triton#5060](https://github.com/facebookexperimental/triton/pull/5060), branch `opt/pr4683-prequantized-forward-draft` in this fork. Open/draft when this record was made.
- [Optimization-only diff](https://github.com/raikonenfnu/tlx_triton/compare/32f95cd2b...c338ba12e04c256a591cd8cb451b74aaf90c9a78). `32f95cd2b` carries the original prerequisite on upstream main `03a938c74`; `c338ba12e` is the optimization commit. The original attribution is preserved.
- **Primary attribution result:** [all 20 shapes with prepared Q/K/V on both sides and identical exponent modes](https://github.com/facebookexperimental/triton/pull/5060#issuecomment-6861366247).
- [Original-core versus final-core table and historical ablations](https://github.com/facebookexperimental/triton/pull/5060#issuecomment-6861183496). Read the baseline qualifications before comparing its 1.89x with the controlled 1.65x.
- [Lei's latest scope request](https://github.com/facebookexperimental/triton/pull/4683#issuecomment-6691037268): prioritize prequantized MXFP8 Q/K/V plus scales; report the BF16-input full attention call alongside it. His preceding comment asks collaborators to optimize the core; upstream projection/preparation fusion needs model-specific sequence metadata.
- **Preserved source snapshots, portable comparison drivers, raw JSON, reports and checksums:** [this directory](.). This is a separate documentation/evidence branch based on the PR head; it does not alter the optimization PR. Read its README before using historical reports.

## What problem this is, and what the baseline actually computes

Packed noncausal cross-attention, dropout 0, window (-1,-1), batch 432, 4 heads, head dimension 128, softmax scale `1/sqrt(128)`. There are ten short-KV and ten long-KV cases. Total Q is about 24.7k–32.8k; total KV about 259k–265k short or 3.12m–3.18m long. Detailed total/max lengths and all per-shape numbers are in the linked tables and preserved JSON.

**The traces did not retain exact offsets.** Benchmarks use deterministic long-tail sequence-length proxies matching the original batch size, totals and maxima (`lengths` seeds 3/5, data seeds 7/11/23). They do not reproduce the exact original model distribution.

| Path | Public input | Attention arithmetic / timed work |
|---|---|---|
| Original #4683 BF16 full call | BF16 Q/K/V | Quantize/rearrange K/V, quantize Q inside attention, scaled-MFMA MXFP8 core |
| Original core-only | BF16 Q, already prepared K/V | Fused Q quantization plus MXFP8 attention |
| Prepared baseline used for clean attribution | Prepared Q/K/V and E8M0 scales | Initial prepared-Q implementation, original scheduling/waits/metadata/fill |
| Optimized prepared path | Same prepared buffers/scales | Balanced persistent schedule, wait overlap, metadata/fill changes |
| **AITER baseline in this work** | **BF16 Q/K/V** | **Direct BF16 ASM-v3**, `fmha_fwd_hd128_bf16_group`; no MXFP8 conversion |

AITER has other MXFP8 implementations, but **we have not done an MXFP8-versus-MXFP8 AITER comparison here**. Call the comparison what it is. Use AITER's native `module_fmha_v3_varlen_fwd.fmha_v3_varlen_fwd`, not a higher-overhead public Python wrapper.

The implementation descends from [#3824 / D122867768](https://github.com/facebookexperimental/triton/pull/3824), the dense `tlx.ops.flash_attn_mxfp8` gfx950 ping-pong core. At the inspected upstream main, TLX ops has dense MXFP8 attention and separate packed-varlen FP16/BF16 attention, not a combined public gfx950 varlen-MXFP8 op. #4683 and #5060 live under `third_party/tlx/tutorials/amd_mxfp8_varlen_attention/`; public TLX ops integration remains separate work.

## What was implemented, and why

1. **Prepared-Q producer and input API.** `prepare_q` creates E4M3 `[total_q,4,128]` and uint8 E8M0 scales `[total_q,4,4]`. Its quantizer matches the fused 32-query-row × 32-feature recipe, restarting groups at each sequence and repeating the scale per row. The consumer also supports independent per-row scales. `forward_prequantized` accepts Q8/Q scales and the existing prepared K/V ABI, without needing BF16 source tensors. Preparation is outside core timing; this is not free conversion.
2. **Cost-balanced static persistent schedule.** The original is already persistent. Round-robin item assignment roughly equalizes `(query tile, head)` counts, but not the number of KV tiles processed. `prepare_work_schedule` uses known CPU lengths, estimates cost as `ceil(k_len/256)`, and assigns expensive items across bounded bins while preserving original item counts per CTA. The tested global assignment beat the XCD-local alternative. There is **no atomic work queue**. Reuse the frozen `ForwardWorkSchedule` with matching offsets; CPU construction/upload is excluded. Historical construction was roughly 0.6–0.9 ms plus upload, so rebuilding per call can erase the benefit.
3. **Async-copy/softmax overlap.** Move `async_load_wait_group(0)` after register-only softmax and before the following LDS consumer. Retain all required waits, commit-group semantics, physical buffer alternation, stage and iteration boundaries. This overlaps existing work; it is not permission to remove synchronization.
4. **Short scheduled metadata carry.** For workloads averaging fewer than 4096 KV tokens per sequence, carry decoded next-item metadata and K/V pointers between persistent iterations instead of reloading them. Unscheduled and long paths retain their original loop form.
5. **Skip provably unnecessary empty fill.** The schedule records `needs_empty_fill`; omit `_fill_empty` only when CPU lengths prove no query-bearing sequence has empty KV. Default/unknown cases stay conservative. No device-to-host read is added. CPU lengths must agree with GPU offsets; the implementation does not verify all offset values by synchronizing.
6. **LSE correctness fix.** Both exponent paths use probability log2 shift 4. Subtract 4 when reporting natural-log LSE, rather than the fast path's BF16 exponent-bias-containing offset. This fixes an approximately 88-unit offset; it is not a claimed performance optimization.
7. **Exponent policy in measurements.** Short uses exact exp. Long enables an **existing** approximate-exp mode, not a newly implemented approximation. The primary prepared-baseline table holds that mode constant on both sides. Enabling it on short previously introduced extra accuracy failures, so we did not adopt that policy.

K/V preparation itself has **not** been optimized. It reads BF16 packed `[tokens,heads,128]`, quantizes into E4M3 plus E8M0 scales, pads each sequence to a multiple of 256 keys, writes head-major K and transposed/permuted V in layouts the matrix core consumes. The BF16 path allocates/prepares these on every call. Q conversion is fused in that path. Projection producing prepared layouts directly remains a model integration opportunity.

## Expected performance: use these comparisons separately

These are geometric means, not guarantees for another compiler/device or sequence distribution. Hardware was idle MI350X gfx950, 256 CUs, ROCm 7.2.1, PyTorch 2.12.0+rocm7.2, source-built TLX compiler `46d1f37fa31c3765bbf773ff5cc0eefe66fcdafd`. Clocks were **not locked**. BM128, 512 CTAs, 256-MiB cold-cache flush; primary comparisons alternate two runners, 15 warmups, 100 samples × 3 rounds. BF16 full-call runs use 10 warmups and 50 × 3 samples.

**Clean kernel attribution: prepared Q/K/V and the same exponent mode on both sides**

| Family | Prepared baseline effective TB/s | After opt effective TB/s | Speedup | Independent repeat on other GPU |
|---|---:|---:|---:|---:|
| Short, ten shapes | 2.631 | 2.820 | **1.072x** | **1.077x** |
| Long, ten shapes | 3.107 | 5.138 | **1.654x** | **1.646x** |

Primary short individual speedups are ~1.046–1.111x; long ~1.503–1.879x. Outputs **and LSE** are bitwise identical across before/after in all 20 cases, confirmed on the other GPU. Absolute latency distributions and exact per-case shapes are in the linked prepared-baseline comment and `evidence/prepared_baseline_*.json`.

A separate pair isolating **prepared Q alone**, with K/V already prepared and unchanged scheduling/waits, measured **1.023x short / 1.010x long**. Prepared Q is not the largest gain. Schedule balancing is the largest identified long gain.

**Earlier original-core versus final configuration, with mixed Q input and long exponent mode**

| Family | Original effective TB/s | Final effective TB/s | Speedup |
|---|---:|---:|---:|
| Short | 2.568 | 2.814 | **1.096x** |
| Long | 2.568 | 4.866 | **1.894x** |

The baseline has BF16 Q and exact exp; after has prepared Q and fast exp on long only. **Do not report 1.894x as kernel optimization gain at identical operands/exponent mode.** The 1.654x result answers that question. Do not divide ratios from separate runs to claim an isolated fast-exp gain; that ablation has not been measured cleanly.

**BF16-input full attention call, including per-call K/V preparation**

| Family | Original BF16 ms | Optimized BF16 ms | Prepared optimized ms | AITER BF16 ms |
|---|---:|---:|---:|---:|
| Short | 0.349 | 0.337 | 0.134 | 0.436 |
| Long | 3.646 | 2.965 | 0.885 | 2.247 |

These displayed latency columns are medians aggregated geometrically from separate two-runner pairs; compare paired ratios, not ratios of rounded cross-pair columns. Paired BF16 improvement over original is **1.032x short / 1.261x long**. Against AITER, optimized BF16 is **1.299x faster short**, but takes **1.304x AITER latency long**. Prepared versus optimized BF16 call is **2.518x short / 3.350x long**, with preparation moved outside the timed interval. Schedule creation is excluded from both optimized paths. This is attention-call timing, not projection or whole-model latency. K/V preparation now accounts for roughly 70% of long BF16 latency by a rough diagnostic subtraction; components were not independently added to construct call timings.

**TB/s definition:** effective logical main-pass K/V payload-and-scale bytes / call latency:

`KV_bytes = H*(2*D + 8)*sum(ceil(q_i/BM)*ceil(k_i/256)*256)` with H4/D128/BM128.

It excludes Q/output, metadata, redundant prefetch and cache effects. **It is not measured HBM bandwidth.** Higher effective TB/s here means the same modeled useful traversal completed sooner; it does not prove higher physical HBM utilization.

## Evidence for individual mechanisms

These were different stages/configurations of tuning, not a single cumulative factorial experiment. Do not add percentages or multiply these values to reconstruct the final result.

| Change | Short gain | Long gain | Evidence / limits |
|---|---:|---:|---|
| Balanced schedule | ~3–5% | ~34–41% | Earlier multi-runner screening, indicative attribution; largest identified long contributor |
| Copy wait after register softmax | 1.017x | 1.089x | Direct same-prepared-input pairs |
| Metadata carry + empty-fill omission | 1.0203x GPU0 / 1.0192x GPU1 | ~1.003x | Direct pairs; combined change, not isolated components |
| Prepared Q alone, latest run | 1.023x | 1.010x | Same-mode direct pairs, conversion excluded |

The original grid has 4 warps/CTA at BM128, ~68,544 bytes LDS/CTA and 256 VGPR configuration in the inspected kernel, allowing two CTAs/CU / eight resident waves. Balancing improves active work distribution, not necessarily peak residency. Reducing BM without redesigning buffering can reduce useful warps while retaining a similar K/V LDS footprint.

## Correctness and review status

- **49 targeted GPU pytest cases pass** on the publication worktree: both exponent modes, BM64/128/256, BF16/prepared Q, scheduled/unscheduled, independent row scales, irregular and empty lengths, invalid schedules, and direct natural-log LSE regression.
- The final metadata/fill update is bitwise identical to its preceding implementation across 20 shapes × seeds 7/11/23, output and LSE. All 30 long checks pass the original AITER gates.
- **Known inherited short LSE failures remain:** seed 7 `control_short_0` and `candidate_short_3`; seed 23 `candidate_short_2`. Output relative L2 threshold is 0.08, LSE maximum absolute threshold 0.10. We did not relax gates. A passing edge-test suite does not mean every production accuracy case passes.
- Latest clean prepared-baseline/after pairs match output and LSE bitwise, on both GPUs; this preserves existing numerics and failures, not proof that every baseline result is accurate enough.
- Changed-file pre-commit passes. Whole-tree checks encounter pre-existing original harness Ruff/shebang failures; do not silently rewrite upstream reproduction code while formatting unrelated changes.
- Only Python changed. Repository instructions say **do not rebuild native/compiler code for Python-only changes**. Native/compiler changes require the prescribed rebuild before tests.
- Preserve block-uniform Triton/TLX semantics. Do not introduce hidden lane-varying scalar hardware IDs, remove async waits based only on timing, or treat sampled ATT stalls as permission to weaken synchronization.

## What was tried and not retained

The handoff branch preserves experimental source snapshots under `experiments/` and raw JSON/reports under `evidence/`. Reports are chronological: statements such as “metadata carry experimental” were superseded by the accepted/validated final implementation above. A slowdown is evidence for the tested configuration, not proof that an entire idea is impossible.

| Hypothesis / variant | Observation | Decision |
|---|---|---|
| XCD-local rather than global cost balancing | Smaller long gains | Keep global schedule |
| Skip redundant final next-item K/V prefetch | ~0.991x one screen, ~1.002x another | No consistent gain; retain existing waits/prefetch semantics |
| Predecoded CPU metadata descriptors | ~0.985–0.992x | No gain versus carry-based change |
| One work item per CTA / nonpersistent alternative | ~0.954x screen, ~0.991x confirmation | Not adopted |
| BM64 / more CTAs alone | Lost latency hiding; K/V LDS footprint remained substantial while warps decreased | Needs buffering/occupancy redesign, not just smaller query tiles |
| AGPR enablement / freer AGPR allocation | ~0.916–0.919x | Not adopted |
| Reduction-tree rewrite / spill sinking | ~0.950x / ~0.932x | Not adopted |
| Relax compiler scheduling barriers | ~0.979x | Not adopted; no correctness waits removed |
| Scale exponent/log arithmetic | ~0.962–0.969x | Not adopted |
| Wider full-Q load / full-Q plus packed scales | ~0.785x / ~0.725x | Not adopted |
| Packed Q scales | Initial ~0.6% gain shrank to ~0.1% on all short shapes | Not adopted |
| Earlier/later Q payload prefetch | 1.00185x / 0.99070x | No consistent useful gain |
| Exact-domain quadratic/cubic exp corrections | 0.784x / 0.576x | Extra vector/register cost outweighed approximation improvement |
| Corrected fast-exp polynomials | 0.591x / 0.482x; original approximate short mode only ~1.016x and prior extra accuracy failures | Keep exact exp for short; do not loosen gates |
| HIP graph replay for short reader/attention comparison | ~5–8 µs slower on this stack | Not adopted; not a universal graph result |
| Captured external/internal event timing | External unsupported; internal elapsed handles invalid | Do not treat these attempts as valid measurements |
| Batch eight independent copies together | **1.318x throughput** across ten short shapes; bitwise checked; ~0.8–1.0 ms call latency | Valid separate throughput/latency tradeoff, **not a fixed-shape kernel optimization** |

## Roofline: what we know, and a correction future work must preserve

The board reported **6.810 TB/s** memory bandwidth. Dense MXFP8 architectural matrix peak used in analysis is **4.614 PFLOP/s**, not sparse/FP4 9.2 PFLOP/s. Do not substitute a retail product's bandwidth or capacity without checking this machine's configuration.

- Useful attention FLOPs: `4*H*D*sum(q_i*k_i)` (FMA = 2).
- Executed main matrix work also includes query and KV padding; row-sum MFMA adds ~6.25% beyond QK/PV.
- BM128 main-matrix/requested-KV intensity is ~248 FLOP/byte, below the ~678 FLOP/byte peak compute/bandwidth balance. Thus peak matrix throughput alone is not the right performance target; memory delivery, vector work, and latency hiding matter.
- For representative `control_long_0`, useful work ~374 GFLOP, padded QK/PV ~938 GFLOP plus row sums ~59 GFLOP. That gives ~0.216 ms at matrix peak versus ~0.524 ms for an estimated 3.569 GB of reads at board peak bandwidth. These are simplified bounds, not a complete performance predictor; do not mix a profiled dispatch's latency with a different benchmark median.

**We previously called the short read reference “shape-aware roofline”; that was too strong.** It is a **byte-count-matched contiguous read-and-checksum benchmark**, not a derived physical ceiling, nor matching attention's addressing/LDS/scale/output paths.

- Best recorded large read: ~5.115 TB/s for 4 GiB; another GPU's word reader ~5.04 TB/s. Short transfers achieve ~2.89 TB/s at 256 MiB, 3.12 at 320 MiB, 3.35 at 384 MiB.
- Latest short attention/reference geometric-mean ratio is ~87.26%. This **does not establish that only ~15% headroom remains**. A better reader can raise the reference, and caching can allow logical effective bytes to exceed a reference's physical rate.
- Readers tested contiguous uint32 reductions, block sizes 4/32/128/256 KiB, 4/8 warps. Word versus byte reductions did not explain most size dependence. Checksums consume the loaded bytes.
- Holding short sizes out, fitting only 1/4-GiB measurements gives `t = 45.7 µs + bytes/(5.326 TB/s)`. Predictions are within ~0.8–3.7% of the short reference measurements. This supports consistency of that implementation's scaling, **not inevitability of its fitted overhead**.
- **Stronger references are still unfinished work:** persistent independent-load readers and async global-to-LDS double/triple-buffer readers were proposed, but no stronger-reference result was validated or claimed. Start here if making a “near read roofline” argument.
- Derived `FetchSize` under-reported a known 4-GiB read by almost exactly 2x on this stack. Estimated read bytes apply local factor `1.999998748303243`. Retain raw counters and revalidate this calibration after runtime/kernel changes; do not present it as a universal correction. Counter collection latency is not normal benchmark latency.
- An untimed ~0.610-ms compute queue prefix before the cache flush changed representative timings <0.4%, arguing against host submission gaps being the main observed short-call effect. That control does not prove the reader is optimal.

## rocprofv3 ATT / ASM evidence and how to continue safely

Full bundles were collected, decoded, validated and packaged using `.agents/skills/amd-att-trace/scripts/att_trace.py`, profiler `/opt/rocm-7.2.1/bin/rocprofv3`, external official trace decoder 0.1.6. On this host the decoder is under `/home/stwinata/nod/tlx_work/tmp/pr4683/att_tools/decoder/opt/rocm/lib/`.

Read the current ATT skill before collecting. Use an idle GPU, **ROCR UUID**, a fresh output directory, matching-dispatch selection, and verify the actual kernel/global dispatch in the DB. Do not assume profiler GPU index equals physical board index. Recent captures use kernel regex `_mxfp8_varlen_kernel`, matching dispatch 4 after three warmups, CU0, SE mask 0x1, SIMD mask 0xF. Original short capture global dispatch was 36; final capture global dispatch was 33 because fill launches disappeared.

Evidence: metadata-carry captures reduce sampled scalar-global-load hits **416 → 368**; final nonempty traces have five attention dispatches and **zero fill dispatches**, versus five fills previously. Earlier traces show waits around metadata-to-pointer dependencies, Q payload/scales and K/V completion. Representative short sampled resident-wave occupancy is ~93–95% of the observed peak in one shader engine's eight CUs. That is **not whole-GPU occupancy or MFMA utilization**. Stalls overlap across waves; never sum them into a GPU runtime percentage or compare different-GPU trace cycles as a paired latency result.

Large binary ATT bundles are **not** in the small Git evidence branch. On the original host, artifact root is `/home/stwinata/nod/tlx_work/tmp/pr4683`:

| Local archive under artifact root | SHA256 |
|---|---|
| `att/next_short_integrated_control1.tar.gz` | `d61006b4d75713fe2ce41bd58fbe659fe2ac451b6a6a8c8bd79f8604b34279c0` |
| `att/next_short_integrated_candidate4.tar.gz` | `4c6cac6996ecbdd09f8e8f43c1c5382e417b51c86dad23c47c35ae7fea891a8f` |
| `att/short_base_control1.tar.gz` | `ed66d1cee3c581dd0e22c99852c2538ed09cb716f8444874afadcf1c35f0297f` |
| `att/short_base_candidate4.tar.gz` | `bbdd6525fac65ee1f1434553a1599d388f1922b4186475c97af47c2ca68db947` |

Archive directories contain decoded `ui_output_agent_*_dispatch_*` data as recorded in preserved reports. ASM/launch metadata for the final long entry-point comparison are at `asm/next_integrated_runtime/control_long_0/`; they match the preceding long implementation apart from compilation hash. Archive the full ATT bundles to durable storage before deleting host temporary data; a decoded CSV alone is not a complete trace bundle.

For ordinary counters, `FetchSize` and `WriteSize` together exceeded hardware counter capacity here: collect separately. Avoid timing from instrumented runs. A hung profiler finalization process was encountered; terminate only a verified process belonging to your experiment, never another user's GPU process or the device.

## Reproduce / resume

The handoff evidence branch includes portable `bench_prepared_baseline.py`, `bench_pr_bf16_comparison.py`, original baseline and initial prepared baseline sources, all later experiment snapshots, raw JSON and `SHA256SUMS`. Driver changes from the measured local scripts only replace host-specific paths/import helpers and make expected PCI explicit; argument parsing was checked. Full timed measurements in the JSON came from the original versions; validate a smoke case if adapting further.

```bash
# Fetch from this fork, then use an isolated worktree at the evidence commit linked above.
# Its base is the exact optimization commit; do not merge the evidence branch into the kernel PR.
git fetch https://github.com/raikonenfnu/tlx_triton.git docs/pr4683-forward-handoff
git worktree add /tmp/tlx-forward-handoff FETCH_HEAD
cd /tmp/tlx-forward-handoff

# Activate a working source-built TLX/ROCm environment first.
# Set ROCR_VISIBLE_DEVICES to the chosen board UUID and map HIP/CUDA logical device 0.
# Determine its KFD id and PCI address; verify there is no competing owner.
python notes/pr4683-forward-handoff/bench_prepared_baseline.py \
  --family short --kfd-id YOUR_KFD_ID --expected-pci YOUR_PCI_ADDRESS \
  --json /tmp/prepared-short.json
# Repeat --family long. --skip-q-only runs only prepared-baseline vs after-opt.
python notes/pr4683-forward-handoff/bench_pr_bf16_comparison.py \
  --family short --kfd-id YOUR_KFD_ID --expected-pci YOUR_PCI_ADDRESS \
  --json /tmp/bf16-short.json
```

The published `bench_prequantized_forward.py` reproduces the older mixed-input table; it does **not** by itself reproduce the clean prepared-baseline attribution table. Use the new preserved driver for that. The BF16 driver also needs AITER native modules built at the pinned source below.

For targeted regressions, preimport the installed source-built compiler to avoid loading incompatible runtime Python from the PR worktree, then use pytest importlib mode:

```python
import triton
import pytest
raise SystemExit(pytest.main([
    '--import-mode=importlib', '-s', '--tb=short',
    '/path/to/optimization/worktree/python/test/unit/language/test_tlx_amd_gfx950.py',
    '-k', 'mxfp8_prequantized',
]))
```

Original host inventory (paths are references, not portable prerequisites):

| Resource | Value |
|---|---|
| Venv | `/home/stwinata/nod/tlx_work/tmp/tlx_venv` |
| Environment | `source /home/stwinata/nod/tlx_work/tmp/pr4683/env.sh` (defaults to GPU0; check before use) |
| Installed compiler checkout | `/home/stwinata/nod/tlx_work/tlx_triton`, source-built at `46d1f37fa31c...` |
| Publication worktree | `/home/stwinata/nod/tlx_work/tmp/pr4683_publish`, clean at `c338ba12e...` when handoff started |
| Development worktree | `/home/stwinata/nod/tlx_work/tmp/pr4683_source`, `opt/pr4683-prequantized-forward`; uncommitted development changes, use published commit as authority |
| AITER source | `/home/stwinata/nod/tlx_work/tmp/aiter_source`, `2988812e134a7dbb08948092a9d89df48a2585d3` |
| GPU0 | UUID `GPU-49b26f9df75a3f63`, PCI `0000:05:00.0`, KFD 39903 |
| GPU1 | UUID `GPU-0274a6afb9b94834`, PCI `0000:15:00.0`, KFD 22482 |

Check GPU ownership afresh. Two GPUs were authorized for this optimization work; availability and ownership are dynamic. A source checkout alone is not evidence that its compiler binary was rebuilt. The venv has local ROCm interoperability fixes: a numeric-output `hipconfig` shim and a torch-bundled rocprofiler SDK replaced with a symlink to the system SDK to avoid duplicate SDK loading. These were venv-local; inspect rather than blindly repeating them on a new stack. Do not expose GitHub tokens in logs.

## Prioritized continuation work

1. **Reproduce the controlled table first.** Check same operands, exponent mode, offsets, BM/grid, source hashes and device identity. Preserve medians/MAD and correctness. Use two-runner alternating A/B, then confirm promising small gains on the other GPU and more than one data seed.
2. **Fix inherited short LSE accuracy failures without weakening gates.** Understand quantization/normalization error; compare exact and existing fast modes carefully. This remains a correctness limitation of the draft.
3. **Validate stronger read references before asserting short memory headroom.** Build persistent readers with independent loads in flight and async global-to-LDS buffered readers; verify all bytes are consumed/checksummed, waits are valid, emitted ASM retains loads and scratch/occupancy are acceptable. Tune grid/block/unroll/buffer depth at the exact short byte counts and 1/4 GiB. Check cold-cache and queue protocol; recalibrate counters. A better reader means the prior 87% ratio understated available headroom.
4. **For BF16 end-to-end, optimize/fuse K/V preparation.** Measure the quantizer/transpose/padding path directly with ATT/counters, then investigate vectorized stores, padding/metadata costs and producer fusion. A projection-to-prepared-layout path requires model-specific sequence metadata. Prequantization alone relocates work; report where its cost went. Keep this separate from core-only gains.
5. **For larger fixed-shape prepared-core gains, redesign buffering and residency together.** More independent requests in flight, smaller K/V LDS footprint, better scale traffic or a small-query specialization may help. BM64 alone already disappointed. Account for re-reading K/V, padding, VGPR/LDS/scratch and softmax vector work; compare useful FLOPs and latency as well as effective TB/s. Use ATT and ASM to test a specific mechanism.
6. **Production scheduling contract:** benchmark amortization across changing sequence lengths, reusable schedules and model metadata. Evaluate GPU-side schedule creation/dynamic work assignment only when it solves real metadata availability/cost; it is not already implemented. Do not hide CPU length extraction or a GPU synchronization outside reported end-to-end cost.
7. **Optional integration:** expose a reviewed varlen MXFP8 op in TLX ops, and/or add a separate AITER MXFP8 baseline if the exact ragged semantics, scale recipe and supported layouts can be matched. Neither is part of the validated result yet.

Before promoting a new winner: preserve valid Triton block semantics and async completion; pass targeted edge tests and the original numerical gates or explicitly report inherited failures; benchmark the full 20-shape portfolio; distinguish prepared-core and BF16 full-call gains; keep a reversible source snapshot and raw results. Do not replace the current winner based on a four-shape screen or a sub-percent fluctuation.
