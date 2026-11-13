Core-throughput results across all 20 production-shape cases, with the optimization breakdown below.

**Metric and scope.** TB/s here is **effective logical K/V payload-and-scale throughput**, not measured HBM bandwidth or a hardware roofline. For H=4, D=128, BM=128:

`KV_bytes = 4 * (2*128 + 8) * sum(ceil(q_i/128) * ceil(k_i/256) * 256)`

`effective_KV_TBps = KV_bytes / median_core_seconds / 1e12`

This counts main-pass tiled K/V reads and scales, excluding Q/output traffic, metadata, redundant prefetch and cache effects. The same byte count is used before/after for each shape.

**Before:** #4683 head `d3774785447d0385c0027dcef799be5188d462ed`, BF16 Q / already prepared K/V core, exact exp. **After:** this PR's prepared Q/K/V entry point with balanced scheduling, wait overlap, and short metadata/fill improvements; exact exp on short cases and the existing approximate-exp mode on long cases. Q/K/V preparation and schedule construction/upload are excluded. Thus, the combined long result includes an exponent-mode change and is not a same-precision BF16 end-to-end or MXFP8-versus-MXFP8 AITER comparison.

Direct alternating two-runner A/B on MI350X, batch 432, H4/D128, BM128, 512 CTAs, 256-MiB cold-L2 flush, 15 warmups, 100 samples × 3 rounds, seed 7, unlocked clocks. Q/KV columns show total tokens (maximum tokens per sequence), using #4683's deterministic proxy offsets; the original trace offsets are unavailable. Latencies are medians in µs.

**Short KV**

| Case | Q total (max) | KV total (max) | Before µs | After µs | Before TB/s | After TB/s | Δ TB/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `control_short_0`† | 24,688 (192) | 264,119 (1,004) | 148.8 | 131.6 | 2.450 | 2.772 | +0.321 | 1.131x |
| `control_short_1` | 27,248 (160) | 264,237 (828) | 155.9 | 143.3 | 2.768 | 3.011 | +0.243 | 1.088x |
| `control_short_2` | 29,232 (240) | 264,686 (748) | 139.5 | 128.5 | 2.518 | 2.735 | +0.217 | 1.086x |
| `control_short_3` | 31,856 (224) | 263,664 (788) | 165.0 | 145.6 | 2.680 | 3.037 | +0.357 | 1.133x |
| `control_short_4` | 32,736 (240) | 263,726 (748) | 139.2 | 129.4 | 2.606 | 2.804 | +0.197 | 1.076x |
| `candidate_short_0` | 26,672 (224) | 259,407 (964) | 148.3 | 133.6 | 2.488 | 2.763 | +0.275 | 1.111x |
| `candidate_short_1` | 27,200 (224) | 264,751 (748) | 138.5 | 128.8 | 2.512 | 2.700 | +0.188 | 1.075x |
| `candidate_short_2` | 30,576 (240) | 264,690 (748) | 140.2 | 128.8 | 2.532 | 2.756 | +0.224 | 1.088x |
| `candidate_short_3`† | 30,928 (224) | 264,436 (748) | 140.6 | 128.9 | 2.539 | 2.770 | +0.230 | 1.091x |
| `candidate_short_4` | 32,784 (240) | 264,520 (748) | 139.8 | 129.2 | 2.602 | 2.814 | +0.213 | 1.082x |
| **Geometric mean** | — | — | — | — | **2.568** | **2.814** | **+0.246** | **1.096x** |

**Long KV**

| Case | Q total (max) | KV total (max) | Before µs | After µs | Before TB/s | After TB/s | Δ TB/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `control_long_0` | 24,688 (192) | 3,135,104 (32,738) | 1514.1 | 825.6 | 2.496 | 4.578 | +2.082 | 1.834x |
| `control_long_1` | 27,248 (160) | 3,142,170 (21,144) | 1568.8 | 814.6 | 2.627 | 5.059 | +2.432 | 1.926x |
| `control_long_2` | 29,232 (240) | 3,161,784 (16,354) | 1754.6 | 877.0 | 2.296 | 4.594 | +2.298 | 2.001x |
| `control_long_3` | 31,856 (224) | 3,146,362 (18,851) | 1686.8 | 874.2 | 2.487 | 4.800 | +2.312 | 1.930x |
| `control_long_4` | 32,736 (240) | 3,139,507 (16,354) | 1532.7 | 833.7 | 2.727 | 5.014 | +2.287 | 1.838x |
| `candidate_long_0` | 26,672 (224) | 3,117,994 (29,926) | 1638.4 | 799.1 | 2.327 | 4.772 | +2.445 | 2.050x |
| `candidate_long_1` | 27,200 (224) | 3,164,011 (16,354) | 1487.4 | 812.4 | 2.689 | 4.924 | +2.235 | 1.831x |
| `candidate_long_2` | 30,576 (240) | 3,164,659 (16,354) | 1538.4 | 823.0 | 2.659 | 4.971 | +2.312 | 1.869x |
| `candidate_long_3` | 30,928 (224) | 3,180,031 (16,354) | 1542.9 | 829.9 | 2.690 | 5.000 | +2.311 | 1.859x |
| `candidate_long_4` | 32,784 (240) | 3,156,939 (16,354) | 1537.4 | 844.4 | 2.731 | 4.973 | +2.241 | 1.821x |
| **Geometric mean** | — | — | — | — | **2.568** | **4.866** | **+2.297** | **1.894x** |

**What each optimization improves**

The following are **separate historical before/after tuning comparisons**, converted to the same effective-byte metric. Their baselines differ, so they are not sequential stages of one run; the deltas must not be added or multiplied to reconstruct the final table. Wait-overlap and metadata/fill measurements use two-runner pairs. Scheduling numbers are indicative results from an earlier multi-runner screen, with approximately 3–5% short and 34–41% long integrated gains across the recorded seed/device runs; they are not a clean factorial attribution of the final result.

| Optimization | Short KV TB/s before → after | Short speedup | Long KV TB/s before → after | Long speedup |
|---|---:|---:|---:|---:|
| Cost-balanced persistent schedule | 2.650 → 2.753 | 1.039x | 3.360 → 4.510 | 1.342x |
| Overlap async K/V copies with register-only softmax | 2.620 → 2.663 | 1.017x | 3.198 → 3.483 | 1.089x |
| Reuse short metadata + omit provably unnecessary empty fill | 2.751 → 2.807 | 1.020x | 4.142 → 4.156 | 1.003x |

- **Scheduling:** the original kernel is already persistent. We change which items each CTA owns, using KV-tile counts to reduce the long tail of uneven sequence work. The schedule is built once and reused; rebuilding it on every call adds a separate cost.
- **Wait overlap:** postpone the async-copy wait until before the next LDS consumer, allowing register-only softmax to overlap the copies. Completion waits and buffer-reuse ordering remain intact.
- **Metadata/fill:** carry decoded short-case metadata between persistent items and remove the extra fill launch only when the schedule proves no query-bearing sequence has empty KV. This final change also gives 1.9% short improvement on GPU 1; long performance is essentially unchanged. ATT validates the traces and confirms disappearance of unnecessary fill launches; sampled scalar-global-load hits drop 416 → 368. These trace counts are diagnostic, not whole-GPU utilization percentages.
- **Prepared Q API:** adds an explicit FP8-Q/E8M0-scale input path and `prepare_q`. It moves Q conversion outside the measured core. We do not have a clean isolated gain attributable to prepared Q alone; it should not receive the combined throughput gain.
- **Existing fast exp:** enabled only for long cases in the final comparison, rather than introduced by this PR. Its separate contribution is not isolated in these tables. The fast-exp LSE offset correction is a correctness fix, not a claimed performance gain.

† The seed-7 `control_short_0` and `candidate_short_3` cases retain the inherited LSE accuracy-gate failure. Performance numbers do not imply those gates pass. All ten short outputs/LSE are bitwise equal to the original exact-exp path in the final comparison; long-mode validation and the additional seed-23 inherited short failure are documented in the PR description. The draft does not relax the gates.

For context, the separate **BF16-input full-attention-call** measurements in the PR description include K/V preparation and show **1.03x short / 1.26x long** speedup over #4683. The core table above excludes that preparation. K/V preparation itself has not yet been optimized.

Reproduce the final core table with `bench_prequantized_forward.py` and the baseline worktree command in the PR description. Runtime: source-built TLX compiler `46d1f37fa31c`, PyTorch 2.12.0+rocm7.2, ROCm 7.2.1. The added BF16 results and this comment do not change the implementation.
