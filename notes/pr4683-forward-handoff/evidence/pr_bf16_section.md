### BF16-input end-to-end attention alongside prepared inputs

Following [Lei's request on #4683](https://github.com/facebookexperimental/triton/pull/4683#issuecomment-6691037268), this separate measurement includes the BF16-input attention call as well as the prepared-input path. These are measured full calls, not sums of separately timed components.

- **Original BF16:** #4683 `forward` with BF16 Q/K/V; K/V allocation and quantization run on every call, with Q quantized inside attention. BM128, 512 CTAs, exact exp.
- **PR BF16:** this PR's `forward` on BF16 Q/K/V, including the same per-call K/V preparation and fused Q quantization, with the reusable balanced schedule. Exact exp for short cases, the existing fast-exp mode for long cases, matching the prepared runner.
- **PR prepared:** `forward_prequantized` with Q/K/V and scales prepared before timing, using the same schedule and exponent mode as PR BF16.
- **AITER BF16:** native ASM-v3 `fmha_v3_varlen_fwd` on BF16 Q/K/V, without a BF16-to-MXFP8 preparation stage. AITER source `2988812e134a7dbb08948092a9d89df48a2585d3`.

Schedule construction/upload is excluded from both PR paths and must be amortized by reusing matching sequence metadata. This is end-to-end **attention-call** timing; it excludes QKV projection and model execution. BF16 and prepared paths produce bitwise-identical output and LSE for all 20 cases in this run. The same two inherited short LSE-gate failures marked † above remain; all ten long cases pass the original AITER gates.

Protocol: same source-built compiler/runtime and seeded shapes as above; 256-MiB cold-L2 flush, 10 warmups, 50 samples × 3 rounds, unlocked clocks. Short cases ran on physical GPU 0 and long cases on GPU 1, both idle MI350X devices. Three separate two-runner alternating comparisons per shape avoid multi-runner position bias: original BF16 vs PR BF16; PR BF16 vs prepared; PR BF16 vs AITER. The PR BF16/prepared columns below come from their direct pair; original and AITER columns come from their respective pairs. Small between-pair timing variation means ratios of these display columns need not equal the paired speedups reported below. Latencies are median ± MAD in **µs**.

**Short KV**

| Case | Original BF16 µs | PR BF16 µs | PR prepared µs | AITER BF16 µs |
|---|---:|---:|---:|---:|
| `control_short_0`† | 350.9 ± 3.1 | 332.5 ± 2.8 | 133.4 ± 2.8 | 392.4 ± 4.3 |
| `control_short_1` | 371.4 ± 3.6 | 357.5 ± 3.4 | 143.1 ± 2.9 | 418.8 ± 5.2 |
| `control_short_2` | 339.3 ± 3.8 | 329.4 ± 2.9 | 130.2 ± 2.3 | 436.0 ± 5.9 |
| `control_short_3` | 385.1 ± 4.2 | 363.8 ± 2.8 | 145.1 ± 2.3 | 460.3 ± 6.0 |
| `control_short_4` | 339.7 ± 3.4 | 333.8 ± 3.3 | 131.1 ± 2.2 | 462.6 ± 6.0 |
| `candidate_short_0` | 350.9 ± 3.6 | 333.4 ± 2.7 | 134.4 ± 2.4 | 408.9 ± 6.1 |
| `candidate_short_1` | 339.5 ± 3.9 | 328.8 ± 3.0 | 130.0 ± 2.3 | 417.2 ± 5.5 |
| `candidate_short_2` | 340.0 ± 3.2 | 332.3 ± 2.5 | 131.4 ± 2.0 | 450.8 ± 7.1 |
| `candidate_short_3`† | 340.2 ± 3.4 | 330.8 ± 2.7 | 131.2 ± 2.4 | 453.7 ± 6.6 |
| `candidate_short_4` | 340.1 ± 3.4 | 334.4 ± 2.4 | 131.2 ± 2.1 | 467.9 ± 6.6 |
| **Geometric mean latency** | 349.4 | 337.5 | 134.0 | 436.2 |

**Long KV**

| Case | Original BF16 µs | PR BF16 µs | PR prepared µs | AITER BF16 µs |
|---|---:|---:|---:|---:|
| `control_long_0` | 3,599.7 ± 84.0 | 2,904.3 ± 15.3 | 856.0 ± 11.1 | 2,169.1 ± 15.7 |
| `control_long_1` | 3,566.9 ± 72.7 | 2,944.0 ± 10.9 | 886.7 ± 13.1 | 2,191.1 ± 44.6 |
| `control_long_2` | 3,950.6 ± 58.6 | 2,928.1 ± 8.9 | 884.5 ± 12.1 | 2,261.3 ± 58.1 |
| `control_long_3` | 3,782.5 ± 87.8 | 3,039.1 ± 15.3 | 894.9 ± 23.8 | 2,253.3 ± 46.6 |
| `control_long_4` | 3,552.8 ± 63.0 | 2,954.0 ± 12.7 | 910.2 ± 15.8 | 2,285.1 ± 75.5 |
| `candidate_long_0` | 3,622.3 ± 64.6 | 2,878.4 ± 13.1 | 853.8 ± 10.5 | 2,200.2 ± 41.9 |
| `candidate_long_1` | 3,612.6 ± 68.9 | 3,006.5 ± 14.5 | 858.0 ± 21.3 | 2,204.9 ± 35.1 |
| `candidate_long_2` | 3,641.6 ± 72.6 | 3,057.8 ± 19.1 | 870.8 ± 25.4 | 2,261.9 ± 47.9 |
| `candidate_long_3` | 3,577.8 ± 68.1 | 2,970.4 ± 15.9 | 911.7 ± 19.9 | 2,326.6 ± 73.8 |
| `candidate_long_4` | 3,568.2 ± 66.1 | 2,968.1 ± 14.1 | 925.9 ± 14.6 | 2,320.1 ± 80.4 |
| **Geometric mean latency** | 3,645.6 | 2,964.6 | 885.0 | 2,246.8 |

Paired geometric-mean BF16-input speedup over the original implementation is **1.03x short / 1.26x long**. Relative to AITER, PR BF16 is **1.30x faster on short cases**, but takes **1.30x AITER's latency on long cases**. Prepared inputs reduce this PR's attention-call latency by **2.52x short / 3.35x long**, with operand preparation moved outside that timed path. The core gains therefore improve the BF16 path, while K/V preparation still prevents an end-to-end BF16 win over AITER on long cases.

