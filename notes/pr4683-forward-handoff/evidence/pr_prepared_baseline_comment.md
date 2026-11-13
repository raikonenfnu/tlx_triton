A cleaner attribution: **prepared Q/K/V as the baseline, then all kernel optimizations**, with the exponent mode held constant.

Prepared Q is **not** the biggest isolated optimization in the new measurements. A separate direct A/B changing only BF16-Q-in-core to prepared Q (K/V already prepared in both) gives **1.023x short / 1.010x long** geometric-mean speedup. This excludes running `prepare_q`; it does not measure a BF16 end-to-end saving. The larger core gain comes from the other changes, especially scheduling on long KV.

**Prepared baseline:** #4683 plus the initial prepared-Q entry point and the LSE correctness fix; original persistent item assignment, original async waits, original metadata handling and empty-fill behavior. **After opt:** the published PR implementation (`c338ba12e04c256a591cd8cb451b74aaf90c9a78`) with cost-balanced scheduling, copy/softmax overlap, and metadata/fill optimizations. Both consume the **same prepared Q/K/V buffers and scales**, with BM128 and 512 CTAs. Both use **exact exp for short / the existing fast-exp mode for long**. Thus this comparison includes neither Q-preparation differences nor an exponent-mode switch.

The byte metric remains **effective logical main-pass K/V payload-and-scale TB/s**, not measured HBM bandwidth or a roofline:

`bytes = 4 * (2*128 + 8) * sum(ceil(q_i/128) * ceil(k_i/256) * 256)`

Fresh direct alternating two-runner A/B, MI350X, batch 432, H4/D128, seed 7, 256-MiB cold-L2 flush, 15 warmups, 100 samples × 3 rounds, unlocked clocks. Primary short cases ran on GPU 0 and long cases on GPU 1. Operand preparation and schedule construction/upload are excluded; matching schedules are reused. Latencies are median ± MAD in µs. All 20 cases produce bitwise-identical output and LSE across prepared baseline and after-opt, and also match the same-mode BF16-Q runner.

**Short KV — prepared inputs on both sides**

| Case | Q total (max) | KV total (max) | Prepared baseline µs | After opt µs | Baseline TB/s | After opt TB/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `control_short_0`† | 24,688 (192) | 264,119 (1,004) | 144.1 ± 1.6 | 132.0 ± 2.6 | 2.531 | 2.762 | 1.092x |
| `control_short_1` | 27,248 (160) | 264,237 (828) | 153.3 ± 1.7 | 142.4 ± 2.8 | 2.814 | 3.029 | 1.076x |
| `control_short_2` | 29,232 (240) | 264,686 (748) | 135.8 ± 1.4 | 128.2 ± 2.5 | 2.588 | 2.742 | 1.059x |
| `control_short_3` | 31,856 (224) | 263,664 (788) | 161.3 ± 2.1 | 145.2 ± 2.0 | 2.741 | 3.045 | 1.111x |
| `control_short_4` | 32,736 (240) | 263,726 (748) | 136.4 ± 1.5 | 130.4 ± 2.2 | 2.660 | 2.782 | 1.046x |
| `candidate_short_0` | 26,672 (224) | 259,407 (964) | 145.8 ± 1.6 | 131.9 ± 2.3 | 2.531 | 2.798 | 1.106x |
| `candidate_short_1` | 27,200 (224) | 264,751 (748) | 135.0 ± 1.4 | 128.1 ± 2.4 | 2.576 | 2.717 | 1.055x |
| `candidate_short_2` | 30,576 (240) | 264,690 (748) | 136.4 ± 1.3 | 128.8 ± 2.1 | 2.601 | 2.756 | 1.059x |
| `candidate_short_3`† | 30,928 (224) | 264,436 (748) | 136.4 ± 1.6 | 128.3 ± 1.8 | 2.617 | 2.783 | 1.063x |
| `candidate_short_4` | 32,784 (240) | 264,520 (748) | 136.6 ± 1.7 | 129.6 ± 2.1 | 2.662 | 2.806 | 1.054x |
| **Geometric mean** | — | — | — | — | **2.631** | **2.820** | **1.072x** |

**Long KV — prepared inputs on both sides**

| Case | Q total (max) | KV total (max) | Prepared baseline µs | After opt µs | Baseline TB/s | After opt TB/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `control_long_0` | 24,688 (192) | 3,135,104 (32,738) | 1247.2 ± 12.9 | 773.7 ± 18.7 | 3.030 | 4.885 | 1.612x |
| `control_long_1` | 27,248 (160) | 3,142,170 (21,144) | 1309.2 ± 10.5 | 767.5 ± 9.5 | 3.148 | 5.370 | 1.706x |
| `control_long_2` | 29,232 (240) | 3,161,784 (16,354) | 1441.4 ± 48.8 | 768.8 ± 10.1 | 2.795 | 5.240 | 1.875x |
| `control_long_3` | 31,856 (224) | 3,146,362 (18,851) | 1394.3 ± 17.3 | 803.0 ± 9.0 | 3.009 | 5.225 | 1.736x |
| `control_long_4` | 32,736 (240) | 3,139,507 (16,354) | 1239.9 ± 15.1 | 825.1 ± 11.5 | 3.371 | 5.066 | 1.503x |
| `candidate_long_0` | 26,672 (224) | 3,117,994 (29,926) | 1380.2 ± 19.1 | 734.5 ± 7.4 | 2.763 | 5.191 | 1.879x |
| `candidate_long_1` | 27,200 (224) | 3,164,011 (16,354) | 1226.5 ± 8.8 | 802.3 ± 9.2 | 3.262 | 4.986 | 1.529x |
| `candidate_long_2` | 30,576 (240) | 3,164,659 (16,354) | 1295.3 ± 13.4 | 781.7 ± 14.4 | 3.158 | 5.234 | 1.657x |
| `candidate_long_3` | 30,928 (224) | 3,180,031 (16,354) | 1288.5 ± 14.3 | 818.5 ± 9.6 | 3.221 | 5.070 | 1.574x |
| `candidate_long_4` | 32,784 (240) | 3,156,939 (16,354) | 1240.5 ± 13.9 | 817.5 ± 22.2 | 3.385 | 5.137 | 1.517x |
| **Geometric mean** | — | — | — | — | **3.107** | **5.138** | **1.654x** |

These replace the mixed-input/mixed-exponent comparison **for attributing kernel optimization gains**. The earlier 1.096x short / 1.894x long table still describes its stated original-core-versus-final configuration, but cannot isolate the kernel changes from prepared Q and the long-case exponent-mode change. The separate Q-only experiment and this experiment have different paired measurements; their speedups are not a single cumulative run.

† The inherited seed-7 short LSE failures against the AITER gate (`control_short_0`, `candidate_short_3`) remain. Bitwise equivalence demonstrates that these optimizations preserve the baseline's numerics; it does not resolve those inherited failures. Preparation costs and BF16 end-to-end results remain separately reported in the PR description.

Baseline forward-source SHA256: `8deb59532255a5ac80ccb9c3868cf56e0693b3a55c01cce7284bfcb7e85c78ec`; after-opt forward-source SHA256: `d01ab3e0c14f452ea5b366342b702bbd106df20a2690bbae39c20b535ed7aceb`. Both share the original core-source SHA256 `bff4a76cb97ca346fdba5793b79ecf0eb289eba050c3f48e161197f8e1b4ee46`. Compiler/runtime match the earlier tables: TLX `46d1f37fa31c`, PyTorch 2.12.0+rocm7.2, ROCm 7.2.1.

Independent repeat of the prepared-baseline/after-opt pair on the other GPU: **short KV 1.077x**, all ten shapes still bitwise equal.

Independent repeat of the prepared-baseline/after-opt pair on the other GPU: **long KV 1.646x**, all ten shapes still bitwise equal.
