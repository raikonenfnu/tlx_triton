# Fastest validated TLX A4W4 snapshot on gfx950

This branch preserves the fastest reproducible TLX A4W4 state measured during
the comparison with AITER. It intentionally excludes later, unfinished
experiments involving AITER-pre-shuffled inputs, explicit load contiguity,
layout-propagation changes, and LLVM AGPR allocation.

## Frozen state

- Base commit: `427ff66f6` (`[AMD][TLX] Select shape-matched A4W4 kernels`)
- Snapshot branch: `raikonen/a4w4-fastest-validated-20260731`
- Target: `gfx950:sramecc+:xnack-`, 256 compute units
- ROCm: 7.2.26015
- PyTorch: 2.11.0+rocm7.2
- Triton: 3.8.0+fb
- AITER baseline: public `gemm_a4w4` wrapper selecting tuned assembly
- AITER reference commit recorded by the reproducer:
  `3ccc08d20c465c2617b393177d45405245032528`
- TLX inter-wave/dispatcher source SHA256:
  `7d36527b2ff39c23352fec76ca3ce97f1177be5827b0d8733eb5411f507b94ac`
- TLX intra-wave source SHA256:
  `aa4826b9510c0f9297cc7fbc5f05e76b7d1d52576895d95aa0b839ff2cb9cf13`

The only source change after the base commit is allowing LLVM's post-register-
allocation machine scheduler to run. The prior kernel set
`TRITON_DISABLE_POST_MISCHED=1` at import time. Removing that process-global
override improved the key `2048x4096x8192` result from approximately 49.84 us
to 49.60 us.

There are no compiler-source modifications relative to the base commit in this
snapshot. Consequently, checking out this branch freezes both the kernel and
the compiler state used by the validated result.

## Kernel selection

The public `matmul()` wrapper dispatches by the number of logical `256x256`
output tiles:

```text
small grid  -> 128x128 skinny kernel with bounded split-K
medium grid -> four-wave 128x256 intra-wave kernel
large grid  -> four-wave 256x256 intra-wave kernel
```

This replaces the single eight-wave `256x256` strategy used by the original
baseline. The smaller four-wave tiles expose more CTAs on occupancy-starved
shapes and avoid the phase-barrier overhead of the eight-wave inter-wave
pipeline.

## Reproduction

Build the checked-out source according to the repository instructions, install
an AITER build with tuned gfx950 A4W4 assembly kernels, and expose one gfx950:

```bash
HIP_VISIBLE_DEVICES=0 python \
  third_party/tlx/tutorials/gfx9_gemm/inter_wave/a4w4/aiter_baseline_repro.py \
  --tlx-inter-wave-source \
    third_party/tlx/tutorials/gfx9_gemm/inter_wave/a4w4/matmul_kernel.py \
  --tlx-intra-wave-source \
    third_party/tlx/tutorials/gfx9_gemm/intra_wave/a4w4/matmul_kernel.py \
  --warmups 20 \
  --samples 100 \
  --timing-rounds 3 \
  --flush-mib 256 \
  --json /tmp/a4w4_fastest_validated.json
```

The reproducer:

- checks exact BF16 equality against an FP32-dequantized reference;
- alternates AITER and TLX runners;
- flushes 256 MiB before each timed sample;
- records GPU-event latency, runtime metadata, source hashes, AITER kernel
  selection, dispatch choice, and correctness in JSON;
- excludes AITER's one-time B/scale pre-shuffle from timing;
- includes output allocation and public Python wrapper overhead for both paths.

## Validated measurements

All outputs were deterministic and exactly matched the BF16 reference.
Latency is the cold-L2 public-wrapper median; lower is better.

| M x N x K | AITER us | best TLX us | AITER / TLX | TLX latency vs AITER |
|---|---:|---:|---:|---:|
| 256 x 4096 x 4096 | 13.20 | 17.64 | 0.748x | +33.6% |
| 256 x 8192 x 4096 | 16.12 | 20.32 | 0.793x | +26.1% |
| 512 x 4096 x 4096 | 15.04 | 18.92 | 0.795x | +25.8% |
| 512 x 8192 x 4096 | 20.84 | 21.76 | 0.958x | +4.4% |
| 2048 x 4096 x 8192 | 46.48 | **49.60** | 0.937x | +6.7% |
| 2048 x 8192 x 4096 | 60.98 | 51.92 | 1.174x | -14.9% |
| 2048 x 8192 x 8192 | 84.76 | 91.00 | 0.931x | +7.4% |

The `2048x8192x4096` apparent win needs repetition: AITER varied from roughly
53 to 61 us across otherwise similar runs. It should not yet be treated as a
stable TLX win.

The broad-suite rows were measured with the base commit's hash-pinned
dispatcher. The 49.60 us key-shape row is the additional post-RA-scheduler
measurement preserved by this snapshot.

## Distinction from the AITER-exact clone

A separate diagnostic kernel matched AITER's `128x256` geometry and
pre-shuffled B/scale ABI. It was useful for identifying compiler issues, but
its fastest correct result was approximately 102.4 us and it is not the
fastest TLX implementation. Those diagnostic changes are deliberately absent
from this branch.

The remaining diagnostic findings were:

1. Triton/TLX failed to infer a true 16-byte per-thread A-load contiguity fact.
2. LLVM initially selected the wrong VGPR/AGPR form for scaled MFMA chains.
3. Physical packed scale layouts were not preserved through
   reshape/transpose layout propagation.
4. AITER transports packed B scales through LDS, while the diagnostic TLX path
   emitted byte loads and `v_perm_b32` reconstruction.

These are follow-up compiler investigations rather than prerequisites for
reproducing the fastest validated raw-layout kernel.
