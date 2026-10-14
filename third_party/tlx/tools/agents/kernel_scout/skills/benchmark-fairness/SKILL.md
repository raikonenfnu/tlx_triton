---
name: benchmark-fairness
description: Define comparable GPU kernel measurements in TLX Kernel Scout, including input semantics, correctness, timing, and throughput accounting.
---

Use the controller's shared harness in `benchmarking/`. Backend discovery selects implementations. It does not supply correctness verdicts, measured latency, or throughput.

For each comparison, preserve the target's operation, dtype, realized sequence lengths,
head grouping, strides, scale, mask alignment, output dtype, and allocation policy.
The initial policy is `returned_tensor`: each timed call returns the canonical output tensor.
Include allocation, required packing, padding, mask creation, layout conversion, and output extraction
inside the call. Descriptor construction and JIT compilation can precede measurement. A reusable
scratch buffer can match the library's steady-state behavior. A cached answer cannot.
Do not move data-dependent computation into `build(case, params)`.

GEMM inputs are A row-major and B in the declared layout. Attention has scale `head_dim**-0.5`,
zero dropout, no bias, and inference-only forward semantics. Dense tensors use BHSD.
Packed tensors use THD and int32 cumulative lengths. Causal packed attention is bottom-right
aligned: query i sees key j when j <= i + kv_length - q_length. Fully masked rows return zero.
GQA repeats each KV head over its consecutive query-head group. Do not change FP16 to BF16
when one backend has a faster BF16 kernel. Use a separate target.

The harness generates identical CPU-seeded data for each backend, transfers it to the selected GPU,
and checks at least two seeds against an independent FP32 oracle. It checks tensor shape, dtype,
device, finiteness, input mutation, and output aliasing. Treat generated adapters as trusted Python
that must be inspected. These checks cannot prove arbitrary code is honest or side-effect free.
Adapters may import existing backend kernels and select launch parameters. They may not read the
oracle, change the harness, patch timing functions, or use a different backend silently.

Use eager `triton.testing.do_bench` for all backends. `warmup` and `rep` are milliseconds.
Compile and autotune before recorded samples. Use identical warmup, repetition, quantiles,
and independent repeats. This function clears cache before measured calls. Label the protocol
accordingly. Do not compare these results with graph replay, hot-cache batches, profiler timing,
or numbers copied from a PR. Graph timing needs a separate campaign and a separate harness contract.
Check device ownership before and after measurement. The harness does not change clocks.
An idle device check cannot rule out power, temperature, or another job arriving between checks. Record unstable samples and investigate them before claiming a win.

Choose winners by checked latency. Use p20–p80 overlap and the target's separation threshold
for ties. Report best observed, evaluated candidates, search budget, and failed or unsupported
variants. A finite scout search does not prove that a backend's entire kernel space was exhausted.

For compute targets, report useful FLOPs / seconds / 1e12. GEMM uses 2MNK. Attention uses
4 * query_heads * head_dim * visible_query_key_pairs. Causal counts include the diagonal and
use the actual rectangular lengths. For memory targets, report logical input/output bytes /
seconds / 1e12, labeled logical TB/s. This is not a measured HBM bandwidth counter.

Weekly controls are frozen measurements. Label their dates. Check exact target and device/runtime
compatibility before reusing them. Remeasure all backends in a fresh campaign when hardware,
PyTorch/driver runtime, timing, input semantics, or target cases change. Save source/compiler pins
and reproducer requests. Refuse to reproduce against changed pins.
