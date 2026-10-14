---
name: register-backend
description: Add a hardware-scoped kernel discovery backend to TLX Kernel Scout with consistent adapters, measurement contracts, and provenance.
---

Create `backends/<id>/backend.json` and `instructions.md`. Use a lowercase directory id.
Declare vendors, architectures, and supported operations. An empty architecture list means
any architecture of the declared vendors. It does not establish implementation availability.
Restrict vendor-specific libraries explicitly. Unsupported hardware must be reported, not
routed to another library under the backend's name.

The controller discovers manifests automatically. Add optional seed candidates and a dispatch
in `backends/seeds.py` so offline smoke runs work. A backend without seeds can use Codex or Claude
to propose an adapter. Extend operation validation, common inputs, and the FP32 oracle together
when introducing a new operation. Do not add backend-owned timing code.

Adapters export `build(case, params)` -> `run(inputs)` -> canonical output tensor.
The common input dictionary uses a/b for mm, q/k/v for attention, and also cu_q/cu_k for packed
attention. Include every conversion required by the target in run. Preserve canonical output
layout. A tuple-returning library must extract its output explicitly and document any auxiliary
work that the API computes. Source selection must stay within the registered backend.

Instructions should name checked public entry points, private tuning surfaces where applicable,
shape and dtype limits, numerical options, algorithm selection, cache/autotune behavior,
fallback detection, provenance, and a bounded search sequence. Link source evidence. Explain
how to distinguish a missing dependency from an unsupported case or a failed candidate.
Do not infer a launch API from another library's naming.

Record source Git SHA when available, package version, dirty state, and a source hash.
If a package has no recoverable Git SHA, report unknown and retain its hash. Do not invent a SHA.
Extend worker provenance for the new package. Configure an isolated Python runtime with
`--runtimes` if dependencies require another environment. check compatible input/timing and
device contracts. Do not install dependencies during discovery.

Add unit tests for manifest eligibility and adapter contract failures. Run one real GPU
correctness/timing smoke case when the dependency and hardware are available. Report whether
live agent discovery ran separately from fake-provider tests.
