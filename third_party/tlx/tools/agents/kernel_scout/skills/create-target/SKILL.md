---
name: create-target
description: Register a TLX Kernel Scout problem with shared inputs, an independent reference, benchmark adapters, and measured readiness checks.
---

Use this skill when adding target cases or a new operation to Kernel Scout.
The application root is two directories above this skill directory.
Read [target authoring](references/target-authoring.md) for commands, required fields, and the extension map.
Read the sibling [fairness skill](../benchmark-fairness/SKILL.md) before defining the measured work.

Identify the operation's semantics before selecting kernels.
State shapes, dtypes, input strides, output layout, mask rules, numerical tolerance, and allocation policy.
Use realized sequence lengths rather than a range that changes between backend runs.
Retain source evidence for production shapes and expected semantics.

For an existing operation, use `appctl.py target-init` to create a starter configuration that passes config checks.
Replace starter cases with the requested workload.
Use the shared inputs, FP32 oracle, timing, and throughput accounting.
Select compute or memory metrics from the measured work.
Keep latency visible when a small workload limits occupancy or exposes launch overhead.

For a new operation, extend validation, inputs, oracle, work counts, descriptions, and backend capability manifests together.
Keep timing in the existing worker.
Use existing backend entry points before adding kernel code.
Declare unsupported hardware and cases explicitly.
Follow the sibling [backend registration skill](../register-backend/SKILL.md) for new implementations.

Registration owns the reusable benchmark implementation, not just its JSON shapes.
Keep input generation, the oracle, and timing in the shared harness.
Do not generate a separate benchmark program for each backend or campaign.
Test deterministic values, declared strides, output semantics, and work counts before GPU work.
For a new operation, include an independent small numerical example in the unit tests.

Run `appctl.py run --mode registration` with persistent workers on supported hardware.
Check a small case and a representative boundary case first.
Check every added case before declaring the complete target ready.
Inspect `registration.json`, the report, and failed candidate logs.
A ready case needs a checked TLX implementation and a checked AITER or PyTorch reference.
Unavailable dependencies, unstable timings, and failed backend comparisons need investigation.
Unsupported implementations remain explicit coverage gaps.
Do not label a missing adapter as proof that the backend cannot support the problem.
Run a pinned reproducer from the report.

Prepare and inspect reusable backend adapters during registration when discovery is needed.
Use target-scoped discovery and retain its source-pinned candidate cache for routine campaigns.
Keep fresh correctness checks and measurements in weekly and nightly runs.
Reuse immutable inputs and FP32 references only within a worker's current case.
Keep each backend in its own persistent worker to separate compiler and library state.
Keep input-dependent packing and conversions inside the timed call.
Registration artifacts record scope and source pins. They do not replace later correctness checks.

A changed target contract needs a new weekly baseline before nightly reuse.
Keep the prior target and report revisions with their evidence.
Adding a target does not install a schedule or publish an issue.
Use the controller's explicit publication command when the user requests publication.
