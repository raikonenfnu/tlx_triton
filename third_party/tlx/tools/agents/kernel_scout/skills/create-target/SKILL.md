---
name: create-target
description: Create a TLX Kernel Scout target problem or extend its shared benchmark harness for a new operation.
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

Check config validation and core unit tests before GPU work.
Run a small case through the shared harness on supported hardware.
Check actual input fingerprints, output correctness, source pins, and timing stability.
Check a pinned reproducer after choosing a valid candidate.
Record whether live discovery ran and which dependencies were available.

A changed target contract needs a new weekly baseline before nightly reuse.
Keep the prior target and report revisions with their evidence.
Adding a target does not install a schedule or publish an issue.
Use the controller's explicit publication command when the user requests publication.
