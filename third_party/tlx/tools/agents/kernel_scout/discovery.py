"""Bounded read-only source research with optional executable adapter proposals."""

import hashlib
import json
from pathlib import Path

from coding_agents import query_json
from contracts import NAME, REPOSITORY, ROOT

SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["complete", "blocked"]},
        "summary": {"type": "string"},
        "candidates": {
            "type": "array", "items": {
                "type": "object", "properties": {
                    "id": {"type":
                           "string"}, "params": {"type": "object", "properties": {}, "additionalProperties": False},
                    "adapter_source": {"type": "string"}, "rationale": {"type": "string"}
                }, "required": ["id", "params", "adapter_source", "rationale"], "additionalProperties": False
            }
        },
    },
    "required": ["status", "summary", "candidates"],
    "additionalProperties": False,
}


def validate_proposals(response, directory, remaining):
    if (set(response) != {"status", "summary", "candidates"} or response["status"] not in {"complete", "blocked"}
            or not isinstance(response["summary"], str)):
        raise ValueError("invalid discovery response")
    proposals = response["candidates"]
    if not isinstance(proposals, list) or len(proposals) > remaining:
        raise ValueError("discovery exceeded its remaining candidate budget")
    if response["status"] == "blocked" and proposals:
        raise ValueError("blocked discovery cannot provide verified proposals")
    seen, result = set(), []
    for proposal in proposals:
        if set(proposal) != {"id", "params", "adapter_source", "rationale"}:
            raise ValueError("invalid candidate fields")
        name, source = proposal["id"], proposal["adapter_source"]
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in seen:
            raise ValueError("candidate ids must be distinct valid names")
        if not isinstance(source, str) or not source.strip() or len(source.encode()) > 100_000:
            raise ValueError("candidate needs bounded Python source")
        if proposal["params"] != {} or not isinstance(proposal["rationale"], str):
            raise ValueError("put proposed parameters in adapter source; params must be empty")
        compile(source, name, "exec")
        seen.add(name)
        path = directory / f"agent_{name}.py"
        path.write_text(source + "\n")
        result.append({
            "id": f"agent_{name}", "params": {}, "adapter": str(path), "adapter_sha256":
            hashlib.sha256(path.read_bytes()).hexdigest(), "rationale": proposal["rationale"]
        })
    return result


def discover(provider, backend, target, case, feedback, directory, remaining, *, model=None, context=None, timeout=300,
             research=None):
    instructions = (ROOT / "backends" / backend / "instructions.md").read_text()
    fairness = (ROOT / "skills/benchmark-fairness/SKILL.md").read_text()
    registration = (ROOT / "skills/register-backend/SKILL.md").read_text()
    workspace = REPOSITORY
    if backend == "aiter" and research and research.get("package", {}).get("status") == "available":
        package = research["package"]
        workspace = Path(package.get("repository") or package["module"])
    prompt = f"""Find the best observed existing kernel variants for this backend and exact target.
Read source before selecting entry points. Do not install dependencies, edit source, run benchmarks,
or publish memory. Return at most {remaining} adapter proposals. Return an empty list if no justified
candidate exists. Return status=blocked for tool, permission, authentication, or source-read failures.
Return status=complete only after source inspection. A blocked discovery must have no candidates.
Each adapter exports build(case, params), which returns run(inputs).
build runs before input generation. run returns the canonical tensor and includes all operation work.
Adapters can import existing kernels; do not rewrite a kernel or substitute another backend.
The controller checks outputs and measures candidates. It owns memory checkpoints.
Use unique ids. Hardcode tunable parameters in source and return params={{}}.
The adapters execute as trusted local Python; do not use files, network, shell commands, or instrumentation.

Backend instructions:
{instructions}
Benchmark skill:
{fairness}
Registration contract:
{registration}
Installed runtime and source identity:
{json.dumps(research, indent=2)}
Research workspace: {workspace}
Target and case:
{json.dumps({'target': target, 'case': case}, indent=2)}
Measured feedback (not a request to infer unmeasured performance):
{json.dumps(feedback, indent=2)}
"""
    response = query_json(provider, prompt, workspace=workspace, schema=SCHEMA, model=model, context_path=context,
                          timeout_seconds=timeout)
    return response, validate_proposals(response, directory, remaining)
