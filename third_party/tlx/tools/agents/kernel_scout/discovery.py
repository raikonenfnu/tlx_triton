"""Bounded read-only source research with optional executable adapter proposals."""

import hashlib
import json
from pathlib import Path

from coding_agents import query_json
from contracts import NAME, REPOSITORY, ROOT, digest, harness_digest, read_json, write_json

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


class DiscoveryCache:
    """Reuse source-scoped adapters, never measurements or correctness verdicts."""

    def __init__(self, root, *, refresh=False):
        self.root, self.refresh, self.responses = Path(root), refresh, {}

    def key(self, backend, target, research):
        paths = [
            ROOT / "discovery.py", ROOT / "backends" / backend / "instructions.md",
            ROOT / "backends" / backend / "backend.json", *sorted((ROOT / "skills").rglob("*.md"))
        ]
        return digest({
            "target": target, "backend": backend, "research": research, "harness": harness_digest(), "instructions":
            {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in paths}
        })

    def load(self, key):
        if key not in self.responses and not self.refresh:
            path = self.root / f"{key}.json"
            if path.exists():
                saved = read_json(path)
                if saved["key"] != key or saved["response_sha256"] != digest(saved["response"]):
                    raise ValueError("discovery cache identity or content differs")
                self.responses[key] = saved["response"]
        return self.responses.get(key)

    def save(self, key, status, summary, candidates):
        proposals, seen = [], set()
        if status == "complete":
            for candidate in candidates:
                if "adapter" not in candidate or candidate["adapter_sha256"] in seen:
                    continue
                source = Path(candidate["adapter"]).read_text()
                if hashlib.sha256(source.encode()).hexdigest() != candidate["adapter_sha256"]:
                    raise ValueError("discovered adapter changed before caching")
                seen.add(candidate["adapter_sha256"])
                proposals.append({
                    "id": f"cached_{len(proposals)}", "params": {}, "adapter_source": source.removesuffix("\n"),
                    "rationale": candidate["rationale"]
                })
        response = {"status": status, "summary": summary, "candidates": proposals}
        self.responses[key] = response
        # Retry blocked research on the next run, rather than saving it as an empty search.
        if status == "complete":
            write_json(self.root / f"{key}.json",
                       {"key": key, "response": response, "response_sha256": digest(response)})


def discover(provider, backend, target, case, feedback, directory, remaining, *, model=None, context=None, timeout=300,
             research=None, scope="case"):
    instructions = (ROOT / "backends" / backend / "instructions.md").read_text()
    fairness = (ROOT / "skills/benchmark-fairness/SKILL.md").read_text()
    registration = (ROOT / "skills/register-backend/SKILL.md").read_text()
    workspace = REPOSITORY
    if backend == "aiter" and research and research.get("package", {}).get("status") == "available":
        package = research["package"]
        workspace = Path(package.get("repository") or package["module"])
    if backend == "pytorch" and research and research.get("runtime", {}).get("torch_module"):
        workspace = Path(research["runtime"]["torch_module"]).parent
    # Feedback already has full provenance on disk. Avoid sending repeated runtime and device dumps.
    compact = [{
        "candidate": f["candidate"], "result": {
            k: v
            for k, v in f["result"].items()
            if k in {
                "status", "error", "latency_ms", "p20_ms", "p80_ms", "relative_spread", "numerical_errors",
                "adapter_metadata"
            }
        }
    }
               for f in feedback]
    scope_instruction = (
        "Research all cases in target.cases. Return adapters that derive dimensions, dtype, and causal mode from case.\n"
        "Guard unsupported cases with NotImplementedError. The controller will check every adapter on each case.\n"
        "This is the target's only research phase. Do not assume the current case represents every shape."
        if scope == "target" else "Research the current case.")
    prompt = f"""Find existing kernel variants for this backend and target.
{scope_instruction}
Read source before selecting entry points. Do not install dependencies, edit source, run benchmarks,
or publish memory. Return at most {remaining} adapter proposals. Return an empty list if no justified
candidate exists. Return status=blocked for tool, permission, authentication, or source-read failures.
Return status=complete only after source inspection. A blocked discovery must have no candidates.
Each adapter exports build(case, params), which returns run(inputs).
build runs before input generation. run returns the canonical tensor and includes all operation work.
Adapters can import existing kernels; do not rewrite a kernel or substitute another backend.
The controller checks outputs and measures candidates. It owns memory checkpoints.
Use unique ids. Hardcode tunable parameters in source and return params={{}}.
Do not propose aliases of evaluated entry points with the same configuration. An empty candidate list is successful research.
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
{json.dumps(compact, indent=2)}
"""
    response = query_json(provider, prompt, workspace=workspace, schema=SCHEMA, model=model, context_path=context,
                          timeout_seconds=timeout)
    return response, validate_proposals(response, directory, remaining)
