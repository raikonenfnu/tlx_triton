"""Query Codex or Claude Code with explicit context and a JSON object contract."""

import contextlib
import json
import math
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal


def query_json(
    provider: Literal["codex", "claude"],
    prompt: str,
    *,
    workspace: Path,
    schema: dict[str, Any],
    model: str | None = None,
    context_path: Path | None = None,
    timeout_seconds: float = 300,
    executable: str | None = None,
) -> dict[str, Any]:
    """Return a structured response from a fresh invocation with restricted tools."""
    if provider not in {"codex", "claude"}:
        raise ValueError("provider must be codex or claude")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if schema.get("type") != "object":
        raise ValueError("this example requires a JSON object schema")
    workspace = workspace.resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("workspace must be a directory")
    if context_path is not None:
        context = context_path.read_text(encoding="utf-8")
        if not context.strip():
            raise ValueError("project context must not be empty")
        prompt = ("Use this project context for the task below.\n"
                  "The controller owns repository changes and memory checkpoints.\n"
                  "Return structured findings only.\n\n"
                  f"{context}\n\nTask:\n{prompt}")

    with tempfile.TemporaryDirectory(prefix="coding-agent-result-") as directory:
        output_dir = Path(directory)
        if provider == "codex":
            schema_path = output_dir / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            answer_path = output_dir / "answer.json"
            command = [
                executable or "codex",
                "--ask-for-approval",
                "never",
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--cd",
                str(workspace),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(answer_path),
            ]
        else:
            command = [
                executable or "claude",
                "--print",
                "--no-session-persistence",
                "--output-format",
                "json",
                "--json-schema",
                json.dumps(schema),
                "--tools",
                "Read,Glob,Grep",
                "--permission-mode",
                "dontAsk",
                "--permission-prompts",
                "none",
                "--strict-mcp-config",
                "--mcp-config",
                '{"mcpServers":{}}',
            ]
        if model is not None:
            command.extend(["--model", model])
        if provider == "codex":
            command.append("-")
        completed = _execute(command, prompt, workspace, timeout_seconds)
        if completed.returncode:
            raise RuntimeError(f"{provider} exited with code {completed.returncode}")
        if provider == "codex":
            result = json.loads(answer_path.read_text(encoding="utf-8"))
        else:
            envelope = json.loads(completed.stdout)
            if not isinstance(envelope, dict):
                raise ValueError("Claude response envelope must be an object")
            if envelope.get("is_error") or envelope.get("subtype") != "success":
                raise RuntimeError("Claude returned an unsuccessful result envelope")
            result = envelope.get("structured_output")
        if not isinstance(result, dict):
            raise TypeError("agent response must contain a JSON object")
        return result


def _execute(command: list[str], prompt: str, workspace: Path,
             timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    with subprocess.Popen(
            command,
            cwd=workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.name == "posix",
    ) as process:
        try:
            stdout, stderr = process.communicate(prompt, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            process.communicate()
            raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
