"""CPU tests for controller contracts and external publication state transitions."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from appctl import baseline_key, validate_baseline, worker_environment
from coding_agents import query_json
from contracts import applicable, backends, digest, load_target, select_winner, work, write_json
from discovery import validate_proposals
from publication import publish


@pytest.fixture
def target():
    return load_target(ROOT / "targets/amd/gfx950/varlen_f16_fa.json")


def test_targets_and_vendor_eligibility(target):
    registry = backends()
    for path in (ROOT / "targets").rglob("*.json"):
        load_target(path)
    assert applicable(registry["aiter"], target, {"vendor": "amd", "arch": "gfx950"})
    assert not applicable(registry["aiter"], target, {"vendor": "nvidia", "arch": "90"})
    assert not applicable(registry["aiter"], target, {"vendor": "amd", "arch": "gfx942"})
    assert target["cases"][0]["q_lengths"] == [1000] * 10
    assert len(target["cases"]) == 6


@pytest.mark.parametrize("change", [
    lambda t: t.update(output_policy="preallocated"),
    lambda t: t["timing"].update(rep_ms=float("nan")),
    lambda t: t["correctness"].update(seeds=[1, 1]),
    lambda t: t["cases"][0].update(kv_lengths=[100]),
    lambda t: t["cases"][0].update(kv_heads=3),
    lambda t: t["cases"][0].update(q_lengths=[-1] * 10),
    lambda t: t["cases"][0].update(id="../escape"),
])
def test_invalid_target_rejected(target, change, tmp_path):
    change(target)
    path = tmp_path / "target.json"
    write_json(path, target)
    with pytest.raises(ValueError):
        load_target(path)


def test_work_counts_rectangular_causal_and_gqa():
    case = {"q_lengths": [4, 2], "kv_lengths": [2, 4], "heads": 4, "kv_heads": 2, "head_dim": 128, "causal": True}
    # First sequence: 0+0+1+2 pairs. Second: 3+4 pairs.
    flops, byte_count = work(case, "flash_attn_varlen")
    assert flops == 4 * 4 * 128 * 10
    assert byte_count == 2 * 128 * (2 * 6 * 4 + 2 * 6 * 2)
    assert work({"m": 2, "n": 3, "k": 4}, "mm") == (48, 52)


def measurement(latency, low=None, high=None, status="ok"):
    return {
        "status": status, "latency_ms": latency, "p20_ms": latency if low is None else low, "p80_ms":
        latency if high is None else high
    }


def test_winner_uses_correct_stable_results_and_uncertainty():
    rows = {
        "wrong": measurement(0.1, status="failed"), "noisy": measurement(0.2, status="unstable"), "a": measurement(1),
        "b": measurement(1.2)
    }
    assert select_winner(rows, 0.03) == "a"
    rows["b"] = measurement(1.1, 1.01, 1.2)
    assert select_winner(rows, 0.03) == "a / b (tie)"
    assert select_winner({}, 0.03) == "none"


def test_baseline_key_accepts_tlx_changes_rejects_device_or_protocol(target):
    device = {"vendor": "amd", "arch": "gfx950", "physical_gpu": 0, "host": "machine"}
    runtime = {"torch": "2", "torch_git": "a", "hip": "7", "cuda": None, "python": "3.12", "compiler_sha256": "old"}
    key = baseline_key([target], device, runtime)
    assert baseline_key([target], device, {**runtime, "compiler_sha256": "new"}) == key
    assert baseline_key([target], {**device, "physical_gpu": 1}, runtime) != key
    changed = copy.deepcopy(target)
    changed["timing"]["rep_ms"] += 1
    assert baseline_key([changed], device, runtime) != key


def test_baseline_reproducer_digest_checked(tmp_path):
    path = tmp_path / "request.json"
    write_json(path, {"expected_identity": {"source": "a"}})
    baseline = {
        "campaign":
        "c", "week":
        "2026-10-12", "baseline_key":
        "k", "rows": [{
            "results": {
                "torch":
                {"status": "ok", "reproducer": str(path), "request_sha256": digest(json.loads(path.read_text()))}
            }
        }]
    }
    validate_baseline(baseline, campaign="c", week="2026-10-12", key="k")
    with pytest.raises(ValueError, match="does not match"):
        validate_baseline(baseline, campaign="c", week="2026-10-19", key="k")
    write_json(path, {"expected_identity": {"source": "changed"}})
    with pytest.raises(ValueError, match="changed"):
        validate_baseline(baseline, campaign="c", week="2026-10-12", key="k")


def test_worker_environment_selects_same_physical_gpu():
    env = worker_environment(3, {"pythonpath": "/source:/torch"})
    assert env["ROCR_VISIBLE_DEVICES"] == env["SCOUT_PHYSICAL_GPU"] == "3"
    assert env["HIP_VISIBLE_DEVICES"] == env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PYTHONPATH"] == "/source:/torch"
    nvidia = worker_environment(3, {"vendor": "nvidia"})
    assert nvidia["CUDA_VISIBLE_DEVICES"] == "3" and "ROCR_VISIBLE_DEVICES" not in nvidia


def test_proposal_source_and_budget_validation(tmp_path):
    response = {
        "status":
        "complete", "summary":
        "one variant", "candidates": [{
            "id": "one", "params": {}, "adapter_source": "def build(case, params):\n    return lambda x: x['q']",
            "rationale": "source caller"
        }]
    }
    proposals = validate_proposals(response, tmp_path, 1)
    assert proposals[0]["adapter_sha256"]
    assert Path(proposals[0]["adapter"]).is_file()
    with pytest.raises(ValueError, match="budget"):
        validate_proposals(response, tmp_path, 0)
    response["candidates"][0]["id"] = "../../escape"
    with pytest.raises(ValueError, match="ids"):
        validate_proposals(response, tmp_path, 1)


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_fake_provider_protocol(provider, tmp_path):
    cli = tmp_path / provider
    cli.write_text('''#!/usr/bin/env python3
import json,sys
prompt=sys.stdin.read()
answer={"summary":prompt,"candidates":[]}
if "--output-last-message" in sys.argv:
    from pathlib import Path
    Path(sys.argv[sys.argv.index("--output-last-message")+1]).write_text(json.dumps(answer))
else:
    print(json.dumps({"subtype":"success","is_error":False,"structured_output":answer}))
''')
    cli.chmod(0o700)
    prompt = "literal $(do-not-execute) `backticks`"
    result = query_json(provider, prompt, workspace=tmp_path, schema={"type": "object"}, executable=str(cli))
    assert result["summary"] == prompt


def test_claude_error_envelope_rejected(tmp_path):
    cli = tmp_path / "claude"
    cli.write_text('#!/usr/bin/env python3\nimport json\nprint(json.dumps({"subtype":"error","is_error":True}))\n')
    cli.chmod(0o700)
    with pytest.raises(RuntimeError, match="unsuccessful"):
        query_json("claude", "query", workspace=tmp_path, schema={"type": "object"}, executable=str(cli))


class FakeGitHub:

    def __init__(self):
        self.issues = {}
        self.comments = {}
        self.events = []
        self.corrupt_verification = False

    def all(self, resource):
        if resource.startswith("issues?"):
            return list(copy.deepcopy(self.issues).values())
        number = int(resource.split("/")[1])
        return [copy.deepcopy(c) for c in self.comments.values() if c["issue"] == number]

    def request(self, method, resource, payload=None):
        self.events.append((method, resource))
        parts = resource.split("/")
        if method == "POST" and resource == "issues":
            number = len(self.issues) + 1
            self.issues[number] = {"number": number, "state": "open", "html_url": f"issue/{number}", **payload}
            return copy.deepcopy(self.issues[number])
        if len(parts) == 3 and method == "POST":
            ident = len(self.comments) + 1
            self.comments[ident] = {"id": ident, "issue": int(parts[1]), "html_url": f"comment/{ident}", **payload}
            return copy.deepcopy(self.comments[ident])
        collection = self.comments if parts[1] == "comments" else self.issues
        ident = int(parts[-1])
        if method == "PATCH":
            collection[ident].update(payload)
        result = copy.deepcopy(collection[ident])
        if method == "GET" and self.corrupt_verification:
            result["body"] = "corrupted"
        return result


def report(mode="weekly", week="2026-10-12", date="2026-10-14"):
    return {"campaign": "abc", "week": week, "date": date, "mode": mode, "device": {"arch": "gfx950"}}


def test_publication_rotation_and_daily_upsert():
    api = FakeGitHub()
    previous = publish(api, report(week="2026-10-05"), "old week")
    current = publish(api, report(), "baseline")
    assert api.issues[previous["number"]]["state"] == "closed"
    # Verification of the new issue must precede closure of the previous one.
    assert api.events.index(("GET", "issues/2")) < api.events.index(("PATCH", "issues/1"))
    publish(api, report(), "baseline retry")
    assert len(api.issues) == 2
    publish(api, report(mode="nightly"), "day one")
    publish(api, report(mode="nightly"), "day one retry")
    daily = [c for c in api.comments.values() if "date=2026-10-14" in c["body"]]
    assert len(daily) == 1 and "day one retry" in daily[0]["body"]
    assert current["number"] == 2


def test_failed_publication_verification_preserves_previous_week():
    api = FakeGitHub()
    previous = publish(api, report(week="2026-10-05"), "old week")
    api.corrupt_verification = True
    with pytest.raises(RuntimeError, match="not verified"):
        publish(api, report(), "new week")
    assert api.issues[previous["number"]]["state"] == "open"


def test_nightly_requires_owned_open_weekly_issue():
    with pytest.raises(RuntimeError, match="weekly baseline"):
        publish(FakeGitHub(), report(mode="nightly"), "night")


@pytest.mark.skipif(os.environ.get("SCOUT_TEST_GPU") != "1", reason="set SCOUT_TEST_GPU=1 for source-built GPU smoke")
def test_gpu_same_inputs_and_mutation_rejected(tmp_path, target):
    # Run through the actual worker, not a mock timer.
    case = {**target["cases"][0], "id": "small", "q_lengths": [33, 65], "kv_lengths": [33, 65]}
    timing = {**target["timing"], "warmup_ms": 5, "rep_ms": 10, "repeats": 3}
    target = {**target, "timing": timing}
    request = {"backend": "tlx_ops", "target": target, "case": case, "candidate": {"id": "public_api", "params": {}}}
    path, output = tmp_path / "request.json", tmp_path / "out.json"
    write_json(path, request)
    subprocess.run([sys.executable, str(ROOT / "benchmarking/worker.py"), str(path), str(output)], check=True)
    result = json.loads(output.read_text())
    assert result["status"] in {"ok", "unstable"}
    assert result["correctness_seeds"] == [17, 29]
    assert len(result["samples_ms"]) == 3
    bad = {
        "status":
        "complete", "summary":
        "mutation", "candidates": [{
            "id":
            "bad", "params": {}, "rationale":
            "negative test", "adapter_source":
            "def build(case, params):\n    def run(x):\n        x['q'].zero_()\n        return x['q']\n    return run"
        }]
    }
    request["candidate"] = validate_proposals(bad, tmp_path, 1)[0]
    write_json(path, request)
    failed = subprocess.run([sys.executable, str(ROOT / "benchmarking/worker.py"), str(path), str(output)], check=False)
    assert failed.returncode == 1
    assert json.loads(output.read_text())["status"] == "failed"


def test_blocked_discovery_is_not_unsupported(tmp_path):
    response = {"status": "blocked", "summary": "sandbox denied source access", "candidates": []}
    assert validate_proposals(response, tmp_path, 2) == []
    response["candidates"] = [{"id": "unverified"}]
    with pytest.raises(ValueError, match="blocked"):
        validate_proposals(response, tmp_path, 2)


def test_controller_freezes_controls_and_preserves_venv(tmp_path, monkeypatch, target):
    import argparse

    import appctl

    target_path = tmp_path / "target.json"
    write_json(target_path, target)
    device = {"vendor": "amd", "arch": "gfx950", "physical_gpu": 0, "name": "GPU"}
    identity = {
        "device": device, "runtime": {"compiler_sha256": "compiler", "torch": "test"}, "aiter":
        {"status": "unavailable"}
    }
    identity["runtime"]["triton_module"] = str(appctl.REPOSITORY / "python/triton/__init__.py")
    monkeypatch.setattr(appctl, "probe", lambda *args: identity)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)
    calls = []

    def fake_scout(backend, t, case, directory, runtime, args):
        assert runtime["python"] == str(venv_python)  # Do not resolve the venv's interpreter symlink.
        calls.append(backend)
        directory.mkdir(parents=True)
        request = directory / "request.json"
        write_json(request, {"expected_identity": identity})
        return {
            "status": "ok", "latency_ms": 1, "p20_ms": 0.9, "p80_ms": 1.1, "tflops": 1, "logical_tbps": 1, "runtime":
            identity["runtime"], "backend_pin": {"git_sha": "source"}, "candidate": {"id": "seed"}, "reproducer":
            str(request), "request_sha256": digest(json.loads(request.read_text())), "reproducer_command": "command"
        }

    monkeypatch.setattr(appctl, "scout", fake_scout)
    args = argparse.Namespace(targets=[target_path], case=["one_x_1000"], state=tmp_path / "state",
                              python=str(venv_python), pythonpath="/source", runtimes=None, date="2026-10-14",
                              execute=True, mode="weekly", gpu=0, provider="builtin", max_candidates=3, rounds=0,
                              agent_timeout=10, worker_timeout=10, report_pointer=tmp_path / "pointer.json")
    appctl.run(args)
    assert calls == ["pytorch", "tlx_ops", "tlx_tutorials"]
    calls.clear()
    args.mode, args.date = "nightly", "2026-10-15"
    appctl.run(args)
    assert calls == ["tlx_ops", "tlx_tutorials"]
    report_path = json.loads(args.report_pointer.read_text())["report"]
    nightly = json.loads(Path(report_path).read_text())
    controls = nightly["rows"][0]["results"]
    assert controls["pytorch"]["frozen_date"] == controls["tlx_ops_baseline"]["frozen_date"] == "2026-10-14"
    args.mode = "weekly"
    with pytest.raises(ValueError, match="immutable"):
        appctl.run(args)
    args.mode, args.case = "nightly", ["ten_x_1000"]
    with pytest.raises(FileNotFoundError):
        appctl.run(args)


def test_report_escapes_table_cells():
    from reporting import render
    r = report()
    r.update(
        device={"name": "GPU",
                "arch": "gfx950", "physical_gpu": 0}, provider="builtin", columns=["pytorch"], budget={}, rows=[{
                    "description":
                    "shape | value", "case": {"id":
                                              "one"}, "target": {"metric": "compute", "timing": {"tie_fraction": 0.03}},
                    "results": {"pytorch": {"status": "failed", "error": "a | b\nline"}}
                }])
    text = render(r)
    assert "shape \\| value" in text and "a \\| b line" in text
    assert "| none |" in text


def test_weekly_publication_does_not_close_future_or_unrelated_issue():
    api = FakeGitHub()
    future = publish(api, report(week="2026-10-19"), "future")
    api.issues[2] = {"number": 2, "state": "open", "body": "ordinary user issue"}
    publish(api, report(), "older")
    assert api.issues[future["number"]]["state"] == "open"
    assert api.issues[2]["state"] == "open"


def test_timeout_terminates_worker(tmp_path):
    from appctl import execute
    with pytest.raises(RuntimeError, match="exceeded"):
        execute([sys.executable, "-c", "import time; time.sleep(10)"], env=os.environ.copy(),
                log=tmp_path / "worker.log", timeout=0.1)


def test_kfd_ownership_finds_queues_and_vram(tmp_path):
    from benchmarking.environment import kfd_owners
    for pid in (101, 102, 103):
        (tmp_path / str(pid)).mkdir()
    queue = tmp_path / "101/queues/0"
    queue.mkdir(parents=True)
    (queue / "gpuid").write_text("999")
    (tmp_path / "102/vram_999").write_text("4096")
    (tmp_path / "103/vram_999").write_text("0")
    assert kfd_owners(tmp_path, 999) == {101, 102}
    assert kfd_owners(tmp_path, 111) == set()


def test_scout_confirms_selected_candidate_and_records_blocked_search(tmp_path, monkeypatch, target):
    from types import SimpleNamespace

    import appctl
    args = SimpleNamespace(provider="codex", max_candidates=3, rounds=1, model=None, context=None, agent_timeout=5)
    requests = []

    def evaluate(backend, target, case, candidate, directory, runtime, args):
        requests.append((candidate["id"], directory.name))
        return {"status": "ok", "latency_ms": 1, "candidate": candidate}

    monkeypatch.setattr(appctl, "evaluate", evaluate)
    monkeypatch.setattr(
        appctl, "discover", lambda *a, **kw:
        ({"status": "blocked", "summary": "source tools denied", "candidates": []}, []))
    result = appctl.scout("tlx_ops", target, target["cases"][0], tmp_path / "ops", {}, args)
    assert requests == [("public_api", "r0_public_api"), ("public_api", "confirmation")]
    assert result["discovery_status"] == "blocked"
    assert result["status"] == "ok"  # Measured seed remains valid, search coverage is partial.
    result = appctl.scout("tlx_tutorials", target, target["cases"][0], tmp_path / "tutorials", {}, args)
    assert result["status"] == "blocked"


def test_deployment_uses_argument_lists_and_success_pointer(tmp_path):
    spec = importlib.util.spec_from_file_location("scout_deploy", ROOT / "deploy/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {
        "python": "/venv/bin/python", "targets": ["/source/target with spaces.json"], "state": "/state with spaces",
        "provider": "claude", "model": "model", "rounds": 1
    }
    argv = module.commands(config, "nightly", tmp_path / "pointer.json")
    assert "/source/target with spaces.json" in argv
    assert argv[argv.index("--state") + 1] == "/state with spaces"
    assert argv[argv.index("--report-pointer") + 1] == str(tmp_path / "pointer.json")


def test_pci_guard_rejects_wrong_physical_gpu():
    from types import SimpleNamespace

    from benchmarking.environment import check_pci_device
    props = SimpleNamespace(pci_domain_id=0, pci_bus_id=0x66, pci_device_id=0)
    check_pci_device(props, "0000:66:00.0")
    with pytest.raises(RuntimeError, match="visibility is inconsistent"):
        check_pci_device(props, "0000:16:00.0")
    with pytest.raises(RuntimeError, match="visibility is inconsistent"):
        check_pci_device(SimpleNamespace(), "0000:66:00.0")


@pytest.mark.parametrize("status", ["failed", "unstable", "unavailable", "unsupported"])
def test_scout_keeps_failure_categories(status, tmp_path, monkeypatch, target):
    from types import SimpleNamespace

    import appctl
    monkeypatch.setattr(appctl, "evaluate", lambda *args: {"status": status, "error": "diagnostic"})
    args = SimpleNamespace(provider="builtin", max_candidates=1, rounds=0)
    result = appctl.scout("tlx_ops", target, target["cases"][0], tmp_path / "ops", {}, args)
    assert result["status"] == status and result["error"] == "diagnostic"
