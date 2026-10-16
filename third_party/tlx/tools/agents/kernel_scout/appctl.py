"""Controller for bounded discovery, fair measurements, and weekly/nightly reports."""

import argparse
import contextlib
import datetime as dt
import fcntl
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from backends.seeds import candidates
from benchmarking.environment import GPUOccupiedError, rocr_device
from contracts import (
    OPERATIONS,
    REPOSITORY,
    ROOT,
    applicable,
    backends,
    digest,
    enforce_equal_inputs,
    harness_digest,
    load_target,
    read_json,
    write_json,
)
from discovery import DiscoveryCache, discover, validate_proposals
from gpu_scheduler import wait_for_gpu
from publication import GitHub, publish
from registration import registration_result
from reporting import describe, render, reproduce_command
from target_templates import create_target
from worker_pool import WorkerPool


def execute(command, *, env, log, timeout):
    """Bound the worker and its descendants, and preserve diagnostics outside source."""
    with Path(log).open("w") as output, subprocess.Popen(command, env=env, cwd=REPOSITORY, stdout=output,
                                                         stderr=subprocess.STDOUT, start_new_session=True) as process:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError(f"worker exceeded {timeout}s; see {log}")


def worker_environment(gpu, runtime):
    env = os.environ.copy()
    env["SCOUT_PHYSICAL_GPU"] = str(gpu)
    if runtime.get("vendor", "amd") == "amd":
        # AMD SMI and HSA can enumerate devices in different orders.
        env.update({"ROCR_VISIBLE_DEVICES": rocr_device(gpu), "HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"})
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.pop("ROCR_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
    env.pop("GPU_DEVICE_ORDINAL", None)
    if runtime.get("python"):
        env["PATH"] = str(Path(runtime["python"]).absolute().parent) + os.pathsep + env.get("PATH", "")
    if runtime.get("pythonpath"):
        env["PYTHONPATH"] = runtime["pythonpath"]
    return env


@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another scout owns {path}") from None
        yield


def probe(runtime, gpu, directory):
    output, log = directory / "device.json", directory / "probe.log"
    output.unlink(missing_ok=True)
    command = [runtime["python"], str(ROOT / "benchmarking/worker.py"), "--probe", str(output)]
    code = execute(command, env=worker_environment(gpu, runtime), log=log, timeout=120)
    if code:
        if output.is_file():
            failure = read_json(output)
            if failure.get("status") == "busy":
                raise GPUOccupiedError(failure["error"])
        raise RuntimeError(f"GPU probe failed; see {log}")
    return read_json(output)


def admitted_probe(runtime, gpu, directory, args, deadline):
    return wait_for_gpu(gpu, runtime["vendor"], wait_seconds=max(0, deadline - time.monotonic()),
                        poll_seconds=getattr(args, "gpu_poll_seconds", 60),
                        status_path=directory / "gpu_admission.json", ready=lambda: probe(runtime, gpu, directory))


def baseline_key(targets, device, runtime, *, execution="isolated", screening=False, finalists=2):
    # TLX source/compiler revisions intentionally evolve during the week.
    # The physical device, target, PyTorch, driver, and timing contract must match.
    return digest({
        "targets": targets, "device": device, "harness_sha256": harness_digest(), "execution": execution, "screening":
        screening, "finalists": finalists, "runtime": {
            k: runtime.get(k)
            for k in ("torch", "torch_git", "torch_num_threads", "hip", "cuda", "python", "timing_sha256",
                      "performance_environment")
        }
    })


def validate_baseline(baseline, *, campaign, week, key):
    if baseline["campaign"] != campaign or baseline["week"] != week or baseline["baseline_key"] != key:
        raise ValueError("weekly baseline does not match this campaign, week, hardware, runtime, or target contract")
    for row in baseline["rows"]:
        for result in row["results"].values():
            if result.get("status") == "ok":
                path = Path(result["reproducer"])
                if not path.is_file() or digest(read_json(path)) != result["request_sha256"]:
                    raise ValueError("weekly baseline reproducer is missing or changed")


def resume_progress(path, report, targets):
    """Reuse only a complete case prefix with identical source, runtime, and search pins."""
    previous = read_json(path)
    if previous.get("complete") is not False or previous.get("resume_key") != report["resume_key"]:
        raise ValueError("resume requires an incomplete report with identical source, runtime, and search pins")
    for key in ("mode", "date", "campaign", "baseline_key"):
        if previous[key] != report[key]:
            raise ValueError(f"resume report differs in {key}")
    expected = [(target, case) for target in targets for case in target["cases"]]
    rows = previous["rows"]
    if len(rows) > len(expected):
        raise ValueError("resume contains more cases than this campaign")
    for row, (target, case) in zip(rows, expected):
        if row["target"] != target or row["case"] != case or set(row["results"]) != set(report["columns"]):
            raise ValueError("resume rows must be a complete ordered case prefix")
        if any(
                r.get("failure_kind") == "infrastructure" or "GPUOccupiedError" in r.get("error", "")
                for r in row["results"].values()):
            raise ValueError("resume contains an infrastructure failure")
        enforce_equal_inputs(row["results"])
    validate_baseline(previous, campaign=report["campaign"], week=report["week"], key=report["baseline_key"])
    started = dt.datetime.fromisoformat(previous["started_utc"])
    if started.tzinfo is None:
        raise ValueError("resume start time must include its timezone")
    report.update(rows=rows, started_utc=previous["started_utc"], resume_from=str(Path(path).resolve()),
                  resumed_cases=len(rows))
    return max(0, (dt.datetime.now(dt.timezone.utc) - started).total_seconds())


def evaluate(backend, target, case, candidate, directory, runtime, args, *, pool=None, timing_phase="full"):
    started = time.monotonic()
    runtime = {**runtime, "vendor": target["hardware"]["vendor"]}
    directory.mkdir(parents=True, exist_ok=True)
    request_path, output = directory / "request.json", directory / "measurement.json"
    request = {
        "backend": backend, "target": target, "case": case, "candidate": candidate, "repository": str(REPOSITORY),
        "worker_runtime": runtime, "physical_gpu": args.gpu, "timing_phase": timing_phase
    }
    if pool is not None:
        request["input_cache"] = str(directory.parents[3] / "inputs" / target["id"] / case["id"])
    write_json(request_path, request)
    try:
        if pool is None:
            execute([runtime["python"],
                     str(ROOT / "benchmarking/worker.py"),
                     str(request_path),
                     str(output)], env=worker_environment(args.gpu, runtime), log=directory / "worker.log",
                    timeout=args.worker_timeout)
        else:
            pool.evaluate(runtime, args.gpu, request_path, output, directory / "worker.log")
        result = read_json(output)
    except RuntimeError as error:
        if pool is not None:
            raise
        result = {"status": "failed", "error": str(error)}
    if result.get("failure_kind") == "infrastructure":
        raise RuntimeError(result["error"])
    if result["status"] == "ok":
        request["expected_identity"] = {
            key: result[key]
            for key in ("device", "runtime", "backend_pin", "harness_sha256")
        }
        write_json(request_path, request)
        result.update({
            "reproducer": str(request_path), "request_sha256": digest(request), "reproducer_command":
            reproduce_command(runtime["python"], request_path)
        })
    result["worker_seconds"] = time.monotonic() - started
    return result


def scout(backend, target, case, directory, runtime, args, *, research=None, cache=None, pool=None):
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    search = backend != "aiter" or getattr(args, "aiter_search", False)
    scope = getattr(args, "discovery", "target")
    stats = {"scope": scope, "agent_calls": 0, "discovery_seconds": 0, "cache_hit": False}
    extra_measurements = []

    def finish(result):
        measured = [f["result"] for f in feedback] + extra_measurements + [result]
        result["search"] = {
            **stats, "elapsed_seconds": time.monotonic() - started, "worker_seconds":
            sum(m.get("worker_seconds", 0) for m in measured), "phase_seconds": {
                key: sum(m.get("phase_seconds", {}).get(key, 0)
                         for m in measured)
                for key in ("setup", "preparation", "correctness", "compiler_cleanup", "benchmark_and_checks")
            }
        }
        return result

    seeds = candidates(backend, target["operation"], search=search)
    feedback = []
    discovery_status, discovery_summary = "not_requested", ""
    proposals = seeds[:args.max_candidates]
    remaining = args.max_candidates - len(proposals)
    cache_key = cache.key(backend, target, research) if cache is not None and search else None
    stats["cache_key"] = cache_key
    saved = cache.load(cache_key) if cache_key else None
    if saved is not None:
        stats["cache_hit"] = True
        discovery_status, discovery_summary = saved["status"], saved["summary"]
        cached_dir = directory / "cached"
        cached_dir.mkdir()
        response = {**saved, "candidates": saved["candidates"][:remaining]}
        proposals += validate_proposals(response, cached_dir, remaining)
        remaining = 0
    generated = []
    evaluation_options = {"pool": pool} if pool is not None else {}
    screening = getattr(args, "screen_candidates", False) and len(proposals) > 1
    if screening:
        evaluation_options["timing_phase"] = "screen"
    for round_number in range(args.rounds + 1):
        for candidate in proposals:
            result = evaluate(backend, target, case, candidate, directory / f"r{round_number}_{candidate['id']}",
                              runtime, args, **evaluation_options)
            feedback.append({"candidate": candidate, "result": result})
        if not search or scope == "off" or args.provider == "builtin" or not remaining or round_number == args.rounds:
            break
        proposal_dir = directory / f"discovery_r{round_number}"
        proposal_dir.mkdir()
        discovery_started = time.monotonic()
        print(f"{target['id']}/{case['id']}: {backend} researching ({scope})", flush=True)
        response, proposals = discover(args.provider, backend, target, case, feedback, proposal_dir, remaining,
                                       model=args.model, context=args.context, timeout=args.agent_timeout,
                                       research=research, scope=scope)
        stats["agent_calls"] += 1
        stats["discovery_seconds"] += time.monotonic() - discovery_started
        generated.extend(proposals)
        write_json(directory / f"discovery_{round_number}.json", response)
        discovery_status, discovery_summary = response["status"], response["summary"]
        remaining -= len(proposals)
        if not proposals:
            break
    if cache_key and stats["agent_calls"]:
        cache.save(cache_key, discovery_status, discovery_summary, generated)
    write_json(directory / "search.json", feedback)
    selection = feedback
    if screening:
        eligible = [f for f in feedback if f["result"]["status"] in {"ok", "unstable"}]
        finalists = sorted(eligible, key=lambda f: f["result"]["latency_ms"])[:getattr(args, "finalists", 2)]
        selection = []
        for finalist in finalists:
            candidate = finalist["candidate"]
            result = evaluate(backend, target, case, candidate, directory / f"final_{candidate['id']}", runtime, args,
                              **({"pool": pool} if pool is not None else {}))
            extra_measurements.append(result)
            selection.append({"candidate": candidate, "result": result})
        write_json(directory / "finalists.json", selection)
        stats.update(screened_candidates=len(feedback), full_candidates=len(selection))
    valid = [f for f in selection if f["result"]["status"] == "ok"]
    if not valid:
        if discovery_status == "blocked":
            return finish({"status": "blocked", "error": discovery_summary, "evaluated_candidates": len(feedback)})
        if not feedback:
            return finish(
                {"status": "unsupported", "error": "no seed adapter; enable an agent to research this backend"})
        statuses = {f["result"]["status"] for f in (selection or feedback)}
        status = next(s for s in ("failed", "unstable", "unavailable", "unsupported") if s in statuses)
        return finish({
            "status": status, "error": (selection or feedback)[0]["result"].get("error", "no stable correct candidate"),
            "evaluated_candidates": len(feedback)
        })
    best = min(valid, key=lambda f: f["result"]["latency_ms"])
    # Independent remeasurement after selection reduces tuning-selection bias.
    winner = evaluate(backend, target, case, best["candidate"], directory / "confirmation", runtime, args,
                      **({"pool": pool} if pool is not None else {}))
    winner["evaluated_candidates"] = len(feedback)
    winner["search_manifest"] = str(directory / "search.json")
    winner["discovery_status"] = discovery_status
    winner["discovery_summary"] = discovery_summary
    return finish(winner)


def run(args):
    job_started = time.monotonic()
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    aiter_search = getattr(args, "aiter_search", False)
    targets = [load_target(path) for path in args.targets]
    if args.case:
        targets = [{**t, "cases": [c for c in t["cases"] if c["id"] in args.case]} for t in targets]
        targets = [t for t in targets if t["cases"]]
        if not targets:
            raise ValueError("case filter selected no cases")
    if len({t["id"] for t in targets}) != len(targets):
        raise ValueError("target ids must be unique")
    # Resolving the executable's symlink would discard the virtual environment.
    runtime = {
        "python": os.path.abspath(args.python), "pythonpath": args.pythonpath, "vendor":
        targets[0]["hardware"]["vendor"]
    }
    runtimes = read_json(args.runtimes) if args.runtimes else {}
    date = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(dt.timezone.utc).date()
    week = (date - dt.timedelta(days=date.weekday())).isoformat()
    args.state = args.state.resolve()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = args.state / "runs" / f"{date}_{args.mode}_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    registry = backends()
    if not args.execute:
        print({
            "mode": args.mode, "targets": [t["id"] for t in targets], "backends": list(registry), "gpu": args.gpu,
            "provider": args.provider, "discovery": getattr(args, "discovery", "target"), "directory": str(directory),
            "note": "pass --execute to discover and measure; publication is a separate command"
        })
        return
    # One lock per physical GPU across state directories used by this user.
    gpu_lock = Path.home() / ".cache/tlx-kernel-scout" / f"gpu-{args.gpu}.lock"
    with locked(gpu_lock), locked(args.state / "controller.lock"), contextlib.ExitStack() as stack:
        execution_mode = getattr(args, "execution", "isolated")
        pool = stack.enter_context(WorkerPool(directory / "workers", worker_environment, args.worker_timeout)) \
            if execution_mode == "persistent" else None
        admission_deadline = time.monotonic() + getattr(args, "gpu_wait_seconds", 1800)
        identity = admitted_probe(runtime, args.gpu, directory, args, admission_deadline)
        if not Path(identity["runtime"]["triton_module"]).is_relative_to(REPOSITORY / "python"):
            raise ValueError("base runtime must import source TLX from this checkout; check PYTHONPATH")
        device = identity["device"]
        for target in targets:
            hardware = target["hardware"]
            if device["vendor"] != hardware["vendor"] or device["arch"] not in hardware["architectures"]:
                raise ValueError(f"target {target['id']} does not apply to {device['arch']}")
        campaign = digest({"arch": device["arch"], "targets": targets})[:16]
        override_pins = {}
        for backend, override in runtimes.items():
            runtime_dir = directory / f"probe_{backend}"
            runtime_dir.mkdir()
            pin = admitted_probe({**runtime, **override}, args.gpu, runtime_dir, args, admission_deadline)
            if pin["device"] != device:
                raise ValueError(f"{backend} runtime selected a different device")
            override_pins[backend] = pin
        if pool is not None:
            pool.register(runtime, args.gpu, identity)
            for backend, override in runtimes.items():
                pool.register({**runtime, **override}, args.gpu, override_pins[backend])
        key = digest({
            "contract":
            baseline_key(targets, device, identity["runtime"], execution=execution_mode,
                         screening=getattr(args, "screen_candidates", False), finalists=getattr(args, "finalists", 2)),
            "runtimes":
            runtimes, "registry":
            registry, "aiter":
            identity["aiter"], "override_pins":
            override_pins, "aiter_search":
            aiter_search
        })
        baseline_path = args.state / "baselines" / campaign / f"{week}.json"
        controls = {}
        if args.mode == "nightly":
            baseline = read_json(baseline_path)
            validate_baseline(baseline, campaign=campaign, week=week, key=key)
            controls = {(row["target"]["id"], row["case"]["id"]): row for row in baseline["rows"]}
        elif args.mode == "weekly" and baseline_path.exists():
            raise ValueError("this week's immutable baseline already exists; use a new state directory to rerun")
        report = {
            "schema_version": 1, "complete": False, "started_utc": started_utc, "mode": args.mode, "date":
            date.isoformat(), "week": week, "campaign": campaign, "baseline_key": key, "device": device, "provider":
            args.provider, "aiter_mode": "search" if aiter_search else "public", "budget": {
                "max_candidates": args.max_candidates, "rounds": args.rounds, "agent_timeout_s": args.agent_timeout,
                "worker_timeout_s": args.worker_timeout
            }, "columns": list(registry) + ["tlx_ops_baseline"], "rows": []
        }
        discovery_scope = getattr(args, "discovery", "target")
        cache = None if discovery_scope == "case" else DiscoveryCache(
            getattr(args, "candidate_cache", None) or args.state / "candidates", refresh=getattr(
                args, "refresh_discovery", False))
        report["discovery_scope"] = discovery_scope
        report["execution_mode"] = execution_mode
        report["screen_candidates"] = getattr(args, "screen_candidates", False)
        report["finalists"] = getattr(args, "finalists", 2)
        run_started = job_started
        context = getattr(args, "context", None)
        context_hash = digest(context.read_text()) if context else None
        preferred = [name for name in ("tlx_ops", "tlx_tutorials", "aiter", "pytorch") if name in registry]
        report["columns"] = preferred + [name for name in registry if name not in preferred] + ["tlx_ops_baseline"]
        if args.mode == "registration":
            report["columns"].remove("tlx_ops_baseline")
        report["resume_key"] = digest({
            "identity": identity, "override_pins": override_pins, "runtime": runtime, "baseline":
            digest(baseline) if args.mode == "nightly" else None, "controller": digest(Path(__file__).read_text()),
            "pool": digest((ROOT / "worker_pool.py").read_text()), "budget": report["budget"], "provider":
            args.provider, "model": getattr(args, "model", None), "context": context_hash, "discovery": discovery_scope,
            "refresh_discovery": getattr(args, "refresh_discovery", False)
        })
        if getattr(args, "resume_from", None):
            elapsed = resume_progress(args.resume_from, report, targets)
            run_started = time.monotonic() - elapsed
            print(f"Resuming after {len(report['rows'])} complete cases", flush=True)
        completed = {(row["target"]["id"], row["case"]["id"]) for row in report["rows"]}
        write_json(directory / "progress.json", report)
        for target in targets:
            for case in target["cases"]:
                if (target["id"], case["id"]) in completed:
                    continue
                if args.case and case["id"] not in args.case:
                    continue
                row = {"target": target, "case": case, "description": describe(target, case), "results": {}}
                for backend, contract in registry.items():
                    backend_identity = override_pins.get(backend, identity)
                    if args.mode == "nightly" and contract.get("cadence", "weekly") == "weekly":
                        result = dict(controls[(target["id"], case["id"])]["results"][backend])
                        result["frozen_date"] = baseline["date"]
                    elif not applicable(contract, target, device):
                        result = {
                            "status": "unsupported", "error": "backend hardware/operation contract excludes this target"
                        }
                    elif backend == "aiter" and backend_identity["aiter"]["status"] == "unavailable":
                        result = {
                            "status": "unavailable", "error":
                            backend_identity["aiter"].get("error", "AITER is not installed in this runtime")
                        }
                    else:
                        backend_runtime = {**runtime, **runtimes.get(backend, {})}
                        research = {
                            "runtime": backend_identity["runtime"], "worker_runtime": backend_runtime, "device": device,
                            "context_sha256": context_hash
                        }
                        if backend == "aiter":
                            research["package"] = backend_identity["aiter"]
                        result = scout(backend, target, case, directory / target["id"] / case["id"] / backend,
                                       backend_runtime, args, research=research, cache=cache, pool=pool)
                    row["results"][backend] = result
                    if result["status"] == "ok" and result.get("timing_phase") == "screen":
                        raise ValueError("screening timings cannot become report results")
                    search_stats = result.get("search", {})
                    print(
                        f"{target['id']}/{case['id']}: {backend} {result['status']} "
                        f"({result.get('evaluated_candidates', 0)} candidates, "
                        f"{search_stats.get('agent_calls', 0)} agent calls, "
                        f"{search_stats.get('elapsed_seconds', 0):.1f}s)", flush=True)
                enforce_equal_inputs(row["results"])
                if args.mode == "nightly":
                    row["results"]["tlx_ops_baseline"] = dict(controls[(target["id"],
                                                                        case["id"])]["results"]["tlx_ops"])
                elif args.mode == "weekly":
                    row["results"]["tlx_ops_baseline"] = dict(row["results"]["tlx_ops"])
                if args.mode != "registration":
                    row["results"]["tlx_ops_baseline"]["frozen_date"] = date.isoformat(
                    ) if args.mode == "weekly" else baseline["date"]
                report["rows"].append(row)
                print(f"measured {target['id']} / {case['id']}", flush=True)
                # Preserve completed rows before the final report and immutable baseline exist.
                report["elapsed_seconds"] = time.monotonic() - run_started
                write_json(directory / "progress.json", report)
                if pool is not None:
                    shutil.rmtree(directory / "inputs" / target["id"] / case["id"], ignore_errors=True)
        if not report["rows"]:
            raise ValueError("case filter selected no cases")
        if pool is not None:
            pool.verify()
        if args.mode == "registration":
            readiness = registration_result(report)
            write_json(directory / "registration.json", readiness)
            report["registration_ready"] = readiness["ready"]
        missing = [
            row["case"]["id"]
            for row in report["rows"]
            if not any(r["status"] == "ok" for backend, r in row["results"].items() if backend != "tlx_ops_baseline")
        ]
        if missing and args.mode != "registration":
            raise RuntimeError(f"cases have no stable correct backend measurement: {missing}")
        report["elapsed_seconds"] = time.monotonic() - run_started
        report["worker_processes"] = len(pool.workers) if pool is not None else None
        report["complete"] = True
        write_json(directory / "progress.json", report)
        write_json(directory / "report.json", report)
        (directory / "report.md").write_text(render(report, directory / "report.json"))
        if args.mode == "weekly":
            write_json(baseline_path, report)
        if args.report_pointer:
            write_json(args.report_pointer, {"report": str(directory / "report.json")})
        print(directory / "report.md")
        if args.mode == "registration" and not report["registration_ready"]:
            raise RuntimeError(f"target registration needs attention: {directory / 'registration.json'}")


def reproduce(args):
    if getattr(args, "report", None):
        report = read_json(args.report)
        if args.row is None or not 1 <= args.row <= len(report["rows"]) or not args.backend:
            raise ValueError("select a valid report row and backend")
        result = report["rows"][args.row - 1]["results"].get(args.backend, {})
        if not result.get("reproducer") or not result.get("request_sha256"):
            raise ValueError("selected result has no verified reproducer")
        args.request = Path(result["reproducer"])
        if digest(read_json(args.request)) != result["request_sha256"]:
            raise ValueError("saved reproducer differs from the report")
    request = read_json(args.request)
    if not request.get("expected_identity"):
        raise ValueError("request is not a verified reproducer")
    runtime = request["worker_runtime"]
    output = args.request.with_name("reproduced.json")
    with locked(Path.home() / ".cache/tlx-kernel-scout" / f"gpu-{request['physical_gpu']}.lock"):
        code = execute(
            [runtime["python"], str(ROOT / "benchmarking/worker.py"),
             str(args.request), str(output)], env=worker_environment(request["physical_gpu"], runtime),
            log=output.with_suffix(".log"), timeout=args.timeout)
    print(output)
    if code:
        raise RuntimeError("reproducer failed; inspect reproduced.json and log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("targets", nargs="+", type=Path)
    initialize = sub.add_parser("target-init", help="create a validated starter target for an existing operation")
    initialize.add_argument("--id", required=True)
    initialize.add_argument("--operation", choices=sorted(OPERATIONS), required=True)
    initialize.add_argument("--vendor", choices=["amd", "nvidia"], required=True)
    initialize.add_argument("--arch", required=True)
    initialize.add_argument("--metric", choices=["compute", "memory"], default="compute")
    initialize.add_argument("--output", type=Path, required=True)
    measure = sub.add_parser("run")
    measure.add_argument("--mode", choices=["registration", "weekly", "nightly"], required=True,
                         help="check target readiness, freeze weekly controls, or compare nightly TLX")
    measure.add_argument("--targets", nargs="+", type=Path, required=True)
    measure.add_argument("--state", type=Path, required=True)
    measure.add_argument("--python", default=sys.executable)
    measure.add_argument("--pythonpath", default=os.environ.get("PYTHONPATH", ""))
    measure.add_argument("--runtimes", type=Path, help="JSON backend -> {python, pythonpath} overrides")
    measure.add_argument("--report-pointer", type=Path, help="write the completed report path for deployment")
    measure.add_argument("--resume-from", type=Path,
                         help="retain complete cases from a matching partial progress.json after interruption")
    measure.add_argument("--provider", choices=["builtin", "codex", "claude"], default="builtin")
    measure.add_argument("--execution", choices=["isolated", "persistent"], default="isolated",
                         help="use fresh processes or serialized runtime workers with shared case preparation")
    measure.add_argument("--screen-candidates", action="store_true",
                         help="select finalists with short timing before full measurement")
    measure.add_argument("--finalists", type=int, default=2,
                         help="number of screened variants to measure with the full timing contract")
    measure.add_argument("--discovery", choices=["target", "case", "off"], default="target",
                         help="research once per target/backend, per case, or use only seeds and saved adapters")
    measure.add_argument("--candidate-cache", type=Path, help="adapter cache directory, defaults to STATE/candidates")
    measure.add_argument("--refresh-discovery", action="store_true",
                         help="replace saved discovery once per target/backend")
    measure.add_argument("--aiter-search", action="store_true", help="search AITER variants instead of public ops only")
    measure.add_argument("--model")
    measure.add_argument("--context", type=Path, help="explicit Megabrain context pack for discovery agents")
    measure.add_argument("--gpu", type=int, default=0)
    measure.add_argument("--gpu-wait-seconds", type=float, default=1800, help="maximum startup wait for a busy GPU")
    measure.add_argument("--gpu-poll-seconds", type=float, default=60, help="interval between GPU ownership checks")
    measure.add_argument("--date", help="ISO UTC report date, defaults to today")
    measure.add_argument("--case", action="append")
    measure.add_argument("--max-candidates", type=int, default=6)
    measure.add_argument("--rounds", type=int, default=2)
    measure.add_argument("--agent-timeout", type=int, default=300)
    measure.add_argument("--worker-timeout", type=int, default=600)
    measure.add_argument("--execute", action="store_true")
    rep = sub.add_parser("reproduce")
    source = rep.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path)
    source.add_argument("--report", type=Path, help="select a saved request from a report")
    rep.add_argument("--row", type=int, help="one-based report row, used with --report")
    rep.add_argument("--backend", help="report backend column, used with --report")
    rep.add_argument("--timeout", type=int, default=600)
    publication = sub.add_parser("publish")
    publication.add_argument("--report", type=Path, required=True)
    publication.add_argument("--repo", required=True)
    publication.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "target-init":
        create_target(args.output, name=args.id, operation=args.operation, vendor=args.vendor, architecture=args.arch,
                      metric=args.metric)
        print(args.output)
    elif args.command == "validate":
        for path in args.targets:
            print(load_target(path)["id"])
        print("backends:", ", ".join(backends()))
    elif args.command == "run":
        if args.finalists < 1:
            parser.error("finalists must be positive")
        if args.refresh_discovery and args.discovery != "target":
            parser.error("refresh discovery requires target discovery")
        if args.refresh_discovery and (args.provider == "builtin" or args.rounds == 0):
            parser.error("refresh discovery requires an agent provider and at least one research round")
        if not math.isfinite(args.gpu_wait_seconds) or args.gpu_wait_seconds < 0:
            parser.error("GPU wait seconds must be finite and nonnegative")
        if not math.isfinite(args.gpu_poll_seconds) or args.gpu_poll_seconds <= 0:
            parser.error("GPU poll seconds must be finite and positive")
        if args.max_candidates < 1 or args.rounds < 0 or min(args.agent_timeout,
                                                             args.worker_timeout) <= 0 or args.gpu < 0:
            parser.error("budgets must be positive; rounds and GPU must be nonnegative")
        run(args)
    elif args.command == "reproduce":
        if args.request and (args.row is not None or args.backend is not None):
            parser.error("row and backend require --report")
        reproduce(args)
    else:
        report = read_json(args.report)
        markdown = render(report, args.report.resolve())
        if args.execute:
            print(publish(GitHub(args.repo), report, markdown)["html_url"])
        else:
            print(f"Preview for {args.repo}:\n{markdown}")


if __name__ == "__main__":
    main()
