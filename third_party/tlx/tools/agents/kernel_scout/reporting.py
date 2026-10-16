"""Render measured evidence, frozen control dates, and pinned reproducer commands."""

import shlex
import sys
from collections import Counter

from contracts import GEMM_OPERATIONS, ROOT, attention_scale, select_winner


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def describe(target, case):
    prefix = f"{target['id']}/{case['id']}, {case['dtype']}"
    if target["operation"] in GEMM_OPERATIONS:
        batch = f", batch={case['batch']} shared_A={case['shared_a']}" if target["operation"] == "bmm" else ""
        bias = ", vector bias" if target["operation"] == "addmm" else ""
        return (f"{prefix}, {target['operation']}, M={case['m']} N={case['n']} K={case['k']}, "
                f"A {case.get('a_layout', 'row_major')} B {case['b_layout']}{batch}{bias}")
    q, k = case["q_lengths"], case["kv_lengths"]
    return (
        f"{prefix}, B={len(q)} Q={min(q)}–{max(q)} KV={min(k)}–{max(k)}, "
        f"H={case['heads']}/{case['kv_heads']} D={case['head_dim']}, causal={case['causal']}, scale={attention_scale(case):g}"
    )


def render(report, report_path=None):
    columns = report["columns"]
    lines = [
        f"# TLX Kernel Scout — {report['date']} ({report['mode']})", "",
        f"Device: {report['device']['name']}, {report['device']['arch']}, physical GPU {report['device']['physical_gpu']}.",
        f"Week: {report['week']}. Campaign: `{report['campaign']}`. Provider: {report['provider']}.",
        f"AITER mode: {report.get('aiter_mode', 'search (legacy report)')}.",
        f"Discovery scope: {report.get('discovery_scope', 'case (legacy report)')}.",
        f"Execution: {report.get('execution_mode', 'isolated')}. Candidate screening: {report.get('screen_candidates', False)}.",
        "Best observed among evaluated candidates; unavailable and unstable results cannot win.",
        "All calls use returned tensors, eager triton.testing.do_bench, and the same seeded logical inputs.",
        "Weekly controls are frozen observations, not simultaneous nightly measurements.", "", "<details>",
        "<summary>Performance</summary>", "", "| # | Problem | " + " | ".join(columns) + " | Winner |",
        "| --- | --- | " + " | ".join("---" for _ in columns) + " | --- |"
    ]
    for number, row in enumerate(report["rows"], 1):
        values = []
        for backend in columns:
            result = row["results"].get(backend, {"status": "unavailable"})
            if result["status"] == "ok":
                metric = "tflops" if row["target"]["metric"] == "compute" else "logical_tbps"
                unit = "TFLOP/s" if metric == "tflops" else "logical TB/s"
                frozen = f"; frozen {result['frozen_date']}" if "frozen_date" in result else ""
                fallback = "; Torch fallback" if result.get("adapter_metadata",
                                                            {}).get("implementation") == "torch" else ""
                reference = "; BF16 reference" if result.get(
                    "adapter_metadata", {}).get("implementation") == "bf16_attention_reference" else ""
                values.append(
                    f"{result[metric]:.3f} {unit}; {result['latency_ms'] * 1000:.2f} µs{frozen}{fallback}{reference}")
            else:
                values.append(result["status"] + (": " + result.get("error", "")[:110] if result.get("error") else ""))
        results = {k: v for k, v in row["results"].items() if report["mode"] != "weekly" or k != "tlx_ops_baseline"}
        winner = select_winner(results, row["target"]["timing"]["tie_fraction"])
        lines.append("| " + " | ".join(cell(v) for v in [number, row["description"], *values, winner]) + " |")
    lines += [
        "",
        "Useful FLOPs exclude padded/masked work. Logical bytes count inputs and output once; they are not measured HBM traffic.",
        "Overlap between p20–p80 timing intervals or the configured minimum separation produces a tie.", "",
        "</details>", "", "<details>", "<summary>Reproducers</summary>", "",
        "Commands verify the saved source, compiler, runtime, and device pins before measuring.",
        "Keep the run directory and source revisions. A moved source checkout is rejected.", "",
        "| # | " + " | ".join(columns) + " |", "| --- | " + " | ".join("---" for _ in columns) + " |"
    ]
    if report_path is not None:
        command = shlex.join([sys.executable, str(ROOT / "appctl.py"), "reproduce", "--report", str(report_path)])
        # Factor shared paths out of each cell. The report selects the exact saved request.
        lines[-2:-2] = [
            "Define this shell function once. Row numbers refer to the performance table.", "", "```bash",
            'repro() { ' + command + ' --row "$1" --backend "$2"; }', "```", ""
        ]
    for number, row in enumerate(report["rows"], 1):
        values = []
        for backend in columns:
            result = row["results"].get(backend, {})
            command = result.get("reproducer_command", "unavailable")
            if report_path is not None and result.get("reproducer"):
                command = shlex.join(["repro", str(number), backend])
            values.append(f"`{command}`" if command != "unavailable" else command)
        lines.append("| " + " | ".join(cell(v) for v in [number, *values]) + " |")
    lines += [
        "", "</details>", "", "<details>", "<summary>Provenance and search coverage</summary>", "",
        "Identical source and variant records are grouped. report.json retains the mapping for each case.", "",
        "| Backend | Source commit | Compiler hash | Variant | Dispatch | Evaluated | Cases |",
        "| --- | --- | --- | --- | --- | --- | --- |"
    ]
    provenance = Counter()
    for row in report["rows"]:
        for backend, result in row["results"].items():
            if result.get("status") != "ok":
                continue
            provenance[tuple(
                str(v) for v in [
                    backend, result["backend_pin"].get("git_sha") or "unknown (package hash in JSON)", result["runtime"]
                    ["compiler_sha256"][:16], result["candidate"]["id"],
                    result.get("adapter_metadata", {}).get("implementation", "implementation defaults"),
                    f"{result.get('evaluated_candidates', 'weekly control')}; discovery {result.get('discovery_status', 'not_requested')}"
                ])] += 1
    for entry, count in provenance.items():
        lines.append("| " + " | ".join(cell(v) for v in (*entry, count)) + " |")
    lines += [
        "",
        f"Candidate budget: {report['budget']}. Discovery and failed candidate details are in report.json and candidate logs.",
        ""
    ]
    searches = [
        row["results"][backend]["search"]
        for row in report["rows"]
        for backend in columns
        if backend != "tlx_ops_baseline" and "frozen_date" not in row["results"].get(backend, {})
        and "search" in row["results"].get(backend, {})
    ]
    if searches:
        lines += [(f"Search overhead: {sum(s['agent_calls'] for s in searches)} agent calls, "
                   f"{sum(s['discovery_seconds'] for s in searches):.1f}s in discovery, "
                   f"{sum(s['worker_seconds'] for s in searches):.1f}s in measurement workers."), ""]
        phases = {
            key: sum(s.get("phase_seconds", {}).get(key, 0)
                     for s in searches)
            for key in ("setup", "preparation", "correctness", "compiler_cleanup", "benchmark_and_checks")
        }
        lines += [f"Worker phase seconds: {phases}. Remaining worker time includes imports and process startup.", ""]
    if "elapsed_seconds" in report:
        lines += [f"Elapsed wall time: {report['elapsed_seconds']:.1f}s, including interruptions after resume.", ""]
    if report.get("resumed_cases"):
        lines += [f"Retained {report['resumed_cases']} completed cases from the same pinned campaign.", ""]
    lines += ["</details>", ""]
    return "\n".join(lines)


def reproduce_command(python, request):
    return shlex.join([str(python), str(ROOT / "appctl.py"), "reproduce", "--request", str(request)])
