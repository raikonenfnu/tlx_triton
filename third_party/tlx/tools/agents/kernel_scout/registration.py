"""Summarize target readiness from real shared-harness measurements."""

from contracts import digest, enforce_equal_inputs


def registration_result(report):
    """Keep checked case coverage explicit, including unsupported implementations."""
    cases = []
    for row in report["rows"]:
        results = {name: result for name, result in row["results"].items() if name != "tlx_ops_baseline"}
        enforce_equal_inputs(results)
        errors = []
        checked = []
        for name, result in results.items():
            status = result["status"]
            if status == "unsupported":
                continue
            if status != "ok":
                errors.append(f"{name}: {status}: {result.get('error', '')}")
                continue
            if result.get("timing_phase") != "full":
                errors.append(f"{name}: full timing is missing")
            elif result.get("correctness_seeds") != row["target"]["correctness"]["seeds"]:
                errors.append(f"{name}: correctness seed coverage differs")
            elif not result.get("reproducer") or not result.get("request_sha256"):
                errors.append(f"{name}: pinned reproducer is missing")
            else:
                checked.append(name)
        if not any(name.startswith("tlx_") for name in checked):
            errors.append("no checked TLX implementation")
        if not any(name in {"aiter", "pytorch"} for name in checked):
            errors.append("no checked reference backend")
        cases.append({
            "target": row["target"]["id"], "case": row["case"]["id"], "target_sha256": digest(row["target"]), "ready":
            not errors, "checked_backends": checked, "errors": errors, "unsupported":
            {name: result.get("error", "")
             for name, result in results.items()
             if result["status"] == "unsupported"}
        })
    return {
        "schema_version": 1, "ready": bool(cases) and all(case["ready"] for case in cases), "scope":
        "Only the listed cases and recorded source/runtime pins were checked.", "cases": cases, "baseline_key":
        report["baseline_key"], "device": report["device"]
    }
