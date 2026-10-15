"""Idempotent weekly issue rotation and nightly comment publication."""

import json
import re
import subprocess

REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
BODY_BYTES = 60_000  # Leave room for ownership markers below GitHub's comment limit.


def split_report(markdown, max_bytes=BODY_BYTES):
    """Split paragraphs and long tables without truncating rows or commands."""
    if len(markdown.encode()) <= max_bytes:
        return [markdown]
    blocks = []
    for paragraph in markdown.split("\n\n"):
        if len(paragraph.encode()) <= max_bytes:
            blocks.append(paragraph)
            continue
        lines = paragraph.splitlines()
        if len(lines) < 3 or not lines[0].startswith("|") or not lines[1].startswith("| ---"):
            raise ValueError("a report paragraph exceeds the publication limit")
        header = "\n".join(lines[:2])
        fragment = header
        for row in lines[2:]:
            if len((header + "\n" + row).encode()) > max_bytes:
                raise ValueError("a report table row exceeds the publication limit")
            if len((fragment + "\n" + row).encode()) > max_bytes:
                blocks.append(fragment)
                fragment = header
            fragment += "\n" + row
        blocks.append(fragment)
    chunks, current = [], ""
    for block in blocks:
        combined = current + "\n\n" + block if current else block
        if len(combined.encode()) > max_bytes:
            chunks.append(current)
            current = block
        else:
            current = combined
    if current:
        chunks.append(current)
    return chunks


def upsert_comment(api, issue, comments, marker, body):
    matches = [c for c in comments if marker in (c.get("body") or "")]
    if len(matches) > 1:
        raise RuntimeError("multiple owned comments match this report part")
    if matches:
        comment = api.request("PATCH", f"issues/comments/{matches[0]['id']}", {"body": body})
    else:
        comment = api.request("POST", f"issues/{issue}/comments", {"body": body})
    verified = api.request("GET", f"issues/comments/{comment['id']}")
    if verified.get("body") != body:
        raise RuntimeError("report comment read-after-write verification failed")
    return verified


def publish_continuations(api, issue, report, chunks, comments):
    prefix = (f"<!-- tlx-kernel-scout:campaign={report['campaign']};week={report['week']};"
              f"mode={report['mode']};")
    if report["mode"] == "nightly":
        prefix += f"date={report['date']};"
    prefix += "part="
    active = set()
    for number, chunk in enumerate(chunks[1:], 2):
        marker = f"{prefix}{number} -->"
        active.add(marker)
        body = marker + f"\n{report['mode'].capitalize()} report continuation {number} of {len(chunks)}.\n\n" + chunk
        upsert_comment(api, issue, comments, marker, body)
    for previous in comments:
        body = previous.get("body") or ""
        if not body.startswith(prefix):
            continue
        marker = body.split(" -->", 1)[0] + " -->"
        if marker not in active:
            upsert_comment(api, issue, comments, marker,
                           marker + "\nThis continuation is superseded by the current report.")


class GitHub:

    def __init__(self, repository):
        if not REPO.fullmatch(repository):
            raise ValueError("repository must be owner/name")
        self.repository = repository

    def request(self, method, resource, payload=None):
        args = ["gh", "api", "--method", method, f"repos/{self.repository}/{resource}"]
        if payload is not None:
            args += ["--input", "-"]
        result = subprocess.run(args, input=json.dumps(payload) if payload is not None else None, text=True,
                                capture_output=True, check=True, timeout=60)
        return json.loads(result.stdout)

    def all(self, resource):
        rows, page = [], 1
        while True:
            separator = "&" if "?" in resource else "?"
            batch = self.request("GET", f"{resource}{separator}per_page=100&page={page}")
            rows.extend(batch)
            if len(batch) < 100:
                return rows
            page += 1


def publish(api, report, markdown):
    if report.get("complete", True) is not True:
        raise ValueError("partial progress cannot be published as a completed report")
    if any(
            result.get("failure_kind") == "infrastructure" or result.get("error", "").startswith("GPUOccupiedError:")
            for row in report.get("rows", [])
            for result in row["results"].values()):
        raise ValueError("infrastructure failures cannot become a published benchmark report")
    if any(
            result.get("timing_phase") == "screen" and result.get("status") == "ok"
            for row in report.get("rows", [])
            for result in row["results"].values()):
        raise ValueError("screening timings cannot be published as final performance")
    chunks = split_report(markdown, BODY_BYTES)
    first = chunks[0]
    if len(chunks) > 1:
        first += f"\n\nThis report continues in {len(chunks) - 1} labeled comments below."
    campaign, week = report["campaign"], report["week"]
    prefix = f"<!-- tlx-kernel-scout:campaign={campaign};week="
    marker = f"{prefix}{week} -->"
    issues = [i for i in api.all("issues?state=all") if "pull_request" not in i and prefix in (i.get("body") or "")]
    matches = [i for i in issues if marker in (i.get("body") or "")]
    if len(matches) > 1:
        raise RuntimeError("multiple owned issues match this week; resolve duplication before publication")
    if report["mode"] == "weekly":
        body = marker + "\n" + first
        if matches:
            issue = api.request("PATCH", f"issues/{matches[0]['number']}", {"body": body, "state": "open"})
        else:
            issue = api.request("POST", "issues",
                                {"title": f"TLX Kernel Scout {report['device']['arch']} — {week}", "body": body})
        verified = api.request("GET", f"issues/{issue['number']}")
        if verified.get("body") != body or verified.get("state") != "open":
            raise RuntimeError("new weekly issue was not verified; previous issues remain open")
        comments = api.all(f"issues/{issue['number']}/comments")
        publish_continuations(api, issue["number"], report, chunks, comments)
        for previous in issues:
            if previous["number"] == issue["number"] or previous["state"] != "open":
                continue
            old_week = (previous.get("body") or "").split(prefix, 1)[1].split(" -->", 1)[0]
            if old_week >= week:
                continue
            link_marker = f"<!-- tlx-kernel-scout:next={issue['number']} -->"
            comments = api.all(f"issues/{previous['number']}/comments")
            if not any(link_marker in (c.get("body") or "") for c in comments):
                api.request("POST", f"issues/{previous['number']}/comments",
                            {"body": link_marker + f"\nContinued in #{issue['number']}."})
            closed = api.request("PATCH", f"issues/{previous['number']}",
                                 {"state": "closed", "state_reason": "completed"})
            if closed.get("state") != "closed":
                raise RuntimeError("previous weekly issue closure was not verified")
        return verified
    if not matches or matches[0]["state"] != "open":
        raise RuntimeError("publish the matching weekly baseline issue before a nightly report")
    issue = matches[0]
    daily = f"<!-- tlx-kernel-scout:campaign={campaign};date={report['date']} -->"
    body = daily + "\n" + first
    comments = api.all(f"issues/{issue['number']}/comments")
    verified = upsert_comment(api, issue["number"], comments, daily, body)
    publish_continuations(api, issue["number"], report, chunks, comments)
    return verified
