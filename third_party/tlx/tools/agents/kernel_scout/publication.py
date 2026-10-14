"""Idempotent weekly issue rotation and nightly comment publication."""

import json
import re
import subprocess

REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


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
    campaign, week = report["campaign"], report["week"]
    prefix = f"<!-- tlx-kernel-scout:campaign={campaign};week="
    marker = f"{prefix}{week} -->"
    issues = [i for i in api.all("issues?state=all") if "pull_request" not in i and prefix in (i.get("body") or "")]
    matches = [i for i in issues if marker in (i.get("body") or "")]
    if len(matches) > 1:
        raise RuntimeError("multiple owned issues match this week; resolve duplication before publication")
    if report["mode"] == "weekly":
        body = marker + "\n" + markdown
        if matches:
            issue = api.request("PATCH", f"issues/{matches[0]['number']}", {"body": body, "state": "open"})
        else:
            issue = api.request("POST", "issues",
                                {"title": f"TLX Kernel Scout {report['device']['arch']} — {week}", "body": body})
        verified = api.request("GET", f"issues/{issue['number']}")
        if verified.get("body") != body or verified.get("state") != "open":
            raise RuntimeError("new weekly issue was not verified; previous issues remain open")
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
    body = daily + "\n" + markdown
    matches = [c for c in api.all(f"issues/{issue['number']}/comments") if daily in (c.get("body") or "")]
    if len(matches) > 1:
        raise RuntimeError("multiple owned comments match this date")
    if matches:
        comment = api.request("PATCH", f"issues/comments/{matches[0]['id']}", {"body": body})
    else:
        comment = api.request("POST", f"issues/{issue['number']}/comments", {"body": body})
    verified = api.request("GET", f"issues/comments/{comment['id']}")
    if verified.get("body") != body:
        raise RuntimeError("nightly comment read-after-write verification failed")
    return verified
