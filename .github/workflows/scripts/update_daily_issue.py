# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Update the community Daily Issue from one dispatched failed-case identity.

Called only by publish_daily_issue.yaml inside its round-scoped concurrency
group. GH_TOKEN and GITHUB_* are standard Actions/gh variables.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DAILY_WORKFLOW = "schedule_daily_test.yaml"
ISSUE_TIMEZONE = timezone(timedelta(hours=8))
HTTP_TIMEOUT_SECONDS = 10
HTTP_ATTEMPTS = 3
MAX_BODY_LENGTH = 60000
MAX_FIELD_LENGTH = 1000
PAGE_SIZE = 100
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
FAILURE_ENTRY_PATTERN = re.compile(r"<!-- daily-failure:[0-9a-f]{64} -->\n- \[[^\n]*\]\((https?://[^\s)]+)\)\n?")


class GitHubClient:
    def __init__(self, repository: str, token: str, api_url: str = "https://api.github.com"):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
            raise ValueError("Invalid repository")
        if not token:
            raise ValueError("GH_TOKEN is required for Daily failure reporting")
        self.repository = repository
        self.token = token
        self.api_url = api_url.rstrip("/")

    def request(self, method: str, path: str, data=None, *, retry: bool = True):
        payload = None if data is None else json.dumps(data).encode("utf-8")
        request = Request(
            f"{self.api_url}/repos/{self.repository}/{path}",
            data=payload,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "vllm-ascend-daily-issue",
            },
        )
        attempts = HTTP_ATTEMPTS if retry else 1
        for attempt in range(attempts):
            try:
                with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                    body = response.read()
                    return json.loads(body) if body else None
            except HTTPError as error:
                if error.code not in RETRYABLE_STATUS or attempt == attempts - 1:
                    raise RuntimeError(f"GitHub {method} failed with HTTP {error.code}") from None
            except (URLError, TimeoutError, OSError):
                if attempt == attempts - 1:
                    raise RuntimeError(f"GitHub {method} failed after {attempts} attempt(s)") from None
            time.sleep(2**attempt)
        raise RuntimeError("GitHub request did not complete")

    def pages(self, path: str, **query):
        page = 1
        while True:
            batch = self.request("GET", f"{path}?{urlencode({**query, 'per_page': PAGE_SIZE, 'page': page})}")
            yield from batch
            if len(batch) < PAGE_SIZE:
                return
            page += 1


def client_from_environment() -> GitHubClient:
    return GitHubClient(
        os.environ["GITHUB_REPOSITORY"],
        os.environ.get("GH_TOKEN", ""),
        os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )


def validate_inputs(inputs: dict, repository: str, server_url: str) -> tuple[str, str]:
    if set(inputs) != {"request_id", "case_name", "run_url"}:
        raise ValueError("Expected only request_id, case_name and run_url")
    match = re.fullmatch(r"daily-([1-9][0-9]*)-([1-9][0-9]*)", inputs["request_id"])
    if not match:
        raise ValueError("Invalid Daily round identity")
    name = inputs["case_name"]
    if not name.strip() or len(name) > MAX_FIELD_LENGTH:
        raise ValueError("Missing or oversized Daily case name")
    prefix = re.escape(f"{server_url}/{repository}/actions/runs/")
    if not re.fullmatch(prefix + r"[1-9][0-9]*/attempts/[1-9][0-9]*", inputs["run_url"]):
        raise ValueError("Execution link must point to this repository's workflow run")
    return match.groups()


def event_id(inputs: dict) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def safe_text(value: str) -> str:
    return html.escape(value, quote=False).replace("@", "@\u200b")


def failure_entry(inputs: dict) -> str:
    label = safe_text(inputs["case_name"]).replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    label = " ".join(label.splitlines())
    return f"<!-- daily-failure:{event_id(inputs)} -->\n- [{label}]({inputs['run_url']})\n"


def append_grouped_failure(body: str, inputs: dict, daily_run_url: str) -> str:
    daily_run_url = daily_run_url.rsplit("/attempts/", 1)[0]
    sections = ("UT", "Nightly")
    section_ends = {name: f"<!-- daily-section:{name}:end -->" for name in sections}
    if not all(marker in body for marker in section_ends.values()):
        # Move entries from the previous flat format into their sections.
        entries = {name: [] for name in sections}
        for match in FAILURE_ENTRY_PATTERN.finditer(body):
            name = "UT" if match[1].rsplit("/attempts/", 1)[0] == daily_run_url else "Nightly"
            entries[name].append(match[0].rstrip() + "\n")
        body = FAILURE_ENTRY_PATTERN.sub("", body).rstrip()
        for name in sections:
            body += f"\n\n## {name}\n\n" + "".join(entries[name]) + "\n" + section_ends[name]
    for name, marker in section_ends.items():
        body = re.sub(rf"(?m)^## {name}\n\s*(?={re.escape(marker)})", "", body)
    # Selected tests execute in the Daily run; Nightly runs are dispatched separately.
    if f"<!-- daily-failure:{event_id(inputs)} -->" in body:
        return body
    name = "UT" if inputs["run_url"].rsplit("/attempts/", 1)[0] == daily_run_url else "Nightly"
    marker = section_ends[name]
    entry = failure_entry(inputs)
    if f"## {name}\n" not in body:
        entry = f"## {name}\n\n" + entry
    return body.replace(marker, entry + "\n" + marker, 1)


def publish_failure(client: GitHubClient, inputs: dict, server_url: str) -> int:
    """Called only inside the update workflow's round-scoped concurrency group."""
    run_id, _ = validate_inputs(inputs, client.repository, server_url)
    root = client.request("GET", f"actions/runs/{run_id}/attempts/1")
    if root.get("path", "").split("@")[0] != f".github/workflows/{DAILY_WORKFLOW}":
        raise ValueError("Run is not a Daily main workflow")
    marker = f"<!-- vllm-ascend-daily:{run_id} -->"
    marker_pattern = re.compile(rf"<!-- vllm-ascend-daily:{run_id}(?::[1-9][0-9]*)? -->")
    matches = [
        issue
        for issue in client.pages("issues", state="all", since=root["created_at"])
        if "pull_request" not in issue and marker_pattern.search(issue.get("body") or "")
    ]
    if len(matches) > 1:
        raise ValueError("Multiple Issues already belong to this Daily round; resolve duplicates before retrying")
    run_url = f"{server_url}/{client.repository}/actions/runs/{run_id}"
    started_at = datetime.fromisoformat(root["run_started_at"]).astimezone(ISSUE_TIMEZONE)
    title = f"Daily test failure on {started_at:%Y-%m-%d %H:%M} - run {run_id} - please check"
    workflow_name = safe_text(root.get("name") or "Schedule Daily Tests")
    header = f"## {workflow_name}\n\n[Run #{run_id}]({run_url})"
    if not matches:
        body = append_grouped_failure(f"{marker}\n\n{header}", inputs, run_url)
        # Creation is not blindly retried: a timeout may have created the Issue.
        # Re-running this writer first lists Issues again to discover that write.
        issue = client.request(
            "POST",
            "issues",
            {"title": title, "body": body},
            retry=False,
        )
        return issue["number"]
    issue = matches[0]
    previous_body = issue.get("body") or ""
    body = marker_pattern.sub(marker, previous_body, count=1)
    body = re.sub(rf"(?m)^\[Daily run\]\({re.escape(run_url)}(?:/attempts/[1-9][0-9]*)?\)$", header, body, count=1)
    body = append_grouped_failure(body, inputs, run_url)
    if body == previous_body and issue.get("title") == title:
        return issue["number"]
    if len(body) > MAX_BODY_LENGTH:
        raise ValueError("Daily Issue body limit reached")
    client.request("PATCH", f"issues/{issue['number']}", {"title": title, "body": body})
    return issue["number"]


def main() -> int:
    inputs = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))["inputs"]
    number = publish_failure(
        client_from_environment(), inputs, os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    )
    print(f"Updated Daily Issue #{number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
