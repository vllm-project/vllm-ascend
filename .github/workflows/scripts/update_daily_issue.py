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
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DAILY_WORKFLOW = "schedule_daily_test.yaml"
HTTP_TIMEOUT_SECONDS = 10
HTTP_ATTEMPTS = 3
MAX_BODY_LENGTH = 60000
MAX_FIELD_LENGTH = 1000
PAGE_SIZE = 100
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


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


def publish_failure(client: GitHubClient, inputs: dict, server_url: str) -> int:
    """Called only inside the update workflow's round-scoped concurrency group."""
    run_id, attempt = validate_inputs(inputs, client.repository, server_url)
    root = client.request("GET", f"actions/runs/{run_id}/attempts/{attempt}")
    if root.get("path", "").split("@")[0] != f".github/workflows/{DAILY_WORKFLOW}":
        raise ValueError("Run is not a Daily main workflow")
    marker = f"<!-- vllm-ascend-daily:{run_id}:{attempt} -->"
    matches = [
        issue
        for issue in client.pages("issues", state="all", since=root["created_at"])
        if "pull_request" not in issue and marker in (issue.get("body") or "")
    ]
    if len(matches) > 1:
        raise ValueError("Multiple Issues already belong to this Daily round; resolve duplicates before retrying")
    entry = failure_entry(inputs)
    if not matches:
        run_url = f"{server_url}/{client.repository}/actions/runs/{run_id}/attempts/{attempt}"
        body = f"{marker}\n# Daily test failures\n\n[Daily run]({run_url})\n\n{entry}"
        # Creation is not blindly retried: a timeout may have created the Issue.
        # Re-running this writer first lists Issues again to discover that write.
        issue = client.request(
            "POST",
            "issues",
            {"title": f"[Daily] Test failures — run {run_id}, attempt {attempt}", "body": body},
            retry=False,
        )
        return issue["number"]
    issue = matches[0]
    body = issue.get("body") or ""
    failure_marker = f"<!-- daily-failure:{event_id(inputs)} -->"
    if failure_marker in body:
        return issue["number"]
    if len(body) + len(entry) + 2 > MAX_BODY_LENGTH:
        raise ValueError("Daily Issue body limit reached")
    client.request("PATCH", f"issues/{issue['number']}", {"body": body + "\n\n" + entry})
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
