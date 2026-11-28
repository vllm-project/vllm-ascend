#!/usr/bin/env python3
#
# Copyright (c) 2026 vLLM Ascend contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reject invalid skip_ci labels or revoke an existing CI waiver."""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from refresh_ci_gate import API_ROOT, _get_json, latest_pull_request_run, rerun_job
from revert_waiver import (
    active_label_event,
    inspect_pr_paths,
    label_actor_has_write_access,
    skip_ci_events,
    skip_ci_label,
)

PAGE_SIZE = 100
GATE_WAIT_SECONDS = 300
GATE_POLL_INTERVAL = 5


def current_pull(repo_url: str, token: str, pr_number: int, signal: dict, *, has_skip_ci: bool = False) -> dict | None:
    pull = _get_json(f"{repo_url}/pulls/{pr_number}", token)
    head = pull["head"]
    base = pull["base"]["ref"]
    head_repo = head.get("repo") or {}
    signal_repo = signal.get("head_repository") or {}
    if (
        pull["state"] == "open"
        and head["sha"] == signal["head_sha"]
        and head["ref"] == signal["head_branch"]
        and head_repo.get("id") == signal_repo.get("id")
        and head_repo.get("id") is not None
        and (base == "main" or base.endswith("-dev") or base.startswith("releases/v"))
        and (skip_ci_label(pull) is not None) == has_skip_ci
    ):
        return pull
    return None


def removal_exists(repo_url: str, token: str, pr_number: int, start: str, end: str) -> bool:
    # Confirm the signal against GitHub. The PR can change the relay workflow.
    return any(
        event["event"] == "unlabeled" and start <= event["created_at"] <= end
        for event in skip_ci_events(repo_url, token, pr_number)
    )


def run_jobs(repo_url: str, token: str, run_id: int) -> dict:
    result = _get_json(f"{repo_url}/actions/runs/{run_id}/jobs?per_page={PAGE_SIZE}", token)
    return {job["name"]: job for job in result["jobs"]}


def is_waiver_gate(job: dict, pr_number: int) -> bool:
    marker = f"Revert CI waiver for PR {pr_number}"
    return any(step["name"] == marker and step["conclusion"] == "success" for step in job.get("steps", []))


def wait_for_gate(repo_url: str, token: str, run_id: int, previous_gate_id: int | None = None) -> dict:
    deadline = time.monotonic() + GATE_WAIT_SECONDS
    while time.monotonic() < deadline:
        gate = run_jobs(repo_url, token, run_id).get("ci-gate")
        if gate and gate["id"] != previous_gate_id and gate["status"] == "completed":
            return gate
        time.sleep(GATE_POLL_INTERVAL)
    raise TimeoutError(f"Could not verify the ci-gate result for run {run_id} in time.")


def send_api_request(url: str, token: str, method: str, body: dict | None = None) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30):
        pass


def trigger_regular_ci(repo_url: str, token: str, pr_number: int) -> None:
    # Same labeled-event mechanism as /run, including the case where ci-run
    # is already on the PR. A GITHUB_TOKEN mutation would not emit a CI run.
    try:
        send_api_request(f"{repo_url}/labels", token, "POST", {"name": "ci-run"})
    except urllib.error.HTTPError as error:
        if error.code != 422:
            raise
    labels_url = f"{repo_url}/issues/{pr_number}/labels"
    try:
        send_api_request(f"{labels_url}/ci-run", token, "DELETE")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    send_api_request(labels_url, token, "POST", {"labels": ["ci-run"]})


def reject_invalid_label(repo: str, token: str, signal: dict) -> None:
    match = re.fullmatch(r"Check skip_ci for PR ([1-9][0-9]*)", signal["display_title"])
    if match is None:
        return
    pr_number = int(match[1])
    repo_url = f"{API_ROOT}/repos/{repo}"
    pull = current_pull(repo_url, token, pr_number, signal, has_skip_ci=True)
    if not pull:
        return

    # Match the live label event to the GitHub-authenticated run actor before
    # removing a label for lack of permission.
    actor_id = (signal.get("actor") or {}).get("id")
    label_event = active_label_event(repo_url, token, pr_number) if actor_id else None
    if (
        label_event
        and (label_event.get("actor") or {}).get("id") == actor_id
        and not label_actor_has_write_access(repo_url, token, label_event)
    ):
        current = current_pull(repo_url, token, pr_number, signal, has_skip_ci=True)
        latest = active_label_event(repo_url, token, pr_number)
        if not current or not latest or latest["id"] != label_event["id"]:
            return
        reason = "this label requires Write, Maintain, or Admin access"
    else:
        current, protected_paths = inspect_pr_paths(f"{repo_url}/pulls/{pr_number}", token, pull)
        if not protected_paths:
            return
        reason = "this PR changes `.github/`"

    label = skip_ci_label(current)
    if not label or current["state"] != "open":
        return

    issue_url = f"{repo_url}/issues/{pr_number}"
    try:
        send_api_request(f"{issue_url}/labels/{quote(label, safe='')}", token, "DELETE")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return  # Someone already removed the label.
        raise
    send_api_request(
        f"{issue_url}/comments",
        token,
        "POST",
        {"body": f"Removed `skip_ci` because {reason}. Regular CI and CODEOWNERS approval apply."},
    )


def revoke(repo: str, token: str, signal: dict) -> None:
    match = re.fullmatch(r"Revoke skip_ci for PR ([1-9][0-9]*)", signal["display_title"])
    if match is None:
        return
    pr_number = int(match[1])
    repo_url = f"{API_ROOT}/repos/{repo}"
    if not current_pull(repo_url, token, pr_number, signal):
        return

    run = latest_pull_request_run(repo, token, signal["head_sha"])
    if run is None or run["created_at"] > signal["created_at"]:
        return

    def still_current() -> bool:
        latest = latest_pull_request_run(repo, token, signal["head_sha"])
        return bool(latest and latest["id"] == run["id"] and current_pull(repo_url, token, pr_number, signal))

    jobs = run_jobs(repo_url, token, run["id"])
    # Gate-only reruns can omit previous jobs from the latest attempt.
    pre_commit = jobs.get("pre-commit", {})
    if pre_commit.get("conclusion") != "skipped" and not is_waiver_gate(jobs.get("ci-gate", {}), pr_number):
        return
    if not removal_exists(repo_url, token, pr_number, run["created_at"], signal["created_at"]):
        return

    gate = wait_for_gate(repo_url, token, run["id"])
    if not is_waiver_gate(gate, pr_number) or not still_current():
        return
    if gate["conclusion"] == "success":
        try:
            rerun_job(repo, token, gate["id"])
        except urllib.error.HTTPError as error:
            if error.code != 409:
                raise
        gate = wait_for_gate(repo_url, token, run["id"], previous_gate_id=gate["id"])
    if not still_current():
        return
    if gate["conclusion"] != "failure":
        raise RuntimeError(f"Could not confirm waiver revocation for PR {pr_number}.")

    trigger_regular_ci(repo_url, os.environ["PAT_TOKEN"], pr_number)
    print(f"Revoked the skip_ci waiver and requested regular CI for PR {pr_number}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reject-invalid-label", action="store_true")
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    handler = reject_invalid_label if parser.parse_args().reject_invalid_label else revoke
    handler(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"], event["workflow_run"])
