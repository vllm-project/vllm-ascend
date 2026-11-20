#!/usr/bin/env python3
#
# Copyright (c) 2026 vLLM Ascend contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Re-run the original pull_request run's gate jobs after a state-changing review.

Run by the ``CI Gate Refresh`` bot workflow (``.github/workflows/bot_ci_gate_refresh.yaml``,
a ``workflow_run`` trigger with a writable base-repo token). The review-event
relay (``.github/workflows/pr_refresh_ci_gate.yaml``) only ships the review
payload as an artifact: ``GITHUB_TOKEN`` is read-only for every run tied to a
fork PR regardless of the triggering event (#22134), so the rerun API calls
must happen here, in the ``workflow_run`` context. All payload fields arrive
as environment variables and are re-validated against live API state below.
Instead of producing a second, review-event
verdict, this re-runs the gate jobs inside the original ``pull_request`` run
so the required ``E2E / ci-gate (pull_request)`` check itself updates in
place. The job is always fast: it never waits for an in-flight CI run,
because the CODEOWNERS gate lives inside the ``ci-gate`` job, which runs
last in the E2E workflow - reviews submitted while the tests run are
already visible when the gate judges, so only reviews that arrive after the
run finished need a re-run:

- APPROVED review on a completed run whose failures are gate-only (every
  code and test job green or skipped, ``ci-gate`` red because an approval
  was missing at the time): re-run the failed jobs. The gate re-judges with
  the current reviews, so the check turns green without any new check name -
  including for docs-only PRs, whose ``select-tests`` skip is legitimate;
- CHANGES_REQUESTED or dismissed review on a green run: re-run the
  ``ci-gate`` job (via the re-run-job API, falling back to a full run
  re-run) so the gate re-arms and the veto lands on the required check
  instead of being masked by stale green;
- a review on a run that is still in flight: no-op - the in-workflow gate
  runs last and will judge with the reviews already submitted;
- anything else (code or test failures, already-green runs, reviews from
  people outside the changed files' CODEOWNERS pool, COMMENTED and bot
  reviews) cannot change the gate verdict and is a no-op: the original run's
  own failure comment already explains what to do.

This job's own failures - no pull_request run for the commit yet, rerun API
error - are written to ``$RUNNER_TEMP/ci_gate_failure_reason.txt``; the
``Package gate result`` step ships them to the PR through the
failure-comment bot, mirroring the ci-gate job's artifact contract.

Environment inputs: ``GITHUB_REPOSITORY``, ``HEAD_SHA``, ``GH_TOKEN``,
``REVIEW_STATE``, ``REVIEW_ACTION``, ``REVIEWER``, ``PR_NUMBER``,
``PR_AUTHOR``, ``BASE_REF`` and ``RUNNER_TEMP``.
"""

import base64
import json
import os
import sys
import urllib.error
import urllib.request

import codeowner_gate as owners_gate

API_ROOT = "https://api.github.com"
WORKFLOW_NAME = "E2E"
IN_FLIGHT_STATUSES = ("queued", "requested", "waiting", "in_progress")


def _get_json(url: str, token: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _post(url: str, token: str) -> None:
    """POST an empty-body rerun request; raise on HTTP errors."""
    request = urllib.request.Request(
        url,
        data=b"",
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request) as response:
        response.read()


def is_gate_job(name: str) -> bool:
    """Whether a failed ``name`` job counts as a real code-CI failure."""
    return (
        name in ("pre-commit", "select-tests")
        or name.startswith("run-selected-tests (vllm@")
        or name.startswith("run-selected-tests-a5 (vllm@")
        or (name.startswith("cpu-ut (vllm@") and " / cpu-ut card" in name)
    )


def latest_pull_request_run(repo: str, token: str, sha: str) -> dict | None:
    """The newest E2E ``pull_request``-event run on ``sha``, if any."""
    data = _get_json(f"{API_ROOT}/repos/{repo}/actions/runs?head_sha={sha}&per_page=100", token)
    runs = [
        r for r in data.get("workflow_runs", []) if r.get("name") == WORKFLOW_NAME and r.get("event") == "pull_request"
    ]
    runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return runs[0] if runs else None


def owner_gate_blocked(jobs: list[dict]) -> bool:
    """Whether a completed run failed on the gate alone: ``ci-gate`` itself
    red while every code and test job is green or skipped, so a re-run can
    only reproduce the same failures or - once the approval arrived - turn
    the gate green. The CODEOWNERS check lives inside ``ci-gate``, so its
    failure surfaces as a ``ci-gate`` failure with no code job red.

    A gate failure caused by a missing test label (not by reviews) also
    matches; re-running the cheap gate job on an owner review is an
    acceptable cost of not having to guess the failure reason from the API.
    """
    gate = next((j for j in jobs if j.get("name") == "ci-gate"), None)
    if gate is None or gate.get("conclusion") != "failure":
        return False
    for j in jobs:
        name = j.get("name", "")
        conclusion = j.get("conclusion")
        if name == "ci-gate":
            continue
        if is_gate_job(name) and conclusion not in (None, "skipped", "success"):
            return False
    return True


def rerun_failed_jobs(repo: str, token: str, run_id: int) -> None:
    _post(f"{API_ROOT}/repos/{repo}/actions/runs/{run_id}/rerun-failed-jobs", token)


def rerun_job(repo: str, token: str, job_id: int) -> None:
    _post(f"{API_ROOT}/repos/{repo}/actions/jobs/{job_id}/rerun", token)


def rerun_run(repo: str, token: str, run_id: int) -> None:
    _post(f"{API_ROOT}/repos/{repo}/actions/runs/{run_id}/rerun", token)


def reviewer_in_owner_pool(pr_number: int, pr_author: str, reviewer: str, base_ref: str) -> bool:
    """Whether ``reviewer`` sits in this PR's approver/veto pool.

    The gate only counts reviews from the CODEOWNERS owners of the changed
    files (and never the PR author's own review), so a review from anyone
    else cannot change the verdict and must not fire a re-run. This
    deliberately reuses codeowner_gate's own parsing and matching so the
    filter and the gate can never drift apart.
    """
    if reviewer == pr_author:
        return False
    data = owners_gate.gh_api("repos/{owner}/{repo}/contents/.github/CODEOWNERS", ref=base_ref)
    rules = owners_gate.parse_codeowners(base64.b64decode(data["content"]).decode())
    pool: set[str] = set()
    for path in owners_gate.get_pr_files(pr_number):
        file_owners, skipped = owners_gate.resolve_owners(rules, path)
        if not skipped:
            pool.update(o.lstrip("@") for o in file_owners)
    return reviewer in pool


def failure_message(verdict: str, detail: str, sha: str) -> str:
    """The actionable PR-comment text when this job itself fails."""
    if verdict == "no-ci":
        return (
            f"A review was submitted for commit {sha}, but there is no E2E run for "
            "this commit to re-run the gate on: pushing a new commit does not start "
            "CI. A commit is tested when the PR is opened or reopened, when a test "
            "label is added, or when someone comments `/run` - none of those has "
            "happened for this commit yet. Fix: comment `/run` on this PR to start "
            "E2E. When it finishes, the gate judges with the reviews already "
            "submitted, so no re-review is needed."
        )
    return (
        f"A review was submitted, but re-running the pull_request CI for this commit ({sha}) "
        f"failed ({detail}). Re-run it manually from the Actions page; no re-review is needed."
    )


def main() -> int:
    repo = os.environ["GITHUB_REPOSITORY"]
    sha = os.environ["HEAD_SHA"]
    token = os.environ["GH_TOKEN"]
    # Event payloads carry lowercase states ("approved"), while the REST API
    # used everywhere below compares uppercase ("APPROVED"). Normalize here.
    action = os.environ["REVIEW_ACTION"].upper()
    state = os.environ["REVIEW_STATE"].upper()
    failure_file = os.path.join(os.environ["RUNNER_TEMP"], "ci_gate_failure_reason.txt")

    def fail(verdict: str, detail: str = "") -> int:
        message = failure_message(verdict, detail, sha)
        with open(failure_file, "w", encoding="utf-8") as f:
            f.write(message + "\n")
        print(f"::error::{message}")
        return 1

    # COMMENTED reviews never change the approval state (the job's if already
    # filters bot reviews), so there is nothing to re-evaluate.
    if action == "SUBMITTED" and state not in ("APPROVED", "CHANGES_REQUESTED"):
        print("Review is comment-only; the gate verdict cannot change. Nothing to re-run.")
        return 0
    veto = action == "DISMISSED" or state == "CHANGES_REQUESTED"

    # Only reviews from the changed files' CODEOWNERS pool (minus the author)
    # can move the gate; re-running for anyone else is pure churn on busy,
    # widely-commented PRs. Fail open when ownership cannot be determined: a
    # harmless no-op re-run is better than a suppressed rescue.
    reviewer = os.environ["REVIEWER"]
    pr_number = int(os.environ["PR_NUMBER"])
    try:
        relevant = reviewer_in_owner_pool(pr_number, os.environ["PR_AUTHOR"], reviewer, os.environ["BASE_REF"])
    except Exception as error:
        print(
            f"::notice::Could not determine CODEOWNERS ownership "
            f"({type(error).__name__}); assuming the reviewer can change the verdict."
        )
        relevant = True
    if not relevant:
        print(
            f"{reviewer} is not a CODEOWNERS owner of the changed files; "
            "their review cannot change the gate verdict. Nothing to re-run."
        )
        return 0

    try:
        run = latest_pull_request_run(repo, token, sha)
        if run is None:
            return fail("no-ci")

        def try_rerun(rerun) -> bool:
            # Another retrigger may already have started the re-run: a conflict
            # means the same work is in flight, which is fine.
            try:
                rerun()
                return True
            except urllib.error.HTTPError as error:
                if error.code == 409:
                    print("A re-run is already in flight; nothing else to do.")
                    return True
                raise

        # A push that landed while CI ran re-runs everything on a new head
        # anyway; re-running the stale head would only churn runners.
        pull = owners_gate.gh_api("repos/{owner}/{repo}/pulls/{pr}", pr=pr_number)
        current_head = (pull.get("head") or {}).get("sha")
        if current_head and current_head != run.get("head_sha"):
            print(
                "A new commit was pushed while waiting for CI; the fresh pull_request run covers it. Nothing to re-run."
            )
            return 0

        # Never wait for an in-flight run: the CODEOWNERS gate lives inside
        # the ci-gate job, which runs last in the workflow, so reviews
        # submitted before the run finishes are judged when it finishes.
        if run.get("status") in IN_FLIGHT_STATUSES:
            print(
                "The pull_request run is still in flight; its gate runs last "
                "and will judge with this review. Nothing to re-run."
            )
            return 0

        if run.get("conclusion") == "success" and not veto:
            print("The pull_request run is already green; nothing to re-run.")
            return 0

        jobs = _get_json(run["jobs_url"] + "?per_page=100", token).get("jobs", [])

        if veto:
            gate_job = next((j for j in jobs if j.get("name") == "ci-gate"), None)
            if run.get("conclusion") == "success":
                # Re-run just the ci-gate job (which contains the CODEOWNERS
                # gate) so the veto lands on the required check without
                # churning UT/E2E runners.
                try:
                    if gate_job is None:
                        try_rerun(lambda: rerun_run(repo, token, run["id"]))
                    else:
                        try_rerun(lambda: rerun_job(repo, token, gate_job["id"]))
                except urllib.error.HTTPError:
                    # Some job states cannot be re-run individually; a full
                    # run re-run always can.
                    try_rerun(lambda: rerun_run(repo, token, run["id"]))
            elif owner_gate_blocked(jobs):
                try_rerun(lambda: rerun_failed_jobs(repo, token, run["id"]))
        elif run.get("conclusion") == "success":
            print("The pull_request run is already green; nothing to re-run.")
        elif owner_gate_blocked(jobs):
            # The run failed on the gate alone: the approval arrived after it
            # finished, so re-running the failed jobs turns the required
            # check green in place.
            try_rerun(lambda: rerun_failed_jobs(repo, token, run["id"]))
            print(f"Re-running the gate jobs of run {run['id']} with the current reviews.")
        else:
            print("The pull_request run failed for code reasons a review cannot fix; leaving it untouched.")
        return 0
    except urllib.error.HTTPError as error:
        return fail("rerun-error", f"HTTP {error.code}")
    except Exception as error:  # noqa: BLE001 - keep the PR comment contract
        return fail("rerun-error", type(error).__name__)


if __name__ == "__main__":
    sys.exit(main())
