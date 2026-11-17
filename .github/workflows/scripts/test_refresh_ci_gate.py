#
# Copyright (c) 2026 vLLM Ascend contributors.
# Licensed under the Apache License, Version 2.0.
#
"""Unit tests for .github/workflows/scripts/refresh_ci_gate.py."""

import base64
import json
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import refresh_ci_gate as rerun  # noqa: E402


class _Response:
    """Minimal context-manager response for the urllib call sites."""

    def __init__(self, data):
        self._data = data
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._data


class _FakeGitHub:
    """In-memory GitHub API for the rerun call sites.

    ``post_status`` maps a rerun endpoint kind ("failed-jobs", "job-rerun",
    "run-rerun") to an HTTP status; the default 201 succeeds. POSTs are
    recorded in ``self.posts`` as (kind, url). ``files``/``codeowners``/
    ``head_sha`` back the ownership-filter endpoints; ``contents_status``
    simulates CODEOWNERS fetch failures.
    """

    def __init__(
        self,
        runs=(),
        jobs_by_run=None,
        post_status=None,
        files=None,
        codeowners="/vllm_ascend/quantization @alice\n",
        head_sha="abc123",
        contents_status=None,
    ):
        self.runs = list(runs)
        self.jobs_by_run = jobs_by_run or {}
        self.post_status = post_status or {}
        self.files = files if files is not None else ["vllm_ascend/quantization/moe.py"]
        self.codeowners = codeowners
        self.head_sha = head_sha
        self.contents_status = contents_status
        self.posts = []

    def _respond(self, obj):
        return _Response(json.dumps(obj).encode())

    def _kind(self, url):
        if "rerun-failed-jobs" in url:
            return "failed-jobs"
        if "/actions/jobs/" in url and url.rstrip("/").endswith("/rerun"):
            return "job-rerun"
        return "run-rerun"

    def install(self, monkeypatch):
        fake = self

        def urlopen(req, **_kw):
            url = req if isinstance(req, str) else req.full_url
            if not isinstance(req, str) and req.get_method() == "POST":
                kind = fake._kind(url)
                fake.posts.append((kind, url))
                status = fake.post_status.get(kind, 201)
                if status != 201:
                    raise urllib.error.HTTPError(url, status, "error", None, None)
                return _Response(b"")
            if "/actions/runs?head_sha=" in url:
                return fake._respond({"workflow_runs": fake.runs})
            for run_id, jobs in fake.jobs_by_run.items():
                if f"/actions/runs/{run_id}/jobs" in url:
                    return fake._respond({"jobs": jobs})
            if "/contents/.github/CODEOWNERS" in url:
                if fake.contents_status:
                    raise urllib.error.HTTPError(url, fake.contents_status, "error", None, None)
                payload = base64.b64encode(fake.codeowners.encode()).decode()
                return fake._respond({"content": payload})
            if "/pulls/1/files" in url:
                return fake._respond([{"filename": path} for path in fake.files])
            if "/pulls/1" in url:
                return fake._respond({"head": {"sha": fake.head_sha}})
            return fake._respond({})

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)


@pytest.fixture
def rerun_env(monkeypatch, tmp_path):
    # REVIEW_STATE/REVIEW_ACTION use the real event payload casing (lowercase)
    # on purpose: main() must normalize before comparing against the
    # uppercase REST API states.
    for key, value in [
        ("GITHUB_REPOSITORY", "o/r"),
        ("HEAD_SHA", "abc123"),
        ("GH_TOKEN", "t0ken"),
        ("GATE_WAIT_SECONDS", "7200"),
        ("REVIEW_STATE", "approved"),
        ("REVIEW_ACTION", "submitted"),
        ("REVIEWER", "alice"),
        ("PR_NUMBER", "1"),
        ("PR_AUTHOR", "yangqi"),
        ("BASE_REF", "main"),
    ]:
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    return tmp_path


def job(name, conclusion):
    return {"name": name, "conclusion": conclusion, "id": 9000}


def green_jobs():
    return [
        job("pre-commit", "success"),
        job("select-tests", "skipped"),
        job("cpu-ut (vllm@x) / cpu-ut card-1", "success"),
        job("run-selected-tests (vllm@x) / a2-1 card-part", "skipped"),
    ]


def owner_blocked_jobs():
    # The pull_request run failed while the owner approval was still missing;
    # ci-gate is its dependent failure. This is the rescue case.
    return green_jobs() + [job("codeowner-approval", "failure"), job("ci-gate", "failure")]


def run(conclusion="failure", run_id=100, status="completed"):
    return {
        "id": run_id,
        "name": "E2E",
        "event": "pull_request",
        "status": status,
        "conclusion": conclusion,
        "created_at": "2026-11-17T00:00:00Z",
        "head_sha": "abc123",
        "jobs_url": f"https://api.github.com/repos/o/r/actions/runs/{run_id}/jobs",
    }


def assert_refused(rerun_env, exit_code, substring):
    assert exit_code == 1
    reason = (rerun_env / "ci_gate_failure_reason.txt").read_text(encoding="utf-8")
    assert substring in reason


class TestApprovedReview:
    def test_mixed_case_review_state_is_normalized(self, rerun_env, monkeypatch):
        # Real payloads are lowercase ("approved"); the script must normalize
        # before comparing against the uppercase REST API states.
        monkeypatch.setenv("REVIEW_STATE", "Approved")
        fake = _FakeGitHub(runs=[run()], jobs_by_run={100: owner_blocked_jobs()})
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert [kind for kind, _ in fake.posts] == ["failed-jobs"]

    def test_failed_owner_gate_is_rerun(self, rerun_env, monkeypatch):
        # The #22106 rescue: only the owner gate and its ci-gate dependent
        # failed, so re-running them turns the required check green in place.
        fake = _FakeGitHub(runs=[run()], jobs_by_run={100: owner_blocked_jobs()})
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert [kind for kind, _ in fake.posts] == ["failed-jobs"]

    def test_green_run_is_untouched(self, rerun_env, monkeypatch):
        jobs = green_jobs() + [job("ci-gate", "success"), job("codeowner-approval", "success")]
        fake = _FakeGitHub(runs=[run(conclusion="success")], jobs_by_run={100: jobs})
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert fake.posts == []

    def test_code_failure_is_not_rerun(self, rerun_env, monkeypatch):
        jobs = green_jobs()[:0] + [
            job("pre-commit", "failure"),
            job("select-tests", "skipped"),
            job("ci-gate", "failure"),
            job("codeowner-approval", "failure"),
        ]
        fake = _FakeGitHub(runs=[run()], jobs_by_run={100: jobs})
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert fake.posts == []

    def test_npu_test_failure_is_not_rerun(self, rerun_env, monkeypatch):
        # Re-running expensive NPU jobs for the same verdict is churn.
        jobs = green_jobs()[:3] + [
            job("run-selected-tests (vllm@x) / a2-1 card-part", "failure"),
            job("ci-gate", "failure"),
            job("codeowner-approval", "failure"),
        ]
        fake = _FakeGitHub(runs=[run()], jobs_by_run={100: jobs})
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert fake.posts == []

    def test_rerun_conflict_is_tolerated(self, rerun_env, monkeypatch):
        fake = _FakeGitHub(runs=[run()], jobs_by_run={100: owner_blocked_jobs()}, post_status={"failed-jobs": 409})
        fake.install(monkeypatch)
        assert rerun.main() == 0

    def test_rerun_api_error_fails(self, rerun_env, monkeypatch):
        fake = _FakeGitHub(runs=[run()], jobs_by_run={100: owner_blocked_jobs()}, post_status={"failed-jobs": 500})
        fake.install(monkeypatch)
        assert_refused(rerun_env, rerun.main(), "Re-run it manually")


class TestVetoReview:
    def test_changes_requested_on_green_run_reruns_owner_job(self, rerun_env, monkeypatch):
        # A veto after everything went green must re-arm the gate: re-run the
        # owner job (plus its ci-gate dependent) without touching UT/E2E.
        monkeypatch.setenv("REVIEW_STATE", "CHANGES_REQUESTED")
        jobs = green_jobs() + [job("ci-gate", "success"), job("codeowner-approval", "success")]
        fake = _FakeGitHub(runs=[run(conclusion="success")], jobs_by_run={100: jobs})
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert [kind for kind, _ in fake.posts] == ["job-rerun"]

    def test_job_rerun_failure_falls_back_to_full_rerun(self, rerun_env, monkeypatch):
        monkeypatch.setenv("REVIEW_STATE", "CHANGES_REQUESTED")
        jobs = green_jobs() + [job("ci-gate", "success"), job("codeowner-approval", "success")]
        fake = _FakeGitHub(runs=[run(conclusion="success")], jobs_by_run={100: jobs}, post_status={"job-rerun": 422})
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert [kind for kind, _ in fake.posts] == ["job-rerun", "run-rerun"]

    def test_dismissed_review_reruns_owner_job(self, rerun_env, monkeypatch):
        monkeypatch.setenv("REVIEW_ACTION", "dismissed")
        monkeypatch.setenv("REVIEW_STATE", "APPROVED")
        jobs = green_jobs() + [job("ci-gate", "success"), job("codeowner-approval", "success")]
        fake = _FakeGitHub(runs=[run(conclusion="success")], jobs_by_run={100: jobs})
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert [kind for kind, _ in fake.posts] == ["job-rerun"]

    def test_changes_requested_on_failed_run_reruns_failed_jobs(self, rerun_env, monkeypatch):
        monkeypatch.setenv("REVIEW_STATE", "CHANGES_REQUESTED")
        fake = _FakeGitHub(runs=[run()], jobs_by_run={100: owner_blocked_jobs()})
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert [kind for kind, _ in fake.posts] == ["failed-jobs"]


class TestOwnershipFilter:
    def test_non_owner_review_is_noop(self, rerun_env, monkeypatch):
        # Reviews from people outside the changed files' owner pool can never
        # move the gate; widely-commented PRs must not churn the rerun.
        monkeypatch.setenv("REVIEWER", "randomguy")
        fake = _FakeGitHub(runs=[run()], jobs_by_run={100: owner_blocked_jobs()})
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert fake.posts == []

    def test_author_review_is_noop_even_when_author_is_owner(self, rerun_env, monkeypatch):
        # The gate never counts the PR author's own review, so theirs must
        # not fire a re-run even when they own the files.
        monkeypatch.setenv("PR_AUTHOR", "yangqi")
        monkeypatch.setenv("REVIEWER", "yangqi")
        fake = _FakeGitHub(
            runs=[run()],
            jobs_by_run={100: owner_blocked_jobs()},
            codeowners="/vllm_ascend/quantization @alice @yangqi\n",
        )
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert fake.posts == []

    def test_codeowners_fetch_failure_fails_open(self, rerun_env, monkeypatch):
        # A re-run is harmless; a wrongly suppressed rescue is not. If
        # ownership cannot be determined, proceed.
        fake = _FakeGitHub(runs=[run()], jobs_by_run={100: owner_blocked_jobs()}, contents_status=500)
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert [kind for kind, _ in fake.posts] == ["failed-jobs"]

    def test_push_while_waiting_skips_rerun(self, rerun_env, monkeypatch):
        # A push that lands while waiting re-runs everything on the new head;
        # re-running the stale head would only churn runners.
        fake = _FakeGitHub(runs=[run()], jobs_by_run={100: owner_blocked_jobs()}, head_sha="newsha")
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert fake.posts == []


class TestNoopsAndErrors:
    def test_commented_review_is_noop(self, rerun_env, monkeypatch):
        monkeypatch.setenv("REVIEW_STATE", "COMMENTED")
        fake = _FakeGitHub(runs=[run()], jobs_by_run={100: owner_blocked_jobs()})
        fake.install(monkeypatch)
        assert rerun.main() == 0
        assert fake.posts == []

    def test_no_ci_run_fails(self, rerun_env, monkeypatch):
        fake = _FakeGitHub(runs=[])
        fake.install(monkeypatch)
        assert_refused(rerun_env, rerun.main(), "CI must run first")

    def test_still_running_ci_times_out(self, rerun_env, monkeypatch):
        monkeypatch.setenv("GATE_WAIT_SECONDS", "0")
        fake = _FakeGitHub(runs=[run(status="in_progress")], jobs_by_run={100: owner_blocked_jobs()})
        fake.install(monkeypatch)
        assert_refused(rerun_env, rerun.main(), "still running")


class TestHelpers:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("pre-commit", True),
            ("select-tests", True),
            ("cpu-ut (vllm@x) / cpu-ut card-1", True),
            ("cpu-ut (vllm@x) / upload-coverage-to-obs", False),
            ("run-selected-tests (vllm@x) / a2-1 card-part", True),
            ("run-selected-tests-upstream (vllm@x) / e2e-upstream_pr", False),
            ("ci-gate", False),
            ("codeowner-approval", False),
        ],
    )
    def test_is_gate_job(self, name, expected):
        assert rerun.is_gate_job(name) is expected

    def test_owner_gate_blocked_requires_owner_failure(self):
        jobs = green_jobs() + [job("codeowner-approval", "success"), job("ci-gate", "failure")]
        assert rerun.owner_gate_blocked(jobs) is False

    def test_owner_gate_blocked_tolerates_cancelled_ci_gate(self):
        jobs = green_jobs() + [job("codeowner-approval", "failure"), job("ci-gate", "cancelled")]
        assert rerun.owner_gate_blocked(jobs) is True
