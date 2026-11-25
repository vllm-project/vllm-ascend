# Copyright (c) 2026 vLLM Ascend contributors. Apache-2.0.
"""Local tests for the AI Review workflow. Run with pytest -q .github/workflows/scripts/test_ai_review.py."""

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ai_review as ai
import pytest


def result(**changes):
    payload = dict(summary="Review complete.", coverage_complete=True, requires_human=False, findings=[], updates=[])
    return json.dumps(payload | changes)


def finding(line=1, severity="MUST-FIX"):
    return dict(path="feature.py", line=line, rule="general", severity=severity, body="Handle empty input.")


@pytest.fixture(scope="session")
def history(tmp_path_factory):
    """A two-commit PR, a later push and a divergent main branch, all in a local repository."""
    repo = tmp_path_factory.mktemp("review-repo")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    def commit(path, content):
        (repo / path).write_text(content)
        git("add", path)
        git("commit", "-qm", path)
        return git("rev-parse", "HEAD")

    git("init", "-q", "-b", "main")
    git("config", "user.name", "Local Test")
    git("config", "user.email", "test@example.invalid")
    git("config", "commit.gpgsign", "false")
    common = commit("README", "base\n")
    git("checkout", "-qb", "feature")
    commit("feature.py", "first_change = 1\n")
    head = commit("second.py", "second_change = 2\n")
    next_head = commit("third.py", "followup_change = 3\n")
    git("checkout", "-q", "main")
    base = commit("base_only.py", "base_only_change = 3\n")
    git("remote", "add", "origin", str(repo))

    kb = repo / ".github/agent-review"
    shutil.copytree(Path(__file__).resolve().parents[2] / "agent-review", kb)
    (kb / "allowlist.txt").write_text("maintainer  # pilot reviewer\n")
    return SimpleNamespace(repo=repo, common=common, base=base, head=head, next_head=next_head)


class PullRequest:
    """GitHub records shared across CLI phases. IDs distinguish users, summaries and inline findings."""

    def __init__(self, history):
        self.pull = dict(state="open", user={"login": "author"}, base={"sha": history.base}, head={"sha": history.head})
        self.labels = {"ai-review"}
        self.comments = {}
        self.inline = {}
        self.statuses = []

    def post(self, body, user_id=1):
        number = 100 + len(self.comments)
        self.comments[number] = dict(id=number, body=body, user={"id": user_id})
        return self.comments[number]

    def command(self, body="/ai-review", actor="author", kind="User"):
        comment = self.post(body, user_id=2)
        return dict(issue={"number": 7}, comment=comment.copy(), sender={"login": actor, "type": kind})

    def label_event(self, action="labeled", actor="maintainer"):
        return dict(
            action=action, pull_request={"number": 7, "head": self.pull["head"].copy()}, sender={"login": actor}
        )

    def summaries(self):
        return [c for c in self.comments.values() if c["user"]["id"] == 1 and c["body"].startswith(ai.STATE_MARKER)]

    def state(self):
        [comment] = self.summaries()
        data = comment["body"].split(ai.STATE_DATA_MARKER)[-1].removesuffix("\n-->")
        return json.loads(data)

    def seed(self, **fields):
        state = ai.State(base=self.pull["base"]["sha"], head=self.pull["head"]["sha"], **fields)
        return self.post(state.body())

    def api(self, endpoint, *args, payload=None, anchor_fallback=False):
        path = endpoint.removeprefix("repos/:owner/:repo/")
        if path == "user":
            return {"id": 1}
        if path == "pulls/7":
            return self.pull | {"labels": [{"name": name} for name in self.labels]}
        if path.startswith("collaborators/"):
            return {"permission": "write" if path.split("/")[1] in {"maintainer", "writer"} else "read"}
        if path == "issues/7/comments?per_page=100":
            assert args == ("--paginate", "--slurp")
            comments = list(self.comments.values())
            return [comments[i : i + 100] for i in range(0, len(comments), 100)]
        if path == "issues/7/comments":
            return self.post(payload["body"])
        if path == "pulls/7/comments":
            number = 500 + len(self.inline)
            self.inline[number] = payload | {"id": number}
            return self.inline[number]
        if path.startswith(("issues/comments/", "pulls/comments/")):
            records = self.comments if path.startswith("issues/") else self.inline
            record = records[int(path.split("/")[-1])]
            if payload is not None:
                assert args == ("--method", "PATCH")
                record.update(payload)
            return record.copy()
        if path.startswith("issues/7/labels/"):
            assert args == ("--method", "DELETE")
            self.labels.remove(path.split("/")[-1])
        elif path == "issues/7/labels":
            self.labels.update(payload["labels"])
        elif path.startswith("statuses/"):
            self.statuses.append((path.split("/")[-1], payload["state"]))
        else:
            pytest.fail(f"Unexpected GitHub request: {endpoint}")


@pytest.fixture
def pr(history, monkeypatch):
    pull = PullRequest(history)
    monkeypatch.setattr(ai, "gh", pull.api)
    return pull


@pytest.fixture
def agent(monkeypatch):
    model = Mock(return_value=result())
    monkeypatch.setattr(ai, "run_agent", model)
    return model


@pytest.fixture
def phase(tmp_path, history, pr, agent, monkeypatch):
    """Invoke the actual CLI entry point. Only the request file carries review inputs across phases."""
    event_file = tmp_path / "event.json"
    outputs = tmp_path / "outputs"
    request = tmp_path / "request.json"
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")

    def invoke(name, event=None, run_id="101"):
        if event is not None:
            event_file.write_text(json.dumps(event))
            monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
            monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment" if "comment" in event else "pull_request_target")
            monkeypatch.setenv("GITHUB_RUN_ID", run_id)
        else:
            monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
        outputs.write_text("")
        monkeypatch.setattr(
            sys, "argv", ["ai_review.py", name, "--workspace", str(history.repo), "--request", str(request)]
        )
        assert ai.main() == 0
        return dict(line.split("=", 1) for line in outputs.read_text().splitlines())

    return invoke


# --- Permissions and command feedback ---


@pytest.mark.parametrize(
    "body,actor,allowed,feedback",
    [
        ("/ai-review F001 fixed", "author", True, "Review request accepted"),
        ("/ai-review", "writer", True, "Review request accepted"),
        ("/ai-review", "reader", False, "Permission denied"),
        ("/ai-review-stop", "author", False, "Permission denied"),
        ("/ai-review-waive all intentional", "author", False, "Permission denied"),
        ("/ai-reviewer", "author", False, None),
    ],
)
def test_command_permissions(pr, phase, body, actor, allowed, feedback):
    event = pr.command(body, actor)
    assert phase("authorize", event) == ({"authorized": "true"} if allowed else {})
    [comment] = pr.comments.values()
    assert comment["body"].startswith(body)
    if feedback is None:
        assert comment["body"] == body
    else:
        assert feedback in comment["body"]
    if allowed:
        assert "[View Actions run](https://github.com/example/project/actions/runs/101)" in comment["body"]


def test_label_checks_the_reviewer_and_removes_it_when_denied(pr, phase):
    assert phase("authorize", pr.label_event(actor="writer")) == {}
    [notice] = pr.comments.values()
    assert "@writer" in notice["body"] and "RFC #21451" in notice["body"]
    assert not pr.labels


def test_bot_command_is_ignored_even_with_write_access(pr, phase):
    event = pr.command("/ai-review-stop", actor="writer", kind="Bot")
    assert phase("authorize", event) == {}
    assert pr.comments[event["comment"]["id"]]["body"] == "/ai-review-stop"
    assert pr.labels == {"ai-review"}


@pytest.mark.parametrize("change,notice", [("closed", "This PR is closed"), ("unlabeled", "AI review is not enabled")])
def test_request_rechecks_live_pr_after_queue(pr, phase, agent, change, notice):
    event = pr.command()
    assert phase("authorize", event) == {"authorized": "true"}
    if change == "closed":
        pr.pull["state"] = "closed"
    else:
        pr.labels.clear()
    original = pr.comments[event["comment"]["id"]]
    original["body"] += "\nEdited while waiting."
    assert phase("prepare", event) == {}
    assert "Edited while waiting." in original["body"] and notice in original["body"]
    assert phase("authorize", pr.command()) == {}
    assert not pr.summaries()
    agent.assert_not_called()


# --- Review, pass and manual follow-up ---


def test_label_then_manual_review_of_all_changes(pr, phase, agent, history):
    start = pr.label_event()
    assert phase("authorize", start) == {"authorized": "true"}
    assert phase("prepare", start) == {"review": "true", "base": history.common, "head": history.head}
    agent.return_value = result(findings=[finding(), finding(line=2), finding(line=None, severity="NIT")])
    phase("review")
    [summary] = pr.summaries()
    assert "DeepSeek Harness headless · POC pilot" in summary["body"]
    assert [comment["line"] for comment in pr.inline.values()] == [1, 2]
    assert pr.state()["status"] == "reviewing" and pr.state()["total"] == 1
    assert pr.statuses[-1] == (history.head, "failure")

    reply = "/ai-review\nF001 fixed in the first commit.\nF002 fixed in the second."
    command = pr.command(reply)
    assert phase("authorize", command) == {"authorized": "true"}
    phase("prepare", command)
    agent.return_value = result(
        updates=[dict(id=fid, status="resolved", evidence="Fixed in feature.py.") for fid in ("F001", "F002")]
    )
    phase("review")
    request = agent.call_args.args[1]
    assert request["reply"] == reply and request["comment_id"] == summary["id"]
    assert request["files"] == ["feature.py", "second.py"]
    assert (
        "first_change" in request["diff"]
        and "second_change" in request["diff"]
        and "base_only_change" not in request["diff"]
    )
    assert [f["id"] for f in request["state"]["findings"]] == ["F001", "F002", "F003"]
    assert [f["status"] for f in pr.state()["findings"]] == ["resolved", "resolved", "open"]
    assert pr.state()["total"] == 2 and pr.statuses[-1] == (history.head, "success")
    assert "ai-review-passed" in pr.labels and pr.summaries() == [summary]

    old_push = pr.label_event("synchronize", "author")
    pr.pull["head"] = {"sha": history.next_head}
    assert phase("authorize", old_push) == {} and "ai-review-passed" in pr.labels
    assert phase("authorize", pr.label_event("synchronize", "author")) == {}
    assert "ai-review-passed" not in pr.labels and agent.call_count == 2 and pr.state()["total"] == 2


@pytest.mark.parametrize("blocked_by", ["incomplete", "human", "truncated", "omitted finding"])
def test_no_pass_when_review_is_blocked(pr, phase, agent, monkeypatch, history, blocked_by):
    if blocked_by == "truncated":
        monkeypatch.setattr(ai, "MAX_DIFF_CHARS", 12)
    if blocked_by == "omitted finding":
        pr.seed(findings=[ai.Finding(id="F001", head=history.head, **finding())])
    phase("prepare", pr.command())
    agent.return_value = result(coverage_complete=blocked_by != "incomplete", requires_human=blocked_by == "human")
    phase("review")
    assert pr.state()["status"] != "passed" and "ai-review-passed" not in pr.labels
    assert pr.statuses[-1][1] in {"error", "failure"}
    if blocked_by == "truncated":
        assert "truncated" in pr.state()["message"] and len(agent.call_args.args[1]["diff"]) == 12


def test_stop_blocks_inflight_publication(pr, phase, agent):
    phase("prepare", pr.command())
    stop = pr.command("/ai-review-stop", "maintainer")
    assert phase("authorize", stop) == {"authorized": "true"}
    assert not pr.labels
    phase("review")
    assert not pr.summaries()
    assert phase("prepare", stop) == {}
    assert pr.state()["status"] == "stopped" and pr.state()["total"] == 0
    assert agent.call_count == 1


def test_waive_one_then_all_for_a_single_commit(pr, phase, agent, history):
    phase("prepare", pr.command())
    agent.return_value = result(findings=[finding(), finding(line=2)])
    phase("review")
    for selector in ("F001", "all"):
        event = pr.command(f"/ai-review-waive {selector} upstream handles this", "maintainer")
        assert phase("authorize", event) == {"authorized": "true"}
        assert phase("prepare", event) == {}
        assert "Waived 1" in pr.comments[event["comment"]["id"]]["body"]
        if selector == "F001":
            assert "ai-review-passed" not in pr.labels
            first, second = pr.state()["findings"]
            assert "[WAIVED by @maintainer]" in pr.inline[first["comment_id"]]["body"]
            assert "[WAIVED" not in pr.inline[second["comment_id"]]["body"]
    assert len(pr.inline) == 2 and all(c["body"].startswith("**[WAIVED by @maintainer]**") for c in pr.inline.values())
    assert all(
        f["waived_head"] == history.head and f["evidence"] == "upstream handles this" for f in pr.state()["findings"]
    )
    assert "ai-review-passed" in pr.labels and pr.state()["total"] == agent.call_count == 1

    pr.pull["head"] = {"sha": history.next_head}
    phase("prepare", pr.command())
    agent.return_value = result(findings=[finding()])
    phase("review")
    assert [f["status"] for f in pr.state()["findings"]] == ["open", "open"]
    assert len(pr.inline) == 2 and "ai-review-passed" not in pr.labels
    assert (pr.state()["round"], pr.state()["total"]) == (1, 2)


@pytest.mark.parametrize(
    "command,feedback",
    [
        ("/ai-review-waive", "Use `/ai-review-waive"),
        ("/ai-review-waive F999 intentionally", "No matching open findings"),
        ("/ai-review-waive F001 intentionally", "Review the current commit"),
    ],
)
def test_invalid_waiver_preserves_summary(pr, phase, history, command, feedback):
    summary = pr.seed(findings=[ai.Finding(id="F001", head=history.head, **finding())])
    before = summary["body"]
    if "F001" in command:
        pr.pull["head"] = {"sha": history.next_head}
    event = pr.command(command, "maintainer")
    assert phase("prepare", event) == {}
    assert feedback in pr.comments[event["comment"]["id"]]["body"]
    assert summary["body"] == before


# --- Published rounds and saved findings ---


@pytest.mark.parametrize(
    "previous_total,expected_limit", [(8, "5-round limit for this commit"), (14, "15-round limit for this session")]
)
def test_last_available_round_is_counted_before_the_next_request_runs(
    pr, phase, agent, history, previous_total, expected_limit
):
    pr.seed(round=4, total=previous_total)
    phase("prepare", pr.command())
    pending = pr.command("/ai-review Check the latest changes.")
    assert phase("authorize", pending) == {"authorized": "true"}
    phase("review")

    before = pr.state()
    assert (before["round"], before["total"]) == (5, previous_total + 1)
    assert phase("prepare", pending) == {}
    assert expected_limit in pr.comments[pending["comment"]["id"]]["body"]
    assert pr.state() == before and agent.call_count == 1
    pr.pull["head"] = {"sha": history.next_head}
    assert bool(phase("prepare", pr.command())) == (previous_total < 14)
    assert pr.state() == before


def test_relabel_resets_counts_once_and_keeps_findings(pr, phase, agent, history):
    pr.seed(
        status="stopped",
        round=5,
        total=15,
        session_run_id="100",
        findings=[ai.Finding(id="F001", head=history.head, **finding())],
    )
    comment = pr.command()
    assert phase("prepare", comment) == {}
    assert "AI review is stopped" in pr.comments[comment["comment"]["id"]]["body"]
    start = pr.label_event()
    for count in (1, 2):
        assert phase("authorize", start, run_id="102")
        assert phase("prepare", start, run_id="102")
        phase("review")
        state = pr.state()
        assert (state["round"], state["total"], state["session_run_id"]) == (count, count, "102")
        assert state["findings"][0]["id"] == "F001"
    assert agent.call_count == 2


@pytest.mark.parametrize("ref", ["base", "head"])
def test_new_commit_during_review_discards_the_result(pr, phase, agent, history, ref):
    if ref == "base":
        pr.pull["base"] = {"sha": history.common}
    reviewed_base = pr.pull["base"]["sha"]
    phase("prepare", pr.command())
    pr.pull[ref] = {"sha": history.base if ref == "base" else history.next_head}
    agent.return_value = result(findings=[finding()])
    phase("review")
    state = pr.state()
    assert (state["base"], state["head"], state["status"], state["total"]) == (
        reviewed_base,
        history.head,
        "discarded",
        1,
    )
    assert not state["findings"] and not pr.inline and "ai-review-passed" not in pr.labels


@pytest.mark.parametrize(
    "bad_result,error",
    [
        (RuntimeError("model unavailable"), RuntimeError),
        ("AI-REVIEW-VERDICT: PASS", ValueError),
        (result(updates=[dict(id="F999", status="resolved", evidence="Fixed.")]), ValueError),
    ],
)
def test_unsuccessful_analysis_can_be_rerun_without_losing_a_round(pr, phase, agent, bad_result, error):
    previous = pr.seed(round=2, total=3)["body"]
    event = pr.command()
    agent.side_effect = [bad_result, result()]
    phase("prepare", event)
    with pytest.raises(error):
        phase("review")
    assert pr.summaries()[0]["body"] == previous
    phase("prepare", event)
    phase("review")
    assert agent.call_count == 2
    assert (pr.state()["round"], pr.state()["total"]) == (3, 4)


@pytest.mark.parametrize("write,expected_total", [("summary", 3), ("status", 4)])
def test_only_a_saved_summary_spends_the_round(pr, phase, monkeypatch, write, expected_total):
    summary = pr.seed(round=2, total=3)
    target = f"issues/comments/{summary['id']}" if write == "summary" else "statuses/"
    phase("prepare", pr.command())

    def fail_write(endpoint, *args, **kwargs):
        if target in endpoint:
            raise RuntimeError("write failed")
        return pr.api(endpoint, *args, **kwargs)

    monkeypatch.setattr(ai, "gh", fail_write)
    with pytest.raises(RuntimeError, match="write failed"):
        phase("review")
    assert pr.state()["total"] == expected_total


def test_head_changed_after_summary_write_gets_no_pass(pr, phase, monkeypatch, history):
    phase("prepare", pr.command())

    def push_after_summary(endpoint, *args, **kwargs):
        response = pr.api(endpoint, *args, **kwargs)
        if endpoint.endswith("/issues/7/comments"):
            pr.pull["head"] = {"sha": history.next_head}
        return response

    monkeypatch.setattr(ai, "gh", push_after_summary)
    phase("review")
    assert pr.state()["total"] == 1 and pr.state()["head"] == history.head
    assert "ai-review-passed" not in pr.labels and pr.statuses == [(history.head, "pending")]


def test_restore_latest_bot_summary_across_comment_pages(pr, phase, agent, history):
    for _ in range(100):
        pr.post("Earlier discussion.", user_id=2)
    pr.seed(round=1, total=2)
    expected = pr.seed(round=2, total=3, findings=[ai.Finding(id="F001", head=history.head, **finding())])
    pr.post(ai.State(status="passed", total=15).body(), user_id=2)
    for _ in range(100):
        pr.post("Later discussion.", user_id=2)
    assert phase("prepare", pr.command())
    phase("review")
    request = agent.call_args.args[1]
    assert request["comment_id"] == expected["id"] and request["state"]["total"] == 3
    assert request["state"]["findings"][0]["id"] == "F001"
    assert "total published rounds 4/15" in expected["body"]


def test_rejected_inline_finding_stays_in_summary_across_reviews(pr, phase, agent, monkeypatch):
    attempts = []

    def reject_anchor(endpoint, *args, **kwargs):
        if endpoint.endswith("/pulls/7/comments"):
            assert kwargs["anchor_fallback"]
            attempts.append(kwargs["payload"]["line"])
            return None
        return pr.api(endpoint, *args, **kwargs)

    monkeypatch.setattr(ai, "gh", reject_anchor)
    agent.return_value = result(findings=[finding()])
    for _ in range(2):
        phase("prepare", pr.command())
        phase("review")
    assert attempts == [1] and not pr.inline
    [saved] = pr.state()["findings"]
    assert saved["id"] == "F001" and saved["comment_id"] is None
    assert phase("prepare", pr.command("/ai-review-waive F001 accepted", "maintainer")) == {}
    assert pr.state()["status"] == "passed" and pr.state()["total"] == 2


# --- External commands ---


@pytest.mark.parametrize(
    "operation,failures,attempts", [("fetch", 2, 3), ("fetch", 3, 3), ("merge-base", 1, 1), ("diff", 1, 1)]
)
def test_only_git_fetch_is_retried(pr, phase, monkeypatch, operation, failures, attempts):
    original = ai.git
    calls = []
    error = subprocess.CalledProcessError(128, ["git", operation])

    def flaky_git(repo, command, *args):
        if command == operation:
            calls.append(args)
            if len(calls) <= failures:
                raise error
        return original(repo, command, *args)

    sleep = Mock()
    monkeypatch.setattr(ai, "git", flaky_git)
    monkeypatch.setattr(ai.time, "sleep", sleep)
    if failures < attempts:
        assert phase("prepare", pr.command())["review"] == "true"
    else:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            phase("prepare", pr.command())
        assert caught.value is error
    assert len(calls) == attempts and len(set(calls)) == 1
    assert [item.args for item in sleep.call_args_list] == [(ai.GIT_FETCH_RETRY_DELAY_SECONDS,)] * (attempts - 1)
    assert not pr.summaries()


@pytest.mark.parametrize("status,inline", [(422, True), (503, True), (422, False)])
def test_github_anchor_fallback(monkeypatch, status, inline):
    failure = subprocess.CompletedProcess(["gh", "api"], 1, stdout="", stderr=f"gh: request failed (HTTP {status})")
    monkeypatch.setattr(ai.subprocess, "run", lambda *args, **kwargs: failure)

    if inline and status == 422:
        assert ai.gh("pulls/1/comments", anchor_fallback=True) is None
    else:
        with pytest.raises(subprocess.CalledProcessError):
            ai.gh("pulls/1/comments", anchor_fallback=inline)


@pytest.fixture
def agent_request(monkeypatch):
    monkeypatch.setenv("AI_REVIEW_KEY", "test-model-secret")
    monkeypatch.setenv("AI_REVIEW_API", "https://test-model.invalid")
    monkeypatch.setenv("AI_REVIEW_MODEL", "test-model")
    monkeypatch.setenv("GH_TOKEN", "test-github-secret")
    return {
        "state": {"head": "abc123", "findings": [{"id": "F003"}]},
        "merge_base": "def456",
        "files": ["example.py"],
        "diff": "+print('{{REPLY}}')",
        "truncated": False,
        "reply": "/ai-review Recheck F003 and keep {{HEAD_SHA}} literal.",
    }


def test_agent_receives_original_input_and_redacts_output(agent_request, tmp_path, monkeypatch):
    agent = tmp_path / "model.py"
    agent.write_text(
        "import json, os, sys\n"
        "assert 'GH_TOKEN' not in os.environ\n"
        "print(json.dumps([sys.stdin.read(), os.environ['AI_REVIEW_KEY']]))\n"
    )
    monkeypatch.setenv("AI_REVIEW_DSH_CMD", shlex.join([sys.executable, str(agent)]))
    workspace = Path(ai.__file__).resolve().parents[3]
    prompt, echoed_key = json.loads(ai.run_agent(workspace, agent_request))

    assert agent_request["diff"] in prompt
    assert agent_request["reply"] in prompt
    assert '"id": "F003"' in prompt
    assert echoed_key == "***REDACTED***"


def test_agent_timeout_is_reported(agent_request, monkeypatch):
    command = [sys.executable, "-c", "import time; time.sleep(1)"]
    monkeypatch.setenv("AI_REVIEW_DSH_CMD", shlex.join(command))
    monkeypatch.setattr(ai, "ROUND_TIMEOUT_SECONDS", 0)
    with pytest.raises(subprocess.TimeoutExpired):
        ai.run_agent(Path(ai.__file__).resolve().parents[3], agent_request)
