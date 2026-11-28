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
import regex as re


def result(**changes):
    payload = dict(
        summary="Review complete.",
        unfinished_work=[],
        coverage_complete=True,
        requires_human=False,
        findings=[],
        updates=[],
    )
    return json.dumps(payload | changes)


def finding(line=1, severity="MUST-FIX"):
    return dict(
        path="feature.py",
        line=line,
        rule="general",
        severity=severity,
        title="Empty input raises an error",
        body="Indexing the first item raises IndexError on empty input.",
        suggestion="Return early when the input is empty.",
    )


def prompt_section(prompt, heading):
    return prompt.split(f"### {heading}\n", 1)[1].split("\n### ", 1)[0]


def prompt_findings(prompt, heading):
    return json.loads(next(line[2:] for line in prompt_section(prompt, heading).splitlines() if line.startswith("> [")))


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
    common = commit("README.md", "Run the installer.\n")
    git("checkout", "-qb", "feature")
    commit("feature.py", "first_change = 1\nsecond_item = 2\nthird_item = 3\n")
    head = commit("second.py", "second_change = 2\n")
    next_head = commit("third.py", "followup_change = 3\n")
    docs_head = commit("README.md", "Run the installer with --unknown-flag.\n")
    git("revert", "--no-edit", docs_head)
    withdrawn_head = git("rev-parse", "HEAD")
    git("checkout", "-q", "main")
    base = commit("base_only.py", "base_only_change = 3\n")
    git("remote", "add", "origin", str(repo))

    kb = repo / ".github/agent-review"
    shutil.copytree(Path(__file__).resolve().parents[2] / "agent-review", kb)
    (kb / "allowlist.txt").write_text("maintainer  # pilot reviewer\npilot-triager\n")
    return SimpleNamespace(
        repo=repo,
        common=common,
        base=base,
        head=head,
        next_head=next_head,
        docs_head=docs_head,
        withdrawn_head=withdrawn_head,
    )


class PullRequest:
    """GitHub records shared across CLI phases. IDs distinguish users, summaries and inline findings."""

    def __init__(self, history):
        self.pull = dict(
            title="Change two features",
            body=None,
            state="open",
            user={"login": "author"},
            base={"sha": history.base, "ref": "main"},
            head={"sha": history.head},
        )
        self.labels = {"ai-review"}
        self.comments = {}
        self.inline = {}
        self.statuses = []

    def post(self, body, user_id=1):
        number = 100 + len(self.comments)
        self.comments[number] = dict(id=number, body=body, user={"id": user_id})
        return self.comments[number]

    def command(self, body="/ai-review", actor="maintainer", kind="User"):
        comment = self.post(body, user_id=2)
        return dict(issue={"number": 7}, comment=comment.copy(), sender={"login": actor, "type": kind})

    def label_event(self, action="labeled", actor="maintainer"):
        return dict(
            action=action, pull_request={"number": 7, "head": self.pull["head"].copy()}, sender={"login": actor}
        )

    def summaries(self):
        return [c for c in self.comments.values() if c["user"]["id"] == 1 and c["body"].startswith(ai.STATE_MARKER)]

    def visible_summary(self):
        [comment] = self.summaries()
        return comment["body"].split(ai.STATE_DATA_MARKER)[0]

    def state(self):
        [comment] = self.summaries()
        data = comment["body"].split(ai.STATE_DATA_MARKER)[-1].removesuffix("\n-->")
        return json.loads(data)

    def seed(self, **fields):
        state = ai.State(base=self.pull["base"]["sha"], head=self.pull["head"]["sha"], **fields)
        return self.post(state.render_summary())

    def api(self, endpoint, *args, payload=None, anchor_fallback=False):
        path = endpoint.removeprefix("repos/:owner/:repo/")
        if path == "user":
            return {"id": 1}
        if path == "pulls/7":
            return self.pull | {"labels": [{"name": name} for name in self.labels]}
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
        ("/ai-review F001 fixed", "pilot-triager", True, "⏳ **Review requested**"),
        ("/ai-review", "author", False, "🔒 **Permission denied**"),
        ("/ai-review", "writer", False, "🔒 **Permission denied**"),
        ("/ai-review-stop", "author", False, "🔒 **Permission denied**"),
        ("/ai-review-stop", "pilot-triager", True, "⏳ **Stop requested**"),
        ("/ai-review-waive all intentional", "author", False, "🔒 **Permission denied**"),
        ("/ai-review-waive all intentional", "pilot-triager", True, "⏳ **Waiver requested**"),
        ("/ai-reviewer", "author", False, None),
    ],
)
def test_command_permissions(pr, phase, body, actor, allowed, feedback):
    event = pr.command(body, actor)
    assert phase("authorize", event) == ({"authorized": "true"} if allowed else {})
    [comment] = pr.comments.values()
    if feedback is None:
        assert comment["body"] == body
    else:
        original, response = comment["body"].split("\n\n---\n")
        assert original == body and len(response.splitlines()) == 1
        assert response.startswith(f"[🤖 AI review bot]: {feedback} · ")
    if allowed:
        assert "[View run](https://github.com/example/project/actions/runs/101)" in comment["body"]
    elif feedback:
        assert "allowlist.txt" in comment["body"] and pr.labels == {"ai-review"}


@pytest.mark.parametrize("actor", ["writer", "unlisted-triager"])
def test_label_checks_the_reviewer_and_removes_it_when_denied(pr, phase, actor):
    assert phase("authorize", pr.label_event(actor=actor)) == {}
    [notice] = pr.comments.values()
    assert notice["body"].startswith(f"@{actor} [🤖 AI review bot]: 🔒 **Permission denied**")
    assert "RFC #21451" in notice["body"]
    assert not pr.labels


def test_triage_label_removal_stops_the_session(pr, phase, history):
    pr.seed(status="passed", round=1, total=1)
    pr.labels = {"ai-review-passed"}  # GitHub already removed ai-review.
    stop = pr.label_event("unlabeled", actor="unlisted-triager")
    assert phase("authorize", stop) == {"authorized": "true"}
    assert not pr.labels
    assert phase("prepare", stop) == {}
    assert pr.state()["status"] == "stopped" and pr.state()["total"] == 1
    assert "🛑 **Review stopped**" in pr.visible_summary()
    assert pr.statuses == [(history.head, "error")]


def test_bot_command_is_ignored_even_when_allowlisted(pr, phase):
    event = pr.command("/ai-review-stop", actor="maintainer", kind="Bot")
    assert phase("authorize", event) == {}
    assert pr.comments[event["comment"]["id"]]["body"] == "/ai-review-stop"
    assert pr.labels == {"ai-review"}


@pytest.mark.parametrize("change,notice", [("closed", "**PR closed**"), ("unlabeled", "**Review not enabled**")])
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
    assert original["body"].count("[🤖 AI review bot]:") == 1
    assert "**Review requested**" not in original["body"]
    assert phase("authorize", pr.command()) == {}
    assert not pr.summaries()
    agent.assert_not_called()


# --- Review, pass and manual follow-up ---


@pytest.mark.parametrize(
    "paths,expected",
    [
        (["vllm_ascend/worker/model_runner.py"], {"model-execution"}),
        (["README.md"], {"doc-quality"}),
        (["README.zh.md"], {"doc-quality"}),
        (["docs/source/getting_started.md", "vllm_ascend/worker/model_runner.py"], {"doc-quality", "model-execution"}),
        (["vllm_ascend/ops/triton/docs/kernels.md"], {"doc-quality"}),
        (["docs/source/locale/zh_CN/LC_MESSAGES/user_guide/getting_started.po"], {"doc-quality"}),
        (["docs/source/locale/de/LC_MESSAGES/guide.po", "examples/README.md"], set()),
        ([".github/agent-review/instructions.md"], {"ci"}),
    ],
)
def test_doc_quality_path_routing(history, paths, expected):
    registry = (history.repo / ".github/agent-review/domains.yaml").read_text()
    assert {domain["name"] for domain in ai.resolve_domains(registry, paths)} == expected


def test_doc_finding_can_be_resolved_after_the_author_withdraws_the_change(pr, phase, agent, history):
    pr.pull["head"]["sha"] = history.docs_head
    assert phase("prepare", pr.command())["doc_quality"] == "true"
    doc = finding() | dict(
        path="README.md",
        rule="L2-doc-quality#upstream-rule",
        title="The installer rejects the documented flag",
        body="The installer exits when it sees --unknown-flag.",
        suggestion="Remove the unsupported flag from the example.",
        replacement={"start_line": 1, "code": "Run the installer."},
    )
    agent.return_value = result(findings=[doc])
    phase("review")
    diff_stat = agent.call_args.args[1]["diff_stat"]
    assert "README.md" in diff_stat and "feature.py" in diff_stat
    [first] = pr.state()["findings"]
    assert first["status"] == "open" and "ai-review-passed" not in pr.labels
    [thread] = pr.inline.values()
    body = thread["body"]
    sections = [doc["title"], "Problem", doc["body"], "Suggested fix", doc["suggestion"], "```diff"]
    positions = [body.index(text) for text in sections]
    assert positions == sorted(positions)
    preview = "-Run the installer with --unknown-flag.\n+Run the installer."
    assert f"{doc['suggestion']}\n\n```diff\n{preview}\n```" in body and "```suggestion" not in body
    assert first["id"] in body and doc["severity"] in body and doc["rule"] in body
    assert ".github/agent-review/badges/must-fix.svg" in body

    pr.pull["head"]["sha"] = history.withdrawn_head
    reply = f"/ai-review {first['id']}: I reverted the README edit and kept the code changes."
    prepared = phase("prepare", pr.command(reply))
    assert prepared == {"review": "true", "head": history.withdrawn_head, "vllm_pin_file": "vllm-main-verified.commit"}
    evidence = "README.md matches the base version. The command no longer mentions --unknown-flag."
    agent.return_value = result(updates=[dict(id=first["id"], status="resolved", evidence=evidence)])
    phase("review")
    request = agent.call_args.args[1]
    assert "README.md" not in request["diff_stat"] and "doc-quality" not in request["domains"]
    for sha in (request["merge_base"], request["state"]["head"]):
        assert ai.git(history.repo, "show", f"{sha}:README.md") == "Run the installer.\n"
    assert request["reply"] == reply and request["state"]["findings"] == [first]
    [reviewed] = pr.state()["findings"]
    assert (reviewed["id"], reviewed["status"], reviewed["evidence"]) == (first["id"], "resolved", evidence)
    assert reviewed["severity"] == first["severity"]
    assert len(pr.inline) == 1 and pr.statuses[-1] == (history.withdrawn_head, "success")
    resolved = thread["body"]
    assert f"**{first['id']}** · Resolved" in resolved
    assert "<details" not in resolved
    assert resolved.index(doc["body"]) < resolved.index(doc["suggestion"]) < resolved.index(preview)
    assert resolved.endswith(f"**Resolution**\n\n{evidence}")
    assert all(f"{level} · 0 open" in pr.visible_summary() for level in ai.SEVERITIES)

    pr.pull["head"]["sha"] = history.docs_head
    phase("prepare", pr.command())
    regression = "The example uses the unsupported flag again."
    agent.return_value = result(updates=[dict(id=first["id"], status="open", evidence=regression)])
    phase("review")
    assert list(pr.inline) == [first["comment_id"]]
    assert "Open" in thread["body"] and evidence not in thread["body"]
    assert thread["body"].index(preview) < thread["body"].index(f"**Latest evidence**\n\n{regression}")


def test_label_then_manual_review_of_all_changes(pr, phase, agent, history):
    start = pr.label_event(actor="pilot-triager")
    assert phase("authorize", start) == {"authorized": "true"}
    assert phase("prepare", start) == {
        "review": "true",
        "head": history.head,
        "vllm_pin_file": "vllm-main-verified.commit",
    }
    assert not pr.statuses
    agent.return_value = result(findings=[finding(), finding(line=2), finding(line=None, severity="NIT")])
    phase("review")
    [summary] = pr.summaries()
    assert "DeepSeek Harness headless · POC pilot" in summary["body"]
    assert [comment["line"] for comment in pr.inline.values()] == [1, 2]
    visible = pr.visible_summary()
    assert visible.startswith(f"{ai.STATE_MARKER}\n\n## 🤖 AI review")
    assert "❌ **Changes required**" in visible and "2 open MUST-FIX findings" in visible
    assert "🔍 **Reviewing**" not in visible
    assert "F003" in visible and finding()["body"] in visible and finding()["suggestion"] in visible
    assert pr.state()["status"] == "reviewing" and pr.state()["total"] == 1
    assert pr.statuses == [(history.head, "pending"), (history.head, "failure")]

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
    assert request["title"] == pr.pull["title"] and request["body"] == ""
    assert request["merge_base"] == history.common and request["state"]["base"] == history.base
    assert request["diff_stat"].strip() == ai.git(history.repo, "diff", "--stat", history.common, history.head).strip()
    assert not {"diff", "files", "truncated"}.intersection(request)
    assert [f["id"] for f in request["state"]["findings"]] == ["F001", "F002", "F003"]
    assert [f["status"] for f in pr.state()["findings"]] == ["resolved", "resolved", "open"]
    assert pr.state()["total"] == 2 and pr.statuses[-1] == (history.head, "success")
    assert "ai-review-passed" in pr.labels and pr.summaries() == [summary]
    assert "✅ **Review passed**" in pr.visible_summary() and "No blocking findings" in pr.visible_summary()

    old_push = pr.label_event("synchronize", "author")
    pr.pull["head"] = {"sha": history.next_head}
    assert phase("authorize", old_push) == {} and "ai-review-passed" in pr.labels
    assert phase("authorize", pr.label_event("synchronize", "author")) == {}
    assert "ai-review-passed" not in pr.labels and agent.call_count == 2 and pr.state()["total"] == 2


def test_reclassifying_a_finding_updates_its_thread_and_verdict(pr, phase, agent, history):
    phase("prepare", pr.command())
    agent.return_value = result(findings=[finding(severity="SHOULD-FIX")])
    phase("review")
    [original] = pr.state()["findings"]
    comment_id = original["comment_id"]
    assert "ai-review-passed" in pr.labels

    evidence = "The supported startup path calls this before input validation and raises."
    phase("prepare", pr.command())
    agent.return_value = result(updates=[dict(id="F001", status="open", severity="MUST-FIX", evidence=evidence)])
    phase("review")
    [updated] = pr.state()["findings"]
    assert updated == original | {"severity": "MUST-FIX", "evidence": evidence}
    assert f"#discussion_r{comment_id}" in pr.visible_summary()
    assert "MUST-FIX" in pr.inline[comment_id]["body"] and evidence in pr.inline[comment_id]["body"]
    assert pr.state()["status"] == "reviewing" and "ai-review-passed" not in pr.labels
    assert pr.statuses[-1] == (history.head, "failure")

    evidence = "The startup caller already handles empty input. Only the diagnostic text is misleading."
    phase("prepare", pr.command())
    agent.return_value = result(
        updates=[dict(id="F001", status="open", severity="SHOULD-FIX", evidence=evidence)],
        findings=[finding(severity="MUST-FIX")],
    )
    phase("review")
    assert pr.state()["findings"] == [original | {"evidence": evidence}]
    assert list(pr.inline) == [comment_id]
    assert "SHOULD-FIX" in pr.inline[comment_id]["body"] and evidence in pr.inline[comment_id]["body"]
    assert pr.state()["status"] == "passed" and "ai-review-passed" in pr.labels
    assert pr.statuses[-1] == (history.head, "success")


def test_repeated_finding_promotes_its_existing_inline_comment(pr, phase, agent):
    phase("prepare", pr.command())
    agent.return_value = result(findings=[finding(severity="SHOULD-FIX")])
    phase("review")
    phase("prepare", pr.command())
    evidence = "The caller now handles empty input."
    agent.return_value = result(updates=[dict(id="F001", status="resolved", evidence=evidence)])
    phase("review")
    [original] = pr.inline.values()
    assert evidence in original["body"]

    phase("prepare", pr.command())
    agent.return_value = result(findings=[finding(severity="MUST-FIX")])
    phase("review")
    [saved] = pr.state()["findings"]
    [comment] = pr.inline.values()
    assert saved["severity"] == "MUST-FIX" and saved["comment_id"] == original["id"] == comment["id"]
    assert "F001" in comment["body"] and "MUST-FIX" in comment["body"]
    assert saved["status"] == "open" and not saved["evidence"] and evidence not in comment["body"]


def test_summary_groups_count_and_link_to_existing_threads(pr, phase, agent, history):
    pr.pull["head"]["sha"] = history.docs_head
    phase("prepare", pr.command())
    changes = {"feature.py": "MUST-FIX", "second.py": "SHOULD-FIX", "third.py": "NIT", "README.md": "MUST-FIX"}
    agent.return_value = result(
        findings=[finding(severity=severity) | {"path": path} for path, severity in changes.items()]
    )
    phase("review")
    phase("prepare", pr.command())
    agent.return_value = result(updates=[dict(id="F004", status="resolved", evidence="Covered by the caller.")])
    phase("review")
    phase("prepare", pr.command("/ai-review-waive F002 intentional"))

    public = pr.visible_summary()
    assert "| 🔴 MUST-FIX | 1 | 1 | 0" in public
    assert "| 🟤 SHOULD-FIX | 0 | 0 | 1" in public
    assert "| ⚪ NIT | 1 | 0 | 0" in public
    assert "Total (4)** | 2 | 1 | 1" in public
    assert finding()["body"] not in public and finding()["suggestion"] not in public
    for saved in pr.state()["findings"]:
        assert f"[{saved['id']}](https://github.com/example/project/pull/7#discussion_r{saved['comment_id']})" in public
        assert f"{saved['path']}:{saved['line']}" in public

    assert "<details open" not in public
    headings = re.findall(r"<details>\s*<summary>(.*?)</summary>", public)
    assert headings[:5] == [
        "🔴 MUST-FIX · 1 open",
        "🟤 SHOULD-FIX · 0 open",
        "⚪ NIT · 1 open",
        "Resolved · 1",
        "Waived · 1",
    ]


@pytest.mark.parametrize("status,severity", [("open", "HIGH"), ("open", None), ("waived", "MUST-FIX")])
def test_invalid_reclassification_is_rejected(status, severity):
    record = ai.Finding(id="F001", status=status, head="abc", **finding(severity="SHOULD-FIX"))
    output = result(updates=[dict(id="F001", status="open", severity=severity, evidence="Reviewed the caller.")])
    with pytest.raises(ValueError):
        ai.parse_result(output, [record])


@pytest.mark.parametrize("field", ["title", "body", "suggestion"])
def test_new_findings_require_each_text_section(field):
    new = finding()
    del new[field]
    with pytest.raises(ValueError):
        ai.parse_result(result(findings=[new]), [])


def test_saved_finding_from_previous_format_remains_readable():
    old = {key: value for key, value in finding().items() if key not in {"title", "suggestion"}}
    saved = ai.State.from_dict({"findings": [old | {"id": "F001", "head": "abc"}]})
    [restored] = saved.findings
    assert old["body"] in restored.inline_body()
    visible = saved.render_summary().split(ai.STATE_DATA_MARKER)[0]
    assert "F001" in visible and old["body"] in visible
    assert "Suggested fix" not in restored.inline_body() and "```diff" not in visible


def test_markdown_preview_cannot_close_the_hidden_review_state():
    preview = "-<!-- old note -->\n+<!-- revised note -->"
    state = ai.State(findings=[ai.Finding(id="F001", head="abc", comment_id=500, preview=preview, **finding())])
    hidden = state.render_summary().split(ai.STATE_DATA_MARKER)[1]
    json_text, closing_marker, remainder = hidden.partition("-->")
    assert closing_marker and remainder == ""
    [stored] = ai.State.from_dict(json.loads(json_text)).findings
    assert stored.preview == preview


def test_multiline_preview_ends_at_the_reported_line(pr, phase, agent):
    new = finding(line=3) | {"replacement": {"start_line": 2, "code": "items = [2, 3]"}}
    phase("prepare", pr.command())
    agent.return_value = result(findings=[new])
    phase("review")
    [comment] = pr.inline.values()
    assert comment["line"] == 3
    assert "-second_item = 2\n-third_item = 3\n+items = [2, 3]" in comment["body"]
    assert "first_change" not in comment["body"]


@pytest.mark.parametrize("start,line", [(2, 1), (1, 9)])
def test_unusable_replacement_keeps_the_text_suggestion(pr, phase, agent, start, line):
    new = finding(line=line) | {"replacement": {"start_line": start, "code": "first_change = 2"}}
    phase("prepare", pr.command())
    agent.return_value = result(findings=[new])
    phase("review")
    [comment] = pr.inline.values()
    assert new["suggestion"] in comment["body"] and "```diff" not in comment["body"]


@pytest.mark.parametrize(
    "complete,work",
    [(True, ["Trace the caller."]), (False, []), (False, "Trace the caller."), (False, [" "]), (False, [42])],
)
def test_invalid_unfinished_work_is_rejected(complete, work):
    with pytest.raises(ValueError, match="unfinished_work"):
        ai.parse_result(result(coverage_complete=complete, unfinished_work=work), [])


def test_unfinished_work_is_required_even_when_the_review_is_complete():
    payload = json.loads(result())
    del payload["unfinished_work"]
    with pytest.raises(ValueError, match="unfinished_work"):
        ai.parse_result(json.dumps(payload), [])


@pytest.mark.parametrize(
    "blocked_by,expected_status",
    [
        ("incomplete", "⏳ **Review incomplete**"),
        ("human", "👤 **Maintainer review required**"),
        ("omitted finding", "❌ **Changes required**"),
    ],
)
def test_no_pass_when_review_is_blocked(pr, phase, agent, history, blocked_by, expected_status):
    if blocked_by == "omitted finding":
        pr.seed(findings=[ai.Finding(id="F001", head=history.head, **finding())])
    phase("prepare", pr.command())
    agent.return_value = result(
        coverage_complete=blocked_by != "incomplete",
        unfinished_work=["Trace the startup caller."] if blocked_by == "incomplete" else [],
        requires_human=blocked_by == "human",
    )
    phase("review")
    assert pr.state()["status"] != "passed" and "ai-review-passed" not in pr.labels
    assert pr.statuses[-1][1] in {"error", "failure"}
    assert expected_status in pr.visible_summary() and "🔍 **Reviewing**" not in pr.visible_summary()


def test_exhausted_rounds_preserve_other_reasons_for_blocking(pr, phase, agent):
    pr.seed(round=4, total=14)
    phase("prepare", pr.command())
    agent.return_value = result(requires_human=True, findings=[finding()])
    phase("review")
    visible = pr.visible_summary()
    assert "⚠️ **Round limit reached**" in visible and "Commit 5/5" in visible and "Session 15/15" in visible
    assert "👤 **Maintainer review required**" in visible and "❌ **Changes required**" in visible


def test_stop_blocks_inflight_publication(pr, phase, agent, history):
    phase("prepare", pr.command())
    [placeholder] = pr.summaries()
    stop = pr.command("/ai-review-stop", "maintainer")

    def stop_during_review(*_):
        assert "🔍 **Reviewing**" in pr.visible_summary()
        assert phase("authorize", stop) == {"authorized": "true"}
        assert not pr.labels and "⏳ **Stop requested**" in pr.comments[stop["comment"]["id"]]["body"]
        return result(findings=[finding()])

    agent.side_effect = stop_during_review
    phase("review")
    assert pr.summaries() == [placeholder] and not pr.inline and pr.state()["total"] == 0
    assert "🔍 **Reviewing**" not in pr.visible_summary()
    assert pr.statuses == [(history.head, "pending"), (history.head, "error")]
    assert phase("prepare", stop) == {}
    assert pr.state()["status"] == "stopped" and pr.state()["total"] == 0
    assert agent.call_count == 1


@pytest.mark.parametrize("push", [False, True])
def test_waived_findings_stay_archived_when_the_same_problem_is_reported_again(pr, phase, agent, history, push):
    phase("prepare", pr.command())
    agent.return_value = result(findings=[finding(), finding(line=2)])
    phase("review")
    for selector in ("F001", "all"):
        event = pr.command(f"/ai-review-waive {selector} upstream handles this", "maintainer")
        assert phase("authorize", event) == {"authorized": "true"}
        assert phase("prepare", event) == {}
        feedback = pr.comments[event["comment"]["id"]]["body"]
        assert feedback.count("[🤖 AI review bot]:") == 1 and "**Waiver requested**" not in feedback
        assert "✅ **Waiver applied**" in feedback and ("F001" if selector == "F001" else "F002") in feedback
        if selector == "F001":
            assert "ai-review-passed" not in pr.labels
            first, second = pr.state()["findings"]
            assert "Waived by @maintainer" in pr.inline[first["comment_id"]]["body"]
            assert "Waived by @maintainer" not in pr.inline[second["comment_id"]]["body"]
    assert len(pr.inline) == 2
    assert all(
        f["waived_head"] == history.head and f["evidence"] == "upstream handles this" for f in pr.state()["findings"]
    )
    assert "ai-review-passed" in pr.labels and pr.state()["total"] == agent.call_count == 1

    archived = pr.state()["findings"]
    archived_comments = {cid: comment["body"] for cid, comment in pr.inline.items()}
    for finding_record in archived:
        body = archived_comments[finding_record["comment_id"]]
        assert f"**{finding_record['id']}** · Waived" in body
        assert "<details" not in body and body.count("Waived by @maintainer") == 1
        assert finding_record["suggestion"] in body
        assert body.endswith(f"**Waived by @maintainer** at `{history.head[:12]}`\n\nupstream handles this")
    if push:
        pr.pull["head"]["sha"] = history.next_head
    phase("prepare", pr.command())
    agent.return_value = result(findings=[finding()])
    phase("review")
    first, second, new = pr.state()["findings"]
    assert [first, second] == archived
    assert (new["id"], new["status"], new["head"]) == ("F003", "open", pr.pull["head"]["sha"])
    assert "<!-- ai-review-finding:F003 -->" in pr.inline[new["comment_id"]]["body"]
    assert len(pr.inline) == 3 and "ai-review-passed" not in pr.labels
    assert (pr.state()["round"], pr.state()["total"]) == (1 if push else 2, 2)

    phase("prepare", pr.command())
    phase("review")
    assert pr.state()["findings"] == [*archived, new] and len(pr.inline) == 3
    assert all(pr.inline[cid]["body"] == body for cid, body in archived_comments.items())


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
    assert f"[🤖 AI review bot]: ⚠️ **Waiver rejected** · {feedback}" in pr.comments[event["comment"]["id"]]["body"]
    assert summary["body"] == before


# --- Published rounds and saved findings ---


@pytest.mark.parametrize(
    "previous_total,expected_limit",
    [(8, "5/5 rounds used for this commit"), (14, "15/15 rounds used for this session")],
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
    original, rejected = pr.comments[pending["comment"]["id"]]["body"].split("\n\n---\n")
    assert original == pending["comment"]["body"] and len(rejected.splitlines()) == 1
    assert rejected.startswith(f"[🤖 AI review bot]: ⚠️ **Round limit reached** · {expected_limit}")
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
    assert "🛑 **Review stopped**" in pr.comments[comment["comment"]["id"]]["body"]
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
        pr.pull["base"]["sha"] = history.common
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
    assert pr.statuses == [(history.head, "pending"), (history.head, "error")]
    assert "🔄 **Review discarded**" in pr.visible_summary()


@pytest.mark.parametrize(
    "bad_result,error",
    [
        (RuntimeError("model unavailable"), RuntimeError),
        ("AI-REVIEW-VERDICT: PASS", ValueError),
        (result(updates=[dict(id="F999", status="resolved", evidence="Fixed.")]), ValueError),
    ],
)
def test_unsuccessful_analysis_can_be_rerun_without_losing_a_round(pr, phase, agent, history, bad_result, error):
    previous = pr.seed(
        status="passed",
        coverage_complete=True,
        round=2,
        total=3,
        last_review={"summary": "Previous result from this commit."},
    )
    saved = pr.state()
    event = pr.command()
    agent.side_effect = [bad_result, result()]
    phase("prepare", event)
    with pytest.raises(error):
        phase("review")
    assert pr.statuses == [(history.head, "pending"), (history.head, "error")]
    assert pr.summaries() == [previous] and pr.state() == saved
    visible = pr.visible_summary()
    assert "❌ **Review failed** · [View logs](https://github.com/example/project/actions/runs/101)" in visible
    assert "Previous result from this commit." in visible and "🔍 **Reviewing**" not in visible
    phase("prepare", event)
    phase("review")
    assert agent.call_count == 2
    assert (pr.state()["round"], pr.state()["total"]) == (3, 4)
    assert "**Review failed**" not in pr.visible_summary()


def test_first_round_placeholder_precedes_findings_and_survives_a_failed_attempt(pr, phase, agent, monkeypatch):
    api = Mock(wraps=pr.api)
    monkeypatch.setattr(ai, "gh", api)
    event = pr.label_event()
    phase("prepare", event)
    [placeholder] = pr.summaries()
    assert pr.visible_summary() == f"{ai.STATE_MARKER}\n\n{ai.SUMMARY_TITLE}"
    assert (pr.state()["round"], pr.state()["total"]) == (0, 0)

    def review(*_):
        assert "🔍 **Reviewing** · Round 1/5" in pr.visible_summary()
        assert "**Review failed**" not in pr.visible_summary() and (pr.state()["round"], pr.state()["total"]) == (0, 0)
        if agent.call_count == 1:
            raise RuntimeError("model unavailable")
        return result(findings=[finding()])

    agent.side_effect = review
    with pytest.raises(RuntimeError, match="model unavailable"):
        phase("review")
    assert "❌ **Review failed**" in pr.visible_summary()
    assert (pr.state()["round"], pr.state()["total"]) == (0, 0)
    phase("prepare", event)
    phase("review")

    writes = [call.args[0] for call in api.call_args_list if "body" in (call.kwargs.get("payload") or {})]
    assert writes[0] == "repos/:owner/:repo/issues/7/comments" and writes.count(writes[0]) == 1
    assert writes[-2:] == [
        "repos/:owner/:repo/pulls/7/comments",
        f"repos/:owner/:repo/issues/comments/{placeholder['id']}",
    ]
    assert pr.summaries() == [placeholder] and "**Review failed**" not in pr.visible_summary()
    assert (pr.state()["round"], pr.state()["total"]) == (1, 1)
    assert "#discussion_r500" in pr.visible_summary()
    [thread] = pr.inline.values()
    assert "/ai-review" not in thread["body"]


@pytest.mark.parametrize("write,expected_total", [("summary", 3), ("status", 4)])
def test_only_a_saved_summary_spends_the_round(pr, phase, monkeypatch, write, expected_total):
    summary = pr.seed(round=2, total=3)
    phase("prepare", pr.command())

    def fail_write(endpoint, *args, payload=None, **kwargs):
        if (
            write == "summary"
            and endpoint.endswith(f"issues/comments/{summary['id']}")
            and payload is not None
            and "✅ **Review passed**" in payload["body"]
        ):
            raise RuntimeError("write failed")
        if write == "status" and "/statuses/" in endpoint and payload["state"] == "success":
            raise RuntimeError("write failed")
        if "/statuses/" in endpoint and payload["state"] == "error":
            raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])
        return pr.api(endpoint, *args, payload=payload, **kwargs)

    monkeypatch.setattr(ai, "gh", fail_write)
    with pytest.raises(RuntimeError, match="write failed"):
        phase("review")
    assert pr.state()["total"] == expected_total


def test_failure_to_post_logs_link_preserves_the_agent_error(pr, phase, agent, monkeypatch):
    phase("prepare", pr.command())
    agent.side_effect = ValueError("invalid model output")

    def fail_notice(endpoint, *args, payload=None, **kwargs):
        if payload and "**Review failed**" in payload.get("body", ""):
            raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])
        return pr.api(endpoint, *args, payload=payload, **kwargs)

    monkeypatch.setattr(ai, "gh", fail_notice)
    with pytest.raises(ValueError, match="invalid model output"):
        phase("review")
    assert pr.state()["total"] == 0


def test_head_changed_after_summary_write_gets_no_pass(pr, phase, monkeypatch, history):
    phase("prepare", pr.command())
    [summary] = pr.summaries()

    def push_after_summary(endpoint, *args, payload=None, **kwargs):
        response = pr.api(endpoint, *args, payload=payload, **kwargs)
        if (
            endpoint.endswith(f"/issues/comments/{summary['id']}")
            and payload
            and "✅ **Review passed**" in payload["body"]
        ):
            pr.pull["head"] = {"sha": history.next_head}
        return response

    monkeypatch.setattr(ai, "gh", push_after_summary)
    phase("review")
    assert pr.state()["total"] == 1 and pr.state()["head"] == history.head
    assert pr.state()["status"] == "passed"
    assert "ai-review-passed" not in pr.labels
    assert pr.statuses == [(history.head, "pending"), (history.head, "error")]


def test_restore_latest_bot_summary_across_comment_pages(pr, phase, agent, history):
    for _ in range(100):
        pr.post("Earlier discussion.", user_id=2)
    pr.seed(round=1, total=2)
    expected = pr.seed(round=2, total=3, findings=[ai.Finding(id="F001", head=history.head, **finding())])
    pr.post(ai.State(status="passed", total=15).render_summary(), user_id=2)
    for _ in range(100):
        pr.post("Later discussion.", user_id=2)
    assert phase("prepare", pr.command())
    phase("review")
    request = agent.call_args.args[1]
    assert request["comment_id"] == expected["id"] and request["state"]["total"] == 3
    assert request["state"]["findings"][0]["id"] == "F001"
    assert "4/15" in expected["body"].split(ai.STATE_DATA_MARKER)[0]


def test_rejected_inline_finding_stays_in_summary_across_reviews(pr, phase, agent, monkeypatch):
    attempts = []

    def reject_anchor(endpoint, *args, **kwargs):
        if endpoint.endswith("/pulls/7/comments"):
            assert kwargs["anchor_fallback"]
            attempts.append(kwargs["payload"]["line"])
            return None
        return pr.api(endpoint, *args, **kwargs)

    monkeypatch.setattr(ai, "gh", reject_anchor)
    new = finding() | {"replacement": {"start_line": 1, "code": "first_change = 2"}}
    agent.return_value = result(findings=[new])
    for _ in range(2):
        phase("prepare", pr.command())
        phase("review")
    assert attempts == [1] and not pr.inline
    [saved] = pr.state()["findings"]
    assert saved["id"] == "F001" and saved["comment_id"] is None
    public = pr.visible_summary()
    assert all(new[field] in public for field in ("title", "body", "suggestion"))
    assert "-first_change = 1\n+first_change = 2" in public and "#discussion_r" not in public
    findings, footer = public.split("Human review is still required.")
    assert "/ai-review" not in findings
    assert (
        "top-level" in footer
        and "https://github.com/example/project/blob/main/.github/agent-review/allowlist.txt" in footer
    )
    assert "`/ai-review F001" in footer
    for command in ("/ai-review", "/ai-review-waive F001 reason", "/ai-review-stop"):
        assert f"`{command}`" in footer
    phase("prepare", pr.command())
    agent.return_value = result(
        updates=[dict(id="F001", status="open", severity="SHOULD-FIX", evidence="Impact narrowed.")]
    )
    phase("review")
    assert pr.state()["findings"][0]["severity"] == "SHOULD-FIX" and attempts == [1] and not pr.inline
    assert "Impact narrowed." in pr.visible_summary()
    assert phase("prepare", pr.command("/ai-review-waive F001 accepted", "maintainer")) == {}
    assert pr.state()["status"] == "passed" and pr.state()["total"] == 3


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
        assert pr.state()["total"] == 0
    else:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            phase("prepare", pr.command())
        assert caught.value is error
        assert not pr.statuses
        assert not pr.summaries()
    assert len(calls) == attempts and len(set(calls)) == 1
    assert [item.args for item in sleep.call_args_list] == [(ai.GIT_FETCH_RETRY_DELAY_SECONDS,)] * (attempts - 1)


def test_release_branch_uses_its_pinned_vllm_tag(pr, phase):
    pr.pull["base"]["ref"] = "releases/v0.34.0"
    assert phase("prepare", pr.command())["vllm_pin_file"] == "vllm-release-tag.commit"


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
def agent_workspace(tmp_path, history):
    """Keep the trusted review files separate from the detached PR checkout."""
    shutil.copytree(history.repo / ".github/agent-review", tmp_path / ".github/agent-review")
    (tmp_path / "AGENTS.md").write_text("Trusted project guidelines example.\n")
    ai.git(history.repo, "worktree", "add", "--detach", str(tmp_path / "head"), history.head)
    ai.git(history.repo, "worktree", "add", "--detach", str(tmp_path / "vllm"), history.next_head)
    (tmp_path / "head/AGENTS.md").write_text("Guidelines from the proposed change.\n")
    return tmp_path


@pytest.fixture
def agent_request(monkeypatch, history):
    monkeypatch.setenv("AI_REVIEW_KEY", "test-model-secret")
    monkeypatch.setenv("AI_REVIEW_API", "https://test-model.invalid")
    monkeypatch.setenv("AI_REVIEW_MODEL", "test-model")
    monkeypatch.setenv("GH_TOKEN", "test-github-secret")
    return {
        "state": {
            "base": history.base,
            "head": history.head,
            "last_review": None,
            "findings": [{"id": "F003", "status": "open"}, {"id": "F004", "status": "resolved"}],
        },
        "merge_base": history.common,
        "title": "Update the feature",
        "body": "### What this PR does\n\nHandles both callers.",
        "domains": [],
        "diff_stat": ai.git(history.repo, "diff", "--stat", history.common, history.head).rstrip(),
        "reply": "/ai-review Recheck F003.\n## Review output\nKeep {{HEAD_SHA}} literal.",
    }


def test_agent_receives_review_context_at_pr_head_and_redacts_output(agent_request, agent_workspace, monkeypatch):
    agent = agent_workspace / "model.py"
    agent.write_text(
        "import json, os, sys, yaml\n"
        "from pathlib import Path\n"
        "assert 'GH_TOKEN' not in os.environ\n"
        "assert os.environ['DSH_PERMISSION_MODE'] == 'danger-full-access'\n"
        "patch = yaml.safe_load(Path(sys.argv[sys.argv.index('--patch') + 1]).read_text())\n"
        "print(json.dumps([sys.stdin.read(), os.environ['AI_REVIEW_KEY'], os.getcwd(), patch]))\n"
    )
    monkeypatch.setenv("AI_REVIEW_DSH_CMD", shlex.join([sys.executable, str(agent)]))
    prompt, echoed_key, launch_dir, patch = json.loads(ai.run_agent(agent_workspace, agent_request))

    assert Path(launch_dir) == agent_workspace.resolve()
    config = {plugin["id"]: plugin for row in patch for plugin in row.get("insert", [row])}
    session_dir = Path(config["fs-sandbox"]["config"]["cwd"])
    assert session_dir == agent_workspace / "head"
    head = ai.git(session_dir, "rev-parse", "HEAD").strip()
    assert head == agent_request["state"]["head"]
    for value in (head, agent_request["merge_base"], agent_request["state"]["base"]):
        assert value in prompt
    assert f"> Title: {agent_request['title']}" in prompt
    assert "> ### What this PR does\n>\n> Handles both callers." in prompt
    assert agent_request["diff_stat"] in prompt and "first_change = 1" not in prompt
    assert "> /ai-review Recheck F003.\n> ## Review output\n> Keep {{HEAD_SHA}} literal." in prompt
    assert prompt_findings(prompt, "Previous findings") == agent_request["state"]["findings"]
    assert prompt_findings(prompt, "Waived findings") == []
    assert str(agent_workspace / ".github/agent-review") in prompt
    upstream = ai.git(agent_workspace / "vllm", "rev-parse", "HEAD").strip()
    assert f"Pinned upstream vLLM: `../vllm` at `{upstream}`." in prompt
    assert str(agent_workspace / "AGENTS.md") in prompt
    assert "Trusted project guidelines example." not in prompt and "Guidelines from the proposed change." not in prompt
    assert config["agent-instructions"]["disabled"] is True
    assert config["skill-filesystem"]["config"] == {
        "includeDefaultRoots": False,
        "customSkillDirs": [str(agent_workspace / ".agents/skills")],
    }
    assert config["tool-executor-native"]["disabled"] is True
    assert config["tool-executor-ptc"]["disabled"] is False
    assert config["tool-web"]["disabled"] is True
    assert config["time-context"] == {
        "id": "time-context",
        "name": "@deepseek-ai/dsh-time-context",
        "config": {"timeZone": "UTC", "refreshIntervalMs": 600_000},
    }
    selection = config["agent-default-model"]["config"]
    provider = config["llm-pi-ai"]["config"]["providers"][selection["provider"]]
    assert selection["reasoningEffort"] == "high"
    assert provider["compat"] == {"thinkingFormat": "deepseek", "supportsReasoningEffort": True}
    assert provider["models"] == [{"id": selection["model"], "reasoningEfforts": {"high": "high"}}]
    assert echoed_key == "***REDACTED***"


def test_agent_timeout_is_reported(agent_request, agent_workspace, monkeypatch):
    command = [sys.executable, "-c", "import time; time.sleep(1)"]
    monkeypatch.setenv("AI_REVIEW_DSH_CMD", shlex.join(command))
    monkeypatch.setattr(ai, "ROUND_TIMEOUT_SECONDS", 0)
    with pytest.raises(subprocess.TimeoutExpired):
        ai.run_agent(agent_workspace, agent_request)


@pytest.mark.parametrize("push", [False, True])
def test_manual_review_continues_unfinished_coverage(pr, phase, agent, history, agent_workspace, push):
    summary = "Checked the startup path in feature.py."
    work = ["Trace the callers in second.py.", "Check whether retry.py handles an empty response."]
    phase("prepare", pr.label_event())
    agent.return_value = result(
        summary=summary, unfinished_work=work, coverage_complete=False, findings=[finding(), finding(line=2)]
    )
    phase("review")
    public_comment = pr.visible_summary()
    assert summary in public_comment and all(item in public_comment for item in work)

    phase("prepare", pr.command("/ai-review-waive F001 accepted", "maintainer"))
    assert "waived" in pr.state()["message"]
    if push:
        pr.pull["head"]["sha"] = history.next_head

    phase("prepare", pr.command())
    assert agent.call_count == 1
    agent.return_value = result()
    phase("review")
    request = agent.call_args.args[1]
    prompt = ai.build_prompt(agent_workspace, request)
    snapshot = prompt_section(prompt, "Previous review snapshot")
    assert history.head in snapshot and history.common in snapshot and "false" in snapshot.lower()
    recalled = prompt_section(prompt, "Previous review summary")
    pending = prompt_section(prompt, "Unfinished work")
    assert summary in recalled and all(item not in recalled for item in work)
    assert all(item in pending for item in work) and summary not in pending
    archived, remaining = request["state"]["findings"]
    assert archived["status"] == "waived"
    assert prompt_findings(prompt, "Previous findings") == [
        {key: value for key, value in remaining.items() if key not in ("title", "suggestion", "preview")}
    ]
    assert prompt_findings(prompt, "Waived findings") == [
        {"id": "F001", "path": "feature.py", "line": 1, "body": archived["body"], "reason": "accepted"}
    ]
    assert pr.state()["last_review"] == {
        "head": pr.pull["head"]["sha"],
        "merge_base": history.common,
        "coverage_complete": True,
        "summary": "Review complete.",
        "unfinished_work": [],
    }
    assert pr.state()["total"] == 2


def test_complete_review_with_no_findings_survives_discard_and_stop(pr, phase, agent, history, agent_workspace):
    summary = "Reviewed both entry points and found no issues."
    phase("prepare", pr.command())
    agent.return_value = result(summary=summary)
    phase("review")
    first_prompt = ai.build_prompt(agent_workspace, agent.call_args.args[1])
    assert history.head not in prompt_section(first_prompt, "Previous review snapshot")

    phase("prepare", pr.command())
    pr.pull["head"]["sha"] = history.next_head
    agent.return_value = result(summary="Discarded review result.")
    phase("review")
    repeat_prompt = ai.build_prompt(agent_workspace, agent.call_args.args[1])
    assert pr.state()["status"] == "discarded"

    stop = pr.command("/ai-review-stop")
    phase("authorize", stop)
    phase("prepare", stop)
    assert pr.state()["status"] == "stopped"
    pr.labels.add("ai-review")
    phase("prepare", pr.label_event(), run_id="102")
    agent.return_value = result()
    phase("review")
    restart_prompt = ai.build_prompt(agent_workspace, agent.call_args.args[1])

    for prompt in (repeat_prompt, restart_prompt):
        snapshot = prompt_section(prompt, "Previous review snapshot")
        assert history.head in snapshot and history.common in snapshot and "true" in snapshot.lower()
        assert summary in prompt_section(prompt, "Previous review summary")
        assert summary not in prompt_section(prompt, "Unfinished work")
        assert "Discarded review result." not in prompt


@pytest.mark.parametrize(
    "status,message,complete,expected",
    [
        ("reviewing", "Still need to check dispatch.", False, "Still need to check dispatch."),
        ("waiting-human", "Remaining work: allocator.", False, "Remaining work: allocator."),
        ("reviewing", "@reviewer waived 1 finding(s) for this commit.", False, ""),
        ("discarded", "The PR changed during review.", False, ""),
        ("stopped", "Review stopped.", False, ""),
        ("reviewing", "Reviewed all code.", True, "Reviewed all code."),
    ],
)
def test_restore_previous_coverage_from_older_summary(status, message, complete, expected):
    state = ai.State.from_dict(
        {
            "status": status,
            "head": "old-head",
            "base": "old-base",
            "total": 1,
            "message": message,
            "coverage_complete": complete,
            "findings": [],
        }
    )
    assert state.last_review == ({"summary": expected} if expected else None)


def test_activity_keeps_original_hidden_json_from_an_older_review(pr, phase, agent, history):
    saved = (
        json.dumps(
            {
                "status": "reviewing",
                "head": history.head,
                "base": history.base,
                "incomplete_review_summary": "Check the remaining caller.",
                "round": 1,
                "total": 2,
                "findings": [],
            },
            indent=2,
        )
        + "\n-->"
    )
    summary = pr.post(f"{ai.STATE_MARKER}\n\nPrevious review{ai.STATE_DATA_MARKER}{saved}")

    def unavailable(*_):
        assert summary["body"].endswith(ai.STATE_DATA_MARKER + saved)
        assert "**Reviewing**" in pr.visible_summary()
        raise RuntimeError("model unavailable")

    agent.side_effect = unavailable
    phase("prepare", pr.command())
    with pytest.raises(RuntimeError, match="model unavailable"):
        phase("review")
    assert "**Review failed**" in pr.visible_summary()
    assert "Check the remaining caller." in pr.visible_summary()
    assert summary["body"].endswith(ai.STATE_DATA_MARKER + saved)


def test_previous_format_retains_model_prose_across_a_waiver():
    summary = "Checked feature.py. Still need to review second.py."
    state = ai.State.from_dict(
        {
            "status": "reviewing",
            "head": "old-head",
            "total": 1,
            "coverage_complete": False,
            "message": "@maintainer waived 1 finding(s) at commit `old-head`.",
            "incomplete_review_summary": summary,
            "findings": [],
        }
    )
    assert state.last_review == {"summary": summary}
