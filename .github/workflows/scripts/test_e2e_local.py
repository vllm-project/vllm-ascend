#
# Copyright (c) 2026 vLLM Ascend contributors.
# Licensed under the Apache License, Version 2.0.
#
FAKE_GH = r"""#!/usr/bin/env python3
# Fake gh: records calls, answers each -q expression the way real gh does.
import json, os, sys

STATE = json.load(open(os.environ["FAKE_GH_STATE"]))
log = open(os.environ["FAKE_GH_LOG"], "a")
log.write(json.dumps(sys.argv[1:]) + "\n")
log.close()

args = sys.argv[1:]
query = args[args.index("-q") + 1] if "-q" in args else None
if args[:2] == ["pr", "diff"]:
    print("diff --git a/x.py b/x.py\n+import os  # changed")
    sys.exit(0)
if args[:2] == ["pr", "comment"]:
    open(os.environ["FAKE_GH_COMMENTS"], "a").write(
        "COMMENT:" + sys.stdin.read() + "\n")
    sys.exit(0)
if args[:2] == ["pr", "edit"]:
    sys.exit(0)
if args[0] == "api":
    tail = " ".join(args[1:])
    if "collaborators" in tail:
        print(json.dumps({"permission": STATE["actor_permission"]}))
        sys.exit(0)
    if "pulls/1/comments" in tail:
        print(json.dumps(STATE.get("review_comments", [])))
        sys.exit(0)
    if "/reviews" in tail:
        open(os.environ["FAKE_GH_REVIEWS"], "a").write(
            open(args[args.index("--input") + 1]).read() + "\n")
        sys.exit(0)
if query == "[.files[].path]":
    print(json.dumps(["x.py", ".github/workflows/ai_review.yml"]))
elif query == ".headRefOid":
    print("cafe123")
elif query == ".comments":
    print(json.dumps(STATE["comments"]))
elif query == ".author.login":
    print(STATE["author"])
elif query == "[.labels[].name]":
    print(json.dumps(STATE["labels"]))
elif query == ".labels" and "-q" not in args:
    print(json.dumps(STATE["labels"]))
else:
    sys.stderr.write("fake gh: unhandled: %s\n" % args)
    sys.exit(1)
"""#
# Copyright (c) 2026 vLLM Ascend contributors.
# Licensed under the Apache License, Version 2.0.
#
"""End-to-end test for ai_review.py with fake `gh` and `dsh` executables.

Runs the real script as a subprocess through its real entry points
(argparse, env vars, subprocess calls) against a sandbox workspace and
canned GitHub responses — the class of bugs unit tests on pure functions
cannot see (argparse typing, template paths, subprocess wiring).
"""

import json
import os
import stat
import subprocess
import sys
import textwrap

import pytest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_review.py")


FAKE_DSH = r'''#!/usr/bin/env python3
"""Fake dsh: reads the prompt file, emits a protocol-valid review."""
import os, sys

prompt = open(sys.argv[-1]).read()
assert "## Knowledge base map" in prompt, "prompt template missing sections"
sys.stdout.write(
    "```finding\npath: x.py\nline: 1\nseverity: NIT\nrule: general\n"
    "body: fine\n```\n\nAI-REVIEW-VERDICT: PASS\n"
    "<!-- ai-review: state=passed | round=1/5 | total=1 | head=cafe123 -->"
)
sys.exit(0)
'''


@pytest.fixture
def e2e(tmp_path, monkeypatch):
    """Sandbox with fake gh/dsh on PATH and a minimal repo workspace."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ws = tmp_path / "ws"
    for sub in (".github/agent-review/L1", ".github/agent-review/L2/ci"):
        (ws / sub).mkdir(parents=True)
    (ws / ".git").mkdir()
    # trusted files the orchestrator reads from the base checkout
    (ws / ".github/agent-review/instructions.md").write_text("# protocol")
    (ws / ".github/agent-review/allowlist.txt").write_text(
        "# operators\nwangxiyuan\n"
    )
    (ws / ".github/agent-review/domains.yaml").write_text(
        "domains:\n  - name: ci\n    description: CI\n    paths:\n      - '.github/**'\n"
    )
    (ws / ".github/agent-review/prompt_template.md").write_text(
        "{{INSTRUCTIONS}}\n{{ENVIRONMENT}}\n## Knowledge base map\n\n{{KB_MAP}}\n"
        "{{WAIVED_SECTION}}\n## Domains this PR touches\n\n{{DOMAINS}}\n"
        "## PR diff (head {{HEAD_SHA}})\n{{DIFF}}\n## Your previous comment\n"
        "{{PREV_COMMENT}}\n## Developer / reviewer reply to address\n{{REPLY}}\n"
        "{{CLOSING_NOTE}}Round {{ROUND}}/{{MAX_ROUNDS}} total {{TOTAL}}/{{MAX_TOTAL}}.\n"
    )
    for name, content in (("gh", FAKE_GH), ("dsh", FAKE_DSH)):
        exe = bin_dir / name
        exe.write_text(content)
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "FAKE_GH_STATE": str(ws / "state.json"),
        "FAKE_GH_LOG": str(ws / "gh_calls.log"),
        "FAKE_GH_COMMENTS": str(ws / "posted_comments.log"),
        "FAKE_GH_REVIEWS": str(ws / "posted_reviews.log"),
        "FAKE_DSH_OUT": str(ws / "dsh_out.md"),
        "AI_REVIEW_KEY": "sk-test",
        "AI_REVIEW_MODEL": "deepseek-v4.1-flash",
    }
    monkeypatch.setenv("PATH", env["PATH"])
    monkeypatch.setenv("HOME", str(tmp_path))
    return {"ws": ws, "env": env, "bin": bin_dir}

def run_orchestrator(e2e, event, actor="wangxiyuan", extra_env=None):
    labels = ["ai-review"]
    (e2e["ws"] / "state.json").write_text(
        json.dumps({"labels": labels, "author": "somebody", "comments": [],
                    "actor_permission": "write", "review_comments": []})
    )
    comment = e2e["ws"] / "comment.md"
    comment.write_text("please review")
    cmd = [
        sys.executable, SCRIPT,
        "--workspace", str(e2e["ws"]),
        "--pr", "1",
        "--event", event,
        "--actor", actor,
        "--comment-body-file", str(comment),
        "--comment-id", "42",
    ]
    full_env = dict(os.environ)
    full_env.update(e2e["env"])
    full_env.update({
        "AI_REVIEW_EVENT_NAME": event.split("/")[0],
        "AI_REVIEW_EVENT_ACTION": event.split("/")[1],
        "AI_REVIEW_DSH_CMD": "dsh --profile headless",
    })
    if extra_env:
        full_env.update(extra_env)
    return subprocess.run(cmd, capture_output=True, text=True, env=full_env)


def posted(e2e, which="posted_comments.log"):
    f = e2e["ws"] / which
    return f.read_text() if f.exists() else ""


def test_label_trigger_full_flow(e2e):
    """labeled → agent runs → inline review posted → pass verdict handled."""
    r = run_orchestrator(e2e, "pull_request_target/labeled")
    if r.returncode != 0:
        print("STDOUT:", r.stdout)
        print("STDERR:", r.stderr)
    assert r.returncode == 0, r.stderr
    assert "AI review" in posted(e2e)
    assert '"event": "COMMENT"' in posted(e2e, "posted_reviews.log")


def test_actor_not_in_allowlist_gets_notice(e2e):
    r = run_orchestrator(e2e, "pull_request_target/labeled", actor="stranger")
    assert r.returncode == 0, r.stderr
    assert "不在 ai-review 灰度名单" in posted(e2e)


def test_engine_failure_posts_notice_and_keeps_budget(e2e):
    # a missing/broken engine must not crash the job or consume the budget
    r = run_orchestrator(
        e2e, "pull_request_target/labeled",
        extra_env={"AI_REVIEW_DSH_CMD": "no-such-engine --profile headless"},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "aborted" in posted(e2e)
