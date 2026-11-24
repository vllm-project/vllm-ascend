#!/usr/bin/env python3
"""Orchestrator for the ai-review POC (RFC #21451).

State machine summary (states persist in the agent's own last PR comment via
a machine-readable signature block, no external storage):

    <!-- ai-review: state=reviewing | round=1/5 | total=1 | head=abc1234 -->

    labeled(ai-review, by write user)   -> round 1 agent review
    author push (synchronize)           -> revoke `ai-review-passed`, re-review
    author comment with /ai-review      -> next round (ignores plain comments)
    reviewer removes label /ai-review-stop -> stopped, the agent stays silent

Termination: pass verdict (MUST-FIX == 0), per-push round cap (5), reviewer
stop, waiting-human decision, whole-PR budget cap (15). The agent never
approves; `ai-review-passed` is not an approve.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import regex as re
import yaml

AI_REVIEW_LABEL = "ai-review"
AI_REVIEW_PASSED_LABEL = "ai-review-passed"
# Slash command style, consistent with the repo's other PR commands (/e2e,
# /rerun, ...): no account is implied. The negative lookahead keeps
# `/ai-review-stop` and hypothetical `/ai-reviewer` from matching.
MENTION_RE = re.compile(r"/ai-review(?![\w-])")
STOP_RE = re.compile(r"/ai-review-stop(?![\w-])")
BOT_MARKER = "ai-review-orchestrator"
SIGNATURE_RE = re.compile(r"<!-- ai-review: ([^>]+) -->")
FIELD_RE = re.compile(r"(\w+)=(\S+)")
MAX_ROUNDS_PER_PUSH = 5
MAX_TOTAL_ROUNDS = 15


def parse_allowlist(text: str) -> set[str]:
    """Usernames from the allowlist; `#` starts a comment anywhere."""
    entries = set()
    for line in text.splitlines():
        user = line.split("#", 1)[0].strip()
        if user:
            entries.add(user)
    return entries


# Cost guards. A several-thousand-line diff (recipes, overlays, .po files)
# dominates the per-round token bill, and an agent left unbounded can wander
# for half an hour; cap both.
MAX_DIFF_CHARS = 80_000
ROUND_TIMEOUT_SECONDS = 600


def render_template(template: str, values: dict[str, str]) -> str:
    """Single-pass {{PLACEHOLDER}} substitution.

    Values are never re-scanned, so a diff containing something that looks
    like a placeholder cannot inject other values; unknown placeholders are
    left verbatim so a template typo is visible in the output.
    """
    return re.sub(
        r"\{\{(\w+)\}\}",
        lambda m: str(values.get(m.group(1), m.group(0))),
        template,
    )


def truncate_diff(diff: str, limit: int = MAX_DIFF_CHARS) -> str:
    """Cap the diff so one huge PR cannot dominate the token bill; the
    agent is told to read the remainder from the head/ checkout."""
    if len(diff) <= limit:
        return diff
    cut = diff[:limit]
    # keep the truncation boundary on a line edge
    cut = cut[: cut.rfind("\n") + 1]
    return (
        cut
        + f"\n\n[diff truncated at {limit} characters; "
        + f"{len(diff) - len(cut)} more characters — read the remaining "
        + "changed files directly from head/ if needed]\n"
    )


def resolve_domains(domains_yaml: str, files: list[str]) -> list[dict]:
    """Match changed PR files against the coarse domains.yaml registry.

    fnmatch's `*` crosses `/`, so a glob like `.github/**` matches nested
    paths — which is what the coarse registry wants. First match wins.
    """
    try:
        registry = yaml.safe_load(domains_yaml) or {}
    except yaml.YAMLError as exc:
        print(f"::warning::domains.yaml is invalid, no L2 domains resolve: {exc}")
        return []
    matched = []
    for domain in registry.get("domains", []):
        globs = domain.get("paths", [])
        if any(fnmatch.fnmatch(f, g) for f in files for g in globs):
            matched.append(domain)
    return matched


# The agent signals "all MUST-FIX resolved" with this marker line, which the
# orchestrator (not the agent itself) turns into the `ai-review-passed` label.

PASS_MARKER = "AI-REVIEW-VERDICT: PASS"

# The agent emits each finding as a fenced block; the orchestrator turns the
# parsed ones into inline review comments anchored on the code line.
FINDING_BLOCK_RE = re.compile(r"```finding\n(.*?)```", re.S)
FIELD_LINE_RE = re.compile(r"^(\w+):\s*(.+)$", re.M)
REQUIRED_FINDING_FIELDS = ("path", "line", "severity", "rule", "body")

# Marker embedded in every inline finding comment so thread replies can be
# attributed to the agent, and so waived findings stay visibly marked.
INLINE_MARKER = "ai-review finding"
WAIVED_PREFIX = "**[WAIVED"
WAIVE_RE = re.compile(r"/ai-review-waive(?![\w-])")

# dsh provider patch written per round from the AI_REVIEW_* configuration
PATCH_FILE = "/tmp/ai-review-patch.yml"


def parse_findings(text: str) -> list[dict]:
    """Parse ```finding blocks into dicts; malformed blocks are dropped.

    `body` is everything after its key line (multi-line allowed); the other
    fields are single `key: value` lines before it.
    """
    findings = []
    for m in FINDING_BLOCK_RE.finditer(text):
        block = m.group(1)
        body_match = re.search(r"^body:\s*(.*)$", block, re.M)
        if not body_match:
            continue
        fields = dict(FIELD_LINE_RE.findall(block[: body_match.start()]))
        # body is the key line's remainder plus everything after it, so it
        # may span multiple lines; strip each piece, then re-join
        fields["body"] = "\n".join(
            part.strip() for part in (body_match.group(1), block[body_match.end() :]) if part.strip()
        )
        if not all(f in fields and fields[f].strip() for f in REQUIRED_FINDING_FIELDS):
            continue
        try:
            line = int(fields["line"])
        except ValueError:
            continue
        if line < 1:
            continue
        findings.append(
            {
                "path": fields["path"].strip(),
                "line": line,
                "severity": fields["severity"].strip(),
                "rule": fields["rule"].strip(),
                "body": fields["body"].strip(),
            }
        )
    return findings


def post_inline_review(pr: int, head_sha: str, findings: list[dict]) -> bool:
    """Post one COMMENT-level review with the findings as inline comments.

    Returns False if GitHub rejected the anchors (e.g. a line outside the
    diff context), in which case the caller falls back to the summary.
    """
    comments = []
    for f in findings:
        body = (
            f"**[{f['severity']}]** rule: `{f['rule']}`\n\n{f['body']}\n\n"
            "🤖 ai-review finding — `ai-review-passed` is not an approve."
        )
        comments.append({"path": f["path"], "line": f["line"], "side": "RIGHT", "body": body})
    payload = {
        "commit_id": head_sha,
        "event": "COMMENT",
        "body": f"🤖 ai-review round: {len(findings)} finding(s) anchored inline.",
        "comments": comments,
    }
    with open("/tmp/ai-review-payload.json", "w") as fp:
        json.dump(payload, fp)
    result = subprocess.run(
        ["gh", "api", "--input", "/tmp/ai-review-payload.json", f"repos/:owner/:repo/pulls/{pr}/reviews"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"::warning::inline review rejected, falling back to summary: {result.stderr.strip()[:300]}", flush=True)
        return False
    return True


@dataclass
class State:
    state: str = "none"
    round: int = 0
    total: int = 0
    head: str = ""

    def signature(self, head: str) -> str:
        self.head = head
        fields = " | ".join(
            f"{k}={v}"
            for k, v in (
                ("state", self.state),
                ("round", f"{self.round}/{MAX_ROUNDS_PER_PUSH}"),
                ("total", self.total),
                ("head", self.head),
            )
        )
        return f"<!-- ai-review: {fields} -->"


class Orchestrator:
    def __init__(self, workspace: str, pr: int, event: str, actor: str, comment_body: str):
        self.workspace = workspace
        self.pr = pr
        self.event = event  # e.g. "pull_request_target/labeled"
        self.actor = actor
        self.comment_body = comment_body
        self.head_sha = self._gh("pr", "view", str(pr), "--json", "headRefOid", "-q", ".headRefOid")
        self.state = self._load_state()

    # ---- thin gh wrappers -------------------------------------------------

    def _gh(self, *args: str) -> str:
        result = subprocess.run(["gh", *args], capture_output=True, text=True)
        if result.returncode != 0:
            # bring-up aid: surface gh's own message instead of losing it
            print(f"::error::gh {' '.join(args)} failed:\n{result.stderr.strip()}", flush=True)
            raise SystemExit(1)
        return result.stdout.strip()

    def _gh_json(self, *args: str):
        return json.loads(self._gh(*args))

    def comment(self, body: str):
        subprocess.run(
            ["gh", "pr", "comment", str(self.pr), "--body-file", "-"],
            input=body + "\n",
            check=True,
            text=True,
        )

    # ---- state ------------------------------------------------------------

    def _load_state(self) -> State:
        comments = self._gh_json(
            "pr",
            "view",
            str(self.pr),
            "--json",
            "comments",
            "-q",
            ".comments",
        )
        for c in reversed(comments):
            # only the orchestrator's own comments may carry state; a user
            # pasting a signature block must not hijack the state machine
            if BOT_MARKER not in c.get("body", ""):
                continue
            m = SIGNATURE_RE.search(c.get("body", ""))
            if not m:
                continue
            fields = dict(FIELD_RE.findall(m.group(1)))
            return State(
                state=fields.get("state", "none"),
                round=int(fields.get("round", "0/5").split("/")[0]),
                total=int(fields.get("total", "0")),
                head=fields.get("head", ""),
            )
        return State()

    # ---- gates ------------------------------------------------------------

    def actor_can_write(self) -> bool:
        perm = self._gh_json("api", f"repos/:owner/:repo/collaborators/{self.actor}/permission")
        return perm.get("permission") in ("admin", "maintain", "write")

    def user_in_allowlist(self, user: str) -> bool:
        path = os.path.join(self.workspace, ".github", "agent-review", "allowlist.txt")
        with open(path) as f:
            entries = parse_allowlist(f.read())
        return user in entries

    # ---- agent round ------------------------------------------------------

    def run_agent_round(self, round_no: int, closing: bool = False):
        """Assemble the prompt and let the agent produce its next comment.

        Progressive disclosure: the prompt carries the instructions and a
        knowledge-base MAP only; the agent reads full checklist files itself
        from the checked-out workspace (they live under
        .github/agent-review/, inside the repo the job checked out).
        """
        with open(os.path.join(self.workspace, ".github", "agent-review", "instructions.md")) as f:
            instructions = f.read()
        kb_map = self._knowledge_base_map()
        diff = truncate_diff(self._gh("pr", "diff", str(self.pr)))
        prev_agent_comment = self._last_agent_comment_text()
        user_reply = self.comment_body or "(n/a — new push or first round)"
        domains = self._touched_domains()
        waived = self._waived_findings()

        closing_note = ""
        if closing:
            closing_note = (
                "\nThis is the FINAL round for this push (round cap reached). "
                "Do NOT raise new findings. Produce a closing summary listing "
                "every unresolved finding and what a human reviewer must decide, "
                "then set state=waiting-human.\n"
            )

        # The prompt skeleton lives in
        # .github/agent-review/prompt_template.md (owner-gated with the rest
        # of the knowledge base); this code only fills in the values.
        template_path = os.path.join(self.workspace, ".github", "agent-review", "prompt_template.md")
        with open(template_path) as f:
            template = f.read()
        waived_section = (
            "\n\n## Waived findings (human decision — FINAL, do not re-raise, "
            "do not count toward MUST-FIX)\n\n" + "\n".join(f"- {w}" for w in waived)
            if waived
            else ""
        )
        domains_section = (
            "\n".join(f"- {d['name']}: {d.get('description', '')} — read `L2/{d['name']}/` first" for d in domains)
            or "(none matched — L1 global rules only)"
        )
        prompt = render_template(
            template,
            {
                "INSTRUCTIONS": instructions.strip(),
                "ENVIRONMENT": (
                    "## Working environment\n\n"
                    "- Your working directory is a full checkout of the"
                    " vllm-ascend repository (BASE branch).\n"
                    "- The PR's own code is checked out read-only under"
                    " `head/`. Finding line numbers refer to files under"
                    " `head/` — when you need context around a changed line,"
                    " read the file at `head/<path>`, never from the base"
                    " checkout.\n"
                    "- Project documentation lives under `head/docs/source/`;"
                    " the contributor guidelines are `head/AGENTS.md`"
                    " (background only — the review rules are the checklists"
                    " in the knowledge base map).\n"
                    "- Knowledge base map paths are relative to the working"
                    " directory (the trusted BASE checkout).\n"
                    "- You have read access to all of it. Do not modify,"
                    " build or execute anything; this is a review-only"
                    " environment.\n"
                ),
                "KB_MAP": kb_map.strip(),
                "WAIVED_SECTION": waived_section,
                "DOMAINS": domains_section,
                "HEAD_SHA": self.head_sha[:12],
                "DIFF": diff.strip(),
                "PREV_COMMENT": prev_agent_comment or "(first round)",
                "REPLY": user_reply,
                "CLOSING_NOTE": closing_note,
                "ROUND": round_no,
                "MAX_ROUNDS": MAX_ROUNDS_PER_PUSH,
                "TOTAL": self.state.total + 1,
                "MAX_TOTAL": MAX_TOTAL_ROUNDS,
            },
        )
        leftover = re.findall(r"\{\{\w+\}\}", prompt)
        if leftover:
            print(
                f"::warning::unfilled prompt placeholders: {sorted(set(leftover))}",
                flush=True,
            )

        prompt_file = "/tmp/ai-review-prompt.md"
        out_file = "/tmp/ai-review-output.md"
        with open(prompt_file, "w") as f:
            f.write(prompt)

        cmd = os.environ.get("AI_REVIEW_DSH_CMD", "dsh --profile headless").split()
        cmd += ["--patch", PATCH_FILE]
        # The agent process gets ONLY the model credential (via the patch's
        # apiKeyEnv) — never GH_TOKEN. Every GitHub API call is made by this
        # orchestrator; the agent's output is parsed text we post ourselves.
        # This keeps a prompt-injected PR diff from reaching any repo token.
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "AI_REVIEW_KEY": os.environ.get("AI_REVIEW_KEY", ""),
            # placeholder: the headless profile boot-gates on the presence of
            # a deepseek-official credential; real requests go to the
            # OpenAI-protocol gateway provider declared in the patch
            "DEEPSEEK_API_KEY": "boot-gate-placeholder",
        }
        base_url = os.environ.get("AI_REVIEW_API", "")
        model = os.environ.get("AI_REVIEW_MODEL", "deepseek-v4.1-flash")
        if not base_url:
            print("::error::AI_REVIEW_API is not set; no endpoint to review against", flush=True)
            raise SystemExit(1)
        # dsh's built-in DeepSeek route speaks the Anthropic Messages
        # protocol with x-api-key auth, which our gateway rejects. The
        # verified recipe: a --patch overlay declaring an OpenAI-protocol
        # provider (Bearer auth) plus the agent-default-model override.
        with open(PATCH_FILE, "w") as f:
            # quote the env-derived values: a URL containing "#" or ":"
            # would otherwise be mangled by YAML comment/key parsing
            f.write(
                "- id: agent-default-model\n"
                "  config:\n"
                "    provider: ai-review-gateway\n"
                f'    model: "{model}"\n'
                "- id: llm-pi-ai\n"
                "  config:\n"
                "    providers:\n"
                "      ai-review-gateway:\n"
                "        apiKeyEnv: AI_REVIEW_KEY\n"
                "        api: openai-completions\n"
                f'        baseURL: "{base_url}"\n'
                "        models:\n"
                f'          - id: "{model}"\n'
            )
        started = time.monotonic()
        print(
            f"::notice::ai-review prompt size: {len(prompt)} chars; "
            f"engine: {cmd[0]}; timeout: {ROUND_TIMEOUT_SECONDS}s",
            flush=True,
        )
        # the base URL is not a secret; print it so endpoint/auth mismatches
        # are diagnosable (the key itself is masked by GitHub)
        print(
            f"::notice::dsh endpoint: {base_url}; model: {model}",
            flush=True,
        )
        try:
            with open(out_file, "w") as out:
                subprocess.run(
                    [*cmd, prompt_file],
                    stdout=out,
                    check=True,
                    env=env,
                    timeout=ROUND_TIMEOUT_SECONDS,
                )
        except subprocess.TimeoutExpired:
            # Budget is NOT consumed: the state signature is re-posted
            # unchanged so the author can retry after the engine is fixed.
            print(
                f"::error::agent round exceeded {ROUND_TIMEOUT_SECONDS}s and was aborted",
                flush=True,
            )
            self.post_notice(
                f"⏱️ AI review round aborted: the agent exceeded "
                f"{ROUND_TIMEOUT_SECONDS // 60} minutes. No review was produced "
                "and the round budget was not consumed. Retry with `/ai-review`.",
                new_state=State(
                    state=self.state.state,
                    round=self.state.round,
                    total=self.state.total,
                ),
            )
            return
        print(
            f"::notice::agent round finished in {time.monotonic() - started:.0f}s",
            flush=True,
        )

        with open(out_file) as f:
            output = f.read()
        # Defense in depth: the agent never sees repo tokens (filtered env
        # above), but the model key could still be coaxed into its output by
        # an injected instruction — redact known secret values before the
        # text ever reaches a public PR comment.
        for secret_name in ("AI_REVIEW_KEY", "AI_REVIEW_API", "GH_TOKEN"):
            secret_value = os.environ.get(secret_name, "")
            if secret_value:
                output = output.replace(secret_value, "***REDACTED***")
        verdict_pass = PASS_MARKER in output

        new_state = State(state="reviewing", round=round_no, total=self.state.total + 1)
        if closing:
            new_state.state = "waiting-human"
        elif verdict_pass:
            new_state.state = "passed"

        if not SIGNATURE_RE.search(output):
            # The agent forgot the signature; append one so the next round can
            # still restore state. Never leave the machine without a marker.
            output += "\n\n" + new_state.signature(self.head_sha)
        else:
            output = SIGNATURE_RE.sub(new_state.signature(self.head_sha), output, count=1)

        # Stop semantics: removing the label mid-round must win. The queued
        # unlabeled handler runs after us and marks state=stopped; here we
        # only need to not post a round the reviewer already cancelled. The
        # completed round is discarded (cost is sunk, the output is dropped).
        labels = self._gh_json("pr", "view", str(self.pr), "--json", "labels", "-q", "[.labels[].name]")
        if AI_REVIEW_LABEL not in labels:
            print("ai-review label was removed mid-round; discarding output", flush=True)
            return

        banner = (
            f"<!-- {BOT_MARKER} -->\n"
            f"🤖 **AI review** (round {round_no}/{MAX_ROUNDS_PER_PUSH}, "
            f"engine: DeepSeek Harness headless, POC gray release — "
            f"`ai-review-passed` is **not** an approve; human review is still required.)\n\n"
        )

        # Inline review comments: findings anchored to code lines land in one
        # COMMENT-level review (never APPROVE); the summary comment keeps the
        # verdict, checklist and state signature. If the review API rejects
        # an anchor (line outside the diff), fall back to a summary-only
        # round so no finding is silently lost.
        findings = parse_findings(output)
        summary_body = FINDING_BLOCK_RE.sub("", output).strip()
        summary = banner + summary_body + "\n"
        # findings move to inline comments only after the review API
        # accepted them; on failure they stay in the summary verbatim
        if findings and post_inline_review(self.pr, self.head_sha, findings):
            findings_text = ""
            summary += (
                f"\n---\n📋 {len(findings)} finding(s) posted as inline review "
                "comments on the diff; this comment carries the verdict and "
                "state only.\n"
            )
        else:
            findings_text = (
                "\n## Findings\n\n"
                + "\n\n".join(
                    f"- **[{f['severity']}]** `{f['path']}:{f['line']}` rule: `{f['rule']}`\n  {f['body']}"
                    for f in findings
                )
                + "\n"
                if findings
                else ""
            )
        summary += findings_text
        self.comment(summary)

        if verdict_pass:
            subprocess.run(
                ["gh", "pr", "edit", str(self.pr), "--add-label", AI_REVIEW_PASSED_LABEL],
                check=True,
            )

    def _knowledge_base_map(self) -> str:
        """One-line summaries of every checklist entry, for progressive
        disclosure: the agent reads the full file only when the PR touches
        that area."""
        lines = []
        kb_dir = os.path.join(self.workspace, ".github", "agent-review")
        for root, _dirs, files in os.walk(kb_dir):
            for name in sorted(files):
                if not name.endswith(".md") or name in (
                    "instructions.md",
                    "README.md",
                    "prompt_template.md",
                ):
                    continue
                path = os.path.join(root, name)
                rel = os.path.relpath(path, self.workspace)
                with open(path) as f:
                    entry_ids = re.findall(r"^-\s+`([\w-]+#\w[\w-]*)`", f.read(), re.M)
                ids = ", ".join(f"`{i}`" for i in entry_ids)
                if ids:
                    lines.append(f"- `{rel}` — rules with IDs: {ids}")
                else:
                    # orientation files carry no citable rule IDs; hint at
                    # when reading them pays off
                    hint = (
                        "orientation map — read when the diff touches modules you cannot place"
                        if "/context/" in rel.replace(os.sep, "/")
                        else "maintenance doc, no rules"
                    )
                    lines.append(f"- `{rel}` — {hint}")
        return "\n".join(lines)

    def _waived_findings(self) -> list[str]:
        """path:line of inline findings a reviewer has waived."""
        waived = []
        for c in self._review_comments():
            body = c.get("body", "")
            if c.get("in_reply_to_id") or INLINE_MARKER not in body:
                continue
            if not body.startswith(WAIVED_PREFIX):
                continue
            waived.append(f"`{c.get('path')}:{c.get('line')}`")
        return waived

    def _touched_domains(self) -> list[dict]:
        registry_path = os.path.join(self.workspace, ".github", "agent-review", "domains.yaml")
        files = self._gh_json("pr", "view", str(self.pr), "--json", "files", "-q", "[.files[].path]")
        with open(registry_path) as f:
            return resolve_domains(f.read(), files)

    def _last_agent_comment_text(self) -> str:
        comments = self._gh_json("pr", "view", str(self.pr), "--json", "comments", "-q", ".comments")
        for c in reversed(comments):
            if BOT_MARKER in c.get("body", ""):
                return c.get("body", "")
        return ""

    def post_notice(self, text: str, new_state: State | None = None):
        body = f"<!-- {BOT_MARKER} -->\n{text}"
        if new_state is not None:
            body += "\n\n" + new_state.signature(self.head_sha)
        self.comment(body)

    # ---- event dispatch ---------------------------------------------------

    def handle(self) -> int:
        name, _, action = self.event.partition("/")
        self.head_sha = self._gh("pr", "view", str(self.pr), "--json", "headRefOid", "-q", ".headRefOid")

        if name == "pull_request_target":
            return self._handle_pr_event(action)
        if name == "issue_comment":
            return self._handle_comment()
        if name == "pull_request_review_comment":
            return self._handle_review_comment()
        return 0

    def _handle_pr_event(self, action: str) -> int:
        labels = self._gh_json(
            "pr",
            "view",
            str(self.pr),
            "--json",
            "labels",
            "-q",
            "[.labels[].name]",
        )

        if action == "labeled":
            # The event payload does not tell us WHICH label; re-fetch the
            # latest label timeline. If our label is gone, the label was not
            # the one we care about.
            if AI_REVIEW_LABEL not in labels:
                return 0
            if not self.actor_can_write():
                print(f"actor {self.actor} lacks write access; ignoring", flush=True)
                return 0
            # The allowlist authorizes OPERATORS (who may apply the label to
            # any PR), not the PR under review.
            if not self.user_in_allowlist(self.actor):
                self.post_notice(
                    f"@{self.actor} 不在 ai-review 灰度名单，无法启动评审（名单见 "
                    "`.github/agent-review/allowlist.txt`，加入请在 RFC #21451 留言）。标签已移除。",
                )
                subprocess.run(
                    ["gh", "pr", "edit", str(self.pr), "--remove-label", AI_REVIEW_LABEL],
                    check=False,
                )
                return 0
            self.run_agent_round(round_no=1)
            return 0

        if action == "unlabeled":
            # We cannot see which label was removed from `labels`; if the
            # ai-review label is no longer present, treat as a stop request
            # only when the previous state was active.
            if AI_REVIEW_LABEL in labels:
                return 0
            if self.state.state in ("reviewing", "waiting-human", "passed"):
                subprocess.run(
                    ["gh", "pr", "edit", str(self.pr), "--remove-label", AI_REVIEW_PASSED_LABEL],
                    check=False,
                )
                self.post_notice(
                    "🛑 AI review 已被终止（label 被移除）。如需重新开始，请重新打 `ai-review` 标签。",
                    new_state=State(state="stopped", round=self.state.round, total=self.state.total),
                )
            return 0

        if action == "synchronize":
            if AI_REVIEW_LABEL not in labels:
                return 0
            if self.state.state == "stopped":
                return 0
            if self.state.state == "passed":
                subprocess.run(
                    ["gh", "pr", "edit", str(self.pr), "--remove-label", AI_REVIEW_PASSED_LABEL],
                    check=False,
                )
            if self.state.total >= MAX_TOTAL_ROUNDS:
                self.post_notice(
                    f"AI review 已达到整个 PR 的轮次预算上限（{MAX_TOTAL_ROUNDS} 轮），"
                    "暂停响应。reviewer 可重新打 `ai-review` 标签重置预算。",
                )
                return 0
            self.run_agent_round(round_no=1)
            return 0

        return 0

    def _handle_comment(self) -> int:
        body = self.comment_body or ""
        if not body or self.state.state == "stopped":
            return 0

        # Reviewer stop command.
        if STOP_RE.search(body):
            if not self.actor_can_write():
                print(f"actor {self.actor} lacks write access; ignoring stop", flush=True)
                return 0
            self.post_notice(
                "🛑 AI review 已被终止。如需重新开始，请重新打 `ai-review` 标签。",
                new_state=State(state="stopped", round=self.state.round, total=self.state.total),
            )
            return 0

        # Reviewer waiver: top-level /ai-review-waive waives ALL open
        # findings, then triggers one re-evaluation round (it costs agent
        # compute, so it counts against the budget).
        if WAIVE_RE.search(body):
            if not self.actor_can_write():
                print(f"actor {self.actor} lacks write access; ignoring waive", flush=True)
                return 0
            return self._waive_all()

        # Only an explicit slash command counts as addressing the agent.
        if not MENTION_RE.search(body):
            return 0
        if self.state.state in ("passed", "none"):
            self.post_notice(
                "AI review 已结束。如需重新评审，请让 reviewer 重新打 `ai-review` 标签。",
            )
            return 0
        if self.state.total >= MAX_TOTAL_ROUNDS:
            self.post_notice(
                f"AI review 已达到整个 PR 的轮次预算上限（{MAX_TOTAL_ROUNDS} 轮），暂停响应。",
            )
            return 0

        next_round = self.state.round + 1
        closing = next_round > MAX_ROUNDS_PER_PUSH
        self.run_agent_round(round_no=next_round, closing=closing)
        return 0

    # ---- waivers and inline threads ---------------------------------------

    def _review_comments(self) -> list[dict]:
        return self._gh_json("api", f"repos/:owner/:repo/pulls/{self.pr}/comments?per_page=100")

    def _edit_review_comment(self, comment_id: int, body: str):
        self._gh_json(
            "api",
            f"repos/:owner/:repo/pulls/{self.pr}/comments/{comment_id}",
            "-X",
            "PATCH",
            "-f",
            f"body={body}",
        )

    def _thread_root(self, comment: dict) -> dict | None:
        """Follow in_reply_to_id to the root comment of a thread (bounded)."""
        root = comment
        for _ in range(10):
            if root.get("in_reply_to_id") is None:
                return root
            root = self._gh_json("api", f"repos/:owner/:repo/pulls/{self.pr}/comments/{root['in_reply_to_id']}")
        return None

    def _waive_comment(self, comment: dict) -> bool:
        """Prefix one inline finding comment with the waived banner."""
        body = comment.get("body", "")
        if INLINE_MARKER not in body or body.startswith(WAIVED_PREFIX):
            return False
        self._edit_review_comment(comment["id"], f"{WAIVED_PREFIX} by @{self.actor}]**\n\n{body}")
        return True

    def _waive_all(self) -> int:
        roots = [
            c
            for c in self._review_comments()
            if INLINE_MARKER in c.get("body", "")
            and not c.get("in_reply_to_id")
            and not c.get("body", "").startswith(WAIVED_PREFIX)
        ]
        for c in roots:
            self._waive_comment(c)
        self.post_notice(
            f"⚖️ @{self.actor} waived all {len(roots)} open finding(s). "
            "Re-evaluating: waived findings will not count toward MUST-FIX."
        )
        if self.state.state in ("reviewing", "waiting-human"):
            if self.state.total >= MAX_TOTAL_ROUNDS:
                return 0
            next_round = self.state.round + 1
            closing = next_round > MAX_ROUNDS_PER_PUSH
            self.run_agent_round(round_no=next_round, closing=closing)
        return 0

    def _handle_review_comment(self) -> int:
        """Replies inside an inline finding thread: `/ai-review-waive` waives
        that single finding; `/ai-review` triggers the next round. Replies to
        threads the agent did not start are ignored."""
        body = self.comment_body or ""
        if not body:
            return 0
        comments = {c["id"]: c for c in self._review_comments()}
        ours = [c for c in comments.values() if INLINE_MARKER in c.get("body", "")]
        if not ours:
            return 0
        # the event's comment itself
        event_comment = comments.get(self._event_comment_id)
        if not event_comment:
            return 0
        root = self._thread_root(event_comment)
        if not root or INLINE_MARKER not in root.get("body", ""):
            return 0

        if WAIVE_RE.search(body):
            if not self.actor_can_write():
                print(f"actor {self.actor} lacks write access; ignoring waive", flush=True)
                return 0
            if self._waive_comment(root):
                self.post_notice(
                    f"⚖️ @{self.actor} waived the finding at `{root.get('path')}:"
                    f"{root.get('line')}`. It will not count toward MUST-FIX."
                )
            return 0

        if MENTION_RE.search(body):
            if self.state.state in ("passed", "none", "stopped"):
                return 0
            if self.state.total >= MAX_TOTAL_ROUNDS:
                self.post_notice(
                    f"AI review 已达到整个 PR 的轮次预算上限（{MAX_TOTAL_ROUNDS} 轮），暂停响应。",
                )
                return 0
            next_round = self.state.round + 1
            closing = next_round > MAX_ROUNDS_PER_PUSH
            self.run_agent_round(round_no=next_round, closing=closing)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment-body-file", default="")
    parser.add_argument("--comment-id", default="0")  # empty on non-comment events
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--actor", default="")
    args = parser.parse_args()

    comment_body = ""
    if args.comment_body_file and os.path.exists(args.comment_body_file):
        with open(args.comment_body_file) as f:
            comment_body = f.read()

    orch = Orchestrator(args.workspace, args.pr, args.event, args.actor, comment_body)
    try:
        comment_id = int(args.comment_id)
    except ValueError:
        comment_id = 0
    orch._event_comment_id = comment_id
    return orch.handle()


if __name__ == "__main__":
    sys.exit(main() or 0)
