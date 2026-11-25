#!/usr/bin/env python3
"""AI review: authorize a request, run the agent, publish and count the result.
Comment examples below use shortened hashes and formatted JSON.

Command feedback (appended to the user's original comment):
    /ai-review I fixed F001. Please check it again.
    ---
    [AI review bot]: Review request accepted. The bot will check the round limit before starting.
    [View Actions run](https://github.com/owner/repo/actions/runs/123)

PR summary (updated in place):
    <!-- ai-review-orchestrator:v2 -->
    🤖 **AI review · DeepSeek Harness headless · POC pilot**
    Status: **reviewing** · Commit: abc123456789
    Round 2/5; total published rounds 3/15.
    This is not an approval. Human review is still required.
    One MUST-FIX remains open.
    **F001 · MUST-FIX · open** `example.py:20` (general)
    Handle empty input.
    <!-- ai-review-state
    {
      "status": "reviewing",
      "base": "def012345678",
      "head": "abc123456789",
      "...",
      "findings": [
        {
          "id": "F001",
          "path": "example.py",
          "line": 20,
          "severity": "MUST-FIX",
          "...",
          "comment_id": 1001
        }
      ]
    }
    -->

Inline comment (created for new findings; updated when waived):
    **[WAIVED by @maintainer]**
    Applies only to commit `abc123456789`.
    <!-- ai-review-finding:F001 -->
    **F001 · MUST-FIX** (general)
    Handle empty input.
    See the AI review summary for the current status. Waive with
    `/ai-review-waive F001 reason` in a top-level comment.
Hidden JSON restores the review state; its findings go into the next AI prompt.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import regex as re
import yaml

AI_REVIEW_LABEL = "ai-review"
AI_REVIEW_PASSED_LABEL = "ai-review-passed"
STATE_MARKER = "<!-- ai-review-orchestrator:v2 -->"
STATE_DATA_MARKER = "\n<!-- ai-review-state\n"
MAX_ROUNDS_PER_HEAD = 5
MAX_TOTAL_ROUNDS = 15
MAX_DIFF_CHARS = 80_000
ROUND_TIMEOUT_SECONDS = 600
GIT_FETCH_MAX_ATTEMPTS = 3
GIT_FETCH_RETRY_DELAY_SECONDS = 5
SEVERITIES = ("MUST-FIX", "SHOULD-FIX", "NIT")


def gh(*args: str, payload=None, anchor_fallback: bool = False):
    """Call the GitHub API and decode its JSON response."""
    command = ["gh", "api", *args]

    if payload is not None:
        command += ["--input", "-"]

    result = subprocess.run(
        command,
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
    )

    if result.returncode:
        if anchor_fallback and "(HTTP 422)" in result.stderr:
            print("Inline anchor rejected; the finding remains in the summary.")
            return None

        print(result.stderr, file=sys.stderr, end="")
        result.check_returncode()

    return json.loads(result.stdout) if result.stdout.strip() else None


def git(workspace: Path, *args: str) -> str:
    """Run a Git command in the workspace and return its standard output."""
    return subprocess.run(
        ["git", "-C", str(workspace), *args],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout


def parse_allowlist(text: str) -> set[str]:
    """Extract allowed usernames, ignoring blank lines and trailing comments."""
    return {user for line in text.splitlines() if (user := line.split("#", 1)[0].strip())}


@dataclass
class Finding:
    """Record one finding's location, severity, status and supporting evidence."""

    id: str
    path: str
    line: int | None
    severity: str
    rule: str
    body: str
    head: str
    status: str = "open"
    evidence: str = ""
    waived_by: str = ""
    waived_head: str = ""
    comment_id: int | None = None

    def inline_body(self) -> str:
        """Render the inline finding, including its commit-specific waiver."""
        body = (
            f"<!-- ai-review-finding:{self.id} -->\n"
            f"**{self.id} · {self.severity}** ({self.rule})\n\n{self.body}\n\n"
            "See the AI review summary for the current status. "
            f"Waive with `/ai-review-waive {self.id} reason` in a top-level comment."
        )
        if self.status == "waived":
            body = f"**[WAIVED by @{self.waived_by}]**\nApplies only to commit `{self.waived_head[:12]}`.\n\n{body}"
        return body


@dataclass
class State:
    """Represent the review progress and findings saved in the summary comment."""

    status: str = "idle"
    base: str = ""
    head: str = ""
    round: int = 0
    total: int = 0
    session_run_id: str = ""
    coverage_complete: bool = False
    requires_human: bool = False
    message: str = ""
    findings: list[Finding] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> State:
        """Restore the saved review state and its finding records."""
        return cls(**{**data, "findings": [Finding(**f) for f in data["findings"]]})

    def decide(self):
        """Choose passed, reviewing or waiting-human from the current review state."""
        if (
            self.coverage_complete
            and not self.requires_human
            and not any(f.status == "open" and f.severity == "MUST-FIX" for f in self.findings)
        ):
            self.status = "passed"
        elif self.requires_human or self.round >= MAX_ROUNDS_PER_HEAD or self.total >= MAX_TOTAL_ROUNDS:
            self.status = "waiting-human"
        else:
            self.status = "reviewing"

    def body(self) -> str:
        """Render the summary comment with findings and hidden state JSON."""
        lines = [
            STATE_MARKER,
            "🤖 **AI review · DeepSeek Harness headless · POC pilot**",
            f"Status: **{self.status}** · Commit: {self.head[:12]}",
            f"Round {self.round}/{MAX_ROUNDS_PER_HEAD}; total published rounds {self.total}/{MAX_TOTAL_ROUNDS}.",
            "This is not an approval. Human review is still required.",
            self.message,
        ]

        for f in self.findings:
            location = f"{f.path}:{f.line}" if f.line is not None else f.path
            lines += [f"\n**{f.id} · {f.severity} · {f.status}** `{location}` ({f.rule})", f.body]
            if f.evidence:
                lines.append(f.evidence)
            if f.status == "waived":
                lines.append(f"Waived by @{f.waived_by} for {f.waived_head[:12]}.")

        # Model text cannot terminate or replace the final machine-owned block.
        data = json.dumps(asdict(self), ensure_ascii=True).replace("<", "\\u003c")
        return "\n\n".join(lines) + STATE_DATA_MARKER + data + "\n-->"


def parse_result(output: str, findings: list[Finding]) -> dict:
    """Parse and validate model findings and updates before publication."""
    result = json.loads(output)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    for key in ("coverage_complete", "requires_human"):
        if type(result.get(key)) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if not isinstance(result.get("summary"), str):
        raise ValueError("summary must be a string")
    if not isinstance(result.get("findings"), list) or not isinstance(result.get("updates"), list):
        raise ValueError("findings and updates must be arrays")

    for finding in result["findings"]:
        if not isinstance(finding, dict) or not all(
            isinstance(finding.get(key), str) and finding[key].strip() for key in ("path", "rule", "body")
        ):
            raise ValueError("Finding requires path, rule and body")
        if finding.get("severity") not in SEVERITIES:
            raise ValueError("Unknown finding severity")
        if "line" not in finding or not (
            finding["line"] is None or type(finding["line"]) is int and finding["line"] > 0
        ):
            raise ValueError("Finding line must be a positive integer or null")

    known = {f.id: f for f in findings}
    seen = set()
    for update in result["updates"]:
        if not isinstance(update, dict) or not all(isinstance(update.get(key), str) for key in ("id", "evidence")):
            raise ValueError("Update requires id and evidence")

        identity = update["id"]
        if identity not in known or identity in seen:
            raise ValueError("Unknown or repeated finding ID")
        if known[identity].status == "waived" or update.get("status") not in ("open", "resolved"):
            raise ValueError("Model cannot change waivers or invent finding states")
        if not update["evidence"].strip():
            raise ValueError("Finding updates require evidence")
        seen.add(identity)

    return result


def merge_findings(state: State, result: dict) -> list[Finding]:
    """Update the finding list and return newly added records for inline comments."""
    known = {f.id: f for f in state.findings}
    for update in result["updates"]:
        finding = known[update["id"]]
        finding.status = update["status"]
        finding.evidence = update["evidence"]

    created = []
    for new in result["findings"]:
        # Exact repeats reuse the existing record; semantic matching belongs to
        # the model, which receives all prior findings and their stable IDs.
        existing = next(
            (
                f
                for f in state.findings
                if (f.path, f.line, f.rule, f.body) == (new["path"], new["line"], new["rule"], new["body"])
            ),
            None,
        )
        if existing:
            if existing.status != "waived":
                existing.status = "open"
                existing.severity = min((existing.severity, new["severity"]), key=SEVERITIES.index)
            continue

        finding = Finding(
            id=f"F{len(state.findings) + 1:03d}",
            head=state.head,
            **{key: new[key] for key in ("path", "line", "severity", "rule", "body")},
        )
        state.findings.append(finding)
        created.append(finding)

    return created


class Orchestrator:
    """Coordinate PR permissions, review preparation and result publication."""

    def __init__(self, workspace: Path, pr: int):
        """Set the workspace and PR for this review."""
        self.workspace = workspace
        self.pr = pr
        self.repo = "repos/:owner/:repo"
        self.summary_comment_id = None

    # --- Authorize requests ---

    def authorize(self, event_name: str, event: dict) -> bool:
        """Check permissions and handle immediate label changes before queuing."""
        command, _ = self.event_command(event_name, event)
        actor = event["sender"]["login"]
        if not command or event_name == "issue_comment" and event["sender"]["type"] == "Bot":
            return False

        pull = self.pull()
        if pull["state"] != "open":
            if command != "push":
                self.feedback(event, "This PR is closed.")
            return False

        if command == "/ai-review":
            if actor != pull["user"]["login"] and not self.can_write(actor):
                self.feedback(
                    event,
                    "Permission denied. Only the PR author or a maintainer with write access can request a review.",
                )
                return False
        elif command != "push" and not self.can_write(actor):
            self.feedback(event, "Permission denied. This action requires repository write access.")
            return False

        enabled = AI_REVIEW_LABEL in {label["name"] for label in pull["labels"]}
        if command == "stop" and enabled:
            return False
        if command not in ("stop", "/ai-review-stop") and not enabled:
            if command != "push":
                self.feedback(
                    event, "AI review is not enabled. Ask an allowlisted maintainer to add the `ai-review` label."
                )
            return False

        if command == "start":
            allowlist = (self.workspace / ".github/agent-review/allowlist.txt").read_text()
            if actor not in parse_allowlist(allowlist):
                self.remove_labels(pull, AI_REVIEW_LABEL)
                self.feedback(
                    event,
                    "You are not on the pilot allowlist. The label was removed. "
                    "See `.github/agent-review/allowlist.txt` or comment on RFC #21451 to join the pilot.",
                )
                return False

        if command == "push":
            if event["pull_request"]["head"]["sha"] == pull["head"]["sha"]:
                self.remove_labels(pull, AI_REVIEW_PASSED_LABEL)
            return False  # Pushes revoke the old pass but never request a review.

        if command in ("stop", "/ai-review-stop"):
            self.remove_labels(pull, AI_REVIEW_LABEL, AI_REVIEW_PASSED_LABEL)
            message = (
                "Review labels removed. The bot will update the summary. To restart, wait for this run to finish, "
                "then ask an allowlisted maintainer to reapply the `ai-review` label."
            )
        elif command == "/ai-review-waive":
            message = "Waiver request accepted. The bot will check the findings and commit before applying it."
        else:
            message = "Review request accepted. The bot will check the round limit before starting."

        run_url = (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        )
        self.feedback(event, f"{message}\n[View Actions run]({run_url})")

        return True

    @staticmethod
    def event_command(event_name: str, event: dict) -> tuple[str, str]:
        """Translate a GitHub event into a review command and its arguments."""
        if event_name == "pull_request_target":
            return {"labeled": "start", "unlabeled": "stop", "synchronize": "push"}[event["action"]], ""

        parts = event["comment"]["body"].strip().split(maxsplit=1)
        command = parts[0]
        if command not in ("/ai-review", "/ai-review-stop", "/ai-review-waive"):
            return "", ""
        return command, parts[1] if len(parts) == 2 else ""

    def can_write(self, actor: str) -> bool:
        """Check whether the actor has write, maintain or admin permission."""
        return gh(f"{self.repo}/collaborators/{actor}/permission")["permission"] in ("write", "maintain", "admin")

    # --- Prepare the review ---

    def prepare(self, event_name: str, event: dict, run_id: str) -> dict | None:
        """Handle queued commands or prepare a review of the current PR."""
        command, argument = self.event_command(event_name, event)
        actor = event["sender"]["login"]

        # The PR can close or its labels can change while the request is queued.
        pull = self.pull()
        if pull["state"] != "open":
            self.feedback(event, "This PR is closed.")
            return None

        enabled = AI_REVIEW_LABEL in {label["name"] for label in pull["labels"]}
        stopping = command in ("stop", "/ai-review-stop")
        if command == "stop" and enabled:
            return None  # The label has already been reapplied to restart.
        if not stopping and not enabled:
            self.feedback(
                event, "AI review is not enabled. Ask an allowlisted maintainer to add the `ai-review` label."
            )
            return None

        # Read the published summary to handle stops, session restarts and waivers.
        self.state = self.load_state()
        if stopping:
            self.remove_labels(pull, AI_REVIEW_LABEL, AI_REVIEW_PASSED_LABEL)
            if self.state.status == "stopped":
                return None

            self.state.status = "stopped"
            self.state.coverage_complete = False
            self.state.message = "Review stopped. An allowlisted reviewer may reapply the ai-review label to restart."
            self.save_summary()
            if self.state.head:
                self.commit_status("error")
            return None

        if command == "start":
            if self.state.session_run_id != run_id:
                self.state.round = self.state.total = 0
                self.state.session_run_id = run_id
        elif self.state.status == "stopped":
            self.feedback(
                event, "AI review is stopped. Ask an allowlisted maintainer to reapply the `ai-review` label."
            )
            return None

        if command == "/ai-review-waive":
            message = self.waive(argument, actor, pull)
            self.feedback(event, message)
            return None

        # A new head resets its own round count and waivers. The session limit still applies.
        changed_head = self.state.head != pull["head"]["sha"]
        if changed_head:
            self.state.round = 0
            for finding in self.state.findings:
                if finding.status == "waived":
                    finding.status = "open"
        self.state.base, self.state.head = pull["base"]["sha"], pull["head"]["sha"]
        if self.state.total >= MAX_TOTAL_ROUNDS:
            self.feedback(
                event,
                f"Reached the {MAX_TOTAL_ROUNDS}-round limit for this session. "
                "Ask an allowlisted maintainer to reapply the `ai-review` label to reset the counts. "
                "Pushing new commits does not reset the session count.",
            )
            return None
        if self.state.round >= MAX_ROUNDS_PER_HEAD:
            self.feedback(
                event,
                f"Reached the {MAX_ROUNDS_PER_HEAD}-round limit for this commit. Push changes and comment `/ai-review` "
                "again, or ask an allowlisted maintainer to reapply the `ai-review` label to reset the counts.",
            )
            return None

        # Mark the review pending and build its diff. Publishing the summary will count the round.
        self.remove_labels(pull, AI_REVIEW_PASSED_LABEL)
        self.state.status = "reviewing"
        self.state.coverage_complete = False
        self.state.requires_human = False

        self.commit_status("pending")

        for attempt in range(1, GIT_FETCH_MAX_ATTEMPTS + 1):
            try:
                git(self.workspace, "fetch", "--no-tags", "origin", self.state.base, self.state.head)
                break
            except subprocess.CalledProcessError:
                if attempt == GIT_FETCH_MAX_ATTEMPTS:
                    raise
                print(
                    f"git fetch failed (attempt {attempt}/{GIT_FETCH_MAX_ATTEMPTS}); "
                    f"retrying in {GIT_FETCH_RETRY_DELAY_SECONDS} seconds.",
                    flush=True,
                )
                time.sleep(GIT_FETCH_RETRY_DELAY_SECONDS)

        merge_base = git(self.workspace, "merge-base", self.state.base, self.state.head).strip()
        diff = git(self.workspace, "diff", "--no-ext-diff", "--no-textconv", merge_base, self.state.head)
        files = (
            git(self.workspace, "diff", "--name-only", "-z", merge_base, self.state.head).rstrip("\0").split("\0")
            if diff
            else []
        )

        return {
            "pr": self.pr,
            "comment_id": self.summary_comment_id,
            "state": asdict(self.state),
            "merge_base": merge_base,
            "files": files,
            "diff": diff[:MAX_DIFF_CHARS],
            "truncated": len(diff) > MAX_DIFF_CHARS,
            "reply": event["comment"]["body"] if event_name == "issue_comment" else "",
        }

    def waive(self, argument: str, actor: str, pull: dict) -> str:
        """Waive findings for the reviewed commit and return feedback for the user."""
        parts = argument.split(maxsplit=1)
        if len(parts) != 2:
            return "Use `/ai-review-waive F003 reason` or `/ai-review-waive all reason`."
        if (self.state.base, self.state.head) != (pull["base"]["sha"], pull["head"]["sha"]):
            return "Review the current commit with `/ai-review` and wait for the result before waiving findings."

        selected = [f for f in self.state.findings if f.status == "open" and parts[0] in ("all", f.id)]
        if not selected:
            return "No matching open findings. Check the finding IDs and their status in the AI review summary."

        for finding in selected:
            finding.status = "waived"
            finding.waived_by = actor
            finding.waived_head = self.state.head
            finding.evidence = parts[1]
            if finding.comment_id is not None:
                gh(
                    f"{self.repo}/pulls/comments/{finding.comment_id}",
                    "--method",
                    "PATCH",
                    payload={"body": finding.inline_body()},
                )

        self.state.message = f"@{actor} waived {len(selected)} finding(s) for this commit."
        self.state.decide()
        self.save_summary()
        self.publish_status()
        return f"Waived {len(selected)} open finding(s) for commit `{self.state.head[:12]}`. See the review summary."

    # --- Publish results ---

    def publish(self, request: dict, output: str):
        """Publish findings or a discard notice, saving the completed round."""
        pull = self.pull()
        if pull["state"] != "open" or AI_REVIEW_LABEL not in {label["name"] for label in pull["labels"]}:
            print("Review stopped or PR closed; no result published.")
            return

        self.summary_comment_id = request["comment_id"]
        self.state = State.from_dict(request["state"])
        self.state.round += 1
        self.state.total += 1

        if (pull["base"]["sha"], pull["head"]["sha"]) != (self.state.base, self.state.head):
            self.state.status = "discarded"
            self.state.message = (
                "The PR changed during review, so this result was discarded. "
                "This published notice counts as one round for the reviewed version. "
                "Comment /ai-review when the new changes are ready."
            )
        else:
            result = parse_result(output, self.state.findings)
            created = merge_findings(self.state, result)
            self.state.coverage_complete = result["coverage_complete"] and not request["truncated"]
            self.state.requires_human = result["requires_human"]
            self.state.message = result["summary"]
            if request["truncated"]:
                self.state.message += "\nThe diff was truncated; this review cannot pass."

            self.state.decide()
            self.publish_inline(created)

        # This comment is the only durable record of a completed round.
        self.save_summary()
        self.publish_status()

    def publish_inline(self, findings: list[Finding]):
        """Publish inline comments for new findings with a code line."""
        for finding in findings:
            if finding.line is None:
                continue

            payload = {
                "commit_id": finding.head,
                "path": finding.path,
                "line": finding.line,
                "side": "RIGHT",
                "body": finding.inline_body(),
            }

            # Only an explicit 422 validation rejection falls back to summary.
            # Network/auth/server failures keep gh's original error and stop.
            comment = gh(f"{self.repo}/pulls/{self.pr}/comments", payload=payload, anchor_fallback=True)
            if comment:
                finding.comment_id = comment["id"]

    def publish_status(self):
        """Update the commit status and pass label if the reviewed commits are current."""
        # Branches and labels can change while comments are being published.
        pull = self.pull()
        if (
            pull["state"] != "open"
            or AI_REVIEW_LABEL not in {label["name"] for label in pull["labels"]}
            or (pull["base"]["sha"], pull["head"]["sha"]) != (self.state.base, self.state.head)
        ):
            return

        if self.state.status == "passed":
            self.commit_status("success")
            gh(f"{self.repo}/issues/{self.pr}/labels", payload={"labels": [AI_REVIEW_PASSED_LABEL]})
        else:
            self.remove_labels(pull, AI_REVIEW_PASSED_LABEL)
            self.commit_status("failure" if self.state.coverage_complete else "error")

    # --- Shared GitHub operations ---

    def load_state(self) -> State:
        """Restore state from the publisher's summary, or start with an empty state."""
        bot_id = gh("user")["id"]
        pages = gh(f"{self.repo}/issues/{self.pr}/comments?per_page=100", "--paginate", "--slurp")
        comments = [comment for page in pages for comment in page]
        for comment in reversed(comments):
            if comment["user"]["id"] != bot_id or not comment["body"].startswith(STATE_MARKER):
                continue

            self.summary_comment_id = comment["id"]
            data = comment["body"].rsplit(STATE_DATA_MARKER, 1)[1].removesuffix("\n-->")
            return State.from_dict(json.loads(data))

        return State()

    def save_summary(self):
        """Create or update the summary comment with the current state."""
        if self.summary_comment_id:
            endpoint, method = f"issues/comments/{self.summary_comment_id}", "PATCH"
        else:
            endpoint, method = f"issues/{self.pr}/comments", "POST"

        comment = gh(f"{self.repo}/{endpoint}", "--method", method, payload={"body": self.state.body()})
        self.summary_comment_id = comment["id"]

    def feedback(self, event: dict, message: str):
        """Append feedback to the command comment, or post it for a label event."""
        body = f"[AI review bot]: {message}"
        if "comment" in event:
            endpoint = f"{self.repo}/issues/comments/{event['comment']['id']}"
            comment = gh(endpoint)
            gh(endpoint, "--method", "PATCH", payload={"body": f"{comment['body']}\n\n---\n{body}"})
        else:
            gh(f"{self.repo}/issues/{self.pr}/comments", payload={"body": f"@{event['sender']['login']} {body}"})

    def pull(self) -> dict:
        """Fetch the current PR details from GitHub."""
        return gh(f"{self.repo}/pulls/{self.pr}")

    def remove_labels(self, pull: dict, *names: str):
        """Remove the requested labels that are present on the PR."""
        current = {label["name"] for label in pull["labels"]}
        for name in names:
            if name in current:
                gh(f"{self.repo}/issues/{self.pr}/labels/{name}", "--method", "DELETE")

    def commit_status(self, status: str):
        """Publish the ai-review commit status for the reviewed head."""
        gh(
            f"{self.repo}/statuses/{self.state.head}",
            payload={
                "state": status,
                "context": "ai-review",
                "description": f"AI review: {self.state.status}; human approval still required",
            },
        )


def build_prompt(workspace: Path, request: dict) -> str:
    """Combine review rules, PR changes, prior findings and the user's comment."""
    kb = workspace / ".github/agent-review"
    mapping = []
    for path in sorted(kb.rglob("*.md")):
        if path.name in ("instructions.md", "README.md", "prompt_template.md"):
            continue
        ids = re.findall(r"^-\s+`([\w-]+#\w[\w-]*)`", path.read_text(), re.M)
        mapping.append(f"- {path.relative_to(workspace)}: {', '.join(ids)}")
    domains = resolve_domains((kb / "domains.yaml").read_text(), request["files"])

    return render_template(
        (kb / "prompt_template.md").read_text(),
        {
            "INSTRUCTIONS": (kb / "instructions.md").read_text(),
            "KB_MAP": "\n".join(mapping),
            "DOMAINS": "\n".join(d["name"] for d in domains) or "L1 only",
            "HEAD_SHA": request["state"]["head"],
            "BASE_SHA": request["merge_base"],
            "FILES": json.dumps(request["files"]),
            "DIFF": request["diff"],
            "TRUNCATED": request["truncated"],
            "FINDINGS": json.dumps(request["state"]["findings"]),
            "REPLY": request["reply"],
        },
    )


def resolve_domains(domains_yaml: str, files: list[str]) -> list[dict]:
    """Select review domains whose path patterns match the changed files."""
    registry = yaml.safe_load(domains_yaml)
    return [
        domain
        for domain in registry["domains"]
        if any(fnmatch.fnmatch(path, pattern) for path in files for pattern in domain["paths"])
    ]


def render_template(template: str, values: dict) -> str:
    """Fill template placeholders once, treating inserted content as data."""
    return re.sub(r"\{\{(\w+)\}\}", lambda match: str(values[match.group(1)]), template)


def agent_patch() -> list[dict]:
    """Build the agent's provider and model settings from the environment."""
    provider = "ai-review-gateway"
    model = os.environ["AI_REVIEW_MODEL"]

    return [
        {"id": "agent-default-model", "config": {"provider": provider, "model": model}},
        {
            "id": "llm-pi-ai",
            "config": {
                "providers": {
                    provider: {
                        "apiKeyEnv": "AI_REVIEW_KEY",
                        "api": "openai-completions",
                        "baseURL": os.environ["AI_REVIEW_API"],
                        "models": [{"id": model}],
                    }
                }
            },
        },
    ]


def run_agent(workspace: Path, request: dict) -> str:
    """Run the agent with a time limit and redact secrets from its output."""
    prompt = build_prompt(workspace, request)

    with tempfile.TemporaryDirectory(prefix="ai-review-") as temporary:
        home = Path(temporary)
        patch_path = home / "patch.yml"
        patch_path.write_text(yaml.safe_dump(agent_patch()))

        command = shlex.split(os.environ["AI_REVIEW_DSH_CMD"])
        command += ["--patch", str(patch_path), "-"]
        env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", "/tmp"),
            "AI_REVIEW_KEY": os.environ["AI_REVIEW_KEY"],
            "DEEPSEEK_API_KEY": "boot-gate-placeholder",
        }

        with subprocess.Popen(
            command,
            cwd=workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as process:
            try:
                output, _ = process.communicate(prompt, timeout=ROUND_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise

            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)

        for key in ("AI_REVIEW_KEY", "AI_REVIEW_API", "GH_TOKEN"):
            if value := os.environ.get(key):
                output = output.replace(value, "***REDACTED***")

        return output


def main() -> int:
    """Run the authorize, prepare or review phase selected by the workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("authorize", "prepare", "review"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--request", type=Path, default=Path("/tmp/ai-review/request.json"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    if args.phase == "review":
        request = json.loads(args.request.read_text())
        orchestrator = Orchestrator(workspace, request["pr"])
        output = run_agent(workspace, request)
        orchestrator.publish(request, output)
        return 0

    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    pr = event["pull_request"]["number"] if "pull_request" in event else event["issue"]["number"]
    orchestrator = Orchestrator(workspace, pr)

    if args.phase == "authorize":
        if orchestrator.authorize(os.environ["GITHUB_EVENT_NAME"], event):
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write("authorized=true\n")
        return 0

    if args.phase == "prepare":
        request = orchestrator.prepare(os.environ["GITHUB_EVENT_NAME"], event, os.environ["GITHUB_RUN_ID"])
        if request:
            args.request.parent.mkdir(parents=True, exist_ok=True)
            args.request.write_text(json.dumps(request))

            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write(f"review=true\nhead={request['state']['head']}\nbase={request['merge_base']}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
