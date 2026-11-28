#!/usr/bin/env python3
"""Coordinate AI review on a GitHub pull request.

authorize checks the operator, prepare restores the PR state, and review runs the agent and publishes its result.
The summary stores the last published round. Activity and failures change its visible text without spending a round.
Command feedback and findings update their original comments. Human approval remains required.

Command receipt (the hidden marker identifies the line to update):
    [🤖 AI review bot]: ⏳ **Review requested** · [View run](.../runs/123) <!-- ai-review-feedback:456 -->

PR summary (hidden state excerpt):
    <!-- ai-review-orchestrator:v2 -->
    ## 🤖 AI review · DeepSeek Harness headless · POC pilot · ✅ **Review passed**
    <!-- ai-review-state
    {"status": "passed", "round": 1, "total": 2, "findings": []}
    -->

Inline finding:
    <!-- ai-review-finding:F001 -->
    ![MUST-FIX](.../must-fix.svg) **F001** · Open
    **Problem** Empty input prevents startup.
    **Suggested fix** Return the default configuration for an empty list.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path

import regex as re
import yaml

AI_REVIEW_LABEL = "ai-review"
AI_REVIEW_PASSED_LABEL = "ai-review-passed"
STATE_MARKER = "<!-- ai-review-orchestrator:v2 -->"
STATE_DATA_MARKER = "\n<!-- ai-review-state\n"
SUMMARY_TITLE = "## 🤖 AI review · DeepSeek Harness headless · POC pilot"
MAX_ROUNDS_PER_HEAD = 5
MAX_TOTAL_ROUNDS = 15
REVIEW_CLOCK_INTERVAL_MS = 10 * 60 * 1000
ROUND_TIMEOUT_SECONDS = 12 * 60
GIT_FETCH_MAX_ATTEMPTS = 3
GIT_FETCH_RETRY_DELAY_SECONDS = 5
SEVERITIES = ("MUST-FIX", "SHOULD-FIX", "NIT")


# --- Basic helpers ---


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


def format_status(icon: str, status: str, detail: str = "") -> str:
    """Format a status label and its optional explanation for a GitHub comment."""
    label = f"{icon} **{status}**"
    return f"{label} · {detail}" if detail else label


def actions_run_url() -> str:
    """Return the GitHub Actions link used for request receipts and failure logs."""
    repo_url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
    return f"{repo_url}/actions/runs/{os.environ['GITHUB_RUN_ID']}"


def read_only_prose(text: str) -> str:
    """Show a native suggestion fence in review prose as an ordinary code block."""
    return re.sub(r"(`{3,}|~{3,})[ \t]*suggestion\b", r"\1text", text, flags=re.I)


# --- Review data and rendering ---


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
    title: str = ""
    suggestion: str = ""
    preview: str = ""

    def details_body(self) -> str:
        """Render the problem, suggested fix, and latest evidence in GitHub Markdown."""
        problem = read_only_prose(self.body)
        sections = [f"**{read_only_prose(self.title)}**", f"**Problem**\n\n{problem}"] if self.title else [problem]
        if self.suggestion:
            sections.append(f"**Suggested fix**\n\n{read_only_prose(self.suggestion)}")
        if self.preview:
            # A Markdown file can itself contain code fences.
            longest = max((len(run) for run in re.findall(r"`+", self.preview)), default=0)
            fence = "`" * max(3, longest + 1)
            sections.append(f"{fence}diff\n{self.preview}\n{fence}")
        sections.append(f"Rule: `{read_only_prose(self.rule)}`")

        if self.status == "waived":
            sections.append(
                f"**Waived by @{self.waived_by}** at `{self.waived_head[:12]}`\n\n{read_only_prose(self.evidence)}"
            )
        elif self.evidence:
            label = "Resolution" if self.status == "resolved" else "Latest evidence"
            sections.append(f"**{label}**\n\n{read_only_prose(self.evidence)}")
        return "\n\n".join(sections)

    def inline_body(self, badge_root: str = "") -> str:
        """Show the severity and current decision on the original inline comment."""
        severity = f"![{self.severity}]({badge_root}/{self.severity.lower()}.svg)" if badge_root else self.severity
        return (
            f"<!-- ai-review-finding:{self.id} -->\n\n"
            f"{severity} **{self.id}** · {self.status.capitalize()}\n\n"
            f"{self.details_body()}"
        )


@dataclass
class State:
    """Represent the review progress and findings saved in the summary comment."""

    # A published "reviewing" result can still have blockers. Live agent activity is rendered separately.
    status: str = "idle"
    base: str = ""
    head: str = ""
    round: int = 0
    total: int = 0
    session_run_id: str = ""
    coverage_complete: bool = False
    requires_human: bool = False
    message: str = ""
    last_review: dict | None = None
    findings: list[Finding] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> State:
        """Restore the saved review state and its finding records."""
        fields = {**data, "findings": [Finding(**f) for f in data["findings"]]}
        old_summary = fields.pop("incomplete_review_summary", "")
        state = cls(**fields)
        if "last_review" in data:
            return state

        # Older comments saved prose but no reviewed merge base or separate task list.
        if not old_summary and state.status in ("passed", "reviewing", "waiting-human"):
            waiver_notice = re.fullmatch(
                r"@\S+ waived \d+ finding\(s\) (?:for this commit|at commit `[^`]+`)\.", state.message
            )
            if not waiver_notice:
                old_summary = state.message
        if old_summary:
            state.last_review = {"summary": old_summary}
            if state.message == old_summary:
                state.message = ""
        return state

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

    def published_outcomes(self, blocking: int) -> list[tuple[str, str, str]]:
        """Describe the published result and any decisions or limits that still apply."""
        if self.status == "stopped":
            return [("🛑", "Review stopped", "")]
        if self.status == "discarded":
            return [("🔄", "Review discarded", "PR changed during review")]
        if self.status == "passed":
            return [("✅", "Review passed", "No blocking findings")]
        if self.status == "idle":
            return [("⏳", "Awaiting review", "")]

        outcomes = []
        if self.status == "waiting-human":
            limits = []
            if self.round >= MAX_ROUNDS_PER_HEAD:
                limits.append(f"Commit {self.round}/{MAX_ROUNDS_PER_HEAD}")
            if self.total >= MAX_TOTAL_ROUNDS:
                limits.append(f"Session {self.total}/{MAX_TOTAL_ROUNDS}")
            if limits:
                outcomes.append(("⚠️", "Round limit reached", " · ".join(limits)))
        if self.requires_human:
            outcomes.append(("👤", "Maintainer review required", ""))
        if blocking:
            noun = "finding" if blocking == 1 else "findings"
            outcomes.append(("❌", "Changes required", f"{blocking} open MUST-FIX {noun}"))
        if self.coverage_complete:
            outcomes.append(("✅", "Static review complete", ""))
        else:
            outcomes.append(("⏳", "Review incomplete", ""))
        return outcomes

    @staticmethod
    def render_finding_group(title: str, findings: list[Finding], pr_url: str) -> str:
        """Link to inline findings and include full details for findings without a thread."""
        heading = f"<details>\n<summary>{title}</summary>"
        if not findings:
            return f"{heading}\n\nNo open findings.\n\n</details>"

        status = findings[0].status
        columns = ["ID", "Location"]
        if status != "open":
            columns += ["Severity", "Resolution" if status == "resolved" else "Waiver"]
        rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        details = []
        for f in findings:
            location = f"{f.path}:{f.line}" if f.line is not None else f.path
            identity = f"[{f.id}]({pr_url}#discussion_r{f.comment_id})" if f.comment_id else f.id
            cells = [identity, f"`{location}`"]
            if status == "resolved":
                cells += [f.severity, f.evidence]
            elif status == "waived":
                cells += [f.severity, f"@{f.waived_by} at `{f.waived_head[:12]}` · {f.evidence}"]
            rows.append("| " + " | ".join(" ".join(cell.splitlines()).replace("|", "\\|") for cell in cells) + " |")
            if not f.comment_id:
                details.append(
                    f"<details>\n<summary>{f.id} · Full finding</summary>\n\n{f.details_body()}\n\n</details>"
                )
        return "\n\n".join([heading, "\n".join(rows), *details, "</details>"])

    def visible_summary(self, pr_url: str, *, activity: str = "") -> str:
        """Show the current activity and the last published result in the summary."""
        if activity and self.status == "idle" and not self.last_review:
            return f"{SUMMARY_TITLE} · {activity}"

        statuses = ("open", "resolved", "waived")
        groups = {severity: {status: [] for status in statuses} for severity in SEVERITIES}
        for finding in self.findings:
            groups[finding.severity][finding.status].append(finding)

        (icon, status, detail), *other_outcomes = self.published_outcomes(len(groups["MUST-FIX"]["open"]))
        if activity:
            lines = [
                f"{SUMMARY_TITLE} · {activity}",
                f"**Last published result** · {format_status(icon, status, detail)}",
            ]
        else:
            lines = [f"{SUMMARY_TITLE} · {format_status(icon, status)}"]
            if detail:
                lines.append(detail)
        rounds = f"Round {self.round}/{MAX_ROUNDS_PER_HEAD} · Total rounds {self.total}/{MAX_TOTAL_ROUNDS}"
        lines.append(f"Commit `{self.head[:12]}` · {rounds}" if self.head else rounds)
        lines.extend(format_status(*outcome) for outcome in other_outcomes)
        if self.message:
            lines.append(self.message)

        repo_url = pr_url.rpartition("/pull/")[0]
        severity_icons = {"MUST-FIX": "🔴", "SHOULD-FIX": "🟤", "NIT": "⚪"}
        counts = ["| Severity | Open | Resolved | Waived |", "| --- | ---: | ---: | ---: |"]
        for severity, group in groups.items():
            cells = " | ".join(str(len(group[status])) for status in statuses)
            counts.append(f"| {severity_icons[severity]} {severity} | {cells} |")
        totals = [sum(len(group[s]) for group in groups.values()) for s in statuses]
        counts.append(f"| **Total ({len(self.findings)})** | " + " | ".join(str(n) for n in totals) + " |")
        lines.append("\n".join(counts))

        for severity, group in groups.items():
            findings = group["open"]
            title = f"{severity_icons[severity]} {severity} · {len(findings)} open"
            lines.append(self.render_finding_group(title, findings, pr_url))
        for status in ("resolved", "waived"):
            if findings := [f for f in self.findings if f.status == status]:
                lines.append(self.render_finding_group(f"{status.title()} · {len(findings)}", findings, pr_url))

        if self.last_review:
            if work := self.last_review.get("unfinished_work"):
                tasks = "\n".join(f"- {item}" for item in work)
                lines.append(f"<details>\n<summary>Unfinished work · {len(work)}</summary>\n\n{tasks}\n\n</details>")
            title = "Last published review"
            if "head" in self.last_review:
                title += f" · {self.last_review['head'][:12]}"
            lines.append(f"<details>\n<summary>{title}</summary>\n\n{self.last_review['summary']}\n\n</details>")
        lines.append("Human review is still required.")
        lines.append(
            "**Review commands**\n\n"
            f"[Allowlisted users only]({repo_url}/blob/main/.github/agent-review/allowlist.txt) · "
            "Post a new top-level PR comment.\n\n"
            "- `/ai-review` — Request another review of the current PR.\n"
            "- `/ai-review F001 F002 fixed in this commit` — Include notes about findings. "
            "The whole PR stays in scope.\n"
            "- `/ai-review-waive F001 reason` — Waive an open finding by ID and explain why.\n"
            "- `/ai-review-stop` — Stop AI review for this PR and remove its review labels."
        )
        return "\n\n".join(lines)

    def render_summary(self, pr_url: str = "") -> str:
        """Render the published summary with its saved review state."""
        return self.attach_state(self.visible_summary(pr_url))

    def attach_state(self, visible: str) -> str:
        """Attach the saved review state to the visible GitHub comment."""
        # Model text cannot terminate or replace the final machine-owned block.
        data = json.dumps(asdict(self), ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e")
        return f"{STATE_MARKER}\n\n{visible}{STATE_DATA_MARKER}{data}\n-->"


# --- Review result processing ---


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
    work = result.get("unfinished_work")
    if not isinstance(work, list) or not all(isinstance(item, str) and item.strip() for item in work):
        raise ValueError("unfinished_work must be an array of nonempty strings")
    if result["coverage_complete"] != (work == []):
        raise ValueError("coverage_complete must be true exactly when unfinished_work is empty")
    if not isinstance(result.get("findings"), list) or not isinstance(result.get("updates"), list):
        raise ValueError("findings and updates must be arrays")

    for finding in result["findings"]:
        if not isinstance(finding, dict) or not all(
            isinstance(finding.get(key), str) and finding[key].strip() for key in ("path", "rule", "title", "body")
        ):
            raise ValueError("Finding requires path, rule, title and body")
        if not isinstance(finding.get("suggestion"), str):
            raise ValueError("Finding suggestion must be a string")
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
        if "severity" in update and update["severity"] not in SEVERITIES:
            raise ValueError("Unknown update severity")
        if not update["evidence"].strip():
            raise ValueError("Finding updates require evidence")
        seen.add(identity)

    return result


def replacement_preview(workspace: Path, finding: Finding, replacement: dict) -> str:
    """Compare a proposed local replacement with the lines in the reviewed Git blob."""
    if not isinstance(replacement, dict) or not isinstance(replacement.get("code"), str):
        return ""
    start = replacement.get("start_line")
    end = finding.line
    if end is None or type(start) is not int or not 1 <= start <= end:
        return ""

    # A finding can point outside the new tree (for example, to a deleted file).
    source = subprocess.run(
        ["git", "-C", str(workspace), "cat-file", "blob", f"{finding.head}:{finding.path}"], capture_output=True
    )
    if source.returncode:
        return ""
    lines = source.stdout.splitlines()
    if end > len(lines):
        return ""
    try:
        before = [line.decode() for line in lines[start - 1 : end]]
    except UnicodeDecodeError:
        return ""  # A binary file has no text preview.
    after = replacement["code"].splitlines()
    diff = difflib.unified_diff(before, after, n=max(len(before), len(after)), lineterm="")
    return "\n".join(list(diff)[3:])  # Skip the file and hunk headers.


def merge_findings(state: State, result: dict, workspace: Path) -> list[Finding]:
    """Update the finding list and return newly added records for inline comments."""
    known = {f.id: f for f in state.findings}
    created = []
    for new in result["findings"]:
        # Waived IDs are archived. An independent report can start a new thread at the same line.
        existing = next(
            (
                f
                for f in state.findings
                if f.status != "waived"
                and (f.path, f.line, f.rule, f.body) == (new["path"], new["line"], new["rule"], new["body"])
            ),
            None,
        )
        if existing:
            if existing.status == "resolved":
                existing.status = "open"
                existing.evidence = ""
            existing.severity = min((existing.severity, new["severity"]), key=SEVERITIES.index)
            continue

        finding = Finding(
            id=f"F{len(state.findings) + 1:03d}",
            head=state.head,
            **{key: new[key] for key in ("path", "line", "severity", "rule", "title", "body", "suggestion")},
        )
        if new.get("replacement") is not None:
            finding.preview = replacement_preview(workspace, finding, new["replacement"])
        state.findings.append(finding)
        created.append(finding)

    # An explicit update to an ID takes precedence over an exact repeat in findings.
    for update in result["updates"]:
        finding = known[update["id"]]
        finding.status = update["status"]
        finding.severity = update.get("severity", finding.severity)
        finding.evidence = update["evidence"]

    return created


# --- Prompt and agent execution ---


def resolve_domains(domains_yaml: str, files: list[str]) -> list[dict]:
    """Select review domains whose path patterns match the changed files."""
    registry = yaml.safe_load(domains_yaml)
    return [
        domain
        for domain in registry["domains"]
        if any(fnmatch.fnmatch(path, pattern) for path in files for pattern in domain["paths"])
    ]


def quote_reference(text: str) -> str:
    """Keep headings and blank lines in supplied text inside a Markdown block quote."""
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def render_template(template: str, values: dict) -> str:
    """Fill template placeholders once, treating inserted content as data."""
    return re.sub(r"\{\{(\w+)\}\}", lambda match: str(values[match.group(1)]), template)


def build_prompt(workspace: Path, request: dict) -> str:
    """Combine review rules, PR changes, prior findings and the user's comment."""
    kb = workspace / ".github/agent-review"
    mapping = []
    for path in sorted(kb.rglob("*.md")):
        if path.name in ("instructions.md", "README.md", "prompt_template.md"):
            continue
        ids = re.findall(r"^-\s+`([\w-]+#\w[\w-]*)`", path.read_text(), re.M)
        mapping.append(f"- {path.relative_to(kb)}: {', '.join(ids)}")

    previous_findings = []
    waived_findings = []
    for finding in request["state"]["findings"]:
        if finding["status"] == "waived":
            waived_findings.append(
                {
                    "id": finding["id"],
                    "path": finding["path"],
                    "line": finding["line"],
                    "body": finding["body"],
                    "reason": finding["evidence"],
                }
            )
        else:
            previous_findings.append({k: v for k, v in finding.items() if k not in ("title", "suggestion", "preview")})

    last_review = request["state"]["last_review"]
    snapshot = "No previous review on record."
    summary = "None."
    unfinished_work = "None."
    if last_review:
        summary = last_review["summary"]
        if "merge_base" in last_review:
            snapshot = (
                f"Reviewed head: {last_review['head']}\nMerge base: {last_review['merge_base']}\n"
                f"Coverage complete: {json.dumps(last_review['coverage_complete'])}"
            )
            unfinished_work = "\n".join(f"- {item}" for item in last_review["unfinished_work"]) or "None."
        else:
            snapshot = "Not recorded by the previous workflow."
            unfinished_work = "Not recorded separately. See the previous review summary."

    return render_template(
        (kb / "prompt_template.md").read_text(),
        {
            "INSTRUCTIONS": (kb / "instructions.md").read_text(),
            "TRUSTED_ROOT": workspace,
            "KB_MAP": "\n".join(mapping),
            "DOMAINS": ", ".join(request["domains"]) or "L1 only",
            "PR_TITLE": request["title"],
            "PR_BODY": quote_reference(request["body"]),
            "BASE_SHA": request["state"]["base"],
            "MERGE_BASE_SHA": request["merge_base"],
            "HEAD_SHA": request["state"]["head"],
            "VLLM_SHA": git(workspace / "vllm", "rev-parse", "HEAD").strip(),
            "DIFF_STAT": request["diff_stat"],
            "PREVIOUS_REVIEW_SNAPSHOT": quote_reference(snapshot),
            "PREVIOUS_REVIEW_SUMMARY": quote_reference(summary),
            "UNFINISHED_WORK": quote_reference(unfinished_work),
            "FINDINGS": json.dumps(previous_findings),
            "WAIVED_FINDINGS": json.dumps(waived_findings, ensure_ascii=False),
            "REPLY": quote_reference(request["reply"]),
        },
    )


def agent_patch(workspace: Path) -> list[dict]:
    """Configure the review model and its trusted project context."""
    provider = "ai-review-gateway"
    model = os.environ["AI_REVIEW_MODEL"]

    return [
        {"id": "fs-sandbox", "config": {"cwd": str(workspace / "head")}},
        {"id": "agent-instructions", "disabled": True},
        {
            "id": "skill-filesystem",
            "config": {"includeDefaultRoots": False, "customSkillDirs": [str(workspace / ".agents/skills")]},
        },
        {"id": "tool-executor-native", "disabled": True},
        {"id": "tool-executor-ptc", "disabled": False},
        {"id": "tool-web", "disabled": True},
        {
            "insert": [
                {
                    "id": "time-context",
                    "name": "@deepseek-ai/dsh-time-context",
                    "config": {"timeZone": "UTC", "refreshIntervalMs": REVIEW_CLOCK_INTERVAL_MS},
                }
            ]
        },
        {"id": "agent-default-model", "config": {"provider": provider, "model": model, "reasoningEffort": "high"}},
        {
            "id": "llm-pi-ai",
            "config": {
                "providers": {
                    provider: {
                        "apiKeyEnv": "AI_REVIEW_KEY",
                        "api": "openai-completions",
                        "baseURL": os.environ["AI_REVIEW_API"],
                        "compat": {"thinkingFormat": "deepseek", "supportsReasoningEffort": True},
                        "models": [{"id": model, "reasoningEfforts": {"high": "high"}}],
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
        patch_path.write_text(yaml.safe_dump(agent_patch(workspace)))

        command = shlex.split(os.environ["AI_REVIEW_DSH_CMD"])
        command += ["--patch", str(patch_path), "-"]
        env = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", "/tmp"),
            # The Actions job container provides isolation for this process.
            "DSH_PERMISSION_MODE": "danger-full-access",
            "AI_REVIEW_KEY": os.environ["AI_REVIEW_KEY"],
            "DEEPSEEK_API_KEY": "boot-gate-placeholder",
        }

        with subprocess.Popen(
            command,
            # Start the CLI from trusted files. Its session uses the PR worktree.
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


# --- Review orchestration ---


class Orchestrator:
    """Coordinate PR permissions, review preparation and result publication."""

    # --- Initialization and events ---

    def __init__(self, workspace: Path, pr: int):
        """Set the workspace and PR for this review."""
        self.workspace = workspace
        self.pr = pr
        self.repo = "repos/:owner/:repo"
        repository = os.environ.get("GITHUB_REPOSITORY")
        self.badge_root = (
            f"https://raw.githubusercontent.com/{repository}/main/.github/agent-review/badges" if repository else ""
        )
        self.summary_comment_id = None

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

    # --- Review requests ---

    def authorize(self, event_name: str, event: dict) -> bool:
        """Check permissions and handle immediate label changes before queuing."""
        command, _ = self.event_command(event_name, event)
        actor = event["sender"]["login"]
        if not command or event_name == "issue_comment" and event["sender"]["type"] == "Bot":
            return False

        pull = self.pull()
        if pull["state"] != "open":
            if command != "push":
                self.feedback(event, format_status("⚠️", "PR closed"))
            return False

        repo_url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
        # Pushes and label removals sync changes that already happened on GitHub.
        if command == "start" or event_name == "issue_comment":
            allowlist = (self.workspace / ".github/agent-review/allowlist.txt").read_text()
            if actor not in parse_allowlist(allowlist):
                detail = (
                    f"[Pilot allowlist]({repo_url}/blob/main/.github/agent-review/allowlist.txt) required. "
                    "Request access on [RFC #21451](https://github.com/vllm-project/vllm-ascend/issues/21451)."
                )
                if command == "start":
                    self.remove_labels(pull, AI_REVIEW_LABEL)
                    detail = "Label removed. " + detail
                self.feedback(event, format_status("🔒", "Permission denied", detail))
                return False

        enabled = AI_REVIEW_LABEL in {label["name"] for label in pull["labels"]}
        if command == "stop" and enabled:
            return False
        if command not in ("stop", "/ai-review-stop") and not enabled:
            if command != "push":
                self.feedback(
                    event, format_status("⚠️", "Review not enabled", "Ask an allowlisted reviewer to add `ai-review`.")
                )
            return False

        if command == "push":
            if event["pull_request"]["head"]["sha"] == pull["head"]["sha"]:
                self.remove_labels(pull, AI_REVIEW_PASSED_LABEL)
            return False  # Pushes revoke the old pass but never request a review.

        run_link = f"[View run]({actions_run_url()})"
        if command in ("stop", "/ai-review-stop"):
            self.remove_labels(pull, AI_REVIEW_LABEL, AI_REVIEW_PASSED_LABEL)
            message = format_status(
                "⏳", "Stop requested", f"{run_link} · To restart, wait for this run to finish and reapply `ai-review`."
            )
        elif command == "/ai-review-waive":
            message = format_status("⏳", "Waiver requested", run_link)
        else:
            message = format_status("⏳", "Review requested", run_link)
        self.feedback(event, message)
        return True

    def prepare(self, event_name: str, event: dict, run_id: str) -> dict | None:
        """Handle queued commands or prepare a review of the current PR."""
        command, argument = self.event_command(event_name, event)
        actor = event["sender"]["login"]

        # Recheck the queued request against the current PR and labels.
        pull = self.pull()
        if pull["state"] != "open":
            self.feedback(event, format_status("⚠️", "PR closed"))
            return None

        enabled = AI_REVIEW_LABEL in {label["name"] for label in pull["labels"]}
        stopping = command in ("stop", "/ai-review-stop")
        if command == "stop" and enabled:
            return None  # The label has already been reapplied to restart.
        if not stopping and not enabled:
            self.feedback(
                event, format_status("⚠️", "Review not enabled", "Ask an allowlisted reviewer to add `ai-review`.")
            )
            return None

        # Restore the published state before handling a stop, restart or waiver.
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
                event, format_status("🛑", "Review stopped", "Ask an allowlisted reviewer to reapply `ai-review`.")
            )
            return None

        if command == "/ai-review-waive":
            message = self.waive(argument, actor, pull)
            self.feedback(event, message)
            return None

        # A new head resets its own round count. Enforce both the session and commit limits.
        if self.state.head != pull["head"]["sha"]:
            self.state.round = 0
        self.state.base, self.state.head = pull["base"]["sha"], pull["head"]["sha"]
        if self.state.total >= MAX_TOTAL_ROUNDS:
            self.feedback(
                event,
                format_status(
                    "⚠️",
                    "Round limit reached",
                    f"{self.state.total}/{MAX_TOTAL_ROUNDS} rounds used for this session. "
                    "Ask an allowlisted reviewer to reapply `ai-review` to restart.",
                ),
            )
            return None
        if self.state.round >= MAX_ROUNDS_PER_HEAD:
            self.feedback(
                event,
                format_status(
                    "⚠️",
                    "Round limit reached",
                    f"{self.state.round}/{MAX_ROUNDS_PER_HEAD} rounds used for this commit. "
                    "Push changes and comment `/ai-review`, or ask an allowlisted reviewer to reapply `ai-review`.",
                ),
            )
            return None

        # Fetch the reviewed commits and collect the context for this round.
        self.remove_labels(pull, AI_REVIEW_PASSED_LABEL)
        self.state.status = "reviewing"
        self.state.coverage_complete = False
        self.state.requires_human = False

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
        diff_stat = git(self.workspace, "diff", "--stat", "--no-ext-diff", "--no-textconv", merge_base, self.state.head)
        files = git(self.workspace, "diff", "--name-only", "-z", merge_base, self.state.head).split("\0")[:-1]
        registry = self.workspace / ".github/agent-review/domains.yaml"
        is_release = pull["base"]["ref"].startswith("releases/")

        # Reserve the summary before any finding is posted, then build the agent request.
        if self.summary_comment_id is None:
            self.save_summary(State().attach_state(SUMMARY_TITLE))

        return {
            "pr": self.pr,
            "title": pull["title"],
            "body": pull["body"] or "",
            "comment_id": self.summary_comment_id,
            "state": asdict(self.state),
            "merge_base": merge_base,
            "vllm_pin_file": "vllm-release-tag.commit" if is_release else "vllm-main-verified.commit",
            "domains": [domain["name"] for domain in resolve_domains(registry.read_text(), files)],
            "diff_stat": diff_stat.rstrip("\n"),
            "reply": event["comment"]["body"] if event_name == "issue_comment" else "",
        }

    def waive(self, argument: str, actor: str, pull: dict) -> str:
        """Archive selected open findings and record who waived them and why."""
        parts = argument.split(maxsplit=1)
        if len(parts) != 2:
            return format_status(
                "⚠️", "Waiver rejected", "Use `/ai-review-waive F001 reason` or `/ai-review-waive all reason`."
            )
        if (self.state.base, self.state.head) != (pull["base"]["sha"], pull["head"]["sha"]):
            return format_status(
                "⚠️", "Waiver rejected", "Review the current commit with `/ai-review` before waiving findings."
            )

        selected = [f for f in self.state.findings if f.status == "open" and parts[0] in ("all", f.id)]
        if not selected:
            return format_status("⚠️", "Waiver rejected", "No matching open findings. Check the IDs in the summary.")

        for finding in selected:
            finding.status = "waived"
            finding.waived_by = actor
            finding.waived_head = self.state.head
            finding.evidence = parts[1]
            self.update_inline(finding)

        self.state.message = f"@{actor} waived {len(selected)} finding(s) at commit `{self.state.head[:12]}`."
        self.state.decide()
        self.save_summary()
        self.publish_status()
        detail = selected[0].id if len(selected) == 1 else f"{len(selected)} findings"
        return format_status("✅", "Waiver applied", detail)

    # --- Review execution and publication ---

    def review(self, request: dict):
        """Run the agent, publish the result and report failures against the last saved round."""
        self.state = State.from_dict(request["state"])
        self.summary_comment_id = request["comment_id"]
        try:
            self.commit_status("pending")
            next_round = self.state.round + 1
            self.show_activity(format_status("🔍", "Reviewing", f"Round {next_round}/{MAX_ROUNDS_PER_HEAD}"))
            output = run_agent(self.workspace, request)
            self.publish(request, output)
        except Exception:
            self.state.status = "error"
            with suppress(subprocess.CalledProcessError):
                self.show_activity(format_status("❌", "Review failed", f"[View logs]({actions_run_url()})"))
            with suppress(subprocess.CalledProcessError):
                self.commit_status("error")
            raise

    def publish(self, request: dict, output: str):
        """Publish findings or a discard notice, saving the completed round."""
        state = self.state
        pull = self.pull()
        if pull["state"] != "open" or AI_REVIEW_LABEL not in {label["name"] for label in pull["labels"]}:
            print("Review stopped or PR closed; no result published.")
            state.status = "stopped"
            detail = "PR closed" if pull["state"] != "open" else "AI review label removed"
            self.show_activity(format_status("🛑", "Review stopped", detail))
            self.commit_status("error")
            return

        state.round += 1
        state.total += 1

        if (pull["base"]["sha"], pull["head"]["sha"]) != (state.base, state.head):
            state.status = "discarded"
            state.message = (
                "The PR changed during review, so this result was discarded. "
                "This published notice counts as one round for the reviewed version. "
                "Comment /ai-review when the new changes are ready."
            )
        else:
            result = parse_result(output, state.findings)
            previous_comments = [(f, f.inline_body()) for f in state.findings if f.comment_id is not None]
            created = merge_findings(state, result, self.workspace)
            state.coverage_complete = result["coverage_complete"]
            state.requires_human = result["requires_human"]
            state.message = ""
            state.last_review = {
                "head": state.head,
                "merge_base": request["merge_base"],
                "coverage_complete": state.coverage_complete,
                "summary": result["summary"],
                "unfinished_work": result["unfinished_work"],
            }

            state.decide()
            for finding, old_body in previous_comments:
                if finding.inline_body() != old_body:
                    self.update_inline(finding)
            self.publish_inline(created)

        # Persist the round before updating labels and commit status; those writes can fail independently.
        self.save_summary()
        self.publish_status()

    def publish_status(self):
        """Update the commit status and pass label if the reviewed commits are current."""
        # Branches and labels can change while comments are being published.
        pull = self.pull()
        if (
            pull["state"] != "open"
            or AI_REVIEW_LABEL not in {label["name"] for label in pull["labels"]}
            or (pull["base"]["sha"], pull["head"]["sha"]) != (self.state.base, self.state.head)
        ):
            # Finish the status on the reviewed commit without changing the PR's labels.
            self.state.status = "discarded"
            self.commit_status("error")
            return

        if self.state.status == "passed":
            self.commit_status("success")
            gh(f"{self.repo}/issues/{self.pr}/labels", payload={"labels": [AI_REVIEW_PASSED_LABEL]})
        else:
            self.remove_labels(pull, AI_REVIEW_PASSED_LABEL)
            self.commit_status("failure" if self.state.coverage_complete else "error")

    # --- Summary persistence ---

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

    def save_summary(self, body: str | None = None):
        """Write an explicit comment body or persist the current working state by default."""
        if self.summary_comment_id:
            endpoint, method = f"issues/comments/{self.summary_comment_id}", "PATCH"
        else:
            endpoint, method = f"issues/{self.pr}/comments", "POST"

        if body is None:
            body = self.state.render_summary(self.pr_url)
        comment = gh(f"{self.repo}/{endpoint}", "--method", method, payload={"body": body})
        self.summary_comment_id = comment["id"]

    def show_activity(self, message: str):
        """Show activity without persisting the working state or spending a round."""
        comment = gh(f"{self.repo}/issues/comments/{self.summary_comment_id}")
        saved_state = comment["body"].rsplit(STATE_DATA_MARKER, 1)[1]
        published_state = State.from_dict(json.loads(saved_state.removesuffix("\n-->")))
        visible = published_state.visible_summary(self.pr_url, activity=message)
        # Keep the saved JSON exactly as GitHub returned it, including older state formats.
        body = f"{STATE_MARKER}\n\n{visible}{STATE_DATA_MARKER}{saved_state}"
        self.save_summary(body)

    # --- Comments ---

    def feedback(self, event: dict, message: str):
        """Update one command receipt or post feedback for a label event."""
        prefix = "[🤖 AI review bot]:"
        body = f"{prefix} {message}"
        if "comment" not in event:
            gh(f"{self.repo}/issues/{self.pr}/comments", payload={"body": f"@{event['sender']['login']} {body}"})
            return

        comment_id = event["comment"]["id"]
        endpoint = f"{self.repo}/issues/comments/{comment_id}"
        current = gh(endpoint)["body"]
        marker = f"<!-- ai-review-feedback:{comment_id} -->"
        line = f"{body} {marker}"

        # Replace only the bot's line so edits around the receipt survive.
        pattern = rf"(?m)^{re.escape(prefix)} .*?{marker}$"
        updated, replaced = re.subn(pattern, lambda _: line, current, count=1)
        if not replaced:
            updated = f"{current}\n\n---\n{line}"
        gh(endpoint, "--method", "PATCH", payload={"body": updated})

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
                "body": finding.inline_body(self.badge_root),
            }

            # Only an explicit 422 validation rejection falls back to summary.
            # Network/auth/server failures keep gh's original error and stop.
            comment = gh(f"{self.repo}/pulls/{self.pr}/comments", payload=payload, anchor_fallback=True)
            if comment:
                finding.comment_id = comment["id"]

    def update_inline(self, finding: Finding):
        """Refresh the original inline comment when the finding has one."""
        if finding.comment_id is not None:
            gh(
                f"{self.repo}/pulls/comments/{finding.comment_id}",
                "--method",
                "PATCH",
                payload={"body": finding.inline_body(self.badge_root)},
            )

    # --- GitHub operations ---

    @property
    def pr_url(self) -> str:
        """Return the URL for this pull request."""
        return f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/pull/{self.pr}"

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


# --- CLI entry point ---


def main() -> int:
    """Run the authorize, prepare or review phase selected by the workflow."""
    parser = argparse.ArgumentParser(description="Run one phase of AI review for a GitHub pull request.")
    parser.add_argument("phase", choices=("authorize", "prepare", "review"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--request", type=Path, default=Path("/tmp/ai-review/request.json"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    if args.phase == "review":
        request = json.loads(args.request.read_text())
        orchestrator = Orchestrator(workspace, request["pr"])
        orchestrator.review(request)
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
                output.write(f"review=true\nhead={request['state']['head']}\n")
                output.write(f"vllm_pin_file={request['vllm_pin_file']}\n")
                if "doc-quality" in request["domains"]:
                    output.write("doc_quality=true\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
