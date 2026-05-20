from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import regex as re


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def collect_commit_range(*, repo: Path, start_ref: str, end_ref: str) -> list[dict[str, str]]:
    log_output = _run_git(
        repo,
        "log",
        "--reverse",
        "--format=%H%x1f%s%x1f%b%x1e",
        f"{start_ref}..{end_ref}",
    )
    commits: list[dict[str, str]] = []
    for record in log_output.split("\x1e"):
        if not record.strip():
            continue
        record = record.rstrip("\n")
        parts = record.split("\x1f", 2)
        if len(parts) == 2:
            sha, subject = parts
            body = ""
        elif len(parts) == 3:
            sha, subject, body = parts
        else:
            raise ValueError(f"Unexpected git log record format: {record!r}")
        commits.append(
            {
                "sha": sha.strip(),
                "subject": subject.strip(),
                "body": body.strip(),
            }
        )
    return commits


def _clean_summary_value(value: str) -> str:
    return value.strip().strip("`").strip()


def parse_final_summary_markdown(markdown: str) -> dict[str, Any]:
    partial_stop_fields = {
        "reason": "reason",
        "unresolved failures": "unresolved_failures",
        "saved patch": "patch_path",
        "saved failure summary": "summary_path",
        "repository state": "repository_state",
    }
    result: dict[str, Any] = {
        "status": "unknown",
        "upstream_range": "",
        "reached_commit": "",
        "steps_completed": 0,
        "steps_total": 0,
        "partial_stop": {},
    }
    partial: dict[str, str] = {}
    in_partial = False

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line == "### Partial Stop":
            in_partial = True
            continue
        if in_partial:
            if line.startswith("### "):
                in_partial = False
            elif line.startswith("- "):
                item = line[2:]
                if ":" not in item:
                    continue
                key, value = item.split(":", 1)
                normalized_key = key.strip().lower()
                cleaned_value = _clean_summary_value(value)
                if normalized_key == "stopped at":
                    partial["stopped_at"] = cleaned_value
                    step_match = re.match(r"([^,]+)", cleaned_value)
                    if step_match:
                        partial["step_id"] = _clean_summary_value(step_match.group(1))
                elif normalized_key in partial_stop_fields:
                    partial[partial_stop_fields[normalized_key]] = cleaned_value
                continue

        if line.startswith("Status:"):
            result["status"] = _clean_summary_value(line.split(":", 1)[1]).lower()
            continue
        if line.startswith("Upstream range:"):
            result["upstream_range"] = _clean_summary_value(line.split(":", 1)[1])
            continue
        if line.startswith("Reached upstream commit:"):
            result["reached_commit"] = _clean_summary_value(line.split(":", 1)[1])
            continue

        if steps_match := re.match(r"^Steps:\s*(\d+)\s*/\s*(\d+)\s*$", line):
            result["steps_completed"] = int(steps_match.group(1))
            result["steps_total"] = int(steps_match.group(2))

    if partial:
        result["partial_stop"] = partial

    return result


def _message_content_items(message: dict[str, Any]) -> list[Any]:
    content = message.get("content", [])
    if isinstance(content, list):
        return content
    if isinstance(content, (str, dict)):
        return [content]
    return []


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _format_tool_input(value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("command"), str):
        return value["command"]
    return _stringify_content(value)


def format_claude_stream_event(event: dict[str, Any]) -> list[str]:
    event_type = event.get("type", "event")
    if event_type == "system":
        subtype = event.get("subtype", "")
        session_id = event.get("session_id", "")
        suffix = f" session_id={session_id}" if session_id else ""
        return [f"[system] {subtype}{suffix}".rstrip()]

    if event_type in {"assistant", "user"}:
        lines: list[str] = []
        message = event.get("message") if isinstance(event.get("message"), dict) else event
        for item in _message_content_items(message):
            if isinstance(item, str):
                lines.extend([f"[{event_type}]", item])
                continue
            if not isinstance(item, dict):
                lines.extend([f"[{event_type}]", _stringify_content(item)])
                continue

            item_type = item.get("type", "content")
            if item_type == "text":
                lines.extend([f"[{event_type}]", _stringify_content(item.get("text", ""))])
            elif item_type == "tool_use":
                lines.extend(
                    [
                        f"[tool_use] {item.get('name', '')}".rstrip(),
                        _format_tool_input(item.get("input", "")),
                    ]
                )
            elif item_type == "tool_result":
                lines.extend(["[tool_result]", _stringify_content(item.get("content", ""))])
            else:
                lines.extend([f"[{event_type}:{item_type}]", _stringify_content(item)])
        return [line for line in lines if line != ""]

    if event_type == "result":
        subtype = event.get("subtype", "")
        if not subtype:
            subtype = "error" if event.get("is_error") else "success"
        details = [f"[result] {subtype}"]
        if "duration_ms" in event:
            details.append(f"duration_ms={event['duration_ms']}")
        if "total_cost_usd" in event:
            details.append(f"total_cost_usd={event['total_cost_usd']}")
        return [" ".join(details)]

    return [f"[{event_type}] {_stringify_content(event)}"]


def render_claude_stream(stream_path: Path) -> str:
    lines: list[str] = []
    with stream_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.rstrip("\n")
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                lines.append(f"[raw:{line_number}] {raw_line}")
                continue
            lines.extend(format_claude_stream_event(event))
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def render_manual_review_issue(
    *,
    pr_url: str,
    old_commit: str,
    new_commit: str,
    summary: dict[str, Any],
    summary_markdown: str,
) -> str:
    status = summary.get("status", "unknown")
    reached_commit = summary.get("reached_commit", "")
    steps_completed = summary.get("steps_completed", 0)
    steps_total = summary.get("steps_total", 0)
    partial_stop = summary.get("partial_stop") or {}

    lines = [
        "## Summary",
        "",
        "main2main automation stopped before completing all planned steps.",
        "",
        "## Context",
        "",
        f"- Draft PR: {pr_url}",
        f"- Commit range: `{old_commit}`...`{new_commit}`",
        f"- Status: `{status}`",
        f"- Reached commit: `{reached_commit}`",
        f"- Steps completed: `{steps_completed}/{steps_total}`",
        "",
    ]

    if partial_stop:
        lines.extend(["## Partial Stop", ""])
        if partial_stop.get("step_id"):
            lines.append(f"- Step: `{partial_stop['step_id']}`")
        if partial_stop.get("reason"):
            lines.append(f"- Reason: {partial_stop['reason']}")
        if partial_stop.get("patch_path"):
            lines.append(f"- Patch: `{partial_stop['patch_path']}`")
        if partial_stop.get("summary_path"):
            lines.append(f"- Failure summary: `{partial_stop['summary_path']}`")
        lines.append("")

    summary_markdown = summary_markdown.strip()
    if summary_markdown:
        lines.extend(["## Final Summary", "", summary_markdown, ""])

    return "\n".join(lines).rstrip() + "\n"


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helper CLI for main2main auto workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    commits_parser = subparsers.add_parser("collect-commit-range")
    commits_parser.add_argument("--repo", type=Path, required=True)
    commits_parser.add_argument("--start-ref", required=True)
    commits_parser.add_argument("--end-ref", required=True)

    claude_stream_parser = subparsers.add_parser("print-claude-stream")
    claude_stream_parser.add_argument("--input", type=Path, required=True)

    final_summary_parser = subparsers.add_parser("parse-final-summary")
    final_summary_parser.add_argument("--summary-md", type=Path, required=True)

    issue_parser = subparsers.add_parser("render-manual-review-issue")
    issue_parser.add_argument("--pr-url", required=True)
    issue_parser.add_argument("--old-commit", required=True)
    issue_parser.add_argument("--new-commit", required=True)
    issue_parser.add_argument("--summary-md", type=Path, required=True)

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "collect-commit-range":
        _print_json(collect_commit_range(repo=args.repo, start_ref=args.start_ref, end_ref=args.end_ref))
        return

    if args.command == "parse-final-summary":
        _print_json(parse_final_summary_markdown(args.summary_md.read_text(encoding="utf-8")))
        return

    if args.command == "print-claude-stream":
        print(render_claude_stream(args.input), end="")
        return

    if args.command == "render-manual-review-issue":
        summary_markdown = args.summary_md.read_text(encoding="utf-8")
        print(
            render_manual_review_issue(
                pr_url=args.pr_url,
                old_commit=args.old_commit,
                new_commit=args.new_commit,
                summary=parse_final_summary_markdown(summary_markdown),
                summary_markdown=summary_markdown,
            ),
            end="",
        )
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
