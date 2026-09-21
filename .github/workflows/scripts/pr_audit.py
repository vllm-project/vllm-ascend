#!/usr/bin/env python3
"""GitHub-native PR convention audit.

The script only reads GitHub API metadata and files. It never checks out or
executes pull-request head code, so it is safe for pull_request_target use.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import yaml


API_URL = "https://api.github.com"
COMMENT_MARKER = "<!-- pr-audit -->"
META_PATTERN = re.compile(r"<!-- pr-audit-meta: first_failed_at=([^ ]+) -->")
LABEL_COLORS = {
    "pr-audit-pending": "FBCA04",
    "pr-audit-failed": "D73A4A",
    "pr-audit-reminded": "1D76DB",
    "pr-audit-escalated": "B60205",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    message: str
    severity: str = "error"


class GitHubClient:
    def __init__(self, repository: str, token: str) -> None:
        self.repository = repository
        self.token = token

    def request(self, method: str, path: str, body: Any | None = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            API_URL + path,
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": "Bearer %s" % self.token,
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "vllm-ascend-pr-audit",
                **({"Content-Type": "application/json"} if data else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read().decode("utf-8")
                return json.loads(content) if content else None
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError("GitHub API %s %s failed: HTTP %d: %s" % (
                method, path, error.code, detail,
            )) from error

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def get_optional(self, path: str) -> Any | None:
        try:
            return self.get(path)
        except RuntimeError as error:
            if "HTTP 404" in str(error):
                return None
            raise

    def get_pages(self, path: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            separator = "&" if "?" in path else "?"
            result = self.get("%s%sper_page=100&page=%d" % (path, separator, page))
            if not isinstance(result, list):
                return items
            items.extend(item for item in result if isinstance(item, dict))
            if len(result) < 100:
                return items
            page += 1

    def get_pull_request(self, number: int) -> dict[str, Any]:
        return self.get("/repos/%s/pulls/%d" % (self.repository, number))

    def get_issue(self, number: int) -> dict[str, Any] | None:
        return self.get_optional("/repos/%s/issues/%d" % (self.repository, number))

    def get_files(self, number: int) -> list[dict[str, Any]]:
        return self.get_pages("/repos/%s/pulls/%d/files" % (self.repository, number))

    def get_comments(self, number: int) -> list[dict[str, Any]]:
        return self.get_pages("/repos/%s/issues/%d/comments" % (self.repository, number))

    def create_comment(self, number: int, body: str) -> None:
        self.request("POST", "/repos/%s/issues/%d/comments" % (self.repository, number), {"body": body})

    def update_comment(self, comment_id: int, body: str) -> None:
        self.request("PATCH", "/repos/%s/issues/comments/%d" % (self.repository, comment_id), {"body": body})

    def add_label(self, number: int, label: str) -> None:
        label_path = urllib.parse.quote(label, safe="")
        if self.get_optional("/repos/%s/labels/%s" % (self.repository, label_path)) is None:
            try:
                self.request("POST", "/repos/%s/labels" % self.repository, {
                    "name": label,
                    "color": LABEL_COLORS.get(label, "D4C5F9"),
                })
            except RuntimeError as error:
                # A concurrent run may have created the label first.
                if "HTTP 422" not in str(error):
                    raise
        self.request("POST", "/repos/%s/issues/%d/labels" % (self.repository, number), {"labels": [label]})

    def remove_label(self, number: int, label: str) -> None:
        try:
            self.request("DELETE", "/repos/%s/issues/%d/labels/%s" % (
                self.repository, number, urllib.parse.quote(label, safe=""),
            ))
        except RuntimeError as error:
            if "HTTP 404" not in str(error):
                raise


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Audit configuration must be a mapping")
    return config


def linked_issue_numbers(body: str, keywords: list[str]) -> list[int]:
    keyword_pattern = "|".join(re.escape(keyword) for keyword in keywords)
    matches = re.findall(r"(?im)\b(?:%s)\s+(?:[\w.-]+/[\w.-]+)?#(\d+)" % keyword_pattern, body)
    return list(dict.fromkeys(int(number) for number in matches))


def has_code_changes(files: list[dict[str, Any]], extensions: list[str]) -> bool:
    return any(str(file_data.get("filename", "")).lower().endswith(tuple(extensions)) for file_data in files)


def has_test_changes(files: list[dict[str, Any]], substrings: list[str]) -> bool:
    return any(any(substring.lower() in str(file_data.get("filename", "")).lower() for substring in substrings)
               for file_data in files)


def audit_pull_request(
    pull_request: dict[str, Any],
    pr_issue: dict[str, Any] | None,
    files: list[dict[str, Any]],
    config: dict[str, Any],
    client: GitHubClient,
) -> list[CheckResult]:
    title = str(pull_request.get("title") or "")
    body = str(pull_request.get("body") or "")
    title_lower = title.lower()
    results: list[CheckResult] = []

    prefixes = [str(prefix).lower() for prefix in config.get("issue_required_title_prefixes", [])]
    requires_issue = any(prefix in title_lower for prefix in prefixes)
    milestone = (pr_issue or {}).get("milestone") or pull_request.get("milestone")
    references = linked_issue_numbers(body, [str(item) for item in config.get("issue_keywords", [])])
    valid_references = [number for number in references if client.get_issue(number) is not None]
    issue_requirement_passed = bool(valid_references) or (bool(milestone) and config.get("allow_milestone", True))
    if requires_issue:
        results.append(CheckResult(
            "linked_issue",
            issue_requirement_passed,
            "已关联 Issue/RFC。" if issue_requirement_passed else
            "[BugFix]/[Feature] PR 需在描述中加入 Fixes、Closes、Resolves 或 Refs #<issue_number>，或关联 milestone。",
        ))

    title_min_length = int(config.get("title_min_length", 0))
    results.append(CheckResult(
        "title_length", len(title.strip()) >= title_min_length,
        "PR 标题长度符合要求。" if len(title.strip()) >= title_min_length else
        "PR 标题至少需要 %d 个字符。" % title_min_length,
    ))
    body_min_length = int(config.get("body_min_length", 0))
    results.append(CheckResult(
        "body_length", len(body.strip()) >= body_min_length,
        "PR 描述长度符合要求。" if len(body.strip()) >= body_min_length else
        "PR 描述至少需要 %d 个字符。" % body_min_length,
    ))

    additions = sum(int(file_data.get("additions") or 0) for file_data in files)
    max_added_lines = int(config.get("max_added_lines", 0))
    results.append(CheckResult(
        "large_diff", additions <= max_added_lines,
        "新增代码行数为 %d，符合限制。" % additions if additions <= max_added_lines else
        "新增代码 %d 行，超过 %d 行限制。" % (additions, max_added_lines),
    ))

    if has_code_changes(files, [str(extension).lower() for extension in config.get("code_extensions", [])]):
        test_changed = has_test_changes(files, [str(item) for item in config.get("test_path_substrings", [])])
        results.append(CheckResult(
            "tests", test_changed,
            "检测到测试文件变更。" if test_changed else "检测到代码变更，但未检测到测试文件变更。",
            "warning",
        ))

    forbidden = [link for link in config.get("forbidden_links", []) if str(link) in body]
    results.append(CheckResult(
        "forbidden_links", not forbidden,
        "未检测到禁止链接。" if not forbidden else "PR 描述包含禁止链接：%s" % ", ".join(forbidden),
    ))
    return results


def failed_results(results: list[CheckResult], config: dict[str, Any]) -> list[CheckResult]:
    include_warnings = bool(config.get("block_on_warnings", False))
    return [result for result in results if not result.passed and (include_warnings or result.severity == "error")]


def audit_comment(author: str, results: list[CheckResult], first_failed_at: str | None = None) -> str:
    failures = [result for result in results if not result.passed]
    if not failures:
        return "%s\n\n## PR 规范审计：通过\n\n所有已启用的 PR 规范检查均已通过。" % COMMENT_MARKER
    timestamp = first_failed_at or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    lines = [
        COMMENT_MARKER,
        "<!-- pr-audit-meta: first_failed_at=%s -->" % timestamp,
        "## PR 规范审计：待整改",
        "",
        "@%s，请处理以下事项；修改后推送提交或编辑 PR 描述会自动复检：" % author,
        "",
    ]
    for result in failures:
        icon = "⚠️" if result.severity == "warning" else "❌"
        lines.append("- %s %s" % (icon, result.message))
    return "\n".join(lines)


def find_audit_comment(comments: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((comment for comment in comments if COMMENT_MARKER in str(comment.get("body", ""))), None)


def comment_first_failed_at(comment: dict[str, Any] | None) -> str | None:
    if not comment:
        return None
    match = META_PATTERN.search(str(comment.get("body", "")))
    return match.group(1) if match else None


def upsert_audit_comment(client: GitHubClient, number: int, author: str, results: list[CheckResult]) -> None:
    existing = find_audit_comment(client.get_comments(number))
    body = audit_comment(author, results, comment_first_failed_at(existing))
    if existing and existing.get("id"):
        client.update_comment(int(existing["id"]), body)
    else:
        client.create_comment(number, body)


def set_audit_labels(client: GitHubClient, number: int, results: list[CheckResult], config: dict[str, Any]) -> None:
    labels = config.get("labels", {})
    pending = str(labels.get("pending", "pr-audit-pending"))
    failed = str(labels.get("failed", "pr-audit-failed"))
    reminder = str(labels.get("reminded", "pr-audit-reminded"))
    escalated = str(labels.get("escalated", "pr-audit-escalated"))
    errors = [result for result in results if not result.passed and result.severity == "error"]
    if errors:
        client.add_label(number, pending)
        client.add_label(number, failed)
        return
    for label in (pending, failed, reminder, escalated):
        client.remove_label(number, label)


def write_summary(results: list[CheckResult]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = ["## PR Audit", "", "| Check | Result |", "| --- | --- |"]
    lines.extend("| %s | %s %s |" % (result.name, "✅" if result.passed else "❌", result.message)
                 for result in results)
    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write("\n".join(lines) + "\n")


def run_audit(config: dict[str, Any], event_path: str) -> int:
    with open(event_path, encoding="utf-8") as event_file:
        event = json.load(event_file)
    pull_request_event = event.get("pull_request", {})
    number = int(pull_request_event.get("number") or event.get("number"))
    repository = os.environ["GITHUB_REPOSITORY"]
    client = GitHubClient(repository, os.environ["GITHUB_TOKEN"])
    pull_request = client.get_pull_request(number)
    pr_issue = client.get_issue(number)
    results = audit_pull_request(pull_request, pr_issue, client.get_files(number), config, client)
    upsert_audit_comment(client, number, str(pull_request.get("user", {}).get("login") or "contributor"), results)
    set_audit_labels(client, number, results, config)
    write_summary(results)
    return 1 if config.get("enforcement", False) and failed_results(results, config) else 0


def parse_timestamp(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_reminders(config: dict[str, Any]) -> int:
    client = GitHubClient(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
    labels = config.get("labels", {})
    pending = str(labels.get("pending", "pr-audit-pending"))
    reminded = str(labels.get("reminded", "pr-audit-reminded"))
    escalated = str(labels.get("escalated", "pr-audit-escalated"))
    now = dt.datetime.now(dt.timezone.utc)
    for pull_request in client.get_pages("/repos/%s/pulls?state=open" % client.repository):
        label_names = {str(label.get("name")) for label in pull_request.get("labels", []) if isinstance(label, dict)}
        if pending not in label_names:
            continue
        number = int(pull_request["number"])
        existing = find_audit_comment(client.get_comments(number))
        first_failed = comment_first_failed_at(existing)
        if not first_failed:
            continue
        age_hours = (now - parse_timestamp(first_failed)).total_seconds() / 3600
        author = str(pull_request.get("user", {}).get("login") or "contributor")
        if age_hours >= float(config.get("escalate_after_hours", 72)) and escalated not in label_names:
            mention = str(config.get("maintainer_mention") or "").strip()
            client.create_comment(number, "@%s PR 审计问题已超过 %.0f 小时未整改。%s" % (
                author, age_hours, mention,
            ))
            client.add_label(number, escalated)
        elif age_hours >= float(config.get("reminder_after_hours", 24)) and reminded not in label_names:
            client.create_comment(number, "@%s 此 PR 仍有待整改的审计项，请完成修复后推送提交以自动复检。" % author)
            client.add_label(number, reminded)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--event-path")
    parser.add_argument("--remind", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.remind:
        return run_reminders(config)
    if not arguments.event_path:
        parser.error("--event-path is required unless --remind is used")
    return run_audit(config, arguments.event_path)


if __name__ == "__main__":
    raise SystemExit(main())
