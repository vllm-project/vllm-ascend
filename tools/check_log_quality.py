#!/usr/bin/env python3
"""Pre-commit hook: basic log quality checks for vllm_ascend/ Python code."""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import regex as re

LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical"})
WARN_METHODS = frozenset({"warning", "error", "exception", "critical"})
LOGGER_OBJECT_NAMES = frozenset({"logger", "_logger", "logging", "log", "_log"})
RULES_PATH = Path(__file__).with_name("log_quality_rules.toml")

_GIT_EXE = shutil.which("git")

DEFAULT_RULES = {
    "path_prefixes": ["vllm_ascend/"],
    "privacy_identifiers": [
        "messages",
        "password",
        "passwd",
        "api_key",
        "access_token",
        "secret",
        "secret_key",
        "private_key",
        "certificate",
        "token",
    ],
    "privacy_allowlist_names": [
        "num_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "max_tokens",
        "prompt_hash",
        "token_ids",
        "token_id",
    ],
    "vague_exact_messages": ["failed", "error", "timeout", "link failed", "connect failed"],
    "min_message_length": 15,
    "info_failure_keywords": ["fail", "failed", "error", "exception", "timeout", "unable"],
}


@dataclass
class Issue:
    path: Path
    line: int
    level: str  # "error" | "warning"
    rule: str
    message: str


@dataclass
class Rules:
    path_prefixes: list[str] = field(default_factory=lambda: list(DEFAULT_RULES["path_prefixes"]))
    privacy_identifiers: list[str] = field(default_factory=lambda: list(DEFAULT_RULES["privacy_identifiers"]))
    privacy_allowlist_names: list[str] = field(default_factory=lambda: list(DEFAULT_RULES["privacy_allowlist_names"]))
    vague_exact_messages: list[str] = field(default_factory=lambda: list(DEFAULT_RULES["vague_exact_messages"]))
    min_message_length: int = DEFAULT_RULES["min_message_length"]
    info_failure_keywords: list[str] = field(default_factory=lambda: list(DEFAULT_RULES["info_failure_keywords"]))


def _import_toml_module():
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:
        try:
            import tomli

            return tomli
        except ModuleNotFoundError:
            return None


def load_rules(path: Path) -> Rules:
    rules = Rules()
    if not path.is_file():
        return rules

    toml = _import_toml_module()
    if toml is None:
        print(
            "check_log_quality: warning: TOML parser (tomllib or tomli) not found. "
            f"Using default rules instead of loading from {path}.",
            file=sys.stderr,
        )
        return rules

    data = toml.loads(path.read_text(encoding="utf-8"))
    scope = data.get("scope", {})
    hard_fail = data.get("hard_fail", {})
    privacy = hard_fail.get("privacy", {})
    vague = hard_fail.get("vague", {})
    soft_warn = data.get("soft_warn", {})

    if scope.get("path_prefixes"):
        rules.path_prefixes = list(scope["path_prefixes"])
    if privacy.get("identifiers"):
        rules.privacy_identifiers = [str(x).lower() for x in privacy["identifiers"]]
    if privacy.get("allowlist_names"):
        rules.privacy_allowlist_names = [str(x).lower() for x in privacy["allowlist_names"]]
    if vague.get("exact_messages"):
        rules.vague_exact_messages = [str(x).lower() for x in vague["exact_messages"]]
    if vague.get("min_message_length") is not None:
        rules.min_message_length = int(vague["min_message_length"])
    if soft_warn.get("info_failure_keywords"):
        rules.info_failure_keywords = [str(x).lower() for x in soft_warn["info_failure_keywords"]]

    return rules


def repo_relative(path: Path) -> str:
    cwd = Path.cwd()
    try:
        return str(path.resolve().relative_to(cwd.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def in_scope(rel_path: str, prefixes: Iterable[str]) -> bool:
    return any(rel_path.startswith(prefix) for prefix in prefixes)


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _require_git() -> str:
    if _GIT_EXE is None:
        print(
            "check_log_quality: git not found in PATH, cannot run incremental check",
            file=sys.stderr,
        )
        sys.exit(1)
    return _GIT_EXE


def parse_staged_changed_lines(rel_path: str) -> set[int] | None:
    git_exe = _require_git()
    result = _run_git([git_exe, "diff", "--cached", "-U0", "--", rel_path])
    stdout = result.stdout or ""
    if result.returncode != 0:
        return None

    changed: set[int] = set()
    for line in stdout.splitlines():
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if not match:
                continue
            start = int(match.group(1))
            count = int(match.group(2) or "1")
            if count == 0:
                continue
            changed.update(range(start, start + count))
    return changed


def node_intersects_changed(node: ast.AST, changed_lines: set[int] | None) -> bool:
    if changed_lines is None:
        return True
    if not changed_lines:
        return False
    start = getattr(node, "lineno", None)
    if start is None:
        return False
    end = getattr(node, "end_lineno", start) or start
    return any(line in changed_lines for line in range(start, end + 1))


def _is_logger_object(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in LOGGER_OBJECT_NAMES
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr in ("logger", "_logger", "log", "_log")
        )
    return False


def is_logger_call(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in LOG_METHODS:
        if _is_logger_object(func.value):
            return func.attr
    return None


def iter_string_parts(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
        return parts
    return []


def message_text(node: ast.AST) -> str:
    return "".join(iter_string_parts(node)).strip()


def is_allowlisted_name(name: str, allowlist: set[str]) -> bool:
    lowered = name.lower()
    return lowered in allowlist


def message_has_sensitive_field(part: str, identifier: str) -> bool:
    if identifier == "token":
        return bool(
            re.search(r"(?<![a-z0-9_])token\s*=", part, re.IGNORECASE)
            or re.search(r"\{(?<![a-z0-9_])token\}", part, re.IGNORECASE)
        )
    return bool(
        re.search(rf"\b{re.escape(identifier)}\s*=", part, re.IGNORECASE)
        or re.search(rf"\{{{re.escape(identifier)}\}}", part, re.IGNORECASE)
    )


def direct_sensitive_arguments(
    node: ast.Call,
    privacy: set[str],
    allowlist: set[str],
) -> str | None:
    for arg in node.args:
        if isinstance(arg, ast.Name):
            if is_allowlisted_name(arg.id, allowlist):
                continue
            if arg.id.lower() in privacy:
                return arg.id.lower()
        if isinstance(arg, ast.FormattedValue):
            if isinstance(arg.value, ast.Name):
                if is_allowlisted_name(arg.value.id, allowlist):
                    continue
                if arg.value.id.lower() in privacy:
                    return arg.value.id.lower()
    for keyword in node.keywords:
        if keyword.arg and is_allowlisted_name(keyword.arg, allowlist):
            continue
        if keyword.arg and keyword.arg.lower() in privacy:
            return keyword.arg.lower()
        if isinstance(keyword.value, ast.Name):
            if is_allowlisted_name(keyword.value.id, allowlist):
                continue
            if keyword.value.id.lower() in privacy:
                return keyword.value.id.lower()
    return None


def has_diagnostic_carrier(node: ast.Call) -> bool:
    if any(keyword.arg == "exc_info" for keyword in node.keywords):
        return True
    if len(node.args) > 1:
        return True
    if node.args:
        arg = node.args[0]
        if isinstance(arg, ast.JoinedStr):
            if any(isinstance(val, ast.FormattedValue) for val in arg.values):
                return True
        text = message_text(arg)
        if "%" in text or "{" in text:
            return True
    return False


class LogQualityChecker(ast.NodeVisitor):
    def __init__(self, path: Path, rel_path: str, rules: Rules, changed_lines: set[int] | None) -> None:
        self.path = path
        self.rel_path = rel_path
        self.rules = rules
        self.changed_lines = changed_lines
        self.issues: list[Issue] = []

    def _add(
        self,
        node: ast.AST,
        level: str,
        rule: str,
        message: str,
    ) -> None:
        if not node_intersects_changed(node, self.changed_lines):
            return
        line = getattr(node, "lineno", 1) or 1
        self.issues.append(Issue(self.path, line, level, rule, message))

    def visit_Call(self, node: ast.Call) -> None:
        method = is_logger_call(node)
        if method is None:
            self.generic_visit(node)
            return

        self._check_privacy(node)
        if method in WARN_METHODS:
            self._check_vague_message(node, method)
        if method == "info":
            self._check_info_failure_keywords(node)

        self.generic_visit(node)

    def _check_privacy(self, node: ast.Call) -> None:
        privacy = set(self.rules.privacy_identifiers)
        allowlist = set(self.rules.privacy_allowlist_names)
        for part in iter_string_parts(node.args[0]) if node.args else []:
            for identifier in privacy:
                if message_has_sensitive_field(part, identifier):
                    self._add(
                        node,
                        "error",
                        "LQ001-privacy-in-message",
                        "log message appears to log a sensitive field value",
                    )
                    return

        sensitive_arg = direct_sensitive_arguments(node, privacy, allowlist)
        if sensitive_arg:
            self._add(
                node,
                "error",
                "LQ001-privacy-in-argument",
                f"log call must not pass sensitive identifier '{sensitive_arg}'",
            )

    def _check_vague_message(self, node: ast.Call, method: str) -> None:
        if not node.args:
            self._add(
                node,
                "error",
                "LQ004-empty-log-message",
                f"logger.{method}() has no message",
            )
            return

        arg = node.args[0]
        if not isinstance(arg, (ast.Constant, ast.JoinedStr)):
            return

        text = message_text(arg)
        if not text:
            self._add(
                node,
                "error",
                "LQ004-empty-log-message",
                f"logger.{method}() message is empty",
            )
            return

        normalized = text.lower()
        if normalized in self.rules.vague_exact_messages:
            self._add(
                node,
                "error",
                "LQ003-vague-log-message",
                f"logger.{method}() message is too vague: {text!r}",
            )
            return

        if len(text) < self.rules.min_message_length and not has_diagnostic_carrier(node):
            self._add(
                node,
                "error",
                "LQ004-short-log-message",
                f"logger.{method}() message is too short ({len(text)} chars); describe what failed and key parameters",
            )

    def _check_info_failure_keywords(self, node: ast.Call) -> None:
        if not node.args:
            return
        text = message_text(node.args[0]).lower()
        if not text:
            return
        for keyword in self.rules.info_failure_keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                self._add(
                    node,
                    "warning",
                    "info-looks-like-error",
                    f"logger.info() message contains failure keyword '{keyword}'; consider logger.warning/error",
                )
                return


def check_file(path: Path, rules: Rules, incremental: bool) -> list[Issue]:
    rel_path = repo_relative(path)
    if not in_scope(rel_path, rules.path_prefixes):
        return []

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        return [
            Issue(
                path,
                exc.lineno or 1,
                "error",
                "syntax-error",
                f"cannot parse file for log checks: {exc.msg}",
            )
        ]

    changed_lines = parse_staged_changed_lines(rel_path) if incremental else None
    if incremental and changed_lines is not None and not changed_lines:
        status = _run_git([_require_git(), "diff", "--cached", "--name-status", "--", rel_path])
        status_out = status.stdout or ""
        if status_out.startswith("A\t") or status_out.startswith("A "):
            changed_lines = None  # new file: check entire file

    checker = LogQualityChecker(path, rel_path, rules, changed_lines)
    checker.visit(tree)
    return checker.issues


def format_issue(issue: Issue) -> str:
    rel = repo_relative(issue.path)
    return f"{rel}:{issue.line}: [{issue.level}] {issue.rule}: {issue.message}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Basic log quality checks for vllm_ascend/ code.")
    parser.add_argument("files", nargs="*", help="Files to check (from pre-commit).")
    parser.add_argument(
        "--incremental",
        action="store_true",
        default="PRE_COMMIT" in __import__("os").environ,
        help="Only report issues on staged changed lines (default under pre-commit).",
    )
    parser.add_argument(
        "--all-lines",
        action="store_true",
        help="Check entire file contents (disables incremental line filtering).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (for CI).",
    )
    args = parser.parse_args(argv)

    if not args.files:
        return 0

    rules = load_rules(RULES_PATH)
    incremental = args.incremental and not args.all_lines

    errors: list[Issue] = []
    warnings: list[Issue] = []
    for file_arg in args.files:
        path = Path(file_arg)
        if not path.is_file() or path.suffix != ".py":
            continue
        for issue in check_file(path, rules, incremental=incremental):
            if issue.level == "error":
                errors.append(issue)
            else:
                warnings.append(issue)

    for issue in errors:
        print(format_issue(issue))
    for issue in warnings:
        print(format_issue(issue))

    if errors:
        return 1
    if args.strict and warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
