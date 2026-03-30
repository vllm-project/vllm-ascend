#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import argparse
import os
import shlex
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re
import tomllib
import yaml

SYNC_FIELD_NAMES = ("sync-yaml", "sync-target", "sync-class")
SUPPORTED_SYNC_CLASSES = {"env", "cmd"}
EXTRACT_BLOCK_OPEN_RE = re.compile(r"^```{test}\s+\S+\s*$")
SYNC_OPTION_RE = re.compile(r"^:([A-Za-z0-9_-]+):\s*(.*)\s*$")
EXPORT_ENV_LINE_RE = re.compile(r'^export\s+([A-Za-z_][A-Za-z0-9_]*)=(?|"([^"]*)"|\'([^\']*)\'|(.*?))(?:\s+#.*)?$')


class LintFailure(RuntimeError):
    """Raised when a sync block cannot be linted successfully."""

    def __init__(self, message: str, *, line: int | None = None) -> None:
        super().__init__(message)
        self.line = line


@dataclass(frozen=True)
class SyncBlock:
    doc_path: Path
    start_line: int
    content: str
    sync_yaml: str
    sync_target: str
    sync_class: str


@dataclass(frozen=True)
class Diagnostic:
    doc_path: Path
    line: int
    sync_yaml: str
    sync_target: str
    message: str

    def format(self, *, color: bool = False) -> str:
        detail_label = style_text("detail:", "1;33", enabled=color)
        lines = [
            f"{self.doc_path}:{self.line}: yaml sync lint error",
            f"  yaml: {self.sync_yaml}",
            f"  target: {self.sync_target}",
        ]
        details = self.message.splitlines()
        if len(details) <= 1:
            lines.append(f"  {detail_label} {style_text(self.message, '1;31', enabled=color)}")
        else:
            lines.append(f"  {detail_label}")
            lines.extend(f"    {style_text(detail, '1;31', enabled=color)}" for detail in details)
        return "\n".join(lines)


def should_use_color(stream: Any) -> bool:
    if os.getenv("NO_COLOR") is not None:
        return False

    pre_commit_color = os.getenv("PRE_COMMIT_COLOR")
    if pre_commit_color == "always":
        return True
    if pre_commit_color == "never":
        return False

    if os.getenv("FORCE_COLOR") not in {None, "", "0"}:
        return True
    if os.getenv("CLICOLOR_FORCE") not in {None, "", "0"}:
        return True
    if os.getenv("CLICOLOR") == "0":
        return False

    return bool(getattr(stream, "isatty", lambda: False)())


def style_text(text: str, ansi_code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{ansi_code}m{text}\033[0m"


def make_diagnostic(
    doc_path: Path,
    line: int,
    message: str,
    *,
    sync_yaml: str = "-",
    sync_target: str = "-",
) -> Diagnostic:
    return Diagnostic(
        doc_path=doc_path,
        line=line,
        sync_yaml=sync_yaml,
        sync_target=sync_target,
        message=message,
    )


def merge_diagnostics_by_block(diagnostics: Iterable[Diagnostic]) -> list[Diagnostic]:
    grouped: dict[tuple[Path, int, str, str], list[str]] = {}
    for diagnostic in diagnostics:
        key = (
            diagnostic.doc_path,
            diagnostic.line,
            diagnostic.sync_yaml,
            diagnostic.sync_target,
        )
        grouped.setdefault(key, []).append(diagnostic.message)

    return [
        Diagnostic(
            doc_path=doc_path,
            line=line,
            sync_yaml=sync_yaml,
            sync_target=sync_target,
            message="\n".join(messages),
        )
        for (doc_path, line, sync_yaml, sync_target), messages in grouped.items()
    ]


def load_exclude_patterns(repo_root: Path) -> set[str]:
    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.exists():
        return set()

    config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    patterns = config.get("tool", {}).get("check_docs_yaml_sync", {}).get("exclude", [])
    if not isinstance(patterns, list) or not all(isinstance(pattern, str) for pattern in patterns):
        raise LintFailure("[tool.check_docs_yaml_sync].exclude must be a list of strings")
    return set(patterns)


def is_excluded_doc(doc_path: Path, repo_root: Path, exclude_patterns: set[str]) -> bool:
    try:
        relative_path = doc_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return False
    return relative_path in exclude_patterns


def parse_sync_blocks(doc_path: Path) -> tuple[list[SyncBlock], list[Diagnostic]]:
    lines = doc_path.read_text(encoding="utf-8").splitlines()
    blocks: list[SyncBlock] = []
    diagnostics: list[Diagnostic] = []
    found_test_block = False
    i = 0

    while i < len(lines):
        if not EXTRACT_BLOCK_OPEN_RE.match(lines[i]):
            i += 1
            continue
        found_test_block = True

        start_line = i + 1
        i += 1
        options: dict[str, str] = {}
        content_lines: list[str] = []
        script_code = False

        while i < len(lines):
            line = lines[i]
            if line == "```":
                break
            if not script_code:
                option_match = SYNC_OPTION_RE.match(line)
                if option_match:
                    options[option_match.group(1)] = option_match.group(2)
                    i += 1
                    continue
                if line.strip() == "":
                    i += 1
                    script_code = True
                    continue
                script_code = True
            content_lines.append(line)
            i += 1

        if i >= len(lines) or lines[i] != "```":
            diagnostics.append(make_diagnostic(doc_path, start_line, "unclosed MyST code-block"))
            break

        missing = [key for key in SYNC_FIELD_NAMES if key not in options]
        if missing:
            diagnostics.append(
                make_diagnostic(
                    doc_path,
                    start_line,
                    f"sync test block missing required metadata: {', '.join(missing)}",
                    sync_yaml=options.get("sync-yaml", "-"),
                    sync_target=options.get("sync-target", "-"),
                )
            )
        else:
            blocks.append(
                SyncBlock(
                    doc_path=doc_path,
                    start_line=start_line,
                    content="\n".join(content_lines).strip(),
                    sync_yaml=options["sync-yaml"],
                    sync_target=options["sync-target"],
                    sync_class=options["sync-class"],
                )
            )

        i += 1

    if not found_test_block:
        message = (
            "Markdown files should link model test cases. "
            "If not, please add the file to the exclude list in pyproject.toml::[tool.check_docs_yaml_sync]."
        )
        diagnostics.append(make_diagnostic(doc_path, 1, message))

    return blocks, diagnostics


def parse_sync_targets(sync_target: str) -> list[str]:
    targets = sync_target.split()
    if not targets:
        raise LintFailure("sync-target is empty")
    return targets


def parse_target_segments(sync_target: str) -> list[str | int]:
    if not sync_target:
        raise LintFailure("sync-target is empty")

    segments: list[str | int] = []
    token: list[str] = []
    bracket: list[str] = []
    in_bracket = False
    quote_char = ""

    for char in sync_target:
        if in_bracket:
            if quote_char:
                if char == quote_char:
                    quote_char = ""
                else:
                    bracket.append(char)
                continue
            if char in {"'", '"'}:
                quote_char = char
                continue
            if char == "]":
                text = "".join(bracket).strip()
                if not text:
                    raise LintFailure(f"sync-target '{sync_target}' contains an empty bracket accessor")
                segments.append(int(text) if text.isdigit() else text)
                bracket.clear()
                in_bracket = False
                continue
            bracket.append(char)
            continue

        if char == ".":
            if token:
                segments.append("".join(token))
                token.clear()
            continue
        if char == "[":
            if token:
                segments.append("".join(token))
                token.clear()
            in_bracket = True
            continue
        token.append(char)

    if quote_char or in_bracket:
        raise LintFailure(f"sync-target '{sync_target}' has unclosed brackets or quotes")
    if token:
        segments.append("".join(token))
    if not segments:
        raise LintFailure(f"sync-target '{sync_target}' is empty")
    return segments


def resolve_yaml_target(root: Any, sync_target: str) -> Any:
    current = root
    segments = parse_target_segments(sync_target)

    for segment in segments:
        if isinstance(segment, int):
            if not isinstance(current, list):
                raise LintFailure(
                    f"sync-target '{sync_target}' expected list before index [{segment}], got {type(current).__name__}"
                )
            if segment >= len(current):
                raise LintFailure(f"sync-target '{sync_target}' index [{segment}] out of range")
            current = current[segment]
            continue

        if not isinstance(current, dict):
            raise LintFailure(
                f"sync-target '{sync_target}' expected mapping before '{segment}', got {type(current).__name__}"
            )
        if segment not in current:
            raise LintFailure(f"sync-target '{sync_target}' missing key '{segment}'")
        current = current[segment]

    return current


def extract_doc_env_entries(content: str) -> list[str]:
    entries: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        match = EXPORT_ENV_LINE_RE.match(line)
        if not match:
            raise LintFailure(f"env block contains invalid export line: {line}")
        value = str(match.group(2)).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        entries.append(f"{match.group(1)}={value}")
    if not entries:
        raise LintFailure("env block is empty")
    return entries


def extract_yaml_env_entries(resolved: Any, sync_target: str) -> list[str]:
    if not isinstance(resolved, dict):
        raise LintFailure(
            f"sync-target '{sync_target}' must resolve to a mapping for env compare, got {type(resolved).__name__}"
        )
    entries: list[str] = []
    for key, value in resolved.items():
        env_value = str(value).strip()
        if len(env_value) >= 2 and env_value[0] == env_value[-1] and env_value[0] in {"'", '"'}:
            env_value = env_value[1:-1]
        entries.append(f"{key}={env_value}")
    return entries


def strip_comment_lines(content: str) -> str:
    return "\n".join(line for line in content.splitlines() if not line.strip().startswith("#"))


def parse_vllm_command(content: str, *, source: str) -> list[str]:
    try:
        tokens = shlex.split(strip_comment_lines(content).replace("\\\n", " "), posix=True)
    except ValueError as exc:
        raise LintFailure(f"failed to parse {source}: {exc}") from exc

    if len(tokens) < 3 or tokens[0] != "vllm" or tokens[1] != "serve":
        raise LintFailure(f"{source} must be a valid 'vllm serve' command")

    model = tokens[2]
    if model.startswith("--"):
        raise LintFailure(f"{source} must provide model as the third token in 'vllm serve'")

    entries = [f"model={' '.join(str(model).split())}"]
    seen_options: set[str] = set()
    i = 3

    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            if token in seen_options:
                raise LintFailure(f"{source} contains duplicated command parameter: {token}")
            seen_options.add(token)
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                entries.append(f"{token} {' '.join(str(tokens[i + 1]).split())}")
                i += 2
                continue
            entries.append(token)
            i += 1
            continue
        raise LintFailure(f"{source} contains unsupported extra positional argument: {token}")

    return entries


def extract_command_fragment(value: Any, sync_target: str) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value, posix=True)
    if isinstance(value, list) and all(not isinstance(item, (dict, list)) for item in value):
        return [str(item) for item in value]
    raise LintFailure(f"sync-target '{sync_target}' must resolve to a string or scalar list for cmd compare")


def extract_yaml_command(root: Any, sync_target: str) -> list[str]:
    targets = parse_sync_targets(sync_target)
    if len(targets) == 1:
        value = resolve_yaml_target(root, targets[0])
        return parse_vllm_command(
            shlex.join(extract_command_fragment(value, targets[0])),
            source=f"yaml target '{sync_target}'",
        )

    tokens = ["vllm", "serve"]
    for target in targets:
        tokens.extend(extract_command_fragment(resolve_yaml_target(root, target), target))

    return parse_vllm_command(
        shlex.join(tokens),
        source=f"yaml targets '{sync_target}'",
    )


def compare_entries(doc_entries: list[str], yaml_entries: list[str], *, label: str) -> list[str]:
    doc_counter = Counter(doc_entries)
    yaml_counter = Counter(yaml_entries)
    missing = sorted((yaml_counter - doc_counter).elements())
    extra = sorted((doc_counter - yaml_counter).elements())
    differences: list[str] = []
    if missing:
        missing_lines = "\n".join(missing)
        differences.append(f"Code block in doc has missing {label}:\n{missing_lines}")
    if extra:
        extra_lines = "\n".join(extra)
        differences.append(f"Code block in doc has extra {label}:\n{extra_lines}")
    return differences


def lint_block(block: SyncBlock, repo_root: Path) -> list[Diagnostic]:
    if block.sync_class not in SUPPORTED_SYNC_CLASSES:
        raise LintFailure(
            f"unsupported sync-class '{block.sync_class}', expected one of {sorted(SUPPORTED_SYNC_CLASSES)}"
        )

    yaml_path = repo_root / block.sync_yaml
    if not yaml_path.exists():
        return [
            make_diagnostic(
                block.doc_path,
                block.start_line,
                "referenced YAML file does not exist",
                sync_yaml=block.sync_yaml,
                sync_target=block.sync_target,
            )
        ]

    yaml_root = yaml.load(yaml_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    if block.sync_class == "env":
        targets = parse_sync_targets(block.sync_target)
        if len(targets) != 1:
            raise LintFailure("env sync-class expects exactly one sync-target")
        resolved = resolve_yaml_target(yaml_root, targets[0])
        differences = compare_entries(
            extract_doc_env_entries(block.content),
            extract_yaml_env_entries(resolved, targets[0]),
            label="env parameters",
        )
    elif block.sync_class == "cmd":
        differences = compare_entries(
            parse_vllm_command(block.content, source="document"),
            extract_yaml_command(yaml_root, block.sync_target),
            label="command parameters",
        )

    return [
        make_diagnostic(
            block.doc_path,
            block.start_line,
            message,
            sync_yaml=block.sync_yaml,
            sync_target=block.sync_target,
        )
        for message in differences
    ]


def lint_documents(
    doc_paths: Iterable[Path],
    repo_root: Path,
    *,
    exclude_patterns: set[str] | None = None,
) -> list[Diagnostic]:
    exclude_patterns = exclude_patterns or set()
    diagnostics: list[Diagnostic] = []
    for doc_path in doc_paths:
        if is_excluded_doc(doc_path, repo_root, exclude_patterns):
            continue

        blocks, block_diagnostics = parse_sync_blocks(doc_path)
        diagnostics.extend(block_diagnostics)

        for block in blocks:
            try:
                diagnostics.extend(lint_block(block, repo_root))
            except LintFailure as exc:
                diagnostics.append(
                    make_diagnostic(
                        block.doc_path,
                        exc.line or block.start_line,
                        str(exc),
                        sync_yaml=block.sync_yaml,
                        sync_target=block.sync_target,
                    )
                )
    return diagnostics


def resolve_input_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path.cwd().resolve() / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint sync blocks between docs and YAML.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Markdown files to lint.",
    )
    args = parser.parse_args(argv)

    repo_root = Path.cwd().resolve()
    doc_paths = [resolve_input_path(path) for path in args.paths]
    for doc_path in doc_paths:
        if doc_path.is_dir():
            parser.error(f"path must be a markdown file, got directory: {doc_path}")
        if doc_path.suffix != ".md":
            parser.error(f"path must end with .md: {doc_path}")

    diagnostics = lint_documents(doc_paths, repo_root, exclude_patterns=load_exclude_patterns(repo_root))
    if diagnostics:
        use_color = should_use_color(sys.stderr)
        for diagnostic in merge_diagnostics_by_block(diagnostics):
            print(diagnostic.format(color=use_color), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
