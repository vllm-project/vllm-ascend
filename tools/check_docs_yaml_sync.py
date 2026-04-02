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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import regex as re
import yaml

try:
    import tomllib
except ImportError:
    import tomli as tomllib

SYNC_FIELD_NAMES = ("sync-yaml", "sync-target", "sync-class")
EXTRACT_BLOCK_OPEN_RE = re.compile(r"^```{test}\s+\S+\s*$")
SYNC_OPTION_RE = re.compile(r"^:([A-Za-z0-9_-]+):\s*(.*)\s*$")
EXPORT_ENV_LINE_RE = re.compile(r'^export\s+([A-Za-z_][A-Za-z0-9_]*)=(?|"([^"]*)"|\'([^\']*)\'|(.*?))(?:\s+#.*)?$')


# ============================================================================
# Core Data Types
# ============================================================================


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


@dataclass(frozen=True)
class ExtractionResult:
    blocks: tuple[SyncBlock, ...]
    diagnostics: tuple[Diagnostic, ...]


@dataclass(frozen=True)
class EnvAssignment:
    name: str
    value: str

    def as_entry(self) -> str:
        return f"{self.name}={self.value}"


@dataclass(frozen=True)
class EnvCompareData:
    assignments: tuple[EnvAssignment, ...]

    def entries(self) -> list[str]:
        return [assignment.as_entry() for assignment in self.assignments]


@dataclass(frozen=True)
class CommandParameter:
    name: str
    value: str | None = None

    def as_entry(self) -> str:
        if self.value is None:
            return self.name
        return f"{self.name} {self.value}"


@dataclass(frozen=True)
class CmdCompareData:
    model: str
    parameters: tuple[CommandParameter, ...]

    def entries(self) -> list[str]:
        return [f"model={self.model}", *[parameter.as_entry() for parameter in self.parameters]]


# ============================================================================
# Diagnostics and Reporting
# ============================================================================


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


# ============================================================================
# Markdown Block Extraction
# ============================================================================


class MarkdownBlockExtractor:
    """Module A: only extract sync blocks from markdown."""

    def extract(self, doc_path: Path) -> ExtractionResult:
        lines = doc_path.read_text(encoding="utf-8").splitlines()
        blocks: list[SyncBlock] = []
        diagnostics: list[Diagnostic] = []
        found_test_block = False
        line_index = 0

        while line_index < len(lines):
            if not EXTRACT_BLOCK_OPEN_RE.match(lines[line_index]):
                line_index += 1
                continue

            found_test_block = True
            start_line = line_index + 1
            line_index += 1
            options: dict[str, str] = {}
            content_lines: list[str] = []
            script_code_started = False

            while line_index < len(lines):
                line = lines[line_index]
                if line == "```":
                    break
                if not script_code_started:
                    option_match = SYNC_OPTION_RE.match(line)
                    if option_match:
                        options[option_match.group(1)] = option_match.group(2)
                        line_index += 1
                        continue
                    if line.strip() == "":
                        line_index += 1
                        script_code_started = True
                        continue
                    script_code_started = True
                content_lines.append(line)
                line_index += 1

            if line_index >= len(lines) or lines[line_index] != "```":
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

            line_index += 1

        if not found_test_block:
            diagnostics.append(
                make_diagnostic(
                    doc_path,
                    1,
                    "Markdown files should link model test cases. "
                    "If not, please add the file to the exclude list in pyproject.toml::"
                    "[tool.check_docs_yaml_sync].",
                )
            )

        return ExtractionResult(tuple(blocks), tuple(diagnostics))


# ============================================================================
# Loading YAML and Locating configuration
# ============================================================================


class YamlDocumentLoader:
    """Shared helper for YAML-backed converters."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self._cache: dict[Path, Any] = {}

    def path_for(self, relative_path: str) -> Path:
        return self.repo_root / relative_path

    def load(self, relative_path: str) -> Any:
        yaml_path = self.path_for(relative_path)
        if not yaml_path.exists():
            raise LintFailure("referenced YAML file does not exist")
        if yaml_path not in self._cache:
            self._cache[yaml_path] = yaml.load(yaml_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        return self._cache[yaml_path]


class YamlTargetResolver:
    """Shared helper for locating sync-target values inside YAML."""

    def parse_targets(self, sync_target: str) -> list[str]:
        targets = sync_target.split()
        if not targets:
            raise LintFailure("sync-target is empty")
        return targets

    def parse_segments(self, target_path: str) -> list[str | int]:
        if not target_path:
            raise LintFailure("sync-target is empty")

        segments: list[str | int] = []
        token: list[str] = []
        bracket: list[str] = []
        in_bracket = False
        quote_char = ""

        for char in target_path:
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
                        raise LintFailure(f"sync-target '{target_path}' contains an empty bracket accessor")
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
            raise LintFailure(f"sync-target '{target_path}' has unclosed brackets or quotes")
        if token:
            segments.append("".join(token))
        if not segments:
            raise LintFailure(f"sync-target '{target_path}' is empty")
        return segments

    def resolve(self, root: Any, target_path: str) -> Any:
        current = root
        for segment in self.parse_segments(target_path):
            if isinstance(segment, int):
                if not isinstance(current, list):
                    raise LintFailure(
                        f"sync-target '{target_path}' expected list before index [{segment}], "
                        f"got {type(current).__name__}"
                    )
                if segment >= len(current):
                    raise LintFailure(f"sync-target '{target_path}' index [{segment}] out of range")
                current = current[segment]
                continue

            if not isinstance(current, dict):
                raise LintFailure(
                    f"sync-target '{target_path}' expected mapping before '{segment}', got {type(current).__name__}"
                )
            if segment not in current:
                raise LintFailure(f"sync-target '{target_path}' missing key '{segment}'")
            current = current[segment]

        return current


# ============================================================================
# Module B/C: Env Conversion
# ============================================================================


class DocEnvConverter:
    """Module B: convert env exports in docs into compare data."""

    @staticmethod
    def _normalize_env_value(value: Any) -> str:
        normalized = str(value).strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
            return normalized[1:-1]
        return normalized

    def convert(self, block: SyncBlock) -> EnvCompareData:
        assignments: list[EnvAssignment] = []
        for raw_line in block.content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or not line.startswith("export "):
                continue
            match = EXPORT_ENV_LINE_RE.match(line)
            if not match:
                raise LintFailure(f"env block contains invalid export line: {line}")
            assignments.append(
                EnvAssignment(
                    name=match.group(1),
                    value=self._normalize_env_value(match.group(2)),
                )
            )

        if not assignments:
            raise LintFailure("env block is empty")

        return EnvCompareData(tuple(assignments))


class YamlEnvConverter:
    """Module C: convert YAML env mappings into compare data."""

    def __init__(self, loader: YamlDocumentLoader, resolver: YamlTargetResolver) -> None:
        self.loader = loader
        self.resolver = resolver

    def convert(self, block: SyncBlock) -> EnvCompareData:
        target_paths = self.resolver.parse_targets(block.sync_target)
        if len(target_paths) != 1:
            raise LintFailure("env sync-class expects exactly one sync-target")

        yaml_root = self.loader.load(block.sync_yaml)
        resolved = self.resolver.resolve(yaml_root, target_paths[0])
        if not isinstance(resolved, dict):
            raise LintFailure(
                f"sync-target '{target_paths[0]}' must resolve to a mapping for env compare, "
                f"got {type(resolved).__name__}"
            )

        assignments: list[EnvAssignment] = []
        for key, value in resolved.items():
            assignments.append(
                EnvAssignment(
                    name=str(key),
                    value=DocEnvConverter._normalize_env_value(value),
                )
            )
        return EnvCompareData(tuple(assignments))


# ============================================================================
# Module D/E: Command Conversion
# ============================================================================


class VllmServeCommandParser:
    """Shared parser for both doc and YAML command converters."""

    def parse_text(self, content: str, *, source: str) -> CmdCompareData:
        try:
            content_without_comment_lines = "\n".join(
                line for line in content.splitlines() if not line.strip().startswith("#")
            )
            tokens = shlex.split(content_without_comment_lines.replace("\\\n", " "), posix=True)
        except ValueError as exc:
            raise LintFailure(f"failed to parse {source}: {exc}") from exc
        return self.parse_tokens(tokens, source=source)

    def parse_tokens(self, tokens: list[str], *, source: str) -> CmdCompareData:
        if len(tokens) < 3 or tokens[0] != "vllm" or tokens[1] != "serve":
            raise LintFailure(f"{source} must be a valid 'vllm serve' command")

        model = tokens[2]
        if model.startswith("--"):
            raise LintFailure(f"{source} must provide model as the third token in 'vllm serve'")

        parameters: list[CommandParameter] = []
        seen_parameter_names: set[str] = set()
        token_index = 3

        while token_index < len(tokens):
            token = tokens[token_index]
            if not token.startswith("--"):
                raise LintFailure(f"{source} contains unsupported extra positional argument: {token}")
            if token in seen_parameter_names:
                raise LintFailure(f"{source} contains duplicated command parameter: {token}")

            seen_parameter_names.add(token)
            if token_index + 1 < len(tokens) and not tokens[token_index + 1].startswith("--"):
                parameters.append(CommandParameter(token, " ".join(str(tokens[token_index + 1]).split())))
                token_index += 2
                continue

            parameters.append(CommandParameter(token))
            token_index += 1

        return CmdCompareData(
            model=" ".join(str(model).split()),
            parameters=tuple(parameters),
        )

    def extract_fragment(self, value: Any, target_path: str) -> list[str]:
        if isinstance(value, str):
            return shlex.split(value, posix=True)
        if isinstance(value, list) and all(not isinstance(item, (dict, list)) for item in value):
            return [str(item) for item in value]
        raise LintFailure(f"sync-target '{target_path}' must resolve to a string or scalar list for cmd compare")


class DocCmdConverter:
    """Module D: convert markdown vllm serve commands into compare data."""

    def __init__(self, parser: VllmServeCommandParser) -> None:
        self.parser = parser

    def convert(self, block: SyncBlock) -> CmdCompareData:
        return self.parser.parse_text(block.content, source="document")


class YamlCmdConverter:
    """Module E: convert YAML command fragments into compare data."""

    def __init__(
        self,
        loader: YamlDocumentLoader,
        resolver: YamlTargetResolver,
        parser: VllmServeCommandParser,
    ) -> None:
        self.loader = loader
        self.resolver = resolver
        self.parser = parser

    def convert(self, block: SyncBlock) -> CmdCompareData:
        yaml_root = self.loader.load(block.sync_yaml)
        target_paths = self.resolver.parse_targets(block.sync_target)

        if len(target_paths) == 1:
            resolved = self.resolver.resolve(yaml_root, target_paths[0])
            tokens = self.parser.extract_fragment(resolved, target_paths[0])
            return self.parser.parse_tokens(tokens, source=f"yaml target '{block.sync_target}'")

        tokens = ["vllm", "serve"]
        for target_path in target_paths:
            resolved = self.resolver.resolve(yaml_root, target_path)
            tokens.extend(self.parser.extract_fragment(resolved, target_path))
        return self.parser.parse_tokens(tokens, source=f"yaml targets '{block.sync_target}'")


# ============================================================================
# Module F: Data Comparison
# ============================================================================


def compare_entry_sets(doc_entries: list[str], yaml_entries: list[str], *, label: str) -> list[str]:
    doc_counter = Counter(doc_entries)
    yaml_counter = Counter(yaml_entries)
    missing = sorted((yaml_counter - doc_counter).elements())
    extra = sorted((doc_counter - yaml_counter).elements())
    differences: list[str] = []
    if extra:
        differences.append(f"Doc has extra {label}:\n" + "\n".join(extra))
    if missing:
        differences.append(f"Yaml has extra {label}:\n" + "\n".join(missing))
    return differences


class DataComparator(Protocol):
    def compare(self, doc_data: Any, yaml_data: Any) -> list[str]:
        """Compare two standard compare-data objects."""


class EnvComparator:
    """Module F for env data."""

    label = "env parameters"

    def compare(self, doc_data: EnvCompareData, yaml_data: EnvCompareData) -> list[str]:
        return compare_entry_sets(doc_data.entries(), yaml_data.entries(), label=self.label)


class CmdComparator:
    """Module F for command data."""

    label = "command parameters"

    def compare(self, doc_data: CmdCompareData, yaml_data: CmdCompareData) -> list[str]:
        return compare_entry_sets(doc_data.entries(), yaml_data.entries(), label=self.label)


class DocDataConverter(Protocol):
    def convert(self, block: SyncBlock) -> Any:
        """Convert doc block content into standard compare data."""


class YamlDataConverter(Protocol):
    def convert(self, block: SyncBlock) -> Any:
        """Convert YAML target content into standard compare data."""


@dataclass(frozen=True)
class SyncHandler:
    doc_converter: DocDataConverter
    yaml_converter: YamlDataConverter
    comparator: DataComparator

    def lint(self, block: SyncBlock) -> list[str]:
        doc_data = self.doc_converter.convert(block)
        yaml_data = self.yaml_converter.convert(block)
        return self.comparator.compare(doc_data, yaml_data)


@dataclass
class SyncHandlerRegistry:
    handlers: dict[str, SyncHandler] = field(default_factory=dict)

    def register(self, sync_class: str, handler: SyncHandler) -> None:
        self.handlers[sync_class] = handler

    def get(self, sync_class: str) -> SyncHandler | None:
        return self.handlers.get(sync_class)

    def supported_classes(self) -> list[str]:
        return sorted(self.handlers)


# ============================================================================
# Orchestration and Scheduling
# ============================================================================


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


@dataclass
class SyncLinter:
    repo_root: Path
    exclude_patterns: set[str] = field(default_factory=set)
    extractor: MarkdownBlockExtractor = field(default_factory=MarkdownBlockExtractor)
    yaml_loader: YamlDocumentLoader = field(init=False)
    registry: SyncHandlerRegistry = field(init=False)

    def __post_init__(self) -> None:
        self.yaml_loader = YamlDocumentLoader(self.repo_root)
        self.registry = self._build_registry()

    def _build_registry(self) -> SyncHandlerRegistry:
        resolver = YamlTargetResolver()
        command_parser = VllmServeCommandParser()
        registry = SyncHandlerRegistry()
        registry.register(
            "env",
            SyncHandler(
                doc_converter=DocEnvConverter(),
                yaml_converter=YamlEnvConverter(self.yaml_loader, resolver),
                comparator=EnvComparator(),
            ),
        )
        registry.register(
            "cmd",
            SyncHandler(
                doc_converter=DocCmdConverter(command_parser),
                yaml_converter=YamlCmdConverter(self.yaml_loader, resolver, command_parser),
                comparator=CmdComparator(),
            ),
        )
        return registry

    def lint_documents(self, doc_paths: Iterable[Path]) -> list[Diagnostic]:
        diagnostics: list[Diagnostic] = []
        for doc_path in doc_paths:
            if is_excluded_doc(doc_path, self.repo_root, self.exclude_patterns):
                continue

            extraction_result = self.extractor.extract(doc_path)
            diagnostics.extend(extraction_result.diagnostics)

            for block in extraction_result.blocks:
                try:
                    diagnostics.extend(self.lint_block(block))
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

    def lint_block(self, block: SyncBlock) -> list[Diagnostic]:
        handler = self.registry.get(block.sync_class)
        if handler is None:
            raise LintFailure(
                f"unsupported sync-class '{block.sync_class}', expected one of {self.registry.supported_classes()}"
            )

        return [
            make_diagnostic(
                block.doc_path,
                block.start_line,
                message,
                sync_yaml=block.sync_yaml,
                sync_target=block.sync_target,
            )
            for message in handler.lint(block)
        ]


# ============================================================================
# CLI Entry
# ============================================================================


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

    linter = SyncLinter(repo_root, exclude_patterns=load_exclude_patterns(repo_root))
    diagnostics = linter.lint_documents(doc_paths)
    if diagnostics:
        use_color = should_use_color(sys.stderr)
        for diagnostic in merge_diagnostics_by_block(diagnostics):
            print(diagnostic.format(color=use_color), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
