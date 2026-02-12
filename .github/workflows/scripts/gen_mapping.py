import argparse
import ast
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from coverage import CoverageData


@dataclass
class FunctionRange:
    start_line: int
    end_line: int
    signature: str


class FunctionParser(ast.NodeVisitor):
    def __init__(self) -> None:
        self.ranges: list[FunctionRange] = []
        self._class_stack: list[str] = []
        self._function_depth = 0

    def _get_end_lineno(self, node: ast.AST) -> int:
        end_lineno = getattr(node, "end_lineno", None)
        if end_lineno is not None:
            return int(end_lineno)

        max_lineno = getattr(node, "lineno", 0) or 0
        for child in ast.walk(node):
            child_lineno = getattr(child, "lineno", None)
            if child_lineno is not None:
                max_lineno = max(max_lineno, int(child_lineno))
        return max_lineno

    def _add_function_range(self, node: ast.AST, func_name: str) -> None:
        if not hasattr(node, "lineno"):
            return
        start_line = int(node.lineno)
        end_line = self._get_end_lineno(node)
        if self._class_stack:
            signature = f"{'.'.join(self._class_stack)}.{func_name}"
        else:
            signature = func_name
        self.ranges.append(FunctionRange(start_line=start_line, end_line=end_line, signature=signature))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._function_depth == 0:
            self._add_function_range(node, node.name)
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._function_depth == 0:
            self._add_function_range(node, node.name)
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1


def parse_function_ranges(file_path: Path) -> list[FunctionRange]:
    try:
        source = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[WARN] Source file not found: {file_path}", flush=True)
        return []
    except OSError as exc:
        print(f"[WARN] Failed to read file {file_path}: {exc}", flush=True)
        return []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        print(f"[WARN] Failed to parse AST for {file_path}: {exc}", flush=True)
        return []

    parser = FunctionParser()
    parser.visit(tree)
    return parser.ranges


def normalize_path(file_path: str, workspace_root: Path) -> str:
    posix_path = file_path.replace("\\", "/")
    marker = "/vllm_ascend/"
    idx = posix_path.find(marker)
    if idx >= 0:
        return posix_path[idx + 1 :]

    try:
        rel_path = Path(file_path).resolve().relative_to(workspace_root.resolve())
        return rel_path.as_posix()
    except (ValueError, OSError):
        return posix_path


def is_file_level_target(rel_path: str) -> bool:
    return "vllm_ascend/utils" in rel_path or "vllm_ascend/config" in rel_path


def _extract_tests_from_contexts(contexts: Iterable[str]) -> set[str]:
    tests: set[str] = set()
    for context in contexts:
        if not context:
            continue
        test_name = context.split("|", 1)[0].strip()
        if not test_name:
            continue
        tests.add(test_name)
    return tests


def _find_signature_for_line(line_no: int, ranges: list[FunctionRange]) -> str | None:
    for item in ranges:
        if item.start_line <= line_no <= item.end_line:
            return item.signature
    return None


def build_mapping(coverage_path: Path, workspace_root: Path) -> dict[str, dict[str, list[str]]]:
    data = CoverageData(basename=str(coverage_path))
    data.read()

    file_mapping: dict[str, set[str]] = defaultdict(set)
    func_mapping: dict[str, set[str]] = defaultdict(set)

    measured_files = data.measured_files() or []
    for measured in measured_files:
        rel_path = normalize_path(measured, workspace_root)
        if not rel_path.endswith(".py"):
            continue
        if "vllm_ascend/" not in rel_path:
            continue

        try:
            contexts_by_lineno = data.contexts_by_lineno(measured)
        except Exception as exc:
            print(f"[WARN] Failed to load contexts for {measured}: {exc}", flush=True)
            continue

        if not contexts_by_lineno:
            continue

        if is_file_level_target(rel_path):
            for contexts in contexts_by_lineno.values():
                file_mapping[rel_path].update(_extract_tests_from_contexts(contexts))
            continue

        src_path = workspace_root / rel_path
        ranges = parse_function_ranges(src_path)
        if not ranges:
            for contexts in contexts_by_lineno.values():
                file_mapping[rel_path].update(_extract_tests_from_contexts(contexts))
            continue

        for line_no, contexts in contexts_by_lineno.items():
            tests = _extract_tests_from_contexts(contexts)
            if not tests:
                continue
            signature = _find_signature_for_line(line_no, ranges)
            if signature is None:
                continue
            key = f"{rel_path}::{signature}"
            func_mapping[key].update(tests)

    return {
        "file_mapping": {k: sorted(v) for k, v in sorted(file_mapping.items())},
        "func_mapping": {k: sorted(v) for k, v in sorted(func_mapping.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate source-to-test mapping from .coverage data")
    parser.add_argument("--coverage-file", default=".coverage", help="Path to coverage data file")
    parser.add_argument("--output", default="", help="Output mapping file path")
    parser.add_argument("--workspace-root", default=".", help="Workspace root path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    coverage_path = Path(args.coverage_file)
    if not coverage_path.is_absolute():
        coverage_path = workspace_root / coverage_path

    if not coverage_path.exists():
        print(f"[ERROR] Coverage file does not exist: {coverage_path}", flush=True)
        return 1

    try:
        mapping = build_mapping(coverage_path=coverage_path, workspace_root=workspace_root)
    except Exception as exc:
        print(f"[ERROR] Failed to build mapping: {exc}", flush=True)
        return 1

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = workspace_root / output_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = workspace_root / f"mapping_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] Generated mapping: {output_path}", flush=True)
    print(
        "[INFO] Summary: "
        f"file_mapping={len(mapping['file_mapping'])}, "
        f"func_mapping={len(mapping['func_mapping'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
