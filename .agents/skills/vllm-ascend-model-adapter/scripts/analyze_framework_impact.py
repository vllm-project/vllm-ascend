#!/usr/bin/env python3
"""analyze_framework_impact.py — Derive framework dependencies from model entry files.

Usage:
    python analyze_framework_impact.py \
      --vllm-src /path/to/vllm \
      --vllm-ascend-src /path/to/vllm-ascend \
      --entry-file /path/to/model.py \
      [--entry-file /path/to/processor.py]

Outputs a human-readable report plus a machine-readable FRAMEWORK_IMPACT_SUMMARY line.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import deque
from pathlib import Path


def module_name_from_path(vllm_src: Path, path: Path) -> str:
    rel = path.resolve().relative_to(vllm_src.resolve())
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = path.stem
    return ".".join(parts)


def resolve_module_path(vllm_src: Path, module_name: str) -> Path | None:
    module_parts = module_name.split(".")
    module_path = vllm_src.joinpath(*module_parts).with_suffix(".py")
    if module_path.exists():
        return module_path
    package_init = vllm_src.joinpath(*module_parts, "__init__.py")
    if package_init.exists():
        return package_init
    return None


def resolve_relative_module(current_module: str, imported_module: str | None,
                            level: int) -> str:
    current_parts = current_module.split(".")
    base_parts = current_parts[:-level] if level else current_parts
    if imported_module:
        return ".".join(base_parts + imported_module.split("."))
    return ".".join(base_parts)


def discover_imports(vllm_src: Path, module_name: str, module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(errors="ignore"))
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "vllm" or alias.name.startswith("vllm."):
                    imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_module = resolve_relative_module(module_name, node.module, node.level)
            else:
                base_module = node.module or ""

            if not (base_module == "vllm" or base_module.startswith("vllm.")):
                continue

            imported_modules.add(base_module)
            for alias in node.names:
                candidate = f"{base_module}.{alias.name}"
                if resolve_module_path(vllm_src, candidate):
                    imported_modules.add(candidate)

    resolved = {name for name in imported_modules if resolve_module_path(vllm_src, name)}
    return resolved


def collect_dependency_graph(vllm_src: Path, entry_files: list[Path]) -> dict[str, Path]:
    queue = deque()
    dependencies: dict[str, Path] = {}

    for path in entry_files:
        module_name = module_name_from_path(vllm_src, path)
        dependencies[module_name] = path
        queue.append(module_name)

    while queue:
        module_name = queue.popleft()
        module_path = dependencies[module_name]
        for imported in discover_imports(vllm_src, module_name, module_path):
            if imported in dependencies:
                continue
            imported_path = resolve_module_path(vllm_src, imported)
            if not imported_path:
                continue
            dependencies[imported] = imported_path
            queue.append(imported)

    return dependencies


def is_model_or_processor(path: Path, vllm_src: Path) -> bool:
    rel = path.resolve().relative_to(vllm_src.resolve()).as_posix()
    return rel.startswith("vllm/model_executor/models/") or rel.startswith(
        "vllm/transformers_utils/processors/")


def build_ascend_index(vllm_ascend_src: Path) -> list[tuple[Path, list[str]]]:
    python_files = sorted((vllm_ascend_src / "vllm_ascend").rglob("*.py"))
    index = []
    for path in python_files:
        try:
            index.append((path, path.read_text(errors="ignore").splitlines()))
        except OSError:
            continue
    return index


def find_reference_hits(module_name: str, module_path: Path, vllm_src: Path,
                        vllm_ascend_src: Path,
                        ascend_index: list[tuple[Path, list[str]]]) -> list[str]:
    rel_path = module_path.resolve().relative_to(vllm_src.resolve()).as_posix()
    patterns = [
        module_name,
        rel_path,
        ".".join(module_name.split(".")[-2:]),
    ]

    hits = []
    for path, lines in ascend_index:
        for lineno, line in enumerate(lines, 1):
            if any(pattern and pattern in line for pattern in patterns):
                try:
                    rel = path.relative_to(vllm_ascend_src.resolve())
                except ValueError:
                    rel = path
                hits.append(f"{rel}:{lineno}: {line.strip()}")
                break
        if len(hits) == 3:
            break
    return hits


def categorize_framework_modules(dependencies: dict[str, Path], entry_files: list[Path],
                                 vllm_src: Path) -> dict[str, Path]:
    entry_set = {path.resolve() for path in entry_files}
    framework_modules = {}

    for module_name, module_path in dependencies.items():
        if module_path.resolve() in entry_set:
            continue
        if is_model_or_processor(module_path, vllm_src):
            continue
        framework_modules[module_name] = module_path

    return framework_modules


def print_report(entry_files: list[Path], framework_modules: dict[str, Path],
                 coverage: dict[str, list[str]], vllm_src: Path) -> None:
    print("=" * 72)
    print("Framework Impact Report")
    print("=" * 72)
    print("entry_files:")
    for path in entry_files:
        print(f"  - {path}")
    print()
    print("candidate_framework_modules:")
    for module_name, module_path in sorted(framework_modules.items()):
        rel_path = module_path.resolve().relative_to(vllm_src.resolve()).as_posix()
        print(f"  - {module_name} [{rel_path}]")
        hits = coverage[module_name]
        if hits:
            for hit in hits:
                print(f"      coverage: {hit}")
        else:
            print("      coverage: no textual reference found in vllm_ascend/")
    print("=" * 72)

    uncovered = sorted(
        module_name for module_name, hits in coverage.items() if not hits)
    summary = {
        "entry_files": [str(path) for path in entry_files],
        "framework_module_count": len(framework_modules),
        "framework_modules": sorted(framework_modules.keys()),
        "uncovered_modules": uncovered,
    }
    print()
    print("FRAMEWORK_IMPACT_SUMMARY:",
          json.dumps(summary, ensure_ascii=True, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-src", required=True)
    parser.add_argument("--vllm-ascend-src", required=True)
    parser.add_argument("--entry-file", action="append", required=True)
    args = parser.parse_args()

    vllm_src = Path(args.vllm_src).resolve()
    vllm_ascend_src = Path(args.vllm_ascend_src).resolve()
    entry_files = [Path(path).resolve() for path in args.entry_file]

    for path in entry_files:
        if not path.exists():
            raise SystemExit(f"ERROR: entry file not found: {path}")
        try:
            path.relative_to(vllm_src)
        except ValueError as exc:
            raise SystemExit(
                f"ERROR: entry file must live under --vllm-src: {path}"
            ) from exc

    dependencies = collect_dependency_graph(vllm_src, entry_files)
    framework_modules = categorize_framework_modules(dependencies, entry_files, vllm_src)
    ascend_index = build_ascend_index(vllm_ascend_src)

    coverage = {}
    for module_name, module_path in framework_modules.items():
        coverage[module_name] = find_reference_hits(
            module_name, module_path, vllm_src, vllm_ascend_src, ascend_index)

    print_report(entry_files, framework_modules, coverage, vllm_src)


if __name__ == "__main__":
    main()
