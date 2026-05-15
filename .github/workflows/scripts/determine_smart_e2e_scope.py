#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Determine which UT tests to run based on changed files in a PR.

Pipeline:
  1. Diff       — get changed files from git.
  2. Match      — identify affected modules via ut_config.yaml.
  3. Collect    — gather test directories for matched modules.
  4. Scan       — AST-parse test files for @npu_test decorators.
  5. Group      — bucket tests by (num_npus, npu_type).
  6. Filter     — remove blacklisted tests, deduplicate.
  7. Resolve    — map each group to a runner via runner_label.json.
  8. Output     — write test_groups / has_tests / matched_modules.

Usage:
    python determine_smart_e2e_scope.py --diff-base origin/main
    python determine_smart_e2e_scope.py --changed-files file1.py file2.py

Flags:
    --run-all-cpu   Run ALL CPU (undecorated) tests regardless of module
                    filtering.  NPU tests are still filtered by module.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# 1. Constants & types
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_CONFIG_PATH = _SCRIPT_DIR / "ut_config.yaml"
_RUNNER_LABEL_PATH = _SCRIPT_DIR / "runner_label.json"
_BLACKLIST_PATH = _SCRIPT_DIR / "ut_blacklist.yaml"


class RunnerDeviceType(str, Enum):
    """Chip types — values must match runner_label.json ``chip`` field.

    Shared by:
      - tests/ut/conftest.py  (``npu_test`` decorator)
      - .github/workflows/scripts/determine_smart_e2e_scope.py  (AST parser)
    """

    A2 = "a2"
    A3 = "a3"
    _310P = "310p"
    CPU = "cpu"


# Type alias for the key used to group tests by runner requirements.
RunnerKey = tuple[int, RunnerDeviceType]

# Tests without @npu_test decorator route to the CPU runner.
_DEFAULT_KEY: RunnerKey = (0, RunnerDeviceType.CPU)


@dataclass
class RunnerInfo:
    """A self-hosted runner entry parsed from runner_label.json."""

    num_npus: int
    npu_type: RunnerDeviceType
    label: str
    image_tag: str = ""


# ---------------------------------------------------------------------------
# 2. Configuration loading
# ---------------------------------------------------------------------------


def _load_runners() -> list[RunnerInfo]:
    """Load runner definitions from runner_label.json."""
    with open(_RUNNER_LABEL_PATH) as f:
        raw = json.load(f)
    return [
        RunnerInfo(
            num_npus=info["npu_num"],
            npu_type=RunnerDeviceType(info["chip"]),
            label=label,
            image_tag=info.get("image_tag", ""),
        )
        for label, info in raw.items()
    ]


def _load_blacklist() -> set[str]:
    """Load blacklisted test file paths from ut_blacklist.yaml.

    Returns an empty set if the file does not exist.
    """
    if not _BLACKLIST_PATH.exists():
        return set()
    items = yaml.safe_load(_BLACKLIST_PATH.read_text()) or []
    return set(items)


# ---------------------------------------------------------------------------
# 3. Changed files & module matching
# ---------------------------------------------------------------------------


def _get_changed_files(base_ref: str) -> list[str]:
    """Return the list of changed files by diffing against *base_ref*."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def _match_modules(
    changed_files: list[str],
    config: list[dict],
) -> list[str]:
    """Return names of modules whose source dependencies match changed files.

    - ``optional: false`` → always included when there are changes.
    - ``optional: true``  → included only when a changed file falls under
      one of the module's ``source_file_dependencies``.
    """
    if not changed_files:
        return []

    matched: list[str] = []
    for module in config:
        if not module.get("optional", True):
            matched.append(module["name"])
            continue
        deps = module.get("source_file_dependencies", [])
        if any(f == dep or f.startswith(dep + "/") for f in changed_files for dep in deps):
            matched.append(module["name"])
    return matched


def _collect_test_dirs(
    module_names: list[str],
    config: list[dict],
) -> list[str]:
    """Collect test directory paths for the given modules.

    Deduplicates parent-child nested paths: if both ``a/b`` and ``a/b/c``
    are present, only ``a/b`` is kept (the parent already covers the child).
    """
    module_map = {m["name"]: m for m in config}
    raw: set[str] = set()
    for name in module_names:
        for path in module_map[name].get("tests", []):
            raw.add(path.rstrip("/"))

    # Remove paths that are children of any other path in the set
    sorted_paths = sorted(raw)
    result: list[str] = []
    for path in sorted_paths:
        if not any(path != other and path.startswith(other + "/") for other in sorted_paths):
            result.append(path)
    return result


# ---------------------------------------------------------------------------
# 4. AST scanning — extract @npu_test decorator info from test files
# ---------------------------------------------------------------------------


def _resolve_npu_type_from_ast(node: ast.expr) -> RunnerDeviceType:
    """Resolve an AST node to a RunnerDeviceType value.

    Handles:
      - String literal:   ``npu_type="a3"``               → ast.Constant
      - Enum attribute:   ``npu_type=RunnerDeviceType.A3`` → ast.Attribute
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return RunnerDeviceType(node.value)
    if isinstance(node, ast.Attribute) and isinstance(node.attr, str):
        return RunnerDeviceType[node.attr]
    raise ValueError(f"Cannot resolve npu_type from AST node: {ast.dump(node)}")


def _extract_runner_key(node: ast.FunctionDef | ast.ClassDef) -> RunnerKey:
    """Extract (num_npus, npu_type) from ``@npu_test`` on *node*.

    Returns ``_DEFAULT_KEY`` when no ``@npu_test`` decorator is found.
    Default values (``num_npus=1, npu_type=A2``) match
    ``tests/ut/conftest.py::npu_test``.
    """
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if not (isinstance(func, ast.Name) and func.id == "npu_test"):
            continue

        num_npus = 1
        npu_type = RunnerDeviceType.A2
        for kw in decorator.keywords:
            if kw.arg == "num_npus" and isinstance(kw.value, ast.Constant):
                num_npus = kw.value.value
            elif kw.arg == "npu_type":
                npu_type = _resolve_npu_type_from_ast(kw.value)
        return (num_npus, npu_type)

    return _DEFAULT_KEY


def _scan_test_file(filepath: str) -> dict[RunnerKey, list[str]]:
    """Parse a single test file and group node IDs by runner key.

    - Top-level ``test_*`` functions → keyed by their own ``@npu_test``.
    - Classes → keyed by the **class-level** ``@npu_test``; output the
      class node ID (``file::Class``), not individual methods.
    - Nodes without ``@npu_test`` → keyed by ``_DEFAULT_KEY`` (CPU).
    """
    with open(filepath) as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    groups: dict[RunnerKey, list[str]] = defaultdict(list)
    for node in ast.iter_child_nodes(tree):
        is_test_func = isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        is_test_class = isinstance(node, ast.ClassDef) and node.name.startswith("Test")
        if is_test_func or is_test_class:
            key = _extract_runner_key(node)
            groups[key].append(f"{filepath}::{node.name}")

    return dict(groups)


# ---------------------------------------------------------------------------
# 5. Test grouping pipeline — scan → filter → resolve
# ---------------------------------------------------------------------------


def _build_test_groups(
    test_dirs: list[str],
    runners: list[RunnerInfo],
    blacklist: set[str],
    *,
    cpu_test_dirs: list[str] | None = None,
) -> list[dict]:
    """Main grouping pipeline: scan, filter, resolve.

    Steps:
      1. Scan *test_dirs* — group tests by (num_npus, npu_type).
      2. If *cpu_test_dirs* is provided, replace the CPU group with a
         full scan of those directories (--run-all-cpu mode).
      3. Apply blacklist — remove blacklisted tests and deduplicate.
      4. Resolve each group to a runner from runner_label.json.
    """
    # Step 1: scan matched test directories
    all_groups: dict[RunnerKey, list[str]] = defaultdict(list)
    for test_dir in test_dirs:
        _scan_directory(test_dir, all_groups)

    # Step 2: override CPU group when --run-all-cpu
    if cpu_test_dirs is not None:
        cpu_groups: dict[RunnerKey, list[str]] = defaultdict(list)
        for test_dir in cpu_test_dirs:
            _scan_directory(test_dir, cpu_groups)
        all_groups[_DEFAULT_KEY] = cpu_groups[_DEFAULT_KEY]

    # Step 3: blacklist + dedup
    _apply_blacklist(all_groups, blacklist)

    # Step 4: resolve to runners
    return _resolve_to_runners(all_groups, runners)


def _scan_directory(
    test_dir: str,
    groups: dict[RunnerKey, list[str]],
) -> None:
    """Scan a single test directory and populate *groups* in place.

    Output uses the most concise representation that preserves group
    mutual exclusion (which holds because :func:`_collect_test_dirs`
    guarantees the input ``test_dirs`` are not parent-child nested):

      - Directory whose files are ALL undecorated → directory path
        in CPU group.
      - Mixed directory (any file has @npu_test) → decorated node IDs
        per runner group; undecorated files go to CPU group as file paths.
    """
    dir_path = Path(test_dir)
    if not dir_path.exists():
        groups[_DEFAULT_KEY].append(test_dir)
        return

    has_decorated = False
    undecorated_files: list[str] = []

    for test_file in sorted(dir_path.rglob("test_*.py")):
        file_path = str(test_file)
        file_groups = _scan_test_file(file_path)

        if not file_groups:
            undecorated_files.append(file_path)
            continue

        if any(key != _DEFAULT_KEY for key in file_groups):
            has_decorated = True
            for key, node_ids in file_groups.items():
                groups[key].extend(node_ids)
        else:
            undecorated_files.append(file_path)

    if not has_decorated:
        # Pure-CPU directory → collapse to single directory path
        groups[_DEFAULT_KEY].append(test_dir)
    else:
        groups[_DEFAULT_KEY].extend(undecorated_files)


def _apply_blacklist(
    all_groups: dict[RunnerKey, list[str]],
    blacklist: set[str],
) -> None:
    """Remove blacklisted tests from all groups and deduplicate in place.

    Handles three target formats produced by :func:`_scan_directory`:
      - File path  — drop if the path is in the blacklist.
      - Node ID    — drop if the file part (before ``::``) is blacklisted.
      - Directory  — if any blacklisted file lives under it, expand to
                     individual files so the blacklisted ones can be excluded.
    """
    for key in list(all_groups.keys()):
        filtered: list[str] = []
        seen: set[str] = set()

        for target in all_groups[key]:
            file_path = target.split("::")[0]

            if file_path in blacklist:
                continue

            target_path = Path(file_path)
            if blacklist and target_path.is_dir() and any(bl.startswith(file_path + "/") for bl in blacklist):
                for f in sorted(target_path.rglob("test_*.py")):
                    name = str(f)
                    if name not in blacklist and name not in seen:
                        filtered.append(name)
                        seen.add(name)
                continue

            if target not in seen:
                filtered.append(target)
                seen.add(target)

        all_groups[key] = filtered


def _resolve_to_runners(
    all_groups: dict[RunnerKey, list[str]],
    runners: list[RunnerInfo],
) -> list[dict]:
    """Map each (num_npus, npu_type) group to a runner.

    Exits with a descriptive error if any group cannot be matched.
    """
    result: list[dict] = []
    errors: list[str] = []

    for (num_npus, npu_type), tests in sorted(all_groups.items()):
        if not tests:
            continue

        runner = _find_runner(num_npus, npu_type, runners)
        if runner is None:
            errors.append(_format_runner_error(num_npus, npu_type, tests, runners))
            continue

        group = {
            "num_npus": num_npus,
            "npu_type": npu_type.value,
            "runner": runner.label,
            "tests": " ".join(sorted(tests)),
        }
        if runner.image_tag:
            group["image_tag"] = runner.image_tag
        result.append(group)

    if errors:
        print(
            "\nERROR: The following @npu_test decorator combinations "
            "cannot be routed to any runner in runner_label.json:"
            + "".join(errors)
            + "\n\nPlease fix the decorator arguments or add the "
            "missing runner to runner_label.json.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    return result


def _find_runner(
    num_npus: int,
    npu_type: RunnerDeviceType,
    runners: list[RunnerInfo],
) -> RunnerInfo | None:
    """Find the runner that matches the given NPU requirements.

    CPU type: match any CPU runner.
    NPU type: exact match on (npu_type, num_npus).
    """
    if npu_type == RunnerDeviceType.CPU:
        candidates = [r for r in runners if r.npu_type == RunnerDeviceType.CPU]
    else:
        candidates = [r for r in runners if r.npu_type == npu_type and r.num_npus == num_npus]
    return candidates[0] if candidates else None


def _format_runner_error(
    num_npus: int,
    npu_type: RunnerDeviceType,
    tests: list[str],
    runners: list[RunnerInfo],
) -> str:
    """Build a human-readable error message for an unresolvable group."""
    available = [f"{r.label} ({r.npu_type.value} x{r.num_npus})" for r in runners if r.npu_type == npu_type]
    header = f'\n  @npu_test(num_npus={num_npus}, npu_type="{npu_type.value}") — no runner available.'
    runners_line = (
        f"\n    Available {npu_type.value} runners: {', '.join(available)}"
        if available
        else f'\n    No runners defined for chip type "{npu_type.value}".'
    )
    tests_line = "\n    Affected tests:\n" + "\n".join(f"      - {t}" for t in sorted(tests))
    return header + runners_line + tests_line


# ---------------------------------------------------------------------------
# 6. Output
# ---------------------------------------------------------------------------


def _write_output(
    test_groups: list[dict],
    matched_modules: list[str],
    blacklist: set[str],
) -> None:
    """Write step outputs to $GITHUB_OUTPUT (or stdout when running locally)."""
    has_tests = len(test_groups) > 0
    groups_json = json.dumps(test_groups, separators=(",", ":"))

    outputs = {
        "test_groups": groups_json,
        "has_tests": str(has_tests).lower(),
        "matched_modules": ",".join(matched_modules),
    }

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            for key, value in outputs.items():
                f.write(f"{key}={value}\n")
    else:
        for key, value in outputs.items():
            print(f"{key}={value}")

    _print_summary(test_groups, matched_modules, blacklist, has_tests)


def _print_summary(
    test_groups: list[dict],
    matched_modules: list[str],
    blacklist: set[str],
    has_tests: bool,
) -> None:
    """Print a human-readable summary to stderr for CI logs."""
    divider = "=" * 60
    print(f"\n{divider}", file=sys.stderr)
    print("Smart UT Scope Determination Summary", file=sys.stderr)
    print(divider, file=sys.stderr)
    print(f"Matched modules: {matched_modules or '(none)'}", file=sys.stderr)
    print(f"Has tests to run: {has_tests}", file=sys.stderr)

    if blacklist:
        print(f"\n  Blacklisted ({len(blacklist)} tests):", file=sys.stderr)
        for bl in sorted(blacklist):
            print(f"    - {bl}", file=sys.stderr)

    for group in test_groups:
        npu_type = group["npu_type"]
        num_npus = group["num_npus"]
        runner = group["runner"]
        tests = group["tests"].split()
        print(
            f"\n  [{npu_type} x{num_npus}] -> {runner} ({len(tests)} tests):",
            file=sys.stderr,
        )
        for t in tests:
            print(f"    - {t}", file=sys.stderr)

    print(f"{divider}\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Determine UT test scope based on changed files",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--changed-files",
        nargs="+",
        help="List of changed file paths",
    )
    input_group.add_argument(
        "--diff-base",
        type=str,
        help="Git ref to diff against (e.g. origin/main)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="Path to ut_config.yaml",
    )
    parser.add_argument(
        "--run-all-cpu",
        action="store_true",
        help="Run all CPU (undecorated) tests regardless of module filtering",
    )

    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    # --- Pipeline ---
    changed_files = _get_changed_files(args.diff_base) if args.diff_base else args.changed_files
    matched_modules = _match_modules(changed_files, config)
    test_dirs = _collect_test_dirs(matched_modules, config)

    cpu_test_dirs = None
    if args.run_all_cpu:
        all_module_names = [m["name"] for m in config]
        cpu_test_dirs = _collect_test_dirs(all_module_names, config)

    runners = _load_runners()
    blacklist = _load_blacklist()
    test_groups = _build_test_groups(test_dirs, runners, blacklist, cpu_test_dirs=cpu_test_dirs)

    _write_output(test_groups, matched_modules, blacklist)


if __name__ == "__main__":
    main()
