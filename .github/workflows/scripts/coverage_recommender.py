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
"""Recommend pytest targets for a PR based on a coverage test case map."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import regex as re

REPO_NAME = "vllm_ascend"

# Minimum proportion of changed product files a test must cover to be selected
# at the line/function granularity (0.0 => any overlap selects the test).
COVERAGE_DENSITY_THRESHOLD = 0.0

TEST_RE = re.compile(r"^tests/(?:e2e/pull_request|ut)(?:/.+)?/test_\w+\.py$")


class FunctionParser:
    """Parses a Python file to map line numbers to enclosing functions."""

    @staticmethod
    def get_lines_functions(filepath: str, lines: set[int]) -> dict[int, str]:
        """Return {line: func_name} for the given lines based on AST ranges."""
        line_to_function: dict[int, str] = {}
        func_to_lines: dict[str, set[int]] = defaultdict(set)
        try:
            tree = ast.parse(Path(filepath).read_text(encoding="utf-8"), filename=filepath)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_to_lines[node.name].update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        except Exception as e:
            print(f"  Warning: Failed to parse {filepath}: {e}")
            return {}
        for line in lines:
            for func_name, covered in func_to_lines.items():
                if line in covered:
                    line_to_function[line] = func_name
                    break
        return line_to_function


class CoverageMap:
    """Loads test case map (test_name -> covered files/lines) and answers coverage queries."""

    def __init__(self, map_file: Path | str):
        self.data = json.loads(Path(map_file).read_text(encoding="utf-8"))

    def names(self) -> list[str]:
        return list(self.data)

    def covered_files(self, test_name: str) -> dict[str, set[int]]:
        return {path: set(lines) for path, lines in self.data.get(test_name, {}).get("files", {}).items()}

    def covers_any(self, test_name: str, path: str, lines: set[int]) -> bool:
        covered = self.covered_files(test_name).get(path, set())
        return bool(lines & covered) if lines else False

    def strips(self, path: str) -> str:
        return path[len(REPO_NAME) + 1 :] if path.startswith(f"{REPO_NAME}/") else path


class TestSelector:
    """Selects tests covering changed lines, cascading line -> function -> file."""

    def __init__(self, cover: CoverageMap, repo_root: str):
        self.cover = cover
        self.repo_root = Path(repo_root)

    def select_tests(self, changed: dict[str, set[int]]) -> tuple[dict[str, int], str]:
        if not changed:
            return {}, "none"
        scored = self._line_match(changed)
        if scored:
            return scored, "line"
        scored = self._function_match(changed)
        if scored:
            return scored, "function"
        return self._file_match(changed), "file"

    def _line_match(self, changed: dict[str, set[int]]) -> dict[str, int]:
        result: dict[str, int] = {}
        total_covered_checks = len(changed)
        for test_name in self.cover.names():
            hit_files = sum(
                1 for path, lines in changed.items() if self.cover.covers_any(test_name, self.cover.strips(path), lines)
            )
            if hit_files and hit_files / total_covered_checks >= COVERAGE_DENSITY_THRESHOLD:
                result[test_name] = hit_files
        return result

    def _function_match(self, changed: dict[str, set[int]]) -> dict[str, int]:
        grouped: dict[tuple[str, str], set[int]] = defaultdict(set)  # (path, func) -> lines
        for path, lines in changed.items():
            source = self._find_source(path)
            if source is None:
                continue
            for line, func_name in FunctionParser.get_lines_functions(source, lines).items():
                grouped[(path, func_name)].add(line)

        result: dict[str, int] = {}
        for test_name in self.cover.names():
            hit = sum(
                1
                for (path, _func), lines in grouped.items()
                if self.cover.covers_any(test_name, self.cover.strips(path), lines)
            )
            if hit:
                result[test_name] = hit
        return result

    def _file_match(self, changed: dict[str, set[int]]) -> dict[str, int]:
        result: dict[str, int] = {}
        for test_name in self.cover.names():
            files = self.cover.covered_files(test_name)
            if any(files.get(self.cover.strips(path)) for path in changed):
                result[test_name] = 1
        return result

    def _find_source(self, path: str) -> str | None:
        for candidate in (self.repo_root / path, self.repo_root / "vllm-covstub" / path):
            if candidate.is_file():
                return str(candidate)
        return None


def _load_extra_tests(yaml_path: Path | None) -> list[str]:
    """Return extra pytest targets from *yaml_path* (each item needs ``path``)."""
    if yaml_path is None or not yaml_path.is_file():
        return []
    import yaml

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("tests") or []
        items = items if isinstance(items, list) else []
    else:
        items = []

    tests: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        target = str(item.get("path") or "").strip()
        if target and target not in seen:
            seen.add(target)
            tests.append(target)
    return tests


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommend pytest targets for a PR from a coverage test case map.")
    parser.add_argument("--map-file", "-m", required=True, help="test_case_map.json")
    parser.add_argument("--base-sha", required=True, help="Base commit SHA (diff base)")
    parser.add_argument("--head-sha", required=True, help="Head commit SHA (PR head)")
    parser.add_argument("--repo-root", default=".", help="Repo root for function-level matching")
    parser.add_argument("--extra-yaml", type=Path, default=None, help="Optional YAML of extra pytest targets")
    parser.add_argument("--output", type=Path, default=Path("recommended_pytest_paths.txt"), help="Output list file")
    args = parser.parse_args()

    # 1. Load test case map
    print("\n=== Loading Test Case Mapping ===")
    cover = CoverageMap(args.map_file)
    print(f"Loaded {len(cover.names())} test case mappings from {args.map_file}")

    # 2. Get PR diff (unified=0 for exact new-file line numbers)
    print("\n=== Parsing Code Changes ===")
    diff = subprocess.run(
        ["git", "diff", "--unified=0", args.base_sha, args.head_sha, "--", "*.py"],
        capture_output=True,
        text=True,
    )
    if diff.returncode != 0:
        raise RuntimeError(f"git diff failed: {diff.stderr}")
    diff_lines = diff.stdout.split("\n")

    # 3. Parse product-code changes, csrc flag, and test-file adds/deletes.
    changed_product: dict[str, set[int]] = {}
    csrc_change = False
    new_test_files: list[str] = []
    deleted_test_files: list[str] = []

    i = 0
    n = len(diff_lines)
    while i < n:
        line = diff_lines[i]
        if not line.startswith("+++ b/"):
            i += 1
            continue
        new_path = line[6:].strip().removeprefix("b/")

        if new_path.startswith("csrc/"):
            csrc_change = True
        elif new_path.startswith(REPO_NAME) and new_path.endswith(".py"):
            line_numbers: set[int] = set()
            j = i + 1
            while j < n and not diff_lines[j].startswith(("+++ b/", "diff --git")):
                m = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff_lines[j])
                if m:
                    new_start = int(m.group(1))
                    new_count = int(m.group(2)) if m.group(2) else 1
                    line_numbers.update(range(new_start, new_start + new_count))
                j += 1
            if line_numbers:
                changed_product[new_path] = line_numbers
            i = j
            continue

        if TEST_RE.match(new_path):
            new_test_files.append(new_path)
        i += 1

    # Determine deleted test files via the paired --- a/ header.
    for line in diff_lines:
        if line.startswith("--- a/") and TEST_RE.match(line[6:].strip().removeprefix("a/")):
            deleted_test_files.append(line[6:].strip().removeprefix("a/"))

    print(f"  csrc change: {csrc_change}")
    print(f"  changed product files ({len(changed_product)}): {list(changed_product)}")
    print(f"  new test files: {new_test_files}")
    print(f"  deleted test files: {deleted_test_files}")

    # 4. Select tests
    reason = "none"
    selected: dict[str, int]
    if csrc_change:
        reason = "csrc-full"
        print("\n=== CSRC changes detected - running full test suite ===")
        selected = {name: 0 for name in cover.names()}
    else:
        selector = TestSelector(cover, args.repo_root)
        selected, reason = selector.select_tests(changed_product)
        if not changed_product:
            print("\n=== No product source code changes found ===")

    for test in new_test_files:
        selected.setdefault(test, 0)
    for test in deleted_test_files:
        selected.pop(test, None)
        for key in [k for k in selected if k.startswith(test)]:
            selected.pop(key, None)

    # 5. Append extra tests from YAML
    for target in _load_extra_tests(args.extra_yaml):
        if target not in selected:
            selected[target] = 0

    # 6. Write deterministic output
    names = sorted(selected, key=lambda name: (-selected[name], name))
    args.output.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
    print(f"\n=== Recommended {len(names)} tests ({reason}) ===")
    for name in names:
        print(f"  {name}")
    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
