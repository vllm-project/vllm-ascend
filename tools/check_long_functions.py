#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from __future__ import annotations

import ast
import subprocess
import sys

MAX_FUNCTION_LINES = 100


def _get_changed_lines(filepath: str) -> set[int]:
    """Return added line numbers from staged git diff.

    Parameters
    ----------
    filepath : str
        File path to inspect.

    Returns
    -------
    set[int]
        1-indexed added line numbers from staged changes.

    Notes
    -----
    If no staged diff exists (e.g. CI --all-files mode),
    an empty set is returned so that existing functions
    are not incorrectly flagged.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--", filepath],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()

    if not result.stdout.strip():
        return set()

    changed: set[int] = set()
    current_line = 0

    for line in result.stdout.splitlines():
        if line.startswith("@@"):
            # Example:
            # @@ -10,3 +20,8 @@
            parts = line.split()
            new_info = parts[2].lstrip("+")
            current_line = int(new_info.split(",")[0])

        elif line.startswith("+") and not line.startswith("+++"):
            changed.add(current_line)
            current_line += 1

        elif not line.startswith("-"):
            current_line += 1

    return changed


def _has_comment(source_lines: list[str], start_line_1: int, end_line_1: int) -> bool:
    """Check whether a function contains Python comments.

    Parameters
    ----------
    source_lines : list[str]
        Source code lines.
    start_line_1 : int
        1-indexed function start line.
    end_line_1 : int
        1-indexed function end line.

    Returns
    -------
    bool
        True if any valid '#' comment exists.
    """
    start_idx = max(start_line_1 - 1, 0)
    end_idx = min(end_line_1, len(source_lines))

    for i in range(start_idx, end_idx):
        stripped = source_lines[i].strip()

        if not stripped:
            continue
        # Full-line comment
        if stripped.startswith("#"):
            return True
        # Inline comment
        if "#" in stripped:
            in_string = False
            quote = ""
            for ch in stripped:
                if ch in ('"', "'"):
                    if not in_string:
                        in_string = True
                        quote = ch
                    elif ch == quote:
                        in_string = False
                elif ch == "#" and not in_string:
                    return True
    return False


def check_file(filepath: str, changed_lines: set[int]) -> list[str]:
    """Check one Python file for undocumented long functions."""
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError):
        return []

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []

    source_lines = source.splitlines()
    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Only check newly added functions.
        if node.lineno not in changed_lines:
            continue
        if node.end_lineno is None:
            continue
        func_lines = node.end_lineno - node.lineno + 1
        if func_lines <= MAX_FUNCTION_LINES:
            continue
        # Has docstring
        if ast.get_docstring(node) is not None:
            continue
        # Has inline comments
        if _has_comment(source_lines, node.lineno, node.end_lineno):
            continue

        violations.append(
            f"{filepath}:{node.lineno}: "
            f"function '{node.name}' "
            f"is {func_lines} lines "
            f"(>{MAX_FUNCTION_LINES}) "
            f"without comments or docstring"
        )
    return violations


def main() -> int:
    if len(sys.argv) < 2:
        return 0

    all_violations: list[str] = []
    for filepath in sys.argv[1:]:
        changed_lines = _get_changed_lines(filepath)

        # No newly added lines in this file.
        if not changed_lines:
            continue

        all_violations.extend(check_file(filepath, changed_lines))

    if all_violations:
        print(
            "Functions longer than "
            f"{MAX_FUNCTION_LINES} lines "
            "must include comments or a docstring.\n"
            "Add a docstring or inline comments "
            "to explain the function logic.\n"
        )

        for violation in all_violations:
            print(f"  {violation}")

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
