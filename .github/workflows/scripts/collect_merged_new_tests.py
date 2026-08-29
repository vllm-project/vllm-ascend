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
"""Collect new or moved tests from a merged PR and write them as extra-yaml paths.

Use GitHub's three-dot PR diff (``pr-base...pr-head``), the same range as the
Files changed tab: ``git diff $(git merge-base BASE HEAD) HEAD``.

Do not two-dot-diff a merge/squash commit against ``pull_request.base.sha``.
That SHA can lag behind ``main``, so ``git diff stale-base merge-commit``
includes every test that landed on ``main`` while the PR was open.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

_DEFAULT_OUTPUT = Path("merged_new_tests.yaml")
_WATCHED_TEST_DIRS = ("tests/e2e/pull_request/", "tests/ut/")

NameStatus = list[tuple[str, str, str]]


def _as_posix(path: str) -> str:
    return path.replace("\\", "/")


def _in_watched_dirs(path: str) -> bool:
    posix = _as_posix(path)
    return posix.startswith(_WATCHED_TEST_DIRS)


def _is_test_file(path: str) -> bool:
    posix = _as_posix(path)
    name = Path(posix).name
    return _in_watched_dirs(posix) and name.startswith("test_") and name.endswith(".py")


def _is_test_class(node: ast.ClassDef) -> bool:
    if node.name.startswith("Test") and node.name != "Test":
        return True
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id.endswith("TestCase"):
            return True
        if isinstance(base, ast.Attribute) and base.attr.endswith("TestCase"):
            return True
    return False


def _is_class_nodeid(nodeid: str) -> bool:
    parts = nodeid.split("::")
    return len(parts) == 2 and parts[1].startswith("Test") and parts[1] != "Test"


def _case_path(nodeid: str) -> str | None:
    """Return ``file.py::test_name``, dropping any test-class segment."""
    if _is_class_nodeid(nodeid):
        return None
    parts = nodeid.split("::")
    if len(parts) == 2:
        return nodeid
    if len(parts) == 3:
        return f"{parts[0]}::{parts[2]}"
    return None


def collect_nodeids(file_path: str, source: str) -> set[str]:
    posix = _as_posix(file_path)
    nodeids = {posix}
    try:
        tree = ast.parse(source, filename=posix)
    except SyntaxError:
        return nodeids
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_test_class(node):
            class_id = f"{posix}::{node.name}"
            nodeids.add(class_id)
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    nodeids.add(f"{class_id}::{item.name}")
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            nodeids.add(f"{posix}::{node.name}")
    return nodeids


def collect_from_changes(
    changes: NameStatus,
    *,
    read_head: Callable[[str], str],
    read_old: Callable[[str], str | None],
) -> list[tuple[str, str]]:
    """Return ``(path, reason)`` pairs for added, renamed, or moved tests.

    New or renamed files are recorded as ``.py`` paths. New cases in an
    existing file are recorded as ``file.py::test_name``.
    """
    required: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(path: str, reason: str) -> None:
        path = _as_posix(path)
        if path in seen:
            return
        seen.add(path)
        required.append((path, reason))

    for kind, old_path, new_path in changes:
        if not _is_test_file(new_path):
            continue
        if kind in {"A", "C"}:
            add(new_path, "new test file")
            continue
        if kind == "R":
            add(new_path, f"test file path changed from {old_path}")
            continue
        head_ids = collect_nodeids(new_path, read_head(new_path))
        old_ids = collect_nodeids(new_path, read_old(new_path) or "")
        added_ids = head_ids - old_ids
        added_ids.discard(_as_posix(new_path))
        for nodeid in sorted(added_ids):
            case_path = _case_path(nodeid)
            if case_path is None:
                continue
            add(case_path, "test case added or renamed")
    return required


def _git_run(args: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _git_show(ref: str, path: str) -> str | None:
    result = _git_run(["git", "show", f"{ref}:{path}"], check=False)
    if result.returncode != 0:
        return None
    return result.stdout


def _rev_parse(ref: str) -> str:
    result = _git_run(["git", "rev-parse", "--verify", ref], check=True)
    return result.stdout.strip()


def resolve_pr_diff_refs(pr_base: str, pr_head: str) -> tuple[str, str]:
    """Return ``(old_ref, new_ref)`` for GitHub's three-dot PR diff.

    ``old_ref`` is ``merge-base(pr_base, pr_head)`` and ``new_ref`` is the PR
    head. This matches ``git diff pr-base...pr-head`` / the Files changed tab,
    even when ``pr_base`` is a stale ``main`` SHA.
    """
    base = _rev_parse(pr_base)
    head = _rev_parse(pr_head)
    merge_base = _git_run(["git", "merge-base", base, head], check=True)
    return merge_base.stdout.strip(), head


def _git_name_status(old: str, new: str) -> NameStatus:
    result = _git_run(
        [
            "git",
            "diff",
            "--name-status",
            "--find-renames",
            "--diff-filter=ACMR",
            f"{old}...{new}",
            "--",
            "tests/e2e/pull_request",
            "tests/ut",
        ],
        check=True,
    )
    changes: NameStatus = []
    for raw_line in result.stdout.splitlines():
        if not raw_line.strip():
            continue
        status, *paths = raw_line.split("\t")
        kind = status[:1]
        if kind in {"A", "C", "M"} and len(paths) == 1:
            changes.append((kind, paths[0], paths[0]))
        elif kind == "R" and len(paths) == 2:
            changes.append(("R", paths[0], paths[1]))
    return changes


def write_new_tests_yaml(
    items: list[tuple[str, str]],
    output: Path,
    *,
    pr_number: str = "",
    merge_sha: str = "",
) -> None:
    lines = [
        "# New or moved tests collected from a merged PR.",
        "# Compatible with extra_recommended_tests.yaml.",
    ]
    if pr_number:
        lines.append(f"# PR: {pr_number}")
    if merge_sha:
        lines.append(f"# merge: {merge_sha}")
    for nodeid, reason in items:
        lines.append(f"- path: {nodeid}")
        lines.append(f"  # {reason}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_required_changes(old_ref: str, new_ref: str) -> list[tuple[str, str]]:
    return collect_from_changes(
        _git_name_status(old_ref, new_ref),
        read_head=lambda path: _git_show(new_ref, path) or "",
        read_old=lambda path: _git_show(old_ref, path),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write new or moved tests from a merged PR's GitHub three-dot diff.",
    )
    parser.add_argument(
        "--pr-base",
        required=True,
        help="PR base SHA (github.event.pull_request.base.sha)",
    )
    parser.add_argument(
        "--pr-head",
        required=True,
        help="PR head ref (refs/pull/<n>/head), not the merge/squash commit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="YAML file to write",
    )
    parser.add_argument("--pr-number", default="", help="Merged PR number for the YAML header")
    parser.add_argument("--merge-sha", default="", help="Merge commit SHA for the YAML header")
    args = parser.parse_args(argv)

    old_ref, new_ref = resolve_pr_diff_refs(args.pr_base, args.pr_head)
    print(f"Collecting GitHub three-dot PR diff {args.pr_base}...{args.pr_head}")
    print(f"Resolved range {old_ref}..{new_ref}")
    required = collect_required_changes(old_ref, new_ref)
    if not required:
        print("No new or moved tests detected")
        if args.output.exists():
            args.output.unlink()
        return 0

    write_new_tests_yaml(
        required,
        args.output,
        pr_number=args.pr_number,
        merge_sha=args.merge_sha,
    )
    print(f"Wrote {len(required)} test path(s) to {args.output}")
    for nodeid, reason in required:
        print(f"  - {nodeid} ({reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
