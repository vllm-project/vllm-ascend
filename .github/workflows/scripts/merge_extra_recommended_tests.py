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
"""Append extra pytest targets from a YAML file onto a recommended test list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_DEFAULT_LIST_FILE = Path("recommended_pytest_paths.txt")
_YAML_HEADER = """# Extra pytest targets appended to coverage-recommended tests.
# Each item needs a path.
"""


def _raw_extra_items(yaml_path: Path) -> list[object]:
    if not yaml_path.is_file():
        return []
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("tests") or []
        return items if isinstance(items, list) else []
    return []


def _item_path(item: object) -> str | None:
    if not isinstance(item, dict):
        return None
    target = str(item.get("path") or "").strip()
    return target or None


def load_extra_tests(yaml_path: Path) -> list[str]:
    """Return extra pytest targets from *yaml_path*.

    Each item needs ``path``. Missing files, empty documents, and empty lists
    yield no extra tests.
    """
    tests: list[str] = []
    seen: set[str] = set()
    for item in _raw_extra_items(yaml_path):
        if not isinstance(item, dict):
            print(f"Skipping invalid extra test entry: {item!r}", file=sys.stderr)
            continue
        target = _item_path(item)
        if not target:
            print(f"Skipping extra test missing path: {item!r}", file=sys.stderr)
            continue
        if target in seen:
            continue
        seen.add(target)
        tests.append(target)
    return tests


def load_existing_tests(list_file: Path) -> list[str]:
    if not list_file.is_file():
        return []
    tests: list[str] = []
    for raw_line in list_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            tests.append(line)
    return tests


def merge_extra_tests(existing: list[str], extra_tests: list[str]) -> tuple[list[str], list[str]]:
    """Return merged paths and the newly added paths, preserving existing order."""
    seen = set(existing)
    added: list[str] = []
    for target in extra_tests:
        if target in seen:
            continue
        seen.add(target)
        added.append(target)
    return existing + added, added


def write_test_list(list_file: Path, tests: list[str]) -> None:
    list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text("".join(f"{target}\n" for target in tests), encoding="utf-8")


def write_extra_yaml(output: Path, paths: list[str]) -> None:
    lines = [_YAML_HEADER.rstrip(), ""]
    for path in paths:
        lines.append(f"- path: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_extra_yaml_files(existing_path: Path, incoming_path: Path, output_path: Path) -> list[str]:
    """Append incoming extra-yaml paths onto existing yaml. Skip duplicates."""
    merged: list[str] = []
    seen: set[str] = set()
    for item in _raw_extra_items(existing_path):
        target = _item_path(item)
        if target is None or target in seen:
            continue
        seen.add(target)
        merged.append(target)
    added: list[str] = []
    for item in _raw_extra_items(incoming_path):
        target = _item_path(item)
        if target is None or target in seen:
            continue
        seen.add(target)
        merged.append(target)
        added.append(target)
    write_extra_yaml(output_path, merged)
    return added


def _normalize_coverage_name(test_name: str) -> str:
    """Convert a coverage directory name back to a pytest path.

    Coverage dirs replace ``/`` with ``__`` and ``::`` with ``--``. File-level
    keys also drop the ``.py`` suffix, which this restores.
    """
    if test_name == "cpu-ut":
        return test_name
    result = test_name.replace("--", "::").replace("__", "/")
    if "::" not in result:
        result = result + ".py"
    return result


def load_covered_pytest_paths(map_file: Path | None, coverage_dir: Path | None) -> set[str]:
    """Return pytest paths present in coverage (map keys and coverage dirs)."""
    covered: set[str] = set()
    if map_file is not None and map_file.is_file():
        data = json.loads(map_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            covered.update(str(key).replace("\\", "/") for key in data)
    if coverage_dir is not None and coverage_dir.is_dir():
        for child in coverage_dir.iterdir():
            if not child.is_dir() or not (child / "covdata").is_dir():
                continue
            if (child / "covdata" / "FAILED").is_file():
                continue
            covered.add(_normalize_coverage_name(child.name))
    return covered


def _split_pytest_path(path: str) -> tuple[str, str]:
    """Return ``(file.py, rest)`` for a pytest path. ``cpu-ut`` is unchanged."""
    path = path.replace("\\", "/").rstrip("/")
    if path == "cpu-ut":
        return path, ""
    file_part, rest = (path.split("::", 1) + [""])[:2]
    if not file_part.endswith(".py"):
        file_part += ".py"
    return file_part, rest


def extra_path_covered(extra: str, covered: set[str]) -> bool:
    """Return True if *extra* is already represented in coverage.

    File-level coverage covers every case in that file, so ``test.py::test_1``
    is removed when coverage contains ``test.py``. A file-level extra path is
    removed only when that file itself is a coverage key.
    """
    extra_file, extra_rest = _split_pytest_path(extra)
    for name in covered:
        covered_file, covered_rest = _split_pytest_path(name)
        if covered_file != extra_file:
            continue
        if not covered_rest:
            return True
        if not extra_rest:
            continue
        if extra_rest == covered_rest:
            return True
        if extra_rest.startswith(covered_rest + "::") or covered_rest.startswith(extra_rest + "::"):
            return True
    return False


def prune_covered_extra_paths(paths: list[str], covered: set[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    for path in paths:
        if extra_path_covered(path, covered):
            removed.append(path)
        else:
            kept.append(path)
    return kept, removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append extra pytest targets from YAML onto a recommended test list.",
    )
    parser.add_argument(
        "--list-file",
        type=Path,
        default=_DEFAULT_LIST_FILE,
        help="Recommended pytest path list to update",
    )
    parser.add_argument(
        "--extra-yaml",
        type=Path,
        required=True,
        help="YAML file containing extra tests",
    )
    parser.add_argument(
        "--incoming-yaml",
        type=Path,
        default=None,
        help="YAML file whose paths are appended onto --extra-yaml",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=None,
        help="Where to write the merged extra yaml (default: --extra-yaml)",
    )
    parser.add_argument(
        "--prune-covered",
        action="store_true",
        help="Remove extra-yaml paths that already have coverage data",
    )
    parser.add_argument(
        "--map-file",
        type=Path,
        default=None,
        help="test_case_map.json used when --prune-covered is set",
    )
    parser.add_argument(
        "--coverage-dir",
        type=Path,
        default=None,
        help="Assembled coverage task directory used when --prune-covered is set",
    )
    args = parser.parse_args(argv)

    if args.prune_covered:
        output_yaml = args.output_yaml or args.extra_yaml
        covered = load_covered_pytest_paths(args.map_file, args.coverage_dir)
        extra_tests = load_extra_tests(args.extra_yaml)
        kept, removed = prune_covered_extra_paths(extra_tests, covered)
        if not removed:
            print("No extra tests are present in coverage; yaml unchanged")
            return 0
        write_extra_yaml(output_yaml, kept)
        print(f"Removed {len(removed)} path(s) already present in coverage:")
        for target in removed:
            print(f"  {target}")
        return 0

    if args.incoming_yaml is not None:
        output_yaml = args.output_yaml or args.extra_yaml
        added = merge_extra_yaml_files(args.extra_yaml, args.incoming_yaml, output_yaml)
        if not added:
            print(f"No new paths to append; wrote {output_yaml}")
            return 0
        print(f"Appended {len(added)} path(s) to {output_yaml}:")
        for target in added:
            print(f"  {target}")
        return 0

    extra_tests = load_extra_tests(args.extra_yaml)
    if not extra_tests:
        print(f"No extra tests in {args.extra_yaml}; leaving {args.list_file} unchanged")
        return 0

    existing = load_existing_tests(args.list_file)
    merged, added = merge_extra_tests(existing, extra_tests)
    if not added:
        print(f"Extra tests already present in {args.list_file}; nothing to add")
        return 0

    write_test_list(args.list_file, merged)
    print(f"Added {len(added)} extra test(s) from {args.extra_yaml}:")
    for target in added:
        print(f"  {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
