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
"""Build the test case -> covered lines mapping from coverage SQLite data."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

REPO_NAME = "vllm_ascend"


def _normalize_test_name(test_name: str) -> str:
    """Convert a coverage directory name to a standard pytest path.

    - tests__e2e__a__b -> tests/e2e/a/b.py (file-level, restores .py)
    - tests__e2e__a__b--test_foo -> tests/e2e/a/b::test_foo (function-level)
    - cpu-ut -> cpu-ut
    """
    if test_name == "cpu-ut":
        return test_name
    result = test_name.replace("--", "::").replace("__", "/")
    if "::" not in result:
        result = result + ".py"
    return result


def _scan_test_cases(coverage_data_dir: Path) -> list[str]:
    test_cases = []
    for item in coverage_data_dir.iterdir():
        if (item.is_dir() and (item.name.startswith("tests__") or item.name == "cpu-ut")) and (
            item / "covdata"
        ).exists():
            test_cases.append(item.name)
    return sorted(test_cases)


def _get_covered_files_from_file(cov_file: str) -> set[str]:
    files = set()
    conn = sqlite3.connect(cov_file)
    try:
        for (path,) in conn.execute("SELECT path FROM file"):
            if REPO_NAME in path:
                rel = path.split(f"{REPO_NAME}/")[-1] if f"{REPO_NAME}/" in path else path
                files.add(rel)
    finally:
        conn.close()
    return files


def _get_covered_lines_from_file(cov_file: str, filename: str) -> set[int]:
    lines = set()
    conn = sqlite3.connect(cov_file)
    try:
        rows = conn.execute("SELECT id FROM file WHERE path LIKE ?", (f"%{filename}",)).fetchall()
        if not rows:
            return lines
        file_id = rows[0][0]
        for fromno, tono in conn.execute("SELECT DISTINCT fromno, tono FROM arc WHERE file_id = ?", (file_id,)):
            if fromno > 0:
                lines.add(fromno)
            if tono > 0:
                lines.add(tono)
    finally:
        conn.close()
    return lines


def build_test_case_map(coverage_data_dir: Path) -> dict:
    test_case_map = {}
    test_cases = _scan_test_cases(coverage_data_dir)
    print(f"Found {len(test_cases)} test cases")

    for i, test_case in enumerate(test_cases, 1):
        print(f"  [{i}/{len(test_cases)}] Processing {test_case}...")
        covdata_dir = coverage_data_dir / test_case / "covdata"
        file_lines_map: dict[str, set[int]] = defaultdict(set)
        for cov_file in covdata_dir.glob("coverage.*"):
            for filename in _get_covered_files_from_file(str(cov_file)):
                lines = _get_covered_lines_from_file(str(cov_file), filename)
                if lines:
                    file_lines_map[filename].update(lines)

        name = _normalize_test_name(test_case)
        test_case_map[name] = {
            "files": {k: sorted(v) for k, v in file_lines_map.items()},
            "file_count": len(file_lines_map),
            "line_count": sum(len(v) for v in file_lines_map.values()),
        }
        print(f"    -> {len(file_lines_map)} files, {test_case_map[name]['line_count']} lines")

    return test_case_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Build test case mapping from coverage SQLite data.")
    parser.add_argument("--coverage-dir", "-c", required=True, help="Coverage data directory")
    parser.add_argument("--map-file", "-m", default="test_case_map.json", help="Output map file")
    args = parser.parse_args()

    print("\n=== Building Test Case Mapping ===")
    test_case_map = build_test_case_map(Path(args.coverage_dir))
    Path(args.map_file).write_text(json.dumps(test_case_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTest case mapping ({len(test_case_map)} tests) saved to: {args.map_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
