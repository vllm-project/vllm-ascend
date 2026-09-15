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
"""Create a mirror of the tests tree with only selected test files for metadata collection.

``openlibing-metadata-collect`` collects pytest test cases from a directory and
typically relies on the surrounding test tree (conftest.py, __init__.py, data
files). This script mirrors the full ``tests/`` tree as symlinks but keeps only
the selected test files, so pytest collection works exactly as in the real tree.
"""

import argparse
import json
import os
import sys


def resolve_project_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def is_test_file(filename: str) -> bool:
    return filename.startswith("test_") or filename.endswith("_test.py")


def collect_selected_tests(test_groups_json: str) -> set[str]:
    if not test_groups_json:
        return set()
    test_groups = json.loads(test_groups_json)
    selected = set()
    prefix = "tests" + os.sep
    for group in test_groups:
        for test_path in group.get("tests", "").split():
            norm = os.path.normpath(test_path)
            if norm.startswith(prefix):
                norm = norm[len(prefix) :]
            selected.add(norm)
    return selected


def mirror_symlink(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.lexists(dst):
        os.symlink(src, dst)


def prepare_metadata_dir(test_groups_jsons: list[str], output_dir: str) -> None:
    selected = set()
    for test_groups_json in test_groups_jsons:
        selected |= collect_selected_tests(test_groups_json)
    if not selected:
        print("::warning::test_groups is empty, no test files to collect", file=sys.stderr)
        return

    project_root = resolve_project_root()
    tests_root = os.path.join(project_root, "tests")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for root, dirs, files in os.walk(tests_root):
        dirs.sort()
        for filename in sorted(files):
            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, tests_root)

            if is_test_file(filename) and rel_path not in selected:
                continue

            dst = os.path.join(output_dir, rel_path)
            mirror_symlink(abs_path, dst)
            count += 1

    print(f"Mirrored {count} files into {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare metadata directory from selected test groups")
    parser.add_argument(
        "--test-groups",
        action="append",
        required=True,
        help="JSON array of test groups (may be repeated, may be empty)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for mirrored test files")
    args = parser.parse_args()

    prepare_metadata_dir(args.test_groups, args.output_dir)

    print(f"metadata-dir={os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
