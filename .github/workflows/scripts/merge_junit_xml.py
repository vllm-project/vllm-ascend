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
"""Merge multiple JUnit XML files from parallel test runs into a single file."""

import argparse
import os
import sys
from pathlib import Path

from junitparser import JUnitXml


def find_xml_files(input_dir: str) -> list[Path]:
    return sorted(Path(input_dir).rglob("*.xml"))


def merge_junit_xmls(input_dir: str, output: str) -> None:
    xml_files = find_xml_files(input_dir)
    if not xml_files:
        print(f"::warning::No JUnit XML files found under {input_dir}", file=sys.stderr)
        return

    merged = JUnitXml()
    for f in xml_files:
        try:
            suite = JUnitXml.fromfile(str(f))
            merged += suite
        except Exception as e:
            print(f"::warning::Failed to parse {f}: {e}", file=sys.stderr)

    merged.write(output)
    print(f"Merged {len(xml_files)} JUnit XML files into {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge JUnit XML files")
    parser.add_argument("--input-dir", required=True, help="Directory containing JUnit XML files")
    parser.add_argument("--output", required=True, help="Output merged JUnit XML file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    merge_junit_xmls(args.input_dir, args.output)


if __name__ == "__main__":
    main()
