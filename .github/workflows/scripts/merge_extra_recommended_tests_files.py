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
"""Merge incoming extra pytest targets into a YAML-based extra test list."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge incoming extra pytest targets into a YAML-based extra test list.",
    )
    parser.add_argument("--extra-yaml", type=Path, required=True, help="Existing YAML file to update")
    parser.add_argument("--incoming-yaml", type=Path, required=True, help="YAML file whose paths are appended")
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=None,
        help="Where to write the merged yaml (default: --extra-yaml)",
    )
    args = parser.parse_args(argv)

    output_yaml = args.output_yaml or args.extra_yaml
    added = merge_extra_yaml_files(args.extra_yaml, args.incoming_yaml, output_yaml)
    if not added:
        print(f"No new paths to append; wrote {output_yaml}")
        return 0
    print(f"Appended {len(added)} path(s) to {output_yaml}:")
    for target in added:
        print(f"  {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
