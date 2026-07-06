#!/usr/bin/env python3
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
"""Update /root/.cache/vllm-ascend/main/nightly/good_table.csv with a
successful test entry.  Creates the file (with header) if it does not exist;
replaces the existing row for the same test name if it does.

CSV columns:
    name, yaml/path, link, status,
    vLLM Git information, vLLM-Ascend Git information, time
"""

import argparse
import csv
import os
import subprocess
from datetime import datetime, timedelta, timezone

HEADER = [
    "name",
    "yaml/path",
    "link",
    "status",
    "vLLM Git information",
    "vLLM-Ascend Git information",
    "time",
]


def git_head(repo_dir: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "N/A"


def current_timestamp() -> str:
    tz = timezone(timedelta(hours=8))
    ts = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %z")
    # Reformat +0800 → +08:00 to match existing CSV entries
    return ts[:-2] + ":" + ts[-2:]


def load_rows(csv_path: str) -> list[list[str]]:
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # Drop the header row if present
    if rows and rows[0] == HEADER:
        rows = rows[1:]
    return rows


def save_rows(csv_path: str, rows: list[list[str]]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


_DEFAULT_SINGLE_NODE_CONFIG_BASE = "tests/e2e/nightly/single_node/models/configs"
_DEFAULT_MULTI_NODE_CONFIG_BASES = (
    "tests/e2e/nightly/multi_node/internal_dp/config",
    "tests/e2e/nightly/multi_node/external_dp/config",
)


def resolve_test_path(
    test_path: str,
    config_base_path: str,
    scene: str = "single_node",
    repo_dir: str = ".",
) -> str:
    """Return the full relative path for the yaml/path CSV column.

    Upper-level workflows pass config_file_path as a bare filename
    (e.g. ``Qwen3.5-27B-w8a8-A2.yaml``).  When no directory component is
    present we prepend the config base path so the CSV matches the format
    used by the existing hand-curated good_table entries.
    """
    if os.sep in test_path or "/" in test_path:
        return test_path
    if config_base_path.strip():
        return f"{config_base_path.strip()}/{test_path}"
    if scene == "multi_node":
        for base in _DEFAULT_MULTI_NODE_CONFIG_BASES:
            if os.path.isfile(os.path.join(repo_dir, base, test_path)):
                return f"{base}/{test_path}"
        return f"{_DEFAULT_MULTI_NODE_CONFIG_BASES[0]}/{test_path}"
    return f"{_DEFAULT_SINGLE_NODE_CONFIG_BASE}/{test_path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Update good_table.csv on test success")
    parser.add_argument("--cache-csv", required=True)
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--config-base-path", default="")
    parser.add_argument("--scene", default="single_node", choices=["single_node", "multi_node"])
    parser.add_argument("--run-link", required=True)
    parser.add_argument("--vllm-dir", default="/vllm-workspace/vllm")
    parser.add_argument("--vllm-ascend-dir", default="/vllm-workspace/vllm-ascend")
    parser.add_argument("--vllm-ascend-version", default="")
    parser.add_argument("--vllm-version", default="")
    args = parser.parse_args()

    vllm_hash = args.vllm_version.strip() or git_head(args.vllm_dir)
    vllm_ascend_hash = args.vllm_ascend_version.strip() or git_head(args.vllm_ascend_dir)
    timestamp = current_timestamp()
    test_path = resolve_test_path(args.test_path, args.config_base_path, args.scene, args.vllm_ascend_dir)

    new_row = [
        args.test_name,
        test_path,
        args.run_link,
        "success",
        vllm_hash,
        vllm_ascend_hash,
        timestamp,
    ]

    is_new = not os.path.isfile(args.cache_csv)
    rows = load_rows(args.cache_csv)
    rows = [r for r in rows if r and r[0] != args.test_name]
    rows.append(new_row)
    save_rows(args.cache_csv, rows)

    action = "Created" if is_new else "Updated"
    print(f">>> {action} {args.cache_csv}: name={args.test_name} status=success time={timestamp}")


if __name__ == "__main__":
    main()
