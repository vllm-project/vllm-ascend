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
"""Update nightly status helper tables.

On success, updates good_table.csv with the latest passing vllm-ascend commit.
For every status (success/failure), updates env_table.csv so auto-bisect can
replay the runtime vLLM/CANN/torch-npu environment that paired with a commit.

CSV columns:
    name, yaml/path, link, status,
    vLLM Git information, vLLM-Ascend Git information, time
"""

import argparse
import csv
import os
import platform
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

ENV_HEADER = [
    "name",
    "yaml/path",
    "link",
    "status",
    "vLLM Git information",
    "vLLM-Ascend Git information",
    "CANN Version",
    "torch-npu Version",
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


def load_rows(csv_path: str, header: list[str]) -> list[list[str]]:
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # Drop the header row if present
    if rows and rows[0] == header:
        rows = rows[1:]
    return rows


def save_rows(csv_path: str, rows: list[list[str]], header: list[str]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
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


def _installed_package_version(package: str) -> str:
    try:
        out = subprocess.check_output(
            ["python3", "-m", "pip", "show", package],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return ""
    for line in out.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return ""


def current_torch_npu_version() -> str:
    return _installed_package_version("torch_npu") or _installed_package_version("torch-npu") or "unknown"


def current_cann_version() -> str:
    env = os.getenv("CANN_VERSION")
    if env:
        return env.strip()
    ascend_home = os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    machine = platform.machine()
    candidates = [
        os.path.join(ascend_home, f"{machine}-linux", "ascend_toolkit_install.info"),
        os.path.join(ascend_home, "ascend_toolkit_install.info"),
    ]
    for info_file in candidates:
        if not os.path.isfile(info_file):
            continue
        with open(info_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("version="):
                    return line.split("=", 1)[1].strip().strip('"')
    return "unknown"


def update_env_table(env_csv: str, row: list[str]) -> None:
    rows = load_rows(env_csv, ENV_HEADER)
    test_name = row[0]
    vllm_ascend_hash = row[5]
    rows = [r for r in rows if not (len(r) > 5 and r[0] == test_name and r[5] == vllm_ascend_hash)]
    rows.append(row)
    save_rows(env_csv, rows, ENV_HEADER)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update nightly good/env tables")
    parser.add_argument("--cache-csv", required=True)
    parser.add_argument("--env-table", default="")
    parser.add_argument("--status", default="success")
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--config-base-path", default="")
    parser.add_argument("--scene", default="single_node", choices=["single_node", "multi_node"])
    parser.add_argument("--run-link", required=True)
    parser.add_argument("--vllm-dir", default="/vllm-workspace/vllm")
    parser.add_argument("--vllm-ascend-dir", default="/vllm-workspace/vllm-ascend")
    parser.add_argument("--vllm-ascend-version", default="")
    parser.add_argument("--vllm-version", default="")
    parser.add_argument("--cann-version", default="")
    parser.add_argument("--torch-npu-version", default="")
    args = parser.parse_args()

    vllm_hash = args.vllm_version.strip() or git_head(args.vllm_dir)
    vllm_ascend_hash = args.vllm_ascend_version.strip() or git_head(args.vllm_ascend_dir)
    cann_version = args.cann_version.strip() or current_cann_version()
    torch_npu_version = args.torch_npu_version.strip() or current_torch_npu_version()
    timestamp = current_timestamp()
    test_path = resolve_test_path(args.test_path, args.config_base_path, args.scene, args.vllm_ascend_dir)

    status = args.status.strip() or "success"
    env_table = args.env_table.strip() or os.path.join(os.path.dirname(args.cache_csv), "env_table.csv")
    env_row = [
        args.test_name,
        test_path,
        args.run_link,
        status,
        vllm_hash,
        vllm_ascend_hash,
        cann_version,
        torch_npu_version,
        timestamp,
    ]
    update_env_table(env_table, env_row)
    print(
        f">>> Updated {env_table}: name={args.test_name} status={status} "
        f"vllm={vllm_hash} cann={cann_version} torch-npu={torch_npu_version}"
    )

    if status.lower() != "success":
        return

    new_row = [
        args.test_name,
        test_path,
        args.run_link,
        status,
        vllm_hash,
        vllm_ascend_hash,
        timestamp,
    ]

    is_new = not os.path.isfile(args.cache_csv)
    rows = load_rows(args.cache_csv, HEADER)
    rows = [r for r in rows if r and r[0] != args.test_name]
    rows.append(new_row)
    save_rows(args.cache_csv, rows, HEADER)

    action = "Created" if is_new else "Updated"
    print(f">>> {action} {args.cache_csv}: name={args.test_name} status=success time={timestamp}")


if __name__ == "__main__":
    main()
