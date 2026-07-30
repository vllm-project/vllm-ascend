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
"""Update frequency-specific nightly status helper tables.

On success, updates good_table.csv using the stable
``soc + scene + yaml/path`` key. For every status (success/failure), updates
env_table.csv so auto-bisect can replay the matching vLLM/CANN/torch-npu
environment. Updates use file locks and atomic renames because matrix jobs
share the same frequency-specific tables.

CSV columns:
    name, yaml/path, link, status,
    vLLM Git information, vLLM-Ascend Git information, soc, scene, time
"""

import argparse
import csv
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

HEADER = [
    "name",
    "yaml/path",
    "link",
    "status",
    "vLLM Git information",
    "vLLM-Ascend Git information",
    "soc",
    "scene",
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


def load_rows(csv_path: str) -> list[dict[str, str]]:
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{column: row.get(column, "") or "" for column in HEADER} for row in reader]


def save_rows(csv_path: str, rows: list[dict[str, str]]) -> None:
    table_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(table_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".good_table.", suffix=".tmp", dir=table_dir)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, csv_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


@contextmanager
def table_lock(csv_path: str) -> Iterator[None]:
    """Hold an advisory cross-process lock while updating ``csv_path``."""
    lock_path = f"{csv_path}.lock"
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_env_rows(csv_path: str) -> list[list[str]]:
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # Drop the header row if present
    if rows and rows[0] == ENV_HEADER:
        rows = rows[1:]
    return rows


def save_env_rows(csv_path: str, rows: list[list[str]]) -> None:
    table_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(table_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".env_table.", suffix=".tmp", dir=table_dir)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ENV_HEADER)
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, csv_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


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


def _normalise_path(path: str) -> str:
    return path.strip().replace("\\", "/").rstrip("/")


def _same_case(row: dict[str, str], *, soc: str, scene: str, test_path: str) -> bool:
    row_path = _normalise_path(row.get("yaml/path", ""))
    target_path = _normalise_path(test_path)
    if row_path != target_path:
        return False

    # A legacy seven-column row has no soc/scene. Replace it when its path
    # matches so the first update migrates it to the new schema.
    row_soc = row.get("soc", "").strip()
    row_scene = row.get("scene", "").strip()
    if not row_soc and not row_scene:
        return True
    return row_soc == soc and row_scene == scene


def update_table(csv_path: str, new_row: dict[str, str]) -> bool:
    with table_lock(csv_path):
        is_new = not os.path.isfile(csv_path)
        rows = load_rows(csv_path)
        rows = [
            row
            for row in rows
            if not _same_case(
                row,
                soc=new_row["soc"],
                scene=new_row["scene"],
                test_path=new_row["yaml/path"],
            )
        ]
        rows.append(new_row)
        save_rows(csv_path, rows)
        return is_new


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
    with table_lock(env_csv):
        rows = load_env_rows(env_csv)
        test_name = row[0]
        vllm_ascend_hash = row[5]
        rows = [r for r in rows if not (len(r) > 5 and r[0] == test_name and r[5] == vllm_ascend_hash)]
        rows.append(row)
        save_env_rows(env_csv, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update nightly good/env tables")
    parser.add_argument("--cache-csv", required=True)
    parser.add_argument("--env-table", default="")
    parser.add_argument("--status", default="success")
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--config-base-path", default="")
    parser.add_argument("--scene", default="single_node", choices=["single_node", "multi_node"])
    parser.add_argument("--soc", default="")
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

    new_row = {
        "name": args.test_name,
        "yaml/path": test_path,
        "link": args.run_link,
        "status": status,
        "vLLM Git information": vllm_hash,
        "vLLM-Ascend Git information": vllm_ascend_hash,
        "soc": args.soc.strip(),
        "scene": args.scene,
        "time": timestamp,
    }

    is_new = update_table(args.cache_csv, new_row)

    action = "Created" if is_new else "Updated"
    print(f">>> {action} {args.cache_csv}: name={args.test_name} status=success time={timestamp}")


if __name__ == "__main__":
    main()
