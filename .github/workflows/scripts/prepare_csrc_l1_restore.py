#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path


def _append_values(path: Path | None, values: dict[str, str]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        for name, value in values.items():
            stream.write(f"{name}={value}\n")


def _github_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _notice(message: str, *, warning: bool = False) -> None:
    level = "warning" if warning else "notice"
    print(f"::{level}::{message}")


def _tracked_csrc_hash(source_root: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={source_root}",
            "-C",
            str(source_root),
            "ls-files",
            "-s",
            "--",
            "csrc",
            "setup.py",
            "CMakeLists.txt",
            "cmake",
        ],
        check=True,
        capture_output=True,
    )
    manifest = result.stdout
    if not manifest:
        return "adhoc"
    return hashlib.sha256(manifest).hexdigest()


def _unique_suffix() -> str:
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    return f"{run_id}-{attempt}-{time.time_ns()}-{os.getpid()}-{secrets.token_hex(4)}"


def _event_log_path(cache_dir: Path) -> Path:
    runner_temp = Path(os.environ.get("RUNNER_TEMP", str(cache_dir.parent)))
    job = os.environ.get("GITHUB_JOB", "job")
    safe_job = "".join(character if character.isalnum() or character in "_.-" else "_" for character in job)
    return runner_temp / f"csrc-l1-{safe_job}.jsonl"


def prepare(args: argparse.Namespace) -> dict[str, str]:
    cache_dir = Path(args.cache_dir).resolve()
    github_env = _github_path("GITHUB_ENV")

    _append_values(
        github_env,
        {
            "VLLM_ASCEND_BUILD_CACHE_DIR": str(cache_dir),
            "VLLM_ASCEND_BUILD_CACHE_EVENT_LOG": str(_event_log_path(cache_dir)),
        },
    )

    try:
        source_root = Path(args.source_root).resolve(strict=True)
        engine = source_root / "csrc" / "scripts" / "build_cache.py"
        if not engine.is_file():
            _notice("This source snapshot predates the fine-grained csrc cache; building without L1.")
            return {"supported": "false"}

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        csrc_hash = args.csrc_hash or _tracked_csrc_hash(source_root)
        result = subprocess.run(
            [
                "python3",
                str(engine),
                "snapshot-key",
                "--architecture",
                args.architecture,
                "--soc-version",
                args.soc_version,
                "--toolchain-image",
                args.toolchain_image,
                "--csrc-hash",
                csrc_hash,
                "--unique-suffix",
                _unique_suffix(),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        key_parts = result.stdout.splitlines()
        if len(key_parts) != 3 or not all(key_parts):
            raise ValueError("snapshot-key returned an invalid response")
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        _notice(f"Unable to prepare the L1 snapshot identity ({exc}); building without persistent L1.", warning=True)
        return {"supported": "false"}

    return {
        "supported": "true",
        "primary_key": key_parts[0],
        "csrc_hash": csrc_hash,
        "same_csrc_prefix": key_parts[1],
        "compat_prefix": key_parts[2],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fail-open persistent csrc L1 restore metadata.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--soc-version", required=True)
    parser.add_argument("--toolchain-image", default="")
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--csrc-hash", default="")
    return parser.parse_args()


def main() -> int:
    outputs = prepare(_parse_args())
    _append_values(_github_path("GITHUB_OUTPUT"), outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
