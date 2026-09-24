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

"""Path resolution for runtime_config JSON and report/dump roots."""

from __future__ import annotations

import os
from pathlib import Path

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

DEFAULT_CONFIG_FILENAME = "runtime_config.json"


def default_runtime_root() -> Path:
    """Execution-directory runtime root: ``<cwd>/runtime``."""
    return Path(os.getcwd()) / "runtime"


def default_config_dir() -> Path:
    return default_runtime_root() / "config"


def _reject_unsafe_path(path: Path, *, label: str) -> Path:
    """Resolve and reject NUL / empty paths (basic path hygiene)."""
    raw = str(path)
    if not raw or "\x00" in raw:
        raise ValueError(f"invalid {label}: empty or contains NUL")
    resolved = path.expanduser().resolve()
    # Soft sandbox: warn when outside cwd (shared NFS paths are common).
    try:
        cwd = Path.cwd().resolve()
        if resolved != cwd and cwd not in resolved.parents:
            logger.warning(
                "[runtime_config] %s is outside process cwd (%s): %s",
                label,
                cwd,
                resolved,
            )
    except Exception:
        pass
    return resolved


def resolve_runtime_config_path(configured_path: str | None = None) -> Path:
    """Resolve config file path.

    Priority:
    1. Explicit ``runtime_config_path`` / ``runtime-config`` from additional_config
    2. Default ``<cwd>/runtime/config/runtime_config.json``
    """
    if configured_path:
        return _reject_unsafe_path(Path(configured_path), label="runtime_config_path")
    return _reject_unsafe_path(default_config_dir() / DEFAULT_CONFIG_FILENAME, label="runtime_config_path")


def resolve_runtime_report_dir(config_path: Path, configured_report_dir: str | None = None) -> Path:
    if configured_report_dir:
        return _reject_unsafe_path(Path(configured_report_dir), label="runtime_report_dir")
    runtime_root = config_path.parent.parent if config_path.parent.name == "config" else config_path.parent
    return _reject_unsafe_path(runtime_root / "report", label="runtime_report_dir")
