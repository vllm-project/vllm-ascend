#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Runtime control plane (``runtime_config.json``).

Package façade: :class:`RuntimeConfig`, path helpers, and
``additional_config`` bootstrap via :func:`build_runtime_config_from_additional`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_ascend.observability.runtime_config.config import (
    RuntimeConfig,
    resolve_runtime_config_path,
    resolve_runtime_report_dir,
)

# Keys consumed here; must be stripped before AscendConfig pydantic construction.
ADDITIONAL_CONFIG_STRIP_KEYS: frozenset[str] = frozenset(
    {
        "runtime_config",
        "runtime_config_path",
        "runtime-config",
        "runtime_config_hot_reload",
        # Retired: interval is fixed at HOT_RELOAD_INTERVAL_SECONDS when enabled.
        "runtime_config_reload_interval",
        "runtime_report_dir",
        "runtime_dump_dir",
    }
)


def _parse_hot_reload(raw: Any) -> bool:
    """Coerce ``runtime_config_hot_reload`` to bool (default false)."""
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    # Accept 0/1 ints from JSON-ish configs; reject other numbers (old interval).
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    raise ValueError(
        "additional_config.runtime_config_hot_reload must be a bool "
        f"(true enables hot-reload at a fixed 3s interval), got {raw!r}."
    )


@dataclass(frozen=True)
class RuntimeConfigBootstrap:
    """AscendConfig fields derived from ``additional_config`` runtime_* keys."""

    path: str | None
    hot_reload: bool
    runtime_config: RuntimeConfig


def build_runtime_config_from_additional(additional_config: dict[str, Any]) -> RuntimeConfigBootstrap:
    """Validate runtime_* keys, construct :class:`RuntimeConfig`, start non-worker reload."""
    if "runtime_config_reload_interval" in additional_config:
        raise ValueError(
            "additional_config.runtime_config_reload_interval is retired; "
            "use runtime_config_hot_reload=true|false (fixed 3s poll when enabled)."
        )

    raw_path = additional_config.get("runtime_config_path") or additional_config.get("runtime-config")
    if raw_path is not None and not isinstance(raw_path, str):
        raise ValueError(f"additional_config.runtime_config_path must be a string, got {type(raw_path).__name__}.")

    hot_reload = _parse_hot_reload(additional_config.get("runtime_config_hot_reload"))

    raw_overlay = additional_config.get("runtime_config")
    if raw_overlay is not None and not isinstance(raw_overlay, dict):
        raise ValueError(f"additional_config.runtime_config must be a dict, got {type(raw_overlay).__name__}.")

    raw_report_dir = additional_config.get("runtime_report_dir")
    if raw_report_dir is not None and not isinstance(raw_report_dir, str):
        raise ValueError(f"additional_config.runtime_report_dir must be a string, got {type(raw_report_dir).__name__}.")

    raw_dump_dir = additional_config.get("runtime_dump_dir")
    if raw_dump_dir is not None and not isinstance(raw_dump_dir, str):
        raise ValueError(f"additional_config.runtime_dump_dir must be a string, got {type(raw_dump_dir).__name__}.")

    runtime_cfg = RuntimeConfig(
        raw_path,
        report_dir=raw_report_dir,
        hot_reload=hot_reload,
        ensure_file=False,
        dump_dir=raw_dump_dir,
        startup_overlay=raw_overlay,
    )
    runtime_cfg.apply_ascend_log_level()
    runtime_cfg.start_non_worker_background_reload()
    return RuntimeConfigBootstrap(
        path=raw_path,
        hot_reload=hot_reload,
        runtime_config=runtime_cfg,
    )


__all__ = [
    "ADDITIONAL_CONFIG_STRIP_KEYS",
    "RuntimeConfig",
    "RuntimeConfigBootstrap",
    "build_runtime_config_from_additional",
    "resolve_runtime_config_path",
    "resolve_runtime_report_dir",
]
