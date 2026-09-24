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

"""Parse ``additional_config`` runtime_* keys and build :class:`RuntimeConfig`.

Called from :func:`vllm_ascend.ascend_config.init_ascend_config` so AscendConfig
stays free of runtime_guard / JSON control-plane details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_ascend.runtime_config.config import RuntimeConfig

# Keys consumed here; must be stripped before AscendConfig pydantic construction.
ADDITIONAL_CONFIG_STRIP_KEYS: frozenset[str] = frozenset(
    {
        "runtime_config",
        "runtime_config_path",
        "runtime-config",
        "runtime_config_reload_interval",
        "runtime_report_dir",
        "runtime_dump_dir",
    }
)


@dataclass(frozen=True)
class RuntimeConfigBootstrap:
    """AscendConfig fields derived from ``additional_config`` runtime_* keys."""

    path: str | None
    reload_interval_seconds: float
    runtime_config: RuntimeConfig


def build_runtime_config_from_additional(additional_config: dict[str, Any]) -> RuntimeConfigBootstrap:
    """Validate runtime_* keys, construct :class:`RuntimeConfig`, start non-worker reload."""
    raw_path = additional_config.get("runtime_config_path") or additional_config.get("runtime-config")
    if raw_path is not None and not isinstance(raw_path, str):
        raise ValueError(
            f"additional_config.runtime_config_path must be a string, got {type(raw_path).__name__}."
        )

    raw_reload = additional_config.get("runtime_config_reload_interval")
    if raw_reload is None:
        raw_reload = 0
    try:
        reload_interval_seconds = float(raw_reload)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "additional_config.runtime_config_reload_interval must be a number of seconds "
            f"(0 disables hot-reload; default 0), got {raw_reload!r}."
        ) from exc
    if reload_interval_seconds < 0:
        raise ValueError(
            f"additional_config.runtime_config_reload_interval must be >= 0, got {reload_interval_seconds}."
        )

    raw_overlay = additional_config.get("runtime_config")
    if raw_overlay is not None and not isinstance(raw_overlay, dict):
        raise ValueError(
            f"additional_config.runtime_config must be a dict, got {type(raw_overlay).__name__}."
        )

    raw_report_dir = additional_config.get("runtime_report_dir")
    if raw_report_dir is not None and not isinstance(raw_report_dir, str):
        raise ValueError(
            f"additional_config.runtime_report_dir must be a string, got {type(raw_report_dir).__name__}."
        )

    raw_dump_dir = additional_config.get("runtime_dump_dir")
    if raw_dump_dir is not None and not isinstance(raw_dump_dir, str):
        raise ValueError(
            f"additional_config.runtime_dump_dir must be a string, got {type(raw_dump_dir).__name__}."
        )

    runtime_cfg = RuntimeConfig(
        raw_path,
        report_dir=raw_report_dir,
        reload_interval_seconds=reload_interval_seconds,
        ensure_file=False,
        dump_dir=raw_dump_dir,
        startup_overlay=raw_overlay,
    )
    runtime_cfg.apply_ascend_log_level()
    runtime_cfg.start_non_worker_background_reload()
    return RuntimeConfigBootstrap(
        path=raw_path,
        reload_interval_seconds=reload_interval_seconds,
        runtime_config=runtime_cfg,
    )
