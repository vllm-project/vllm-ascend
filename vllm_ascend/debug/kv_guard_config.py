# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-engine KV guard options, usable before AscendConfig initialization."""

from typing import Any, Literal

from pydantic import StrictBool

from vllm_ascend.config_utils import config


@config
class KVGuardConfig:
    """KV address diagnostics configured through additional_config."""

    enabled: StrictBool = False
    action: Literal["log", "raise"] = "log"

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "KVGuardConfig":
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        options = additional_config.get("kv_guard_config", {})
        if not isinstance(options, dict):
            raise ValueError("additional_config.kv_guard_config must be a dict")
        return cls(**options)
