# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-engine KV Trace options, usable before AscendConfig initialization."""

from typing import Any, Literal

from pydantic import StrictBool

from vllm_ascend.config_utils import config


@config
class KVTraceConfig:
    """KV address diagnostics configured through additional_config."""

    enabled: StrictBool = False
    action: Literal["log", "raise"] = "log"

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "KVTraceConfig":
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        options = additional_config.get("kv_trace", {})
        if not isinstance(options, dict):
            raise ValueError("additional_config.kv_trace must be a dict")
        return cls(**options)
