# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Attach an Engram config to vLLM v0.29, which has no native support."""

import os

from vllm.config import VllmConfig
from vllm.config import utils as config_utils


def _resolve_fallback_engram_config(vllm_config):
    from vllm_ascend.compat.deepseek_v41.engram_config import EngramConfig

    additional = vllm_config.additional_config
    raw_config = additional.get("engram_config") if isinstance(additional, dict) else None
    legacy_enabled = os.getenv("VLLM_PLE_CPU_OFFLOAD", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if raw_config is None and not legacy_enabled:
        return None
    if raw_config is None:
        config = EngramConfig()
    elif isinstance(raw_config, EngramConfig):
        config = raw_config
    elif isinstance(raw_config, dict):
        config = EngramConfig(**raw_config)
    else:
        raise TypeError("additional_config['engram_config'] must be a mapping or EngramConfig instance.")

    model_config = vllm_config.model_config
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None and model_config is speculative_config.draft_model_config:
        model_config = speculative_config.target_model_config
    config.verify_model_config(model_config)
    config.verify_parallel_config(vllm_config.parallel_config)
    return config


_original_post_init = VllmConfig.__post_init__

if not getattr(_original_post_init, "_vllm_ascend_engram", False):

    def _post_init_with_engram(self) -> None:
        _original_post_init(self)
        object.__setattr__(
            self,
            "engram_config",
            _resolve_fallback_engram_config(self),
        )

    _post_init_with_engram._vllm_ascend_engram = True  # type: ignore[attr-defined]
    VllmConfig.__post_init__ = _post_init_with_engram


_original_is_init_field = config_utils.is_init_field

if not getattr(_original_is_init_field, "_vllm_ascend_engram", False):

    def _is_init_field_with_engram(cls, name: str) -> bool:
        # vLLM 0.29's replace() walks __dict__ and assumes every key is a
        # dataclass field. Engram is attached after validation, so omit it;
        # additional_config recreates it on the replacement object.
        if cls is VllmConfig and name == "engram_config":
            return False
        return _original_is_init_field(cls, name)

    _is_init_field_with_engram._vllm_ascend_engram = True  # type: ignore[attr-defined]
    config_utils.is_init_field = _is_init_field_with_engram
