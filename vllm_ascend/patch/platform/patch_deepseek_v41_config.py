# SPDX-License-Identifier: Apache-2.0
"""Register downstream DeepSeek V4.1 config classes before model parsing."""

from vllm.config.model import ModelConfig
from vllm.transformers_utils.config import _CONFIG_REGISTRY

from vllm_ascend.deepseek_v41_config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)

_CONFIG_REGISTRY["deepseek_v4.1"] = DeepseekV41Config
_CONFIG_REGISTRY["deepseek_v4.1_text"] = DeepseekV41TextConfig
_CONFIG_REGISTRY["deepseek_v4.1_vision"] = DeepseekV41VisionConfig
_CONFIG_REGISTRY["deepseek_v41"] = DeepseekV41Config
_CONFIG_REGISTRY["deepseek_v41_text"] = DeepseekV41TextConfig
_CONFIG_REGISTRY["deepseek_v41_vision"] = DeepseekV41VisionConfig

_original_is_deepseek_mla = ModelConfig.is_deepseek_mla.fget  # type: ignore[attr-defined]


def _is_deepseek_mla(self: ModelConfig) -> bool:
    if _original_is_deepseek_mla(self):
        return True
    return getattr(self.hf_text_config, "model_type", None) in (
        "deepseek_v4.1_text",
        "deepseek_v41_text",
    )


ModelConfig.is_deepseek_mla = property(_is_deepseek_mla)  # type: ignore[method-assign,assignment]
