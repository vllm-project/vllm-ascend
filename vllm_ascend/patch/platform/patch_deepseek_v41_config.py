# SPDX-License-Identifier: Apache-2.0
"""Recognize legacy V4.1 model types using the pinned upstream configuration."""

from vllm.config.model import ModelConfig
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

from vllm_ascend.config_utils import is_deepseek_v41, normalize_deepseek_v41_config

# The canonical deepseek_v41 entry remains owned by upstream vLLM.
_CONFIG_REGISTRY["deepseek_v4.1"] = DeepseekV41Config
_original_is_deepseek_mla = ModelConfig.is_deepseek_mla.fget


def _is_deepseek_mla(self: ModelConfig) -> bool:
    if is_deepseek_v41(self.hf_config):
        normalize_deepseek_v41_config(self.hf_config)
        return True
    return _original_is_deepseek_mla(self)


ModelConfig.is_deepseek_mla = property(_is_deepseek_mla)
