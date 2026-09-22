# SPDX-License-Identifier: Apache-2.0

from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    Qwen3_5ForConditionalGenerationConfig,
)
from vllm.model_executor.models.interfaces import is_hybrid

from vllm_ascend.models import register_model
from vllm_ascend.models.qwen3_5_text import (
    AscendQwen3_5ForCausalLM,
    AscendQwen3_5MoeForCausalLM,
)


def test_qwen3_5_text_models_are_hybrid():
    assert is_hybrid(AscendQwen3_5ForCausalLM)
    assert is_hybrid(AscendQwen3_5MoeForCausalLM)


def test_qwen3_5_text_models_use_qwen_config_updater():
    register_model()
    assert MODELS_CONFIG_MAP["Qwen3_5ForCausalLM"] is Qwen3_5ForConditionalGenerationConfig
    assert MODELS_CONFIG_MAP["Qwen3_5MoeForCausalLM"] is Qwen3_5ForConditionalGenerationConfig
