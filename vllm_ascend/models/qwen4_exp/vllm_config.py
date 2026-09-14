# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""vLLM configuration hooks for the plugin-owned Qwen4Exp model."""

from vllm.model_executor.models import config as model_config_module
from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig


def _strip_mrope(model_config: object) -> None:
    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    for config in {id(item): item for item in (hf_config, text_config) if item is not None}.values():
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_parameters.pop("mrope_section", None)
            rope_parameters.pop("mrope_interleaved", None)


class Qwen4ExpForConditionalGenerationConfig(Qwen3_5ForConditionalGenerationConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: object) -> None:
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        text_config = vllm_config.model_config.hf_text_config
        if text_config.hc_count <= 1:
            raise ValueError("Qwen4Exp requires hc_count > 1")
        parallel_config = vllm_config.parallel_config
        if (text_config.ple_layer_ids or getattr(text_config, "indexer_n_heads", None) is not None) and (
            parallel_config.enable_dbo or parallel_config.ubatch_size > 1
        ):
            raise NotImplementedError("Qwen4Exp PLE/QSA does not support DBO or microbatching")
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is not None and multimodal_config.language_model_only:
            _strip_mrope(vllm_config.model_config)


class Qwen4ExpForCausalLMConfig(Qwen4ExpForConditionalGenerationConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: object) -> None:
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        _strip_mrope(vllm_config.model_config)


class Qwen4ExpMTPConfig(Qwen4ExpForConditionalGenerationConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: object) -> None:
        Qwen4ExpForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        if not hasattr(vllm_config.model_config.hf_config, "vision_config"):
            _strip_mrope(vllm_config.model_config)


def register_qwen4_exp_vllm_config() -> None:
    model_config_module.MODELS_CONFIG_MAP.update(
        {
            "Qwen4ExpForCausalLM": Qwen4ExpForCausalLMConfig,
            "Qwen4ExpForConditionalGeneration": Qwen4ExpForConditionalGenerationConfig,
            "Qwen4ExpMTP": Qwen4ExpMTPConfig,
        }
    )
