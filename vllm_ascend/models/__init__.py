from vllm import ModelRegistry


def register_model():
    # Qwen3.5 text-only checkpoints use the *ForCausalLM architecture names.
    # Register them explicitly so they are not resolved through the multimodal
    # conditional-generation wrappers, which require a vision_config.
    ModelRegistry.register_model(
        "Qwen3_5ForCausalLM",
        "vllm_ascend.models.qwen3_5_text:AscendQwen3_5ForCausalLM",
    )
    ModelRegistry.register_model(
        "Qwen3_5MoeForCausalLM",
        "vllm_ascend.models.qwen3_5_text:AscendQwen3_5MoeForCausalLM",
    )

    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        Qwen3_5ForConditionalGenerationConfig,
    )

    MODELS_CONFIG_MAP.update(
        {
            "Qwen3_5ForCausalLM": Qwen3_5ForConditionalGenerationConfig,
            "Qwen3_5MoeForCausalLM": Qwen3_5ForConditionalGenerationConfig,
        }
    )

    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
