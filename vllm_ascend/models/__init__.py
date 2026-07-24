from vllm import ModelRegistry


def register_model():
    # Register Ascend-specific model implementations.
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")
    ModelRegistry.register_model(
        "MistralLarge3ForCausalLM",
        "vllm_ascend.models.mistral_large3:MistralLarge3ForCausalLM",
    )
    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTPModel")
    ModelRegistry.register_model(
        "DSparkDraftModel",
        "vllm_ascend.models.deepseek_v4_dspark:DSparkDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
    ModelRegistry.register_model("Qwen3DSparkModel", "vllm_ascend.models.qwen3_dspark:AscendQwen3DSparkModel")
