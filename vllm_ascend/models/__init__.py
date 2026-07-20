from vllm import ModelRegistry


def register_model():
    # Apply the compatibility patch before vLLM resolves and imports its
    # hardware-isolated MiniMax-M3 model module.
    import vllm_ascend.patch_minimax_m3_model  # noqa: F401

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM",
        "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "DeepSeekV4MTPModel",
        "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP",
    )
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3",
        "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM",
    )
