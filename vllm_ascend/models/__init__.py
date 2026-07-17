from vllm import ModelRegistry


def register_model():
    import vllm_ascend.patch.worker.patch_minimax_m3  # noqa: F401

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
