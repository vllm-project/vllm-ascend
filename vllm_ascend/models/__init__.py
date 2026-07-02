from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")

    # DSpark (paper arxiv:2606.19348) draft architecture. hf_config_override
    # rewrites the DSpark checkpoint's architecture to this name, so it must be
    # in the registry before SpeculativeConfig validates the draft ModelConfig.
    ModelRegistry.register_model(
        "DSparkDeepseekV4ForCausalLM",
        "vllm_ascend.models.deepseek_v4_dspark:DSparkDeepseekV4ForCausalLM",
    )
