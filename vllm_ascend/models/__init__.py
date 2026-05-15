from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM",
        "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model(
        "DeepSeekV4MTPModel",
        "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")
