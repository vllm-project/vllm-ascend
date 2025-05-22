from vllm import ModelRegistry

import vllm_ascend.envs as envs


def register_model():
    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "vllm_ascend.models.qwen2_vl:AscendQwen2VLForConditionalGeneration")

    if envs.USE_OPTIMIZED_MODEL:
        ModelRegistry.register_model(
            "Qwen2_5_VLForConditionalGeneration",
            "vllm_ascend.models.qwen2_5_vl:AscendQwen2_5_VLForConditionalGeneration"
        )
    else:
        ModelRegistry.register_model(
            "Qwen2_5_VLForConditionalGeneration",
            "vllm_ascend.models.qwen2_5_vl_without_padding:AscendQwen2_5_VLForConditionalGeneration_Without_Padding"
        )

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "vllm_ascend.models.deepseek_v2:CustomDeepseekV2ForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "vllm_ascend.models.deepseek_v2:CustomDeepseekV3ForCausalLM")

    ModelRegistry.register_model(
        "DeepSeekMTPModel",
        "vllm_ascend.models.deepseek_mtp:CustomDeepSeekMTP")

    ModelRegistry.register_model("Qwen3ForCausalLM",
                                 "vllm_ascend.models.qwen3:Qwen3ForCausalLM")
