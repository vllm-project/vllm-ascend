from vllm import ModelRegistry

import vllm_ascend.envs as envs_ascend


def register_model():
    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "vllm_ascend.models.qwen2_vl:AscendQwen2VLForConditionalGeneration")

    ModelRegistry.register_model(
        "Qwen3VLMoeForConditionalGeneration",
        "vllm_ascend.models.qwen2_5_vl_without_padding:AscendQwen3VLMoeForConditionalGeneration"
    )

    ModelRegistry.register_model(
        "Qwen3VLForConditionalGeneration",
        "vllm_ascend.models.qwen2_5_vl_without_padding:AscendQwen3VLForConditionalGeneration"
    )

    if envs_ascend.USE_OPTIMIZED_MODEL:
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

    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "vllm_ascend.models.qwen3_moe:CustomQwen3MoeForCausalLM")

    # There is no PanguProMoEForCausalLM in vLLM, so we should register it before vLLM config initialization
    # to make sure the model can be loaded correctly. This register step can be removed once vLLM support PanguProMoEForCausalLM.
    ModelRegistry.register_model(
        "PanguProMoEForCausalLM",
        "vllm_ascend.torchair.models.torchair_pangu_moe:PanguProMoEForCausalLM"
    )
    ModelRegistry.register_model(
        "Qwen3NextForCausalLM",
        "vllm_ascend.models.qwen3_next:CustomQwen3NextForCausalLM")
