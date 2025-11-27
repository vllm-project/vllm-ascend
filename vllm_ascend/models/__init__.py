from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "vllm_ascend.models.qwen2_vl:AscendQwen2VLForConditionalGeneration")

    ModelRegistry.register_model(
        "Qwen3VLMoeForConditionalGeneration",
        "vllm_ascend.models.qwen3_vl:AscendQwen3VLMoeForConditionalGeneration")

    ModelRegistry.register_model(
        "Qwen3VLForConditionalGeneration",
        "vllm_ascend.models.qwen3_vl:AscendQwen3VLForConditionalGeneration")

    # There is no PanguProMoEForCausalLM in vLLM, so we should register it before vLLM config initialization
    # to make sure the model can be loaded correctly. This register step can be removed once vLLM support PanguProMoEForCausalLM.
    ModelRegistry.register_model(
        "PanguProMoEForCausalLM",
        "vllm_ascend.torchair.models.torchair_pangu_moe:PanguProMoEForCausalLM"
    )

    ModelRegistry.register_model(
        "Qwen3NextForCausalLM",
        "vllm_ascend.models.qwen3_next:CustomQwen3NextForCausalLM")

    ModelRegistry.register_model(
        "Qwen3NextMTP", "vllm_ascend.models.qwen3_next_mtp:CustomQwen3NextMTP")
