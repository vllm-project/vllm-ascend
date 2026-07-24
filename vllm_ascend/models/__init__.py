from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForCausalLM",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForConditionalGeneration",
        "vllm_ascend.models.minimax_m3.minimax_m3_vl:MiniMaxM3SparseForConditionalGeneration",
    )
    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")
    ModelRegistry.register_model(
        "DSparkDraftModel",
        "vllm_ascend.models.deepseek_v4_dspark:DSparkDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
    ModelRegistry.register_model("Qwen3DSparkModel", "vllm_ascend.models.qwen3_dspark:AscendQwen3DSparkForCausalLM")
