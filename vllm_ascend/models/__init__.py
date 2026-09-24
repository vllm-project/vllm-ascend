from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model(
        "Qwen4ExpForCausalLM",
        "vllm_ascend.models.qwen4_exp:AscendQwen4ExpForCausalLM",
    )
    ModelRegistry.register_model(
        "Qwen4ExpForConditionalGeneration",
        "vllm_ascend.models.qwen4_exp:AscendQwen4ExpForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "Qwen4ExpMTP",
        "vllm_ascend.models.qwen4_exp:AscendQwen4ExpMTP",
    )
    ModelRegistry.register_model(
        "KimiK3ForCausalLM",
        "vllm_ascend.models.kimi_k3:AscendKimiK3ForCausalLM",
    )
    ModelRegistry.register_model(
        "KimiK3ForConditionalGeneration",
        "vllm_ascend.models.kimi_k3:AscendKimiK3ForConditionalGeneration",
    )
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")
    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForCausalLM",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForConditionalGeneration",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForConditionalGeneration",
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
    ModelRegistry.register_model(
        "K3DSparkModel",
        "vllm_ascend.models.kimi_k3_dspark:K3DSparkForCausalLM",
    )
