from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model(
        "KimiLinearForCausalLM",
        "vllm_ascend.models.kimi_k3:AscendKimiLinearForCausalLM",
    )
    # Keep the release-branch text architecture as a compatibility alias for
    # checkpoints whose config predates vLLM's KimiLinear rename.
    ModelRegistry.register_model(
        "KimiK3ForCausalLM",
        "vllm_ascend.models.kimi_k3:AscendKimiLinearForCausalLM",
    )
    ModelRegistry.register_model(
        "KimiK3ForConditionalGeneration",
        "vllm_ascend.models.kimi_k3:AscendKimiK3ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "KimiK3MTPModel",
        "vllm_ascend.models.kimi_k3_mtp:AscendKimiK3MTP",
    )
    ModelRegistry.register_model(
        "K3DSparkModel",
        "vllm_ascend.models.kimi_k3_dspark:AscendK3DSparkForCausalLM",
    )
    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4.model:AscendDeepseekV4ForCausalLM"
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForCausalLM",
    )
    ModelRegistry.register_model(
        "MiniMaxM3SparseForConditionalGeneration",
        "vllm_ascend.models.minimax_m3:MiniMaxM3SparseForConditionalGeneration",
    )
    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4.mtp:DeepSeekV4MTP")
    ModelRegistry.register_model(
        "DSparkDraftModel",
        "vllm_ascend.models.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
    ModelRegistry.register_model("Qwen3DSparkModel", "vllm_ascend.models.qwen3_dspark:AscendQwen3DSparkForCausalLM")
    ModelRegistry.register_model(
        "DFlash2DraftModel",
        "vllm_ascend.models.qwen3_dflash2:DFlash2Qwen3ForCausalLM",
    )
    ModelRegistry.register_model("DeepSeekMTPModel", "vllm_ascend.models.deepseek_mtp:AscendDeepSeekMTP")
    ModelRegistry.register_model("GlmMoeDsaForCausalLM", "vllm_ascend.models.deepseek_mtp:AscendGlmMoeDsaForCausalLM")
    ModelRegistry.register_model(
        "Eagle3LlamaForCausalLM", "vllm_ascend.models.llama_eagle3:AscendEagle3LlamaForCausalLM"
    )
