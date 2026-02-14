from vllm_ascend.dbo.overlap_templates.base import UbatchOverlapBaseTemplate
from vllm_ascend.dbo.overlap_templates.deepseek import DeepseekAllgatherTemplate, DeepseekAlltoallTemplate
from vllm_ascend.dbo.overlap_templates.qwen3_dense import QwenDenseAllgatherTemplate, QwenDenseAlltoallTemplate
from vllm_ascend.dbo.overlap_templates.qwen3_moe import QwenMoEAllgatherTemplate, QwenMoEAlltoallTemplate
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def select_dbo_templates(vllm_config):
    # select according to model name
    model_config = vllm_config.model_config
    architectures = getattr(model_config.hf_config, "architectures", [])

    soc_version = get_ascend_device_type()
    if "DeepseekV2ForCausalLM" in architectures or "DeepseekV3ForCausalLM" in architectures:
        if soc_version in {AscendDeviceType.A3}:
            return DeepseekAlltoallTemplate()
        else:
            return DeepseekAllgatherTemplate()
    elif "Qwen3MoeForCausalLM" in architectures:
        # qwen MoE model
        if soc_version in {AscendDeviceType.A3}:
            return QwenMoEAlltoallTemplate()
        else:
            return QwenMoEAllgatherTemplate()
    elif "Qwen3ForCausalLM" in architectures:
        # qwen dense model
        if soc_version in {AscendDeviceType.A3}:
            return QwenDenseAlltoallTemplate()
        else:
            return QwenDenseAllgatherTemplate()
    else:
        return UbatchOverlapBaseTemplate()
