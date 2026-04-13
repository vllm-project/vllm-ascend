from vllm_ascend.dbo.overlap_templates.bailing_moe_v25 import (
    BailingMoEV25AllgatherTemplate,
    BailingMoEV25AlltoallTemplate,
)
from vllm_ascend.dbo.overlap_templates.base import UbatchOverlapBaseTemplate
from vllm_ascend.dbo.overlap_templates.deepseek import DeepseekAllgatherTemplate, DeepseekAlltoallTemplate
from vllm_ascend.dbo.overlap_templates.glm4_moe import Glm4MoEAllgatherTemplate, Glm4MoEAlltoallTemplate
from vllm_ascend.dbo.overlap_templates.glm_moe_dsa import GlmMoeDsaAllgatherTemplate, GlmMoeDsaAlltoallTemplate
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
    elif "GlmMoeDsaForCausalLM" in architectures:
        # GlmMoeDsa model (MLA+MoE, same architecture as DeepSeek V2/V3)
        if soc_version in {AscendDeviceType.A3}:
            return GlmMoeDsaAlltoallTemplate()
        else:
            return GlmMoeDsaAllgatherTemplate()
    elif "Qwen3MoeForCausalLM" in architectures:
        # qwen MoE model
        if soc_version in {AscendDeviceType.A3}:
            return QwenMoEAlltoallTemplate()
        else:
            return QwenMoEAllgatherTemplate()
    elif "Glm4MoeForCausalLM" in architectures:
        # GLM-4 MoE model
        if soc_version in {AscendDeviceType.A3}:
            return Glm4MoEAlltoallTemplate()
        else:
            return Glm4MoEAllgatherTemplate()
    elif "BailingMoeV2_5ForCausalLM" in architectures:
        # Bailing MoE V2.5 model (hybrid attention: MLA + Linear Attention + MoE)
        if soc_version in {AscendDeviceType.A3}:
            return BailingMoEV25AlltoallTemplate()
        else:
            return BailingMoEV25AllgatherTemplate()
    elif "Qwen3ForCausalLM" in architectures:
        # qwen dense model
        if soc_version in {AscendDeviceType.A3}:
            return QwenDenseAlltoallTemplate()
        else:
            return QwenDenseAllgatherTemplate()
    else:
        return UbatchOverlapBaseTemplate()
