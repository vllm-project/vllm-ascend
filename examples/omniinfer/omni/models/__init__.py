# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm import ModelRegistry
import os
from omni.models.common.config.model_config import model_extra_config

import os
if os.getenv("PROFILING_NAMELIST", None):
    print("<<<Profiler patch environmental variable is enabled, applying profiler patches.")
    from omni.tools.profiler import apply_profiler_patches


def register_model():
    is_A2 = os.getenv("ASCEND_PLATFORM", "A3")=="A2"
    all2all = model_extra_config.operator_opt_config.prefill_moe_all_to_all
    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "omni.models.deepseek.deepseek_v2:CustomDeepseekV2ForCausalLM")
    if is_A2 and not all2all:
        ModelRegistry.register_model(
            "DeepseekV3ForCausalLM",
            "omni.models.deepseek.deepseek_v3_a2:DeepseekV3ForCausalLM")
        ModelRegistry.register_model(
            "PanguUltraMoEForCausalLM",
            "omni.models.pangu.pangu_ultra_moe_a2:PanguUltraMoEForCausalLM")
    else:
        ModelRegistry.register_model(
            "DeepseekV3ForCausalLM",
            "omni.models.deepseek.deepseek_v3:DeepseekV3ForCausalLM")
        ModelRegistry.register_model(
            "PanguUltraMoEForCausalLM",
            "omni.models.pangu.pangu_ultra_moe:PanguUltraMoEForCausalLM")

    ModelRegistry.register_model(
        "DeepSeekMTPModel",
        "omni.models.deepseek.deepseek_mtp:DeepseekV3MTP")
    
    ModelRegistry.register_model(
        "DeepSeekMTPModelDuo",
        "omni.models.deepseek.deepseek_mtp:DeepseekV3MTPDuo")

    ModelRegistry.register_model(
        "DeepSeekMTPModelTres",
        "omni.models.deepseek.deepseek_mtp:DeepseekV3MTPTres")

    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "omni.models.qwen.qwen2:Qwen2ForCausalLM")
    
    ModelRegistry.register_model(
        "EagleQwen2ForCausalLM",
        "omni.models.qwen.qwen2_eagle:EagleQwen2ForCausalLM")

    ModelRegistry.register_model(
        "Eagle3Qwen2ForCausalLM",
        "omni.models.qwen.qwen2_eagle3:Eagle3Qwen2ForCausalLM")

    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "omni.models.qwen.qwen3:Qwen3ForCausalLM")

    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "omni.models.qwen.qwen3_moe:Qwen3MoeForCausalLM"
    )

    ModelRegistry.register_model(
        "LlamaForCausalLM",
        "omni.models.llama.llama:LlamaForCausalLM")

    ModelRegistry.register_model(
        "Qwen2_5_VLForConditionalGeneration",
        "omni.models.qwen.qwen2_5_vl:Qwen2_5_VLForConditionalGeneration")

    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "omni.models.qwen.qwen2_vl:Qwen2VLForConditionalGeneration")

    ModelRegistry.register_model(
        "PanguUltraMoEMTPModel",
        "omni.models.pangu.pangu_ultra_moe_mtp:PanguUltraMoEMTP")

    ModelRegistry.register_model(
        "PanguProMoEForCausalLM",
        "omni.models.pangu.pangu_pro_moe.pangu_moe:PanguProMoEForCausalLM")

    ModelRegistry.register_model(
        "PanguEmbeddedForCausalLM",
        "omni.models.pangu.pangu_dense:PanguEmbeddedForCausalLM")

    ModelRegistry.register_model(
        "InternLM2ForCausalLM",
        "omni.models.internvl.internlm2:InternLM2ForCausalLM")
    
    ModelRegistry.register_model(
        "InternVLChatModel",
        "omni.models.internvl.internvl:InternVLChatModel")
    
    ModelRegistry.register_model(
        "Gemma3ForCausalLM",
        "omni.models.gemma.gemma3:Gemma3ForCausalLM")

    ModelRegistry.register_model(
        "Gemma3ForConditionalGeneration",
        "omni.models.gemma.gemma3_mm:Gemma3ForConditionalGeneration")


    if (
        int(os.getenv("RANDOM_MODE", default='0')) or
        int(os.getenv("CAPTURE_MODE", default='0')) or
        int(os.getenv("REPLAY_MODE", default='0'))
    ):
        from omni.models.mock.mock import mock_model_class_factory

        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
        ModelRegistry.register_model(
            "Qwen2ForCausalLM",
            mock_model_class_factory(Qwen2ForCausalLM))
        if is_A2 and not all2all:
            from omni.models.deepseek.deepseek_v3_a2 import DeepseekV3ForCausalLM
        else:
            from omni.models.deepseek.deepseek_v3 import DeepseekV3ForCausalLM
        ModelRegistry.register_model(
            "DeepseekV3ForCausalLM",
            mock_model_class_factory(DeepseekV3ForCausalLM))