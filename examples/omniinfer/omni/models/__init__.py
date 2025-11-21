# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os

from omni.adaptors.vllm.patches import model_patch
from transformers import AutoConfig
from vllm import ModelRegistry

if os.getenv("PROFILING_NAMELIST", None):
    print("<<<Profiler patch environmental variable is enabled, applying profiler patches.")
    from omni.adaptors.vllm.patches.profiler_patches import apply_profiler_patches

from omni.models.deepseek.deepseek_v32 import DeepseekV32Config

AutoConfig.register("deepseek_v32", DeepseekV32Config)


def register_model():
    is_A2 = os.getenv("ASCEND_PLATFORM", "A3")=="A2"
    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "omni.models.deepseek.deepseek_v2:CustomDeepseekV2ForCausalLM")
    if is_A2:
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
            "DeepseekV32ForCausalLM",
            "omni.models.deepseek.deepseek_v32:DeepseekV32ForCausalLM")
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
        "Qwen3ForCausalLM",
        "omni.models.qwen.qwen3:Qwen3ForCausalLM")
    
    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "omni.models.qwen.qwen3_moe:Qwen3MoeForCausalLM"
    )

    ModelRegistry.register_model(
        "PanguUltraMoEMTPModel",
        "omni.models.pangu.pangu_ultra_moe_mtp:PanguUltraMoEMTP")

    ModelRegistry.register_model(
        "PanguProMoEForCausalLM",
        "omni.models.pangu.pangu_pro_moe.pangu_moe:PanguProMoEForCausalLM")

    ModelRegistry.register_model(
        "PanguEmbeddedForCausalLM",
        "omni.models.pangu.pangu_dense:PanguEmbeddedForCausalLM")

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
        if is_A2:   
            from omni.models.deepseek.deepseek_v3_a2 import \
                DeepseekV3ForCausalLM
        else:
            from omni.models.deepseek.deepseek_v3 import DeepseekV3ForCausalLM
        ModelRegistry.register_model(
            "DeepseekV3ForCausalLM",
            mock_model_class_factory(DeepseekV3ForCausalLM))