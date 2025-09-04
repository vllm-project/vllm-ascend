# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend NPU优化的LongCat Flash模型，通过继承vLLM实现来减少代码重复。"""
from typing import Optional
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig

# 继承原始的vLLM LongCat Flash实现
from vllm.model_executor.models.longcat_flash import (
    FlashConfig,
    LongcatMoe,
    FlashDecoderLayer,
    FlashModel,
    LongcatFlashForCausalLM
)

# 导入Ascend特定的实现
from vllm_ascend.models.deepseek_v2 import CustomDeepseekV2MLAAttention
from vllm_ascend.ops.fused_moe import AscendFusedMoE

logger = init_logger(__name__)


class CustomLongcatMoe(LongcatMoe):
    """Ascend优化的LongcatMoe，使用AscendFusedMoE替代原始FusedMoE"""
    
    def __init__(
        self,
        config: FlashConfig,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype=None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        # 调用父类初始化，获得所有基础组件
        super().__init__(config, num_experts, top_k, hidden_size, intermediate_size, 
                         params_dtype, quant_config, prefix, enable_eplb)
        
        # 只需要重写一个关键组件：使用AscendFusedMoE替代原始FusedMoE
        self.experts = AscendFusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=True,
            params_dtype=params_dtype,
            e_score_correction_bias=self.router.e_score_correction_bias,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            zero_expert_num=self.zero_expert_num,
            zero_expert_type=self.zero_expert_type,
            enable_eplb=self.enable_eplb,
            routed_scaling_factor=config.routed_scaling_factor,
        )


class CustomFlashDecoderLayer(FlashDecoderLayer):
    """Ascend优化的Flash decoder layer，使用CustomDeepseekV2MLAAttention"""

    def __init__(
        self,
        config: FlashConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        # 调用父类初始化，获得所有基础组件
        super().__init__(config, cache_config, quant_config, prefix, enable_eplb)
        
        # 只需要重写两个关键组件：self_attn 和 mlp
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)

        # 关键修改：使用CustomDeepseekV2MLAAttention替代DeepseekV2MLAAttention
        self.self_attn = nn.ModuleList([
            CustomDeepseekV2MLAAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(config.q_lora_rank if hasattr(
                    config, "q_lora_rank") else None),
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=None if "self_attn" in getattr(
                    config, "disable_quant_module", []) else quant_config,
                prefix=f"{prefix}.self_attn.{i}",
            ) for i in range(2)
        ])

        # 使用CustomLongcatMoe替代原始LongcatMoe
        self.mlp = CustomLongcatMoe(
            config=config,
            num_experts=config.n_routed_experts if hasattr(
                config, "n_routed_experts") else
            config.num_experts[self.layer_idx],
            top_k=config.moe_topk
            if hasattr(config, "moe_topk") else config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=(f"{prefix}.mlp"),
        )


class CustomFlashModel(FlashModel):
    """Ascend优化的Flash模型，使用CustomFlashDecoderLayer"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # 调用父类初始化，获得所有基础组件
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        # 只需要重新创建layers，使用CustomFlashDecoderLayer替代原始FlashDecoderLayer
        from vllm.model_executor.models.utils import make_layers
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: CustomFlashDecoderLayer(
                self.config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers")


class CustomLongcatFlashForCausalLM(LongcatFlashForCausalLM):
    """Ascend优化的LongCat Flash因果语言模型"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # 调用父类初始化，但需要替换model为AscendFlashModel
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        # 只需要替换model为Ascend优化版本
        from vllm.model_executor.models.utils import maybe_prefix
        self.model = CustomFlashModel(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "model"))
        
        # 重新设置make_empty_intermediate_tensors以使用新的model
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
