# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend NPU优化的LongCat Flash模型，通过继承vLLM实现来减少代码重复。"""
from typing import Optional
import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.utils import IntermediateTensors

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
        # 不调用父类初始化，直接初始化nn.Module
        nn.Module.__init__(self)
        
        self.hidden_size = config.hidden_size
        # 从 prefix 中提取层索引，prefix 格式应该是 "model.layers.{idx}"
        try:
            self.layer_idx = int(prefix.split(".")[-1])
        except (ValueError, IndexError):
            self.layer_idx = 0  # 默认值
        
        # 为FlashConfig添加CustomDeepseekV2MLAAttention需要的属性
        if not hasattr(config, 'first_k_dense_replace'):
            config.first_k_dense_replace = 0  # LongCat Flash不使用密集层替换
        if not hasattr(config, 'moe_layer_freq'):
            config.moe_layer_freq = 1  # 默认频率
        
        # 初始化input_layernorm
        from vllm.model_executor.layers.layernorm import RMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 初始化post_attention_layernorm  
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
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
                prefix=f"model.layers.{self.layer_idx}.self_attn",
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
    
    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        kv_cache,
        attn_metadata,
        residual=None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # 使用双注意力机制
        attn_outputs = []
        for i, attn_layer in enumerate(self.self_attn):
            attn_output = attn_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
            )
            attn_outputs.append(attn_output)
        
        # 合并注意力输出
        hidden_states = attn_outputs[0] + attn_outputs[1]
        
        # 残差连接
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        
        # MLP层
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual
    
    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype, device
    ):
        return {
            "hidden_states": torch.zeros(
                (batch_size, self.hidden_size),
                dtype=dtype,
                device=device,
            ),
            "residual": torch.zeros(
                (batch_size, self.hidden_size),
                dtype=dtype,
                device=device,
            ),
        }


class CustomFlashModel(FlashModel):
    """Ascend优化的Flash模型，使用CustomFlashDecoderLayer"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # 不调用父类初始化，直接初始化nn.Module
        nn.Module.__init__(self)
        
        config = FlashConfig(**vllm_config.model_config.hf_config.__dict__)
        self.config = config
        lora_config = vllm_config.lora_config
        quant_config = vllm_config.quant_config
        
        self.lora_config = lora_config
        self.quant_config = quant_config
        
        # 初始化embed_tokens
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
        from vllm.model_executor.models.utils import maybe_prefix
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        
        # 创建layers，使用CustomFlashDecoderLayer
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
            
        # 初始化norm
        from vllm.model_executor.layers.layernorm import RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.make_empty_intermediate_tensors = (
            self.layers[0].make_empty_intermediate_tensors
            if self.layers else None)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches,
        attn_metadata,
        intermediate_tensors = None,
    ):
        if get_pp_group().is_first_rank:
            hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                hidden_states,
                attention_mask=None,
                position_ids=positions,
                kv_cache=kv_caches[i - self.start_layer],
                attn_metadata=attn_metadata,
                residual=residual,
            )
        
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })
        
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class CustomLongcatFlashForCausalLM(LongcatFlashForCausalLM):
    """Ascend优化的LongCat Flash因果语言模型"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # 不调用父类初始化，直接初始化nn.Module
        nn.Module.__init__(self)
        
        config = FlashConfig(**vllm_config.model_config.hf_config.__dict__)
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        config.intermediate_size = config.ffn_hidden_size if hasattr(
            config, "ffn_hidden_size") else config.intermediate_size
        self.lora_config = lora_config
        self.quant_config = quant_config

        # 使用我们的CustomFlashModel
        from vllm.model_executor.models.utils import maybe_prefix
        self.model = CustomFlashModel(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "model"))

        # 初始化其他组件
        from vllm.distributed import get_pp_group
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.models.utils import PPMissingLayer
        
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config,
                                          prefix=maybe_prefix(prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
