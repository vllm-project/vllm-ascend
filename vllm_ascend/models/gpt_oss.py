# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 OpenAI and the vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only GPT-OSS model on Ascend NPU."""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer, is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.sequence import IntermediateTensors

from vllm_ascend.ops.fused_moe import AscendFusedMoE
from vllm_ascend.utils import dispose_tensor


class GPTOSSConfig(PretrainedConfig):
    """GPT-OSS model configuration."""
    
    model_type = "gpt_oss"
    
    def __init__(
        self,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        num_experts: int = 128,
        experts_per_token: int = 4,
        sliding_window: int = 128,
        initial_context_length: int = 4096,
        rope_theta: float = 150000.0,
        rope_scaling_factor: float = 32.0,
        rope_ntk_alpha: float = 1.0,
        rope_ntk_beta: float = 32.0,
        swiglu_limit: float = 7.0,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.sliding_window = sliding_window
        self.initial_context_length = initial_context_length
        self.rope_theta = rope_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_ntk_alpha = rope_ntk_alpha
        self.rope_ntk_beta = rope_ntk_beta
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        
        super().__init__(**kwargs)


class GPTOSSAttention(nn.Module):
    """GPT-OSS attention layer with sliding window support."""
    
    def __init__(
        self,
        config: GPTOSSConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // get_tensor_model_parallel_world_size()
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // get_tensor_model_parallel_world_size())
        self.head_dim = config.head_dim
        
        # Sliding window attention (every other layer)
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else None
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        
        # Output projection
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        
        # Attention
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            scale=1.0 / math.sqrt(self.head_dim),
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            sliding_window=self.sliding_window,
        )
        
        # RoPE
        self.rotary_emb = get_rope(
            head_dim=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.initial_context_length * config.rope_scaling_factor,
            base=int(config.rope_theta),
            rope_scaling={
                "type": "yarn",
                "factor": config.rope_scaling_factor,
                "original_max_position_embeddings": config.initial_context_length,
                "alpha": config.rope_ntk_alpha,
                "beta": config.rope_ntk_beta,
            },
        )
        
        # Sink attention weights for streaming attention
        self.sinks = nn.Parameter(
            torch.zeros(self.num_heads, dtype=torch.float32)
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        
        output, _ = self.o_proj(attn_output)
        return output


class GPTOSSMoELayer(nn.Module):
    """GPT-OSS MoE layer with swiglu activation."""
    
    def __init__(
        self,
        config: GPTOSSConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.experts_per_token
        
        # Expert gate
        self.gate = RowParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_experts,
            bias=config.use_bias,
            quant_config=None,  # Gate typically not quantized
            prefix=f"{prefix}.gate",
        )
        
        # MoE experts using AscendFusedMoE
        self.experts = AscendFusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )
        
        # Custom swiglu activation with limit
        self.swiglu_limit = config.swiglu_limit
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        # Get routing logits
        router_logits, _ = self.gate(hidden_states)
        
        # Apply MoE with custom swiglu
        output = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
        )
        
        return output


class GPTOSSDecoderLayer(nn.Module):
    """GPT-OSS decoder layer."""
    
    def __init__(
        self,
        config: GPTOSSConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Attention
        self.self_attn = GPTOSSAttention(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        
        # MoE MLP
        self.mlp = GPTOSSMoELayer(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        
        # Layer norms
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        
        # MLP
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states, attn_metadata)
        
        return hidden_states, residual


class GPTOSSModel(nn.Module):
    """GPT-OSS model."""
    
    def __init__(
        self,
        config: GPTOSSConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_idx = getattr(config, "pad_token_id", None)
        
        # Embeddings
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        
        # Transformer layers
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda layer_idx, prefix: GPTOSSDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        
        # Final layer norm
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states", "residual"],
                                                    config.hidden_size))
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )
        
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class GPTOSSForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """GPT-OSS for causal language modeling."""
    
    def __init__(
        self,
        config: GPTOSSConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional = None,
    ) -> None:
        super().__init__()
        
        self.config = config
        self.lora_config = lora_config
        
        self.model = GPTOSSModel(
            config,
            cache_config,
            quant_config,
            prefix="model",
        )
        
        # Language model head
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                quant_config=quant_config,
                prefix="lm_head",
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.sampler = get_sampler()
        else:
            self.lm_head = PPMissingLayer()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds,
        )
        return model_output
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"), 
            (".qkv_proj", ".v_proj", "v"),
        ]
        
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                continue
            
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias"):
                    name = name[:-5]
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias"):
                    name = name[:-5]
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
