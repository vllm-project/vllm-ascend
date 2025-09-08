# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Optional

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from transformers import GptOssConfig

from vllm.attention import Attention, AttentionType, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_ep_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_pp_group)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import cdiv

# Import the original GPT-OSS classes from vLLM
from vllm.model_executor.models.gpt_oss import (
    GptOssForCausalLM, GptOssModel, OAIAttention, MLPBlock, TransformerBlock
)
from vllm.model_executor.models.utils import (
    extract_layer_index, maybe_prefix, PPMissingLayer, is_pp_missing_parameter
)
from vllm.model_executor.model_loader.weight_utils import (
    AutoWeightsLoader, WeightsMapper
)

from vllm_ascend.ops.fused_moe import AscendFusedMoE


class CustomOAIAttention(OAIAttention):
    """Custom OAI Attention with Ascend optimizations."""

    def __init__(
        self,
        config: GptOssConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config, cache_config, prefix)

    def forward(self, 
                hidden_states: torch.Tensor,
                positions: torch.Tensor,
                kv_cache: Optional[torch.Tensor] = None,
                attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        # Use original forward but with Ascend-optimized attention
        t = self.norm(hidden_states)

        qkv, _ = self.qkv(t)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        v = v.contiguous()
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)

        return output + hidden_states


class CustomMLPBlock(MLPBlock):
    """Custom MLP Block using AscendFusedMoE."""

    def __init__(
        self,
        config: GptOssConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)  # Skip MLPBlock.__init__
        self.layer_idx = layer_idx
        self.num_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, eps=1e-5)
        self.router = torch.nn.Linear(config.hidden_size,
                                      config.num_local_experts,
                                      dtype=torch.bfloat16)
        assert config.intermediate_size % self.world_size == 0
        
        # Use AscendFusedMoE instead of standard FusedMoE
        self.experts = AscendFusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            reduce_results=True,
            renormalize=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            apply_router_weight_on_input=False,
            has_bias=True,
            activation="swigluoai"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.router(t)
        t = self.experts(hidden_states=t, router_logits=g)
        return x + t


class CustomTransformerBlock(TransformerBlock):
    """Custom Transformer Block with Ascend-optimized components."""

    def __init__(
        self,
        config: GptOssConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)  # Skip TransformerBlock.__init__
        self.layer_idx = extract_layer_index(prefix)
        self.attn = CustomOAIAttention(
            config, 
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.attn"
        )
        self.mlp = CustomMLPBlock(
            config,
            self.layer_idx,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp"
        )

    def forward(self, 
                hidden_states: torch.Tensor,
                positions: torch.Tensor,
                kv_cache: Optional[torch.Tensor] = None,
                attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        attn_output = self.attn(hidden_states, positions, kv_cache, attn_metadata)
        output = self.mlp(attn_output)
        return output


class CustomGptOssModel(GptOssModel):
    """Custom GPT-OSS Model with Ascend optimizations."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        nn.Module.__init__(self)  # Skip GptOssModel.__init__
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        
        self.config.hidden_size = self.config.hidden_size
        self.embedding = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.layers = torch.nn.ModuleList([
            CustomTransformerBlock(
                self.config,
                quant_config=self.quant_config,
                cache_config=self.cache_config,
                prefix=maybe_prefix(prefix, f"block.{layer_idx}"),
            ) for layer_idx in range(self.config.num_hidden_layers)
        ])
        self.norm = RMSNorm(self.config.hidden_size, eps=1e-5)

    def forward(self, 
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: Optional[torch.Tensor] = None,
                attn_metadata: Optional[AttentionMetadata] = None,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, positions, 
                     kv_cache=kv_caches[i] if kv_caches else None,
                     attn_metadata=attn_metadata)
        x = self.norm(x)
        return x


class CustomGptOssForCausalLM(GptOssForCausalLM):
    """Custom GPT-OSS For Causal Language Modeling with Ascend optimizations."""
    
    packed_modules_mapping = {"qkv": ["q_proj", "k_proj", "v_proj"]}

    # Use the same weight mapper as the original
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".self_attn.": ".attn.",
            ".post_attention_layernorm.": ".mlp.norm.",
        },
        orig_to_new_suffix={
            ".embed_tokens.weight": ".embedding.weight",
            ".input_layernorm.weight": ".attn.norm.weight",
            ".post_attention_layernorm.weight": ".mlp.norm.weight",

            # MoE MXFP4 weights
            ".gate_up_proj_blocks": ".w13_weight",
            ".down_proj_blocks": ".w2_weight",
            ".gate_up_proj_scales": ".w13_weight_scale",
            ".down_proj_scales": ".w2_weight_scale",

            # MoE other weights
            ".gate_up_proj": ".w13_weight",
            ".down_proj": ".w2_weight",

            # MoE Bias
            ".gate_up_proj_bias": ".w13_bias",
            ".down_proj_bias": ".w2_bias",
        },
    )

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        nn.Module.__init__(self)  # Skip GptOssForCausalLM.__init__
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        # Use CustomGptOssModel instead of GptOssModel
        self.model = CustomGptOssModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
            
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.sampler = get_sampler()

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: Optional[torch.Tensor] = None,
                attn_metadata: Optional[AttentionMetadata] = None,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert intermediate_tensors is None
        assert inputs_embeds is None
        return self.model(input_ids, positions, kv_caches, attn_metadata, 
                         intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
