# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mla import MultiHeadLatentAttention
from vllm.model_executor.layers.quantization import QuantizationConfig


@dataclass
class AscendMLAModules:
    q_a_proj: Optional[torch.nn.Module]
    q_a_layernorm: Optional[torch.nn.Module]
    q_b_proj: Optional[torch.nn.Module]
    q_proj: Optional[torch.nn.Module]
    kv_a_proj_with_mqa: torch.nn.Module
    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    o_proj: torch.nn.Module
    rotary_emb: torch.nn.Module


class AscendMultiHeadLatentAttention(MultiHeadLatentAttention):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        mla_modules: AscendMLAModules,
        num_local_heads: int,
        scaling: float,
        qk_head_dim: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        
    ) -> None:
        super().__init__(
            hidden_size,
            num_heads,
            scale,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )
        self.num_local_heads = num_local_heads
        self.scaling = scaling
        self.qk_head_dim = qk_head_dim

        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            
            rotary_emb=mla_modules.rotary_emb,
            q_a_proj=mla_modules.q_a_proj,
            q_a_layernorm=mla_modules.q_a_layernorm,
            q_proj=mla_modules.q_proj if self.q_lora_rank is None else mla_modules.q_b_proj,
            kv_a_proj_with_mqa=mla_modules.kv_a_proj_with_mqa,
            kv_a_layernorm=mla_modules.kv_a_layernorm,
            kv_b_proj=mla_modules.kv_b_proj,
            o_proj=mla_modules.o_proj,
        )

    def forward_oot(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            enable_shared_expert_dp: bool,
            debug_layer_idx: int,
            first_k_dense_replace: int,
            tp_size: int,
            layers: int,
            kv_cache: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None,
            ) -> torch.Tensor:
        forward_context = get_forward_context()
        if kv_cache is None:
            kv_cache = self.mla_attn.kv_cache[forward_context.virtual_engine]
        num_tokens = hidden_states.shape[0]
        need_gather_q_kv = False
        if enable_shared_expert_dp and debug_layer_idx > first_k_dense_replace and debug_layer_idx < layers:
            # Simulate all gather to calculate output shape
            num_tokens = num_tokens * tp_size
            need_gather_q_kv = True
        if not enable_shared_expert_dp or debug_layer_idx < first_k_dense_replace:
            output_shape = hidden_states.shape
        else:
            rows = num_tokens // tp_size
            if num_tokens % tp_size:
                rows += 1
            output_shape = (rows, hidden_states.shape[1])
        output = torch.empty(output_shape,
                             dtype=hidden_states.dtype,
                             device=hidden_states.device)
        output = self.mla_attn.impl.forward(hidden_states, kv_cache,
                                            forward_context.attn_metadata,
                                            need_gather_q_kv, output)
        output = output.view(-1, output_shape[-1])
        return output
