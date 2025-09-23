# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.mla import MultiHeadLatentAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils import direct_register_custom_op


@dataclass
class AscendMLAModules:
    q_a_proj: Optional[torch.nn.Module]
    q_a_layernorm: Optional[torch.nn.Module]
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
        enable_shared_expert_dp: bool,
        debug_layer_idx: int,
        first_k_dense_replace: int,
        tp_size: int,
        mla_modules: AscendMLAModules,
        num_local_heads: int,
        scaling: float,
        layers: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        q_lora_rank: Optional[int],
        qk_nope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.enable_shared_expert_dp = enable_shared_expert_dp
        self.debug_layer_idx = debug_layer_idx
        self.first_k_dense_replace = first_k_dense_replace
        self.tp_size = tp_size
        self.num_local_heads = num_local_heads
        self.layers = layers
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.prefix = prefix

        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=scaling,
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
            q_proj=mla_modules.q_proj,
            kv_a_proj_with_mqa=mla_modules.kv_a_proj_with_mqa,
            kv_a_layernorm=mla_modules.kv_a_layernorm,
            kv_b_proj=mla_modules.kv_b_proj,
            o_proj=mla_modules.o_proj,
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        need_gather_q_kv = False
        if self.enable_shared_expert_dp and self.debug_layer_idx > self.first_k_dense_replace and self.debug_layer_idx < self.layers:
            # Simulate all gather to calculate output shape
            num_tokens = num_tokens * self.tp_size
            need_gather_q_kv = True
        if not self.enable_shared_expert_dp or self.debug_layer_idx < self.first_k_dense_replace:
            output_shape = hidden_states.shape
        else:
            rows = num_tokens // self.tp_size
            if num_tokens % self.tp_size:
                rows += 1
            output_shape = (rows, hidden_states.shape[1])
        # FIXME: This does not seem right, should make sure the buffer is fixed
        output = torch.empty(output_shape,
                             dtype=hidden_states.dtype,
                             device=hidden_states.device)
        torch.ops.vllm.mla_forward(hidden_states, need_gather_q_kv, output,
                                   self.prefix)
        output = output.view(-1, output_shape[-1])
        return output


def mla_forward(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if forward_context.attn_metadata:
        attn_metadata = forward_context.attn_metadata[self.mla_attn.layer_name]
    else:
        attn_metadata = forward_context.attn_metadata
    kv_cache = self.mla_attn.kv_cache[forward_context.virtual_engine]
    self.mla_attn.impl.forward(self.mla_attn.layer_name, hidden_states,
                               kv_cache, attn_metadata, need_gather_q_kv,
                               output)
    return


def mla_forward_fake(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="mla_forward",
    op_func=mla_forward,
    mutates_args=["output"],
    fake_impl=mla_forward_fake,
    dispatch_key="PrivateUse1",
)
