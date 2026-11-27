#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Qwen2Config
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack

from vllm_ascend.ops.qwen2_decoder import (
    AscendCustomQwen2Decoder,
    AscendQwen2Attention,
    AscendQwen2DecoderLayer,
    optimized_apply_rotary_pos_emb,
)


def _gqa_attention_bmm_310p(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    """Grouped-query attention via batched matmul without K/V head expansion.

    PyTorch SDPA with a 4D additive float mask falls back to a slow path on
    310P and expands K/V across query groups. Reshaping Q as
    ``(B * num_kv_heads, group_size * q_len, head_dim)`` lets NPU bmm consume
    K/V directly and applies the mask in-place.
    """
    bsz, num_heads, q_len, head_dim = query_states.shape
    num_kv_heads = key_states.shape[1]
    kv_len = key_states.shape[2]
    num_groups = num_heads // num_kv_heads
    bh_kv = bsz * num_kv_heads

    q_scaled = (query_states * scaling).reshape(bh_kv, num_groups * q_len, head_dim)
    k_t = key_states.reshape(bh_kv, kv_len, head_dim).transpose(-2, -1)
    v = value_states.reshape(bh_kv, kv_len, head_dim)

    attn_weights = torch.bmm(q_scaled, k_t)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :kv_len]
        attn_weights.view(bsz, num_kv_heads, num_groups, q_len, kv_len).add_(causal_mask.unsqueeze(1))
    attn_probs = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.bmm(attn_probs, v)
    return attn_output.view(bsz, num_heads, q_len, head_dim).transpose(1, 2)


class AscendQwen2Attention310(AscendQwen2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = optimized_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output = _gqa_attention_bmm_310p(
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.scaling,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class AscendQwen2DecoderLayer310(AscendQwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = AscendQwen2Attention310(config=config, layer_idx=layer_idx)


class AscendCustomQwen2Decoder310(AscendCustomQwen2Decoder):
    def _create_optimized_custom_model(self, config):
        model = super()._create_optimized_custom_model(config)
        model.layers = nn.ModuleList(
            [AscendQwen2DecoderLayer310(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        return model
