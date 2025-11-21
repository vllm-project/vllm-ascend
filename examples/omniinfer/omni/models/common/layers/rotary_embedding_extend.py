# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Rotary Positional Embeddings."""
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch_npu
from vllm.logger import init_logger

logger = init_logger(__name__)


class RotaryEmbedding(torch.nn.Module):

    def __init__(self,
                 head_size: int,
                 rotary_dim: int,
                 max_position_embeddings: int = 2048,
                 base: int = 10000,
                 is_neox_style: bool = True,
                 dtype: torch.dtype = None):
        super().__init__()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.max_len = self.max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style

        self.head_size = head_size
        cos, sin = RotaryEmbedding.compute_full_cos_sin(
            self.base, self.rotary_dim, self.max_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def compute_full_cos_sin(
            base: Union[int, float], rotary_dim: int,
            max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the cos and sin cache."""
        inv_freq = RotaryEmbedding.compute_inv_freq(base, rotary_dim)
        t = torch.arange(max_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb).to(dtype=torch.get_default_dtype())
        sin = torch.sin(emb).to(dtype=torch.get_default_dtype())

        return cos, sin

    @staticmethod
    def compute_inv_freq(base: Union[int, float],
                         rotary_dim: int) -> torch.Tensor:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (base**(
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        return inv_freq

    # use small ops
    def apply_rotary_pos_emb(self, x, cos, sin):
        x1, x2 = torch.chunk(x, 2, -1)
        x_new = torch.cat((-x2, x1), dim=-1)
        output = cos * x + sin * x_new
        return output

    def forward(self, position_ids, query, key, cos, sin):
        """
        Args: 
            position_ids: [num_tokens, ]
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_heads * head_size]
        """

        if self.rotary_dim != 128:
            query = query.view(*query.shape[:-1], -1,
                               self.head_size).contiguous()
            key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
            q_embed = self.apply_rotary_pos_emb(query, cos, sin)
            k_embed = self.apply_rotary_pos_emb(key, cos, sin)
            q_embed = q_embed.flatten(-2)
            k_embed = k_embed.flatten(-2)
        else:
            # shape to bsnd
            cos = cos.unsqueeze(1).unsqueeze(1)
            sin = sin.unsqueeze(1).unsqueeze(1)

            query = query.view(query.shape[0], 1, -1, self.head_size)
            key = key.view(key.shape[0], 1, -1, self.head_size)

            q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(
                query, key, cos, sin)

            q_embed = q_embed.view(q_embed.shape[0], -1)
            k_embed = k_embed.view(k_embed.shape[0], -1)

        return q_embed, k_embed

    def forward_cos_sin(self, query, key, cos, sin):
        """
        Args: 
            position_ids: [num_tokens, ]
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_heads * head_size]
        """

        if self.rotary_dim != 128:
            query = query.view(*query.shape[:-1], -1,
                               self.head_size).contiguous()
            key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
            q_embed = self.apply_rotary_pos_emb(query, cos, sin)
            k_embed = self.apply_rotary_pos_emb(key, cos, sin)
            q_embed = q_embed.flatten(-2)
            k_embed = k_embed.flatten(-2)
        else:
            # shape to bsnd
            cos = cos.unsqueeze(1).unsqueeze(1)
            sin = sin.unsqueeze(1).unsqueeze(1)

            query = query.view(query.shape[0], 1, -1, self.head_size)
            key = key.view(key.shape[0], 1, -1, self.head_size)

            q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(
                query, key, cos, sin)

            q_embed = q_embed.view(q_embed.shape[0], -1)
            k_embed = k_embed.view(k_embed.shape[0], -1)

        return q_embed, k_embed


_ROPE_DICT: Dict[Tuple, nn.Module] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
):
    if dtype is None:
        dtype = torch.get_default_dtype()
    key = (head_size, rotary_dim, max_position, base, is_neox_style,
           tuple(rope_scaling.items()) if rope_scaling is not None else None)
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                                 is_neox_style)

    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
