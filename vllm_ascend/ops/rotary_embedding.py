#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from typing import Optional, Tuple
import math
import torch
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)


def rope_forward_oot(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    if self.cos_sin_cache.device != query.device:
        self.cos_sin_cache = self.cos_sin_cache.to(query.device)
    if self.cos_sin_cache.dtype != query.dtype:
        self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)
    if offsets is not None:
        raise NotImplementedError(
            "Batched rotary embedding is currently not supported on NPU.")
    else:
        # TODO: Remove the contiguous in the future.
        query = query.contiguous()
        key = key.contiguous()
        torch_npu._npu_rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
    return query, key


def rope_deepseek_forward_oot(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    if self.cos_sin_cache.device != query.device:
        self.cos_sin_cache = self.cos_sin_cache.to(query.device)
    if self.cos_sin_cache.dtype != query.dtype:
        self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)
    if offsets is not None:
        raise NotImplementedError(
            "Batched rotary embedding is currently not supported on NPU.")
    else:
        # TODO: Remove the contiguous in the future.
        ori_query_shape, ori_key_shape = query.shape, key.shape
        query = query.contiguous().view(query.shape[0], -1)
        key = key.contiguous().view(query.shape[0], -1)
        torch_npu._npu_rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
        query = query.view(ori_query_shape)
        key = key.view(ori_key_shape)

    return query, key

def native_rope_deepseek_forward(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
):
    # seq_len = positions.max() + 1
    seq_len = self.max_position_embeddings

    # x: [bs, num_attention_heads, seq_len, head_size]
    # if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
    #     self._set_cos_sin_cache(seq_len=seq_len, device=query.device, dtype=query.dtype)
    self._set_cos_sin_cache(seq_len=seq_len, device=query.device, dtype=query.dtype)

    cos = self.cos_cached[:seq_len].to(dtype=query.dtype)
    sin = self.sin_cached[:seq_len].to(dtype=query.dtype)

    q_pe, k_pe = apply_rotary_pos_emb(query, key, cos, sin, positions)

    return q_pe, k_pe

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    # Note: use torch instead of math to solve MTP compilation error.
    return (dim * torch.log(torch.tensor(max_position_embeddings) / 
                            (num_rotations * 2 * torch.pi))) / (
                                2 * torch.log(torch.tensor(base)))

def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    # Note: use torch instead of math to solve MTP compilation error.
    low = torch.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = torch.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    # Note: use torch instead of max/min to solve MTP compilation error.
    return torch.clamp(low, min=0), torch.clamp(high, max=dim-1)

def yarn_linear_ramp_mask(min_value, max_value, dim):
    # Note: The if conditional branch is not used here 
    # to solve MTP compilation error.
    max_value += (min_value == max_value).float() * 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_value) / (max_value - min_value)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids]
    sin = sin[position_ids]
    cos = cos[:, None, None, :]
    sin = sin[:, None, None, :]

    if len(q.shape) == 3:
        q = q[:, :, None, :]
    if len(k.shape) == 2:
        k = k[:, None, None, :]
    elif len(k.shape) == 3:
        k = k[:, :, None, :]

    b, h_q, s, d = q.shape
    q = q.view(b, h_q, s, d // 2, 2).transpose(4, 3).reshape(b, h_q, s, d)

    b, h_k, s, d = k.shape
    k = k.view(b, h_k, s, d // 2, 2).transpose(4, 3).reshape(b, h_k, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    q_embed = q_embed.view(b, h_q, d)
    k_embed = k_embed.view(b, h_k, d)

    return q_embed, k_embed

def _set_cos_sin_cache(self, seq_len, device, dtype):
    seq_len = self.max_position_embeddings
    self.max_seq_len_cached = seq_len
    dim = self.rotary_dim

    freq_extra = 1.0 / (
        self.base
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    freq_inter = 1.0 / (
        self.scaling_factor
        * self.base
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )

    low, high = yarn_find_correction_range(
        self.beta_fast,
        self.beta_slow,
        dim,
        self.base,
        self.max_position_embeddings,
    )
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
        device=device, dtype=torch.float32
    )
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    self.register_buffer("inv_freq", inv_freq, persistent=False)

    t = torch.arange(seq_len, device=device, dtype=torch.float32)

    freqs = torch.outer(t, inv_freq)

    # _mscale = float(
    #     yarn_get_mscale(self.scaling_factor, self.mscale)
    #     / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
    # )

    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer(
        "cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False
    )
    self.register_buffer(
        "sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False
    )


# TODO: Patch when aclnn ops avaiable
RotaryEmbedding.forward_oot = rope_forward_oot
# DeepseekScalingRotaryEmbedding.forward = rope_deepseek_forward_oot
DeepseekScalingRotaryEmbedding.forward = native_rope_deepseek_forward
DeepseekScalingRotaryEmbedding._set_cos_sin_cache = _set_cos_sin_cache
DeepseekScalingRotaryEmbedding.max_seq_len_cached = None
