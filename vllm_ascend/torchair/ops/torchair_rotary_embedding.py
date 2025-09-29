#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import enable_custom_op, is_310p


def custom_rotary_embedding_enabled(query, neox_style, head_size):
    return query.dtype == torch.float16 and neox_style and head_size % 32 == 0 and enable_custom_op(
    )


def rope_forward_oot(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
    is_neox_style_override: Optional[bool] = None,
    is_qwen_torchair: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if get_ascend_config(
    ).torchair_graph_config.enabled and not is_qwen_torchair:
        return self.forward_native(
            positions,
            query,
            key,
            offsets,
        )

    query_shape, key_shape = query.shape, key.shape
    if self.cos_sin_cache.device != query.device:
        self.cos_sin_cache = self.cos_sin_cache.to(query.device)
    if self.cos_sin_cache.dtype != query.dtype:
        self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)
    neox_style = self.is_neox_style
    if is_neox_style_override is not None:
        neox_style = is_neox_style_override
    # adopt custom kernel path for rotary_embedding
    if custom_rotary_embedding_enabled(query, neox_style,
                                       self.head_size) and not is_310p():
        query, key = torch.ops._C_ascend.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            neox_style,
        )
        return query.view(query_shape), key.view(key_shape)
    if offsets is not None:
        raise NotImplementedError(
            "Batched rotary embedding is currently not supported on NPU.")
    else:
        # TODO: Remove the contiguous in the future.
        query = query.contiguous().view(query.shape[0], -1)
        key = key.contiguous().view(key.shape[0], -1)
        torch_npu._npu_rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            neox_style,
        )
    return query.view(query_shape), key.view(key_shape)


def native_rope_deepseek_forward(self,
                                 positions: torch.Tensor,
                                 query: torch.Tensor,
                                 key: torch.Tensor,
                                 offsets: Optional[torch.Tensor] = None):
    if len(key.shape) == 2:
        key = key[:, None, :]
    # Note: we implement the non neox_style method with shuffle the last dim and neox style
    # calculation method which is also more compute friendly to the ascend machine
    # https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/main/modeling_deepseek.py
    neox_style = True
    if self.is_neox_style is False:
        b, h_q, d = query.shape
        query = query.view(b, h_q, d // 2, 2).transpose(3,
                                                        2).reshape(b, h_q, d)
        b, h_k, d = key.shape
        key = key.view(b, h_k, d // 2, 2).transpose(3, 2).reshape(b, h_k, d)
    q_pe, k_pe = rope_forward_oot(self, positions, query, key, offsets,
                                  neox_style)
    return q_pe, k_pe


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations,
                             dim,
                             base=10000,
                             max_position_embeddings=2048):
    # Note: use torch instead of math to solve MTP compilation error.
    return (dim * torch.log(
        torch.tensor(max_position_embeddings) /
        (num_rotations * 2 * torch.pi))) / (2 * torch.log(torch.tensor(base)))


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# Find dim range bounds based on rotations
def yarn_find_correction_range(low_rot,
                               high_rot,
                               dim,
                               base=10000,
                               max_position_embeddings=2048):
    # Note: use torch instead of math to solve MTP compilation error.
    low = torch.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = torch.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    # Note: use torch instead of max/min to solve MTP compilation error.
    return torch.clamp(low, min=0), torch.clamp(high, max=dim - 1)


def yarn_linear_ramp_mask(min_value, max_value, dim):
    # Note: The if conditional branch is not used here
    # to solve MTP compilation error.
    max_value += (min_value == max_value).float() * 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) -
                   min_value) / (max_value - min_value)
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


def _set_cos_sin_cache(self, max_seq_len, device, dtype):
    dim = self.rotary_dim

    freq_extra = 1.0 / (self.base**(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    freq_inter = 1.0 / (self.scaling_factor * self.base**(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

    low, high = yarn_find_correction_range(
        self.beta_fast,
        self.beta_slow,
        dim,
        self.base,
        self.max_position_embeddings,
    )
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
        device=device, dtype=torch.float32)
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    self.register_buffer("inv_freq", inv_freq, persistent=False)

    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)

    freqs = torch.outer(t, inv_freq)
    cos_cached = torch.cat([freqs, freqs], dim=-1).cos() * self.mscale
    sin_cached = torch.cat([freqs, freqs], dim=-1).sin() * self.mscale
    cos_cached = cos_cached.to(dtype)
    sin_cached = sin_cached.to(dtype)
    cache = torch.cat([freqs.cos() * self.mscale,
                       freqs.sin() * self.mscale],
                      dim=-1).to(dtype)
    self.register_buffer("cos_sin_cache", cache, persistent=False)
    self.register_buffer("cos_cached", cos_cached, persistent=False)
    self.register_buffer("sin_cached", sin_cached, persistent=False)


def __set_cos_sin_cache(self, seq_len, device, dtype):
    inv_freq = 1.0 / (self.base**(torch.arange(
        0, self.rotary_dim, 2, device=device, dtype=torch.float32) *
                                  (1 / self.rotary_dim)))
    self.register_buffer("inv_freq", inv_freq)

    t = torch.arange(self.max_position_embeddings,
                     device=self.inv_freq.device,
                     dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)

    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos", emb.cos().to(dtype=dtype), persistent=False)
    self.register_buffer("sin", emb.sin().to(dtype=dtype), persistent=False)
    self.embed = F.embedding


_original_re_init = RotaryEmbedding.__init__


def qwen_rope_init_func(
    self,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
) -> None:
    _original_re_init(self, head_size, rotary_dim, max_position_embeddings,
                      base, is_neox_style, dtype)
    if get_ascend_config().torchair_graph_config.enabled:
        __set_cos_sin_cache(self,
                            seq_len=max_position_embeddings,
                            device="npu",
                            dtype=dtype)


def rope_forward(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
    is_neox_style_override: Optional[bool] = None,
    max_seq_len: Optional[int] = None,
    is_prefill: Optional[bool] = True,
    is_qwen_torchair: Optional[bool] = False,
):
    if get_ascend_config().torchair_graph_config.enabled \
            and is_qwen_torchair and not is_prefill:
        if max_seq_len is not None and torch.gt(max_seq_len,
                                                self.max_position_embeddings):
            __set_cos_sin_cache(self,
                                seq_len=max_seq_len,
                                device=query.device,
                                dtype=torch.float32)

        # bsnd/bnsd
        if positions is not None:
            cos = self.embed(positions, self.cos)
            sin = self.embed(positions, self.sin)
            self.cos_embed = cos
            self.sin_embed = sin
        else:
            cos = self.cos_embed
            sin = self.sin_embed

        query = query.view(*query.shape[:-1], -1, self.head_size).contiguous()
        key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()

        cos = cos.unsqueeze(-2).unsqueeze(-2)
        sin = sin.unsqueeze(-2).unsqueeze(-2)

        query = query.unsqueeze(1)
        key = key.unsqueeze(1)

        q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(
            query, key, cos, sin)
        return q_embed.flatten(-2), k_embed.flatten(-2)
    else:
        return rope_forward_oot(self, positions, query, key, offsets,
                                is_neox_style_override,
                                is_qwen_torchair)  # type: ignore


def deepseek_rope_init_func(
    self,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: int,
    is_neox_style: bool,
    scaling_factor: float,
    dtype: torch.dtype,
    *,
    extrapolation_factor: float = 1,
    attn_factor: float = 1,
    beta_fast: int = 32,
    beta_slow: int = 1,
    mscale: float = 1,
    mscale_all_dim: float = 0,
) -> None:
    self.scaling_factor = scaling_factor
    self.extrapolation_factor = extrapolation_factor
    self.attn_factor = attn_factor
    self.beta_fast = beta_fast
    self.beta_slow = beta_slow
    # Get n-d magnitude scaling corrected for interpolation.
    self.mscale = float(
        yarn_get_mscale(self.scaling_factor, float(mscale)) /
        yarn_get_mscale(self.scaling_factor, float(mscale_all_dim)) *
        attn_factor)
    super(DeepseekScalingRotaryEmbedding,
          self).__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    # NOTE: For ascend friendly computing, reorder sin and cos cache
    self.max_seq_len = math.ceil(max_position_embeddings * scaling_factor)
    _set_cos_sin_cache(self, self.max_seq_len, dtype=dtype, device="npu")
