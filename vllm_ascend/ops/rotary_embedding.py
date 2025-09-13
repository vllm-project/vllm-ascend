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
import torch_npu
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)

from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import enable_custom_op, is_310p


def _custom_rotary_embedding_enabled(query, neox_style, head_size):
    return query.dtype == torch.float16 and neox_style and head_size % 32 == 0 and enable_custom_op(
    )


def _rope_forward_oot(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    is_neox_style: bool,
    offsets: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_shape, key_shape = query.shape, key.shape
    if self.cos_sin_cache.device != query.device:
        self.cos_sin_cache = self.cos_sin_cache.to(query.device)
    if self.cos_sin_cache.dtype != query.dtype:
        self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)
    # adopt custom kernel path for rotary_embedding
    if _custom_rotary_embedding_enabled(query, is_neox_style,
                                        self.head_size) and not is_310p():
        query, key = torch.ops._C_ascend.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            is_neox_style,
        )
        return query.view(query_shape), key.view(key_shape)
    if offsets is not None:
        raise NotImplementedError(
            "Batched rotary embedding is currently not supported on NPU.")
    else:
        if self.cos is not None and \
            self.sin is not None:
            # If cos and sin are generated outside, use npu_apply_rotary_pos_emb to avoid redundant calculation.
            # This method requires head_size and rotary_dim equal 128 and neox_style is True
            query = query.contiguous().view(1, query.shape[0], -1,
                                            self.head_size)
            key = key.contiguous().view(1, key.shape[0], -1, self.head_size)
            torch_npu.npu_apply_rotary_pos_emb(query, key, self.cos, self.sin)
        elif self.rotary_dim < self.head_size:
            num_tokens = query.shape[0]
            query = query.view(num_tokens, -1, self.head_size)
            key = key.view(num_tokens, -1, self.head_size)
            q_rot = query[..., :self.rotary_dim]
            q_pass = query[..., self.rotary_dim:]
            k_rot = key[..., :self.rotary_dim]
            k_pass = key[..., self.rotary_dim:]
            q_rot = q_rot.contiguous().view(num_tokens, -1)
            k_rot = k_rot.contiguous().view(num_tokens, -1)
            torch_npu._npu_rotary_embedding(
                positions,
                q_rot,
                k_rot,
                self.head_size,
                self.cos_sin_cache,
                is_neox_style,
            )
            q_rot = q_rot.view(num_tokens, -1, self.rotary_dim)
            k_rot = k_rot.view(num_tokens, -1, self.rotary_dim)
            q = torch.cat((q_rot, q_pass), dim=-1).reshape(query_shape)
            k = torch.cat((k_rot, k_pass), dim=-1).reshape(key_shape)
            return q, k
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
                is_neox_style,
            )
        return query.view(query_shape), key.view(key_shape)


class AscendRotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        self.cos = None
        self.sin = None
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        is_neox_style_override: Optional[bool] = None,
    ):
        is_neox_style = self.is_neox_style
        if is_neox_style_override is not None:
            is_neox_style = is_neox_style_override
        forward_context = get_forward_context()
        is_first_layer = forward_context.is_first_layer
        # Generate cos and sin outside layers to avoid repeated calculation.
        if is_neox_style and self.head_size == 128 and self.cos_sin_cache.shape[
                -1] == 128:
            if is_first_layer:
                cos_sin = self.cos_sin_cache.index_select(0, positions)
                last_dim = cos_sin.size()[-1]
                cos, sin = cos_sin.reshape(-1, 2, last_dim // 2).repeat(
                    1, 1, 2).chunk(2, dim=-2)
                # BSNH
                self.cos = cos.view(1, -1, 1, last_dim).contiguous()
                self.sin = sin.view(1, -1, 1, last_dim).contiguous()
                forward_context.is_first_layer = False
        return _rope_forward_oot(self, positions, query, key, is_neox_style,
                                 offsets)


class AscendDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):

    def __init__(
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
        # Note: we adopt the native huggingface deepseek rope initialization code from
        # https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/main/modeling_deepseek.py for
        # its more ascend compute friendly
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            self._yarn_get_mscale(self.scaling_factor, float(mscale)) /
            self._yarn_get_mscale(self.scaling_factor, float(mscale_all_dim)) *
            attn_factor)
        super(DeepseekScalingRotaryEmbedding,
              self).__init__(head_size, rotary_dim, max_position_embeddings,
                             base, is_neox_style, dtype)

        # NOTE: For ascend friendly computing, reorder sin and cos cache
        self.max_seq_len = math.ceil(max_position_embeddings * scaling_factor)
        self._set_cos_sin_cache(self.max_seq_len,
                                device=NPUPlatform.device_type,
                                dtype=dtype)

    def _yarn_get_mscale(self, scale: float = 1, mscale: float = 1) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _yarn_linear_ramp_mask(self, min_value, max_value, dim):
        # Note: The if conditional branch is not used here
        # to solve MTP compilation error.
        max_value += (min_value == max_value).float() * 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) -
                       min_value) / (max_value - min_value)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Inverse dim formula to find dim based on number of rotations
    def _yarn_find_correction_dim(self,
                                  num_rotations,
                                  dim,
                                  base=10000,
                                  max_position_embeddings=2048):
        # Note: use torch instead of math to solve MTP compilation error.
        return (dim * torch.log(
            torch.tensor(max_position_embeddings) /
            (num_rotations * 2 * torch.pi))) / (2 *
                                                torch.log(torch.tensor(base)))

    # Find dim range bounds based on rotations
    def _yarn_find_correction_range(self,
                                    low_rot,
                                    high_rot,
                                    dim,
                                    base=10000,
                                    max_position_embeddings=2048):
        # Note: use torch instead of math to solve MTP compilation error.
        low = torch.floor(
            self._yarn_find_correction_dim(low_rot, dim, base,
                                           max_position_embeddings))
        high = torch.ceil(
            self._yarn_find_correction_dim(high_rot, dim, base,
                                           max_position_embeddings))
        # Note: use torch instead of max/min to solve MTP compilation error.
        return torch.clamp(low, min=0), torch.clamp(high, max=dim - 1)

    # Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    def _apply_rotary_pos_emb(self,
                              q,
                              k,
                              cos,
                              sin,
                              position_ids,
                              unsqueeze_dim=1):
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

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        q_embed = q_embed.view(b, h_q, d)
        k_embed = k_embed.view(b, h_k, d)

        return q_embed, k_embed

    def _set_cos_sin_cache(self, max_seq_len, device, dtype):
        dim = self.rotary_dim

        freq_extra = 1.0 / (self.base**(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (self.scaling_factor * self.base**(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

        low, high = self._yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(
            low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 -
                                 inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)
        cos_cached = torch.cat([freqs, freqs], dim=-1).cos() * self.mscale
        sin_cached = torch.cat([freqs, freqs], dim=-1).sin() * self.mscale
        cos_cached = cos_cached.to(dtype)
        sin_cached = sin_cached.to(dtype)
        cache = torch.cat(
            [freqs.cos() * self.mscale,
             freqs.sin() * self.mscale], dim=-1).to(dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self,
                positions: torch.Tensor,
                query: torch.Tensor,
                key: torch.Tensor,
                offsets: Optional[torch.Tensor] = None):
        if len(key.shape) == 2:
            key = key[:, None, :]
        # Note: we implement the non neox_style method with shuffle the last dim and neox style
        # calculation method which is also more compute friendly to the ascend machine
        # https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/main/modeling_deepseek.py
        is_neox_style = True
        if self.is_neox_style is False:
            b, h_q, d = query.shape
            query = query.view(b, h_q, d // 2,
                               2).transpose(3, 2).reshape(b, h_q, d)
            b, h_k, d = key.shape
            key = key.view(b, h_k, d // 2, 2).transpose(3,
                                                        2).reshape(b, h_k, d)
        q_pe, k_pe = _rope_forward_oot(self, positions, query, key,
                                       is_neox_style, offsets)
        return q_pe, k_pe
