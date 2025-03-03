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

import torch
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding, _yarn_find_correction_range, _yarn_linear_ramp_mask)


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


def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
    pos_freqs = self.base**(torch.arange(
        0, self.rotary_dim, 2, dtype=torch.float, device="npu") /
                            self.rotary_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

    low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                            self.rotary_dim, self.base,
                                            self.max_position_embeddings)
    # Get n-d rotational scaling corrected for extrapolation
    inv_freq_mask = (1 - _yarn_linear_ramp_mask(
        low, high, self.rotary_dim // 2,
        dtype=torch.float)) * self.extrapolation_factor
    inv_freq = inv_freq_interpolation * (
        1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    return inv_freq

def _compute_cos_sin_cache(self) -> torch.Tensor:
    inv_freq = self._compute_inv_freq(self.scaling_factor)
    t = torch.arange(self.max_position_embeddings * self.scaling_factor,
                        device="npu",
                        dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = (freqs.cos() * self.mscale)
    sin = (freqs.sin() * self.mscale)
    cache = torch.cat((cos, sin), dim=-1)
    return cache


RotaryEmbedding.forward_oot = rope_forward_oot
DeepseekScalingRotaryEmbedding.forward = rope_deepseek_forward_oot
# v0.7.4 do not need. or `from torch_npu.contrib import transfer_to_npu` also cann fix it.
DeepseekScalingRotaryEmbedding._compute_inv_freq = _compute_inv_freq
DeepseekScalingRotaryEmbedding._compute_cos_sin_cache = _compute_cos_sin_cache

