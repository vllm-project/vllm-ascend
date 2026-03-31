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


import torch
import torch_npu

from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding, get_cos_and_sin_slice


def _rope_forward_oot(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    is_neox_style: bool,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_shape, key_shape = query.shape, key.shape
    if self.cos_sin_cache.device != query.device:
        self.cos_sin_cache = self.cos_sin_cache.to(query.device)
    if self.cos_sin_cache.dtype != query.dtype:
        self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)
    cos, sin = get_cos_and_sin_slice()
    if offsets is not None:
        raise NotImplementedError("Batched rotary embedding is currently not supported on NPU.")
    rotary_mode = "half" if is_neox_style else "interleave"
    if self.head_size == 128 and self.cos_sin_cache.shape[-1] == 128:
        query = query.contiguous().view(1, query.shape[0], -1, self.head_size)
        key = key.contiguous().view(1, key.shape[0], -1, self.head_size)
        query, key = torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin, rotary_mode=rotary_mode)
    elif self.rotary_dim < self.head_size:
        num_tokens = query.shape[0]
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)
        q_rot = query[..., : self.rotary_dim]
        q_pass = query[..., self.rotary_dim :]
        k_rot = key[..., : self.rotary_dim]
        k_pass = key[..., self.rotary_dim :]
        if self.rotary_dim == 64:
            q_rot = q_rot.contiguous().view(1, num_tokens, -1, self.rotary_dim)
            k_rot = k_rot.contiguous().view(1, num_tokens, -1, self.rotary_dim)
            q_rot, k_rot = torch_npu.npu_apply_rotary_pos_emb(q_rot, k_rot, cos, sin, rotary_mode=rotary_mode)
        else:
            q_rot = q_rot.contiguous().view(num_tokens, -1)
            k_rot = k_rot.contiguous().view(num_tokens, -1)
            torch_npu._npu_rotary_embedding(
                positions,
                q_rot,
                k_rot,
                self.rotary_dim,
                self.cos_sin_cache,
                is_neox_style,
            )
        q_rot = q_rot.view(num_tokens, -1, self.rotary_dim)
        k_rot = k_rot.view(num_tokens, -1, self.rotary_dim)
        query = torch.cat((q_rot, q_pass), dim=-1).reshape(query_shape)
        key = torch.cat((k_rot, k_pass), dim=-1).reshape(key_shape)
    else:
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


class AscendRotaryEmbedding310(AscendRotaryEmbedding):
    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
        is_neox_style_override: bool | None = None,
    ):
        is_neox_style = self.is_neox_style
        if is_neox_style_override is not None:
            is_neox_style = is_neox_style_override
        return _rope_forward_oot(self, positions, query, key, is_neox_style, offsets)
