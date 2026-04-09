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
from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope

from vllm_ascend.ops.rotary_embedding import (
    AscendMRotaryEmbedding,
    AscendRotaryEmbedding,
    get_cos_and_sin_slice,
)

_mrope_cos_slice: torch.Tensor | None = None
_mrope_sin_slice: torch.Tensor | None = None
_mrope_forward_id: int = 0
_prepared_mrope_forward_id: int = -1
_prepared_mrope_dtype: torch.dtype | None = None
_prepared_mrope_device: torch.device | None = None
_prepared_mrope_interleaved: bool | None = None
_prepared_mrope_section: tuple[int, ...] | None = None


def begin_mrope_forward_310():
    global _mrope_forward_id
    _mrope_forward_id += 1


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


class AscendMRotaryEmbedding310(AscendMRotaryEmbedding):
    def _merge_mrope_cos_sin(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.mrope_section is not None
        if self.mrope_interleaved:
            return (
                apply_interleaved_rope(cos, self.mrope_section),
                apply_interleaved_rope(sin, self.mrope_section),
            )
        return (
            torch.cat([m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))], dim=-1),
            torch.cat([m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))], dim=-1),
        )

    def _get_or_prepare_mrope_cos_sin(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _mrope_cos_slice
        global _mrope_sin_slice
        global _prepared_mrope_forward_id
        global _prepared_mrope_dtype
        global _prepared_mrope_device
        global _prepared_mrope_interleaved
        global _prepared_mrope_section

        current_section = tuple(self.mrope_section or [])
        cache_hit = (
            _prepared_mrope_forward_id == _mrope_forward_id
            and _mrope_cos_slice is not None
            and _mrope_sin_slice is not None
            and _prepared_mrope_dtype == query.dtype
            and _prepared_mrope_device == query.device
            and _prepared_mrope_interleaved == self.mrope_interleaved
            and _prepared_mrope_section == current_section
        )
        if cache_hit:
            return _mrope_cos_slice, _mrope_sin_slice

        assert positions.ndim in (1, 2), "M-RoPE positions must be [num_tokens] or [3, num_tokens]."
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            assert positions.shape[0] == 3, "MRoPE expects positions [3, num_tokens] (T/H/W)."
            cos, sin = self._merge_mrope_cos_sin(cos, sin)
        # `npu_apply_rotary_pos_emb` follows ApplyRotaryPosEmbV2 semantics:
        # q_embed = q * cos + rotate(q) * sin, where cos/sin have full rotary dim.
        # MRoPE merge above gives half-dim cos/sin, so expand to full dim here.
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        num_tokens = positions.shape[-1]
        _mrope_cos_slice = cos.contiguous().view(1, num_tokens, 1, -1)
        _mrope_sin_slice = sin.contiguous().view(1, num_tokens, 1, -1)
        _prepared_mrope_forward_id = _mrope_forward_id
        _prepared_mrope_dtype = query.dtype
        _prepared_mrope_device = query.device
        _prepared_mrope_interleaved = self.mrope_interleaved
        _prepared_mrope_section = current_section
        return _mrope_cos_slice, _mrope_sin_slice

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
        is_neox_style_override: bool | None = None,
    ):
        query_shape, key_shape = query.shape, key.shape
        if self.cos_sin_cache.device != query.device:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device)
        if self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)
        if offsets is not None:
            raise NotImplementedError("Batched rotary embedding is currently not supported on NPU.")

        # For MRoPE, upstream Ascend path (`npu_mrope`) always uses rotary_mode="half".
        # Interleaved behavior is encoded by mrope_section/cos-sin re-layout, not by
        # switching rotary kernel mode.
        rotary_mode = "half"
        num_tokens = query.shape[0]
        cos, sin = self._get_or_prepare_mrope_cos_sin(positions, query)

        # Keep branch layout aligned with rope implementation for better numerical consistency.
        if self.head_size == 128 and self.rotary_dim == self.head_size:
            query = query.contiguous().view(1, num_tokens, -1, self.head_size)
            key = key.contiguous().view(1, num_tokens, -1, self.head_size)
            query, key = torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin, rotary_mode=rotary_mode)
        elif self.rotary_dim < self.head_size:
            query = query.view(num_tokens, -1, self.head_size)
            key = key.view(num_tokens, -1, self.head_size)
            q_rot = query[..., : self.rotary_dim]
            q_pass = query[..., self.rotary_dim :]
            k_rot = key[..., : self.rotary_dim]
            k_pass = key[..., self.rotary_dim :]
            if self.rotary_dim == 64:
                q_rot = q_rot.contiguous().view(1, num_tokens, -1, self.rotary_dim)
                k_rot = k_rot.contiguous().view(1, num_tokens, -1, self.rotary_dim)
            else:
                # Keep separate branch for non-64 partial rotary to match rope control flow.
                q_rot = q_rot.contiguous().view(1, num_tokens, -1, self.rotary_dim)
                k_rot = k_rot.contiguous().view(1, num_tokens, -1, self.rotary_dim)
            q_rot, k_rot = torch_npu.npu_apply_rotary_pos_emb(q_rot, k_rot, cos, sin, rotary_mode=rotary_mode)
            q_rot = q_rot.view(num_tokens, -1, self.rotary_dim)
            k_rot = k_rot.view(num_tokens, -1, self.rotary_dim)
            query = torch.cat((q_rot, q_pass), dim=-1).reshape(query_shape)
            key = torch.cat((k_rot, k_pass), dim=-1).reshape(key_shape)
            return query, key
        else:
            query = query.contiguous().view(1, num_tokens, -1, self.head_size)
            key = key.contiguous().view(1, num_tokens, -1, self.head_size)
            query, key = torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin, rotary_mode=rotary_mode)

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
