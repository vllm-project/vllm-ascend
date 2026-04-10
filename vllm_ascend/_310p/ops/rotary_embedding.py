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
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

from typing import Any

import torch
import torch_npu
from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope

from vllm_ascend.ops.rotary_embedding import (
    AscendMRotaryEmbedding,
    AscendRotaryEmbedding,
    get_cos_and_sin_slice,
)

# Filled once per model forward in NPUModelRunner310._model_forward; read by every MRoPE layer.
_mrope_cos_slice: torch.Tensor | None = None
_mrope_sin_slice: torch.Tensor | None = None


def merge_mrope_cos_sin_for_apply(
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    mrope_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mrope_interleaved:
        return (
            apply_interleaved_rope(cos, mrope_section),
            apply_interleaved_rope(sin, mrope_section),
        )
    return (
        torch.cat([m[i] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1),
        torch.cat([m[i] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1),
    )


def set_mrope_apply_rotary_slices(
    emb: AscendMRotaryEmbedding,
    positions: torch.Tensor,
    target_dtype: torch.dtype,
    target_device: torch.device,
    *,
    capacity_tokens: int | None = None,
) -> None:
    """Build cos/sin views for `npu_apply_rotary_pos_emb` from positions; must run once per forward before layers."""
    global _mrope_cos_slice
    global _mrope_sin_slice

    assert emb.mrope_section is not None
    if emb.cos_sin_cache.device != target_device:
        emb.cos_sin_cache = emb.cos_sin_cache.to(target_device)
    if emb.cos_sin_cache.dtype != target_dtype:
        emb.cos_sin_cache = emb.cos_sin_cache.to(target_dtype)

    assert positions.ndim in (1, 2), "M-RoPE positions must be [num_tokens] or [3, num_tokens]."
    cos_sin = emb.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        assert positions.shape[0] == 3, "MRoPE expects positions [3, num_tokens] (T/H/W)."
        cos, sin = merge_mrope_cos_sin_for_apply(
            cos,
            sin,
            list(emb.mrope_section),
            emb.mrope_interleaved,
        )
    # `npu_apply_rotary_pos_emb` follows ApplyRotaryPosEmbV2 semantics:
    # q_embed = q * cos + rotate(q) * sin, where cos/sin have full rotary dim.
    # MRoPE merge above gives half-dim cos/sin, so expand to full dim here.
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)
    num_tokens = positions.shape[-1]
    cos_view = cos.contiguous().view(1, num_tokens, 1, -1)
    sin_view = sin.contiguous().view(1, num_tokens, 1, -1)

    # Keep stable storage across forwards for graph replay.
    need_alloc = (
        _mrope_cos_slice is None
        or _mrope_sin_slice is None
        or _mrope_cos_slice.device != cos_view.device
        or _mrope_cos_slice.dtype != cos_view.dtype
        or _mrope_cos_slice.shape[-1] != cos_view.shape[-1]
    )
    if need_alloc:
        capacity = capacity_tokens if capacity_tokens is not None else num_tokens
        if capacity < num_tokens:
            capacity = num_tokens
        _mrope_cos_slice = torch.empty(
            (1, capacity, 1, cos_view.shape[-1]),
            dtype=cos_view.dtype,
            device=cos_view.device,
        )
        _mrope_sin_slice = torch.empty(
            (1, capacity, 1, sin_view.shape[-1]),
            dtype=sin_view.dtype,
            device=sin_view.device,
        )
    elif _mrope_cos_slice.shape[1] < num_tokens:
        if capacity_tokens is not None:
            # Keep storage address stable for graph replay; fail-fast instead of reallocating.
            raise RuntimeError(
                f"Prepared MRoPE buffer capacity {_mrope_cos_slice.shape[1]} is smaller than required "
                f"{num_tokens}. Please increase max_num_batched_tokens."
            )
        # Local/unit-test path without fixed capacity may still grow.
        _mrope_cos_slice = torch.empty(
            (1, num_tokens, 1, cos_view.shape[-1]),
            dtype=cos_view.dtype,
            device=cos_view.device,
        )
        _mrope_sin_slice = torch.empty(
            (1, num_tokens, 1, sin_view.shape[-1]),
            dtype=sin_view.dtype,
            device=sin_view.device,
        )

    _mrope_cos_slice[:, :num_tokens].copy_(cos_view)
    _mrope_sin_slice[:, :num_tokens].copy_(sin_view)


def prepare_mrope_cos_sin_slices_from_runner(runner: Any, positions: torch.Tensor) -> None:
    """Resolve MRoPE embedding from the runner and populate `_mrope_cos_slice` / `_mrope_sin_slice`."""
    emb = getattr(runner, "_cached_mrope_emb310", None)
    if emb is None:
        for module in runner.model.modules():
            if isinstance(module, AscendMRotaryEmbedding310):
                emb = module
                runner._cached_mrope_emb310 = emb
                break
    if emb is None:
        raise RuntimeError("uses_mrope is True but no AscendMRotaryEmbedding310 was found in the model.")
    set_mrope_apply_rotary_slices(
        emb,
        positions,
        runner.dtype,
        runner.device,
        capacity_tokens=runner.max_num_tokens,
    )


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
    def _get_mrope_cos_sin_for_apply(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        global _mrope_cos_slice
        global _mrope_sin_slice
        if _mrope_cos_slice is None or _mrope_sin_slice is None:
            raise RuntimeError(
                "MRoPE cos/sin slices are not prepared. "
                "Expected NPUModelRunner310._model_forward after positions are ready for each forward."
            )
        return _mrope_cos_slice[:, :num_tokens], _mrope_sin_slice[:, :num_tokens]

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ):
        query_shape, key_shape = query.shape, key.shape
        if self.cos_sin_cache.device != query.device:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device)
        if self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)

        # For MRoPE, upstream Ascend path (`npu_mrope`) always uses rotary_mode="half".
        # Interleaved behavior is encoded by mrope_section/cos-sin re-layout, not by
        # switching rotary kernel mode.
        rotary_mode = "half"
        num_tokens = query.shape[0]
        cos, sin = self._get_mrope_cos_sin_for_apply(num_tokens)

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
