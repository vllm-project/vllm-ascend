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

from __future__ import annotations

from typing import Any

import torch
import torch_npu
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope

from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding, get_cos_and_sin_slice

# Filled once per model forward in NPUModelRunner310._model_forward; read by every MRoPE layer.
_mrope_cos_slice: torch.Tensor | None = None
_mrope_sin_slice: torch.Tensor | None = None


def _apply_rotary_mrope_torch(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch path aligned with vLLM MRotaryEmbedding.forward_native -> ApplyRotaryEmb."""
    half = cos.shape[-1] // 2
    cos_h = cos[0, :, 0, :half].contiguous()
    sin_h = sin[0, :, 0, :half].contiguous()
    q_out = ApplyRotaryEmb.forward_static(q_rot[0], cos_h, sin_h, is_neox_style)
    k_out = ApplyRotaryEmb.forward_static(k_rot[0], cos_h, sin_h, is_neox_style)
    return q_out.unsqueeze(0), k_out.unsqueeze(0)


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
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    mrope_section: list[int] | None = None,
    mrope_interleaved: bool = False,
    capacity_tokens: int = 0,
) -> None:
    """Build cos/sin views for `npu_apply_rotary_pos_emb` from positions; must run once per forward before layers."""
    global _mrope_cos_slice
    global _mrope_sin_slice

    assert positions.ndim in (1, 2), "M-RoPE positions must be [num_tokens] or [3, num_tokens]."
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        assert positions.shape[0] == 3, "MRoPE expects positions [3, num_tokens] (T/H/W)."
        assert mrope_section is not None
        cos, sin = merge_mrope_cos_sin_for_apply(
            cos,
            sin,
            list(mrope_section),
            mrope_interleaved,
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
    if _mrope_cos_slice is None or _mrope_sin_slice is None:
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

    _mrope_cos_slice[:, :num_tokens].copy_(cos_view)
    _mrope_sin_slice[:, :num_tokens].copy_(sin_view)


# Persistent draft cos/sin buffers, mirroring the global _cos/_sin used by the
# main model. npu_apply_rotary_pos_emb requires cos/sin to be slices of a
# persistently-allocated (stream-registered) buffer; freshly-allocated tensors
# fail with an allocator/stream error ("aclrtAllocatorGetByStream failed").
# These are draft-local so a draft whose rotary dim differs from the main
# model's (e.g. VL main + text dflash draft) works without touching the main
# model's buffers.
_draft_cos: torch.Tensor | None = None
_draft_sin: torch.Tensor | None = None
_draft_rope_dim: int | None = None
_draft_min_capacity_tokens = 0
_full_decode_query_cos: torch.Tensor | None = None
_full_decode_query_sin: torch.Tensor | None = None
_full_decode_context_cos: torch.Tensor | None = None
_full_decode_context_sin: torch.Tensor | None = None
_full_decode_rope_precomputed = False
_full_decode_rope_source = "query"


def configure_draft_rope_capacity_310(capacity_tokens: int) -> None:
    """Reserve stable draft RoPE storage before graph capture.

    DFlash uses the same buffers for profile/capture query batches and runtime
    context RoPE.  Reserving the runner token budget up front prevents a later
    context batch from replacing tensors already bound to a Piecewise graph.
    """
    global _draft_cos, _draft_sin, _draft_min_capacity_tokens

    capacity_tokens = int(capacity_tokens)
    if capacity_tokens <= _draft_min_capacity_tokens:
        return
    _draft_min_capacity_tokens = capacity_tokens

    if _draft_cos is None or _draft_cos.shape[1] >= capacity_tokens:
        return
    old_capacity = _draft_cos.shape[1]
    new_cos = torch.ones(
        1,
        capacity_tokens,
        1,
        _draft_cos.shape[-1],
        dtype=_draft_cos.dtype,
        device=_draft_cos.device,
    )
    new_sin = torch.zeros_like(new_cos)
    new_cos[:, :old_capacity].copy_(_draft_cos)
    assert _draft_sin is not None
    new_sin[:, :old_capacity].copy_(_draft_sin)
    _draft_cos = new_cos
    _draft_sin = new_sin


def _build_draft_cos_sin_slice(
    cos_sin_cache: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin slices for the ``head_size==128`` / ``rotary_dim==64`` NPU
    apply-rotary paths from a rotary's own ``cos_sin_cache``.

    Mirrors ``update_cos_sin`` (persistent buffer + slice) but is draft-local, so
    a draft model whose rotary dim differs from the main model's works correctly
    (e.g. VL main + text dflash draft). Returns tensors shaped
    ``[1, num_tokens, 1, rotary_dim]`` matching ``get_cos_and_sin_slice``. The
    ``[:, :num_tokens]`` slice stays contiguous because the leading dim is 1.
    """
    global _draft_cos, _draft_sin, _draft_rope_dim
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"draft RoPE positions must use int32 or int64 indices, got {positions.dtype}")
    num_tokens = positions.size(0)
    rope_dim = cos_sin_cache.shape[-1]
    if (
        _draft_cos is None
        or _draft_rope_dim != rope_dim
        or _draft_cos.shape[1] < num_tokens
        or _draft_cos.device != cos_sin_cache.device
        or _draft_cos.dtype != cos_sin_cache.dtype
    ):
        capacity = max(
            num_tokens,
            _draft_min_capacity_tokens,
            0 if _draft_cos is None else _draft_cos.shape[1],
        )
        _draft_cos = torch.ones(1, capacity, 1, rope_dim, dtype=cos_sin_cache.dtype, device=cos_sin_cache.device)
        _draft_sin = torch.zeros(1, capacity, 1, rope_dim, dtype=cos_sin_cache.dtype, device=cos_sin_cache.device)
        _draft_rope_dim = rope_dim

    draft_cos = _draft_cos
    draft_sin = _draft_sin
    assert draft_cos is not None and draft_sin is not None

    sel = cos_sin_cache.index_select(0, positions).view(num_tokens, 2, -1).repeat(1, 1, 2)
    draft_cos[:, :num_tokens] = sel.chunk(2, dim=-2)[0]
    draft_sin[:, :num_tokens] = sel.chunk(2, dim=-2)[1]
    return draft_cos[:, :num_tokens], draft_sin[:, :num_tokens]


def _ensure_full_decode_rope_pair(
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    cos_sin_cache: torch.Tensor,
    capacity_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope_dim = cos_sin_cache.shape[-1]
    if (
        cos is None
        or sin is None
        or cos.shape[1] < capacity_tokens
        or cos.shape[-1] != rope_dim
        or cos.device != cos_sin_cache.device
        or cos.dtype != cos_sin_cache.dtype
    ):
        cos = torch.ones(
            1,
            capacity_tokens,
            1,
            rope_dim,
            dtype=cos_sin_cache.dtype,
            device=cos_sin_cache.device,
        )
        sin = torch.zeros_like(cos)
    return cos, sin


def _populate_full_decode_rope_pair(
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> None:
    num_tokens = positions.shape[0]
    cos[:, num_tokens:].fill_(1)
    sin[:, num_tokens:].zero_()
    selected = cos_sin_cache.index_select(0, positions).view(num_tokens, 2, -1).repeat(1, 1, 2)
    cos[:, :num_tokens].copy_(selected.chunk(2, dim=-2)[0])
    sin[:, :num_tokens].copy_(selected.chunk(2, dim=-2)[1])


def prepare_full_decode_draft_rope_310(
    cos_sin_cache: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    query_actual_tokens: int,
    context_positions: torch.Tensor,
    context_actual_tokens: int,
    capacity_tokens: int,
) -> None:
    """Prepare distinct context/query RoPE inputs outside the FULL graph.

    The 310P ACL graph replay fault is a GatherV2 over the draft RoPE cache.
    Context KV insertion and query decoding use different position vectors, so
    each source owns a separate persistent cos/sin pair. Sharing one pair makes
    context K/V use query positions and silently cuts speculative acceptance.
    """
    global _full_decode_query_cos, _full_decode_query_sin
    global _full_decode_context_cos, _full_decode_context_sin
    global _full_decode_rope_precomputed

    for name, positions, actual_tokens in (
        ("query", query_positions, query_actual_tokens),
        ("context", context_positions, context_actual_tokens),
    ):
        if positions.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"FULL draft {name} RoPE positions must use int32 or int64 indices, got {positions.dtype}")
        if positions.ndim != 1:
            raise ValueError(f"FULL draft {name} RoPE positions must be one-dimensional")
        if not 0 <= actual_tokens <= positions.shape[0]:
            raise ValueError(
                f"FULL draft {name} RoPE active extent is invalid: "
                f"actual={actual_tokens}, descriptor={positions.shape[0]}"
            )
        positions[actual_tokens:].zero_()

    query_ptr = query_positions.data_ptr()
    context_ptr = context_positions.data_ptr()
    if query_ptr == context_ptr:
        raise ValueError("FULL draft context/query position storage must differ")

    capacity_tokens = max(
        int(capacity_tokens),
        query_positions.shape[0],
        context_positions.shape[0],
    )
    _full_decode_query_cos, _full_decode_query_sin = _ensure_full_decode_rope_pair(
        _full_decode_query_cos,
        _full_decode_query_sin,
        cos_sin_cache=cos_sin_cache,
        capacity_tokens=capacity_tokens,
    )
    _full_decode_context_cos, _full_decode_context_sin = _ensure_full_decode_rope_pair(
        _full_decode_context_cos,
        _full_decode_context_sin,
        cos_sin_cache=cos_sin_cache,
        capacity_tokens=capacity_tokens,
    )
    _populate_full_decode_rope_pair(
        cos_sin_cache,
        query_positions,
        _full_decode_query_cos,
        _full_decode_query_sin,
    )
    _populate_full_decode_rope_pair(
        cos_sin_cache,
        context_positions,
        _full_decode_context_cos,
        _full_decode_context_sin,
    )
    _full_decode_rope_precomputed = True


def clear_full_decode_draft_rope_310() -> None:
    global _full_decode_rope_precomputed
    global _full_decode_rope_source
    _full_decode_rope_precomputed = False
    _full_decode_rope_source = "query"


def set_full_decode_draft_rope_source_310(source: str) -> str:
    """Select which precomputed pair the next draft rotary calls consume."""
    if source not in ("query", "context"):
        raise ValueError(f"unsupported FULL draft RoPE source: {source}")
    global _full_decode_rope_source
    previous = _full_decode_rope_source
    _full_decode_rope_source = source
    return previous


def get_full_decode_draft_rope_buffers_310() -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    return (
        _full_decode_query_cos,
        _full_decode_query_sin,
        _full_decode_context_cos,
        _full_decode_context_sin,
    )


def _select_draft_cos_sin_slice(
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _full_decode_rope_precomputed:
        return _build_draft_cos_sin_slice(cos_sin_cache, positions)
    if _full_decode_rope_source == "context":
        pair = (_full_decode_context_cos, _full_decode_context_sin)
    else:
        pair = (_full_decode_query_cos, _full_decode_query_sin)
    cos, sin = pair
    if cos is None or sin is None:
        raise RuntimeError(f"FULL draft RoPE source has no persistent buffers: source={_full_decode_rope_source}")
    num_tokens = positions.shape[0]
    if cos.shape[1] < num_tokens or sin.shape[1] < num_tokens:
        raise RuntimeError(
            "FULL draft RoPE buffer is smaller than its position source: "
            f"tokens={num_tokens}, cos={cos.shape[1]}, sin={sin.shape[1]}"
        )
    return cos[:, :num_tokens], sin[:, :num_tokens]


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

    # This flag should set to True when doing drafting.
    if getattr(self, "_is_drafting_update_enabled", False):
        # Draft models build cos/sin from THIS rotary's own cos_sin_cache rather
        # than the global _cos/_sin buffers. The global buffers/_cos_sin_cache
        # belong to the main model, whose rotary dim can differ from the draft's
        # (e.g. a VL main model using MRoPE + a text dflash draft), which would
        # otherwise corrupt cos/sin or raise a shape/index error in
        # update_cos_sin.
        cos, sin = _select_draft_cos_sin_slice(self.cos_sin_cache, positions)
    else:
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


class AscendMRotaryEmbedding310(MRotaryEmbedding):
    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ):
        query_shape, key_shape = query.shape, key.shape

        # MRoPE T/H/W layout is handled in `merge_mrope_cos_sin_for_apply` (mrope_interleaved).
        # Here `rotary_mode` matches vLLM ApplyRotaryEmb: half = neox chunk, interleave = GPT-J pairs.
        rotary_mode = "half" if self.is_neox_style else "interleave"
        num_tokens = query.shape[0]
        if _mrope_cos_slice is None or _mrope_sin_slice is None:
            raise RuntimeError(
                "MRoPE cos/sin slices are not initialized. Call set_mrope_apply_rotary_slices before forward."
            )
        cos, sin = _mrope_cos_slice[:, :num_tokens], _mrope_sin_slice[:, :num_tokens]

        is_partial_rope = self.rotary_dim < self.head_size
        if is_partial_rope:
            query = query.view(num_tokens, -1, self.head_size)
            key = key.view(num_tokens, -1, self.head_size)
            q_pass = query[..., self.rotary_dim :]
            k_pass = key[..., self.rotary_dim :]
            q_rot = query[..., : self.rotary_dim].contiguous().view(1, num_tokens, -1, self.rotary_dim)
            k_rot = key[..., : self.rotary_dim].contiguous().view(1, num_tokens, -1, self.rotary_dim)
        else:
            q_rot = query.contiguous().view(1, num_tokens, -1, self.head_size)
            k_rot = key.contiguous().view(1, num_tokens, -1, self.head_size)

        # `npu_apply_rotary_pos_emb` only supports rotary_dim 64 or 128.
        use_npu_apply = self.rotary_dim in (64, 128)

        if use_npu_apply:
            q_rot, k_rot = torch_npu.npu_apply_rotary_pos_emb(q_rot, k_rot, cos, sin, rotary_mode=rotary_mode)
        else:
            q_rot, k_rot = _apply_rotary_mrope_torch(q_rot, k_rot, cos, sin, self.is_neox_style)

        if is_partial_rope:
            q_rot = q_rot.view(num_tokens, -1, self.rotary_dim)
            k_rot = k_rot.view(num_tokens, -1, self.rotary_dim)
            query = torch.cat((q_rot, q_pass), dim=-1).reshape(query_shape)
            key = torch.cat((k_rot, k_pass), dim=-1).reshape(key_shape)
        else:
            query = q_rot.view(query_shape)
            key = k_rot.view(key_shape)

        return query, key


def prepare_mrope_cos_sin_slices_from_runner(runner: Any, positions: torch.Tensor) -> None:
    """Resolve MRoPE embedding from the runner and populate `_mrope_cos_slice` / `_mrope_sin_slice`."""
    emb = getattr(runner, "_mrope_embedding", None)
    if emb is None:
        emb = next(module for module in runner.model.modules() if isinstance(module, AscendMRotaryEmbedding310))
        runner._mrope_embedding = emb
    assert isinstance(emb, AscendMRotaryEmbedding310)
    set_mrope_apply_rotary_slices(
        emb.cos_sin_cache,
        positions,
        mrope_section=emb.mrope_section,
        mrope_interleaved=emb.mrope_interleaved,
        capacity_tokens=runner.max_num_tokens,
    )


class AscendRotaryEmbedding310(AscendRotaryEmbedding):
    _is_drafting_update_enabled: bool = False

    @classmethod
    def set_rope_position_flag_310p(cls, state: bool):
        cls._is_drafting_update_enabled = state

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
