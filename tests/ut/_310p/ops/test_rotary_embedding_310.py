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

import pytest
import torch

from vllm_ascend._310p.ops import rotary_embedding as rotary_310
from vllm_ascend._310p.ops.rotary_embedding import (
    AscendMRotaryEmbedding310,
    AscendRotaryEmbedding310,
    _build_draft_cos_sin_slice,
    set_mrope_apply_rotary_slices,
)


def _reset_mrope_globals():
    rotary_310._mrope_cos_slice = None
    rotary_310._mrope_sin_slice = None


def _build_mrope_embedding() -> AscendMRotaryEmbedding310:
    emb = AscendMRotaryEmbedding310.__new__(AscendMRotaryEmbedding310)
    emb.mrope_section = [2, 2, 2]
    emb.mrope_interleaved = False
    emb.cos_sin_cache = torch.randn(64, 12, dtype=torch.float32)
    return emb


def test_set_mrope_apply_rotary_slices_populates_globals():
    _reset_mrope_globals()
    emb = _build_mrope_embedding()
    positions = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 4), dtype=torch.long)
    set_mrope_apply_rotary_slices(
        emb.cos_sin_cache,
        positions,
        mrope_section=emb.mrope_section,
        mrope_interleaved=emb.mrope_interleaved,
    )

    assert rotary_310._mrope_cos_slice is not None
    assert rotary_310._mrope_sin_slice is not None
    assert rotary_310._mrope_cos_slice.shape[1] == positions.shape[-1]


def test_set_mrope_apply_rotary_slices_reuses_buffer_address():
    _reset_mrope_globals()
    emb = _build_mrope_embedding()
    positions = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 4), dtype=torch.long)

    set_mrope_apply_rotary_slices(
        emb.cos_sin_cache,
        positions,
        mrope_section=emb.mrope_section,
        mrope_interleaved=emb.mrope_interleaved,
    )
    first_ptr = rotary_310._mrope_cos_slice.data_ptr()

    set_mrope_apply_rotary_slices(
        emb.cos_sin_cache,
        positions,
        mrope_section=emb.mrope_section,
        mrope_interleaved=emb.mrope_interleaved,
    )
    second_ptr = rotary_310._mrope_cos_slice.data_ptr()

    assert first_ptr == second_ptr


def test_ascend_rotary_embedding_310_drafting_flag():
    assert hasattr(AscendRotaryEmbedding310, "_is_drafting_update_enabled")
    assert AscendRotaryEmbedding310._is_drafting_update_enabled is False
    AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
    assert AscendRotaryEmbedding310._is_drafting_update_enabled is True
    AscendRotaryEmbedding310.set_rope_position_flag_310p(False)
    assert AscendRotaryEmbedding310._is_drafting_update_enabled is False


def _reset_draft_globals():
    rotary_310._draft_cos = None
    rotary_310._draft_sin = None
    rotary_310._draft_rope_dim = None
    rotary_310._draft_min_capacity_tokens = 0
    clear_precomputed = getattr(
        rotary_310,
        "clear_full_decode_draft_rope_310",
        None,
    )
    if callable(clear_precomputed):
        clear_precomputed()


def test_build_draft_cos_sin_slice_uses_own_cache():
    # Draft rotary dim (128) may differ from the main model's; the slice must be
    # built from the passed cache, independent of the main model's global buffers.
    _reset_draft_globals()
    try:
        rotary_dim = 128
        cos_sin_cache = torch.randn(64, rotary_dim, dtype=torch.float32)
        positions = torch.arange(5, dtype=torch.long)

        cos, sin = _build_draft_cos_sin_slice(cos_sin_cache, positions)

        assert tuple(cos.shape) == (1, 5, 1, rotary_dim)
        assert tuple(sin.shape) == (1, 5, 1, rotary_dim)
        # npu_apply_rotary_pos_emb requires contiguous cos/sin; the leading-dim-1
        # slice of the persistent buffer stays contiguous.
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        # cos/sin are the two halves derived from the selected cache rows.
        expected = cos_sin_cache.index_select(0, positions).view(5, 2, -1).repeat(1, 1, 2)
        torch.testing.assert_close(cos, expected.chunk(2, dim=-2)[0].reshape(1, 5, 1, rotary_dim))
        torch.testing.assert_close(sin, expected.chunk(2, dim=-2)[1].reshape(1, 5, 1, rotary_dim))
    finally:
        _reset_draft_globals()


def test_build_draft_cos_sin_slice_reuses_buffer_address():
    # Second call with <= capacity must reuse the same persistent buffer.
    _reset_draft_globals()
    try:
        cos_sin_cache = torch.randn(64, 128, dtype=torch.float32)
        cos1, _ = _build_draft_cos_sin_slice(cos_sin_cache, torch.arange(8, dtype=torch.long))
        first_ptr = cos1.data_ptr()
        cos2, _ = _build_draft_cos_sin_slice(cos_sin_cache, torch.arange(4, dtype=torch.long))
        assert cos2.data_ptr() == first_ptr
        assert tuple(cos2.shape) == (1, 4, 1, 128)
    finally:
        _reset_draft_globals()


def test_build_draft_cos_sin_slice_rejects_non_integral_positions():
    _reset_draft_globals()
    try:
        cos_sin_cache = torch.randn(64, 128, dtype=torch.float32)
        positions = torch.tensor([0.0, 1.0], dtype=torch.float32)

        with pytest.raises(
            TypeError,
            match="draft RoPE positions must use int32 or int64 indices",
        ):
            _build_draft_cos_sin_slice(cos_sin_cache, positions)
    finally:
        _reset_draft_globals()


def test_configured_draft_capacity_prevents_runtime_growth_reallocation():
    _reset_draft_globals()
    try:
        rotary_310.configure_draft_rope_capacity_310(64)
        cos_sin_cache = torch.randn(128, 128, dtype=torch.float32)

        cos1, sin1 = _build_draft_cos_sin_slice(
            cos_sin_cache,
            torch.arange(8, dtype=torch.long),
        )
        cos_ptr = cos1.data_ptr()
        sin_ptr = sin1.data_ptr()
        cos2, sin2 = _build_draft_cos_sin_slice(
            cos_sin_cache,
            torch.arange(48, dtype=torch.long),
        )

        assert rotary_310._draft_cos.shape[1] == 64
        assert rotary_310._draft_sin.shape[1] == 64
        assert cos2.data_ptr() == cos_ptr
        assert sin2.data_ptr() == sin_ptr
    finally:
        _reset_draft_globals()


def test_full_decode_precompute_keeps_context_and_query_rope_distinct():
    """Context KV and query tokens must never share one precomputed RoPE slice."""
    _reset_draft_globals()
    prepare = getattr(
        rotary_310,
        "prepare_full_decode_draft_rope_310",
        None,
    )
    select = getattr(
        rotary_310,
        "_select_draft_cos_sin_slice",
        None,
    )
    select_source = getattr(
        rotary_310,
        "set_full_decode_draft_rope_source_310",
        None,
    )
    assert callable(prepare), "dual-source FULL draft RoPE preparation is missing"
    assert callable(select), "precomputed FULL draft RoPE routing is missing"
    assert callable(select_source), "explicit FULL draft RoPE source routing is missing"

    try:
        cos_sin_cache = torch.arange(32 * 8, dtype=torch.float32).view(32, 8)
        query_positions = torch.tensor([1, 2, 19, 23], dtype=torch.int32)
        context_positions = torch.tensor([5, 6, 17, 29], dtype=torch.int32)

        prepare(
            cos_sin_cache,
            query_positions=query_positions,
            query_actual_tokens=2,
            context_positions=context_positions,
            context_actual_tokens=2,
            capacity_tokens=8,
        )

        query_cos, query_sin = select(cos_sin_cache, query_positions)
        previous_source = select_source("context")
        try:
            context_cos, context_sin = select(
                cos_sin_cache,
                context_positions,
            )
        finally:
            select_source(previous_source)
        expected_query = cos_sin_cache.index_select(0, query_positions)
        expected_query = expected_query.view(4, 2, -1).repeat(1, 1, 2)
        expected_context = cos_sin_cache.index_select(0, context_positions)
        expected_context = expected_context.view(4, 2, -1).repeat(1, 1, 2)

        torch.testing.assert_close(
            query_cos,
            expected_query.chunk(2, dim=-2)[0].reshape(1, 4, 1, 8),
        )
        torch.testing.assert_close(
            query_sin,
            expected_query.chunk(2, dim=-2)[1].reshape(1, 4, 1, 8),
        )
        torch.testing.assert_close(
            context_cos,
            expected_context.chunk(2, dim=-2)[0].reshape(1, 4, 1, 8),
        )
        torch.testing.assert_close(
            context_sin,
            expected_context.chunk(2, dim=-2)[1].reshape(1, 4, 1, 8),
        )
        assert query_cos.data_ptr() != context_cos.data_ptr()
        torch.testing.assert_close(query_positions[2:], torch.zeros(2, dtype=torch.int32))
        torch.testing.assert_close(
            context_positions[2:],
            torch.zeros(2, dtype=torch.int32),
        )
    finally:
        _reset_draft_globals()
