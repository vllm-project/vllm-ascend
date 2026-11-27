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

import torch

from vllm_ascend._310p.ops import rotary_embedding as rotary_310
from vllm_ascend._310p.ops.rotary_embedding import (
    AscendMRotaryEmbedding310,
    AscendRotaryEmbedding310,
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


def test_rope_forward_oot_zero_rotary_dim_returns_unchanged():
    emb = AscendRotaryEmbedding310.__new__(AscendRotaryEmbedding310)
    emb.rotary_dim = 0
    positions = torch.arange(4, dtype=torch.long)
    query = torch.randn(4, 16)
    key = torch.randn(4, 16)

    q_out, k_out = rotary_310._rope_forward_oot(emb, positions, query, key, is_neox_style=True)

    assert torch.equal(q_out, query)
    assert torch.equal(k_out, key)


def test_rope_forward_oot_builds_cos_sin_when_slice_missing(monkeypatch):
    emb = AscendRotaryEmbedding310.__new__(AscendRotaryEmbedding310)
    emb.rotary_dim = 128
    emb.head_size = 128
    emb.cos_sin_cache = torch.randn(16, 128, dtype=torch.float32)

    captured = {}

    def fake_apply_rotary_pos_emb(query, key, cos, sin, rotary_mode="half"):
        captured["cos_shape"] = tuple(cos.shape)
        captured["sin_shape"] = tuple(sin.shape)
        captured["rotary_mode"] = rotary_mode
        return query, key

    monkeypatch.setattr(rotary_310, "get_cos_and_sin_slice", lambda: (None, None))
    monkeypatch.setattr(rotary_310.torch_npu, "npu_apply_rotary_pos_emb", fake_apply_rotary_pos_emb, raising=False)

    positions = torch.tensor([0, 2, 5], dtype=torch.long)
    query = torch.randn(3, 256, dtype=torch.float32)
    key = torch.randn(3, 256, dtype=torch.float32)

    q_out, k_out = rotary_310._rope_forward_oot(emb, positions, query, key, is_neox_style=True)

    assert q_out.shape == query.shape
    assert k_out.shape == key.shape
    assert captured == {"cos_shape": (1, 3, 1, 128), "sin_shape": (1, 3, 1, 128), "rotary_mode": "half"}
