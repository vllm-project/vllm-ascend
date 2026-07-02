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

from unittest import mock

import torch

from vllm_ascend._310p.ops.rotary_embedding import (
    AscendMRotaryEmbedding310,
    AscendRotaryEmbedding310,
    _select_apply_rotary_pos_emb_slices,
    _select_mrope_apply_rotary_slices,
)


def _build_mrope_embedding() -> AscendMRotaryEmbedding310:
    emb = AscendMRotaryEmbedding310.__new__(AscendMRotaryEmbedding310)
    emb.mrope_section = [2, 2, 2]
    emb.mrope_interleaved = False
    emb.cos_sin_cache = torch.randn(64, 12, dtype=torch.float32)
    emb.is_neox_style = True
    return emb


def _select_slices(emb: AscendMRotaryEmbedding310, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ref_tensor = torch.empty(positions.shape[-1], 1, 8, dtype=emb.cos_sin_cache.dtype)
    return _select_mrope_apply_rotary_slices(emb, positions, ref_tensor)


def test_select_mrope_apply_rotary_slices_populates_layer_buffers():
    emb = _build_mrope_embedding()
    positions = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 4), dtype=torch.long)
    cos, sin = _select_slices(emb, positions)

    assert cos.shape[1] == positions.shape[-1]
    assert sin.shape[1] == positions.shape[-1]
    assert emb._mrope_apply_cos_buffer.shape[1] == positions.shape[-1]


def test_select_mrope_apply_rotary_slices_reuses_buffer_address():
    emb = _build_mrope_embedding()
    positions = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 4), dtype=torch.long)

    _select_slices(emb, positions)
    first_ptr = emb._mrope_apply_cos_buffer.data_ptr()

    _select_slices(emb, positions)
    second_ptr = emb._mrope_apply_cos_buffer.data_ptr()

    assert first_ptr == second_ptr


def test_select_mrope_apply_rotary_slices_updates_existing_buffer_for_new_positions():
    emb = _build_mrope_embedding()
    first_positions = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=torch.long,
    )
    second_positions = first_positions + 12

    first_cos, first_sin = _select_slices(emb, first_positions)
    first_ptr = emb._mrope_apply_cos_buffer.data_ptr()
    first_cos_snapshot = first_cos.clone()
    first_sin_snapshot = first_sin.clone()

    second_cos, second_sin = _select_slices(emb, second_positions)

    assert emb._mrope_apply_cos_buffer.data_ptr() == first_ptr
    assert not torch.equal(second_cos, first_cos_snapshot)
    assert not torch.equal(second_sin, first_sin_snapshot)
    assert torch.equal(second_cos, emb._mrope_apply_cos_buffer[:, : second_positions.shape[-1]])
    assert torch.equal(second_sin, emb._mrope_apply_sin_buffer[:, : second_positions.shape[-1]])


def test_select_mrope_apply_rotary_slices_grows_buffer_and_reuses_capacity():
    emb = _build_mrope_embedding()
    short_positions = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 4), dtype=torch.long)
    long_positions = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 8), dtype=torch.long)

    _select_slices(emb, short_positions)
    assert emb._mrope_apply_cos_buffer.shape[1] == short_positions.shape[-1]

    _select_slices(emb, long_positions)
    grown_ptr = emb._mrope_apply_cos_buffer.data_ptr()
    assert emb._mrope_apply_cos_buffer.shape[1] == long_positions.shape[-1]

    cos, sin = _select_slices(emb, short_positions)
    assert emb._mrope_apply_cos_buffer.data_ptr() == grown_ptr
    assert emb._mrope_apply_cos_buffer.shape[1] == long_positions.shape[-1]
    assert cos.shape[1] == short_positions.shape[-1]
    assert sin.shape[1] == short_positions.shape[-1]


def test_select_mrope_apply_rotary_slices_supports_1d_positions():
    emb = _build_mrope_embedding()
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    cos, sin = _select_slices(emb, positions)
    expected_cos, expected_sin = emb.cos_sin_cache[positions].chunk(2, dim=-1)
    expected_cos = torch.cat((expected_cos, expected_cos), dim=-1).view(1, positions.shape[-1], 1, -1)
    expected_sin = torch.cat((expected_sin, expected_sin), dim=-1).view(1, positions.shape[-1], 1, -1)

    assert torch.equal(cos, expected_cos)
    assert torch.equal(sin, expected_sin)


def _build_rotary_embedding(
    *,
    head_size: int = 96,
    rotary_dim: int = 96,
    cache_dim: int = 96,
    is_neox_style: bool = True,
) -> AscendRotaryEmbedding310:
    emb = AscendRotaryEmbedding310.__new__(AscendRotaryEmbedding310)
    emb.head_size = head_size
    emb.rotary_dim = rotary_dim
    emb.is_neox_style = is_neox_style
    emb.cos_sin_cache = torch.randn(64, cache_dim, dtype=torch.float32)
    return emb


def test_310p_rotary_embedding_by_cache_path_does_not_select_legacy_slices():
    emb = _build_rotary_embedding()
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    query = torch.randn(positions.numel(), emb.head_size)
    key = torch.randn_like(query)

    with (
        mock.patch("vllm_ascend._310p.ops.rotary_embedding.select_cos_sin_cache") as select_cache,
        mock.patch(
            "vllm_ascend._310p.ops.rotary_embedding.torch_npu._npu_rotary_embedding",
            create=True,
        ) as npu_rotary,
    ):
        out_q, out_k = emb.forward_oot(positions, query, key)

    select_cache.assert_not_called()
    npu_rotary.assert_called_once()
    assert torch.equal(out_q, query)
    assert torch.equal(out_k, key)


def test_310p_apply_rotary_slices_are_selected_only_for_legacy_apply_op():
    emb = _build_rotary_embedding(head_size=128, rotary_dim=128, cache_dim=128)
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    query = torch.randn(positions.numel(), emb.head_size)
    selected_cache = torch.randn(positions.numel(), 128)

    with mock.patch(
        "vllm_ascend._310p.ops.rotary_embedding.select_cos_sin_cache",
        return_value=selected_cache,
    ) as select_cache:
        cos, sin = _select_apply_rotary_pos_emb_slices(emb, positions, query)

    select_cache.assert_called_once_with(emb, positions, query)
    expected_cos, expected_sin = selected_cache.chunk(2, dim=-1)
    expected_cos = torch.cat((expected_cos, expected_cos), dim=-1).view(1, positions.shape[-1], 1, -1)
    expected_sin = torch.cat((expected_sin, expected_sin), dim=-1).view(1, positions.shape[-1], 1, -1)
    assert torch.equal(cos, expected_cos)
    assert torch.equal(sin, expected_sin)
