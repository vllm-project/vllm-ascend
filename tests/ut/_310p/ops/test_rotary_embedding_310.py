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
    set_mrope_apply_rotary_slices(emb, positions, torch.float32, torch.device("cpu"))

    assert rotary_310._mrope_cos_slice is not None
    assert rotary_310._mrope_sin_slice is not None
    assert rotary_310._mrope_cos_slice.shape[1] == positions.shape[-1]


def test_set_mrope_apply_rotary_slices_second_call_replaces():
    _reset_mrope_globals()
    emb = _build_mrope_embedding()
    positions_a = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 4), dtype=torch.long)
    positions_b = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 5), dtype=torch.long)

    set_mrope_apply_rotary_slices(emb, positions_a, torch.float32, torch.device("cpu"))
    cos_a = rotary_310._mrope_cos_slice
    set_mrope_apply_rotary_slices(emb, positions_b, torch.float32, torch.device("cpu"))
    cos_b = rotary_310._mrope_cos_slice

    assert cos_a is not cos_b
    assert cos_b.shape[1] == 5


def test_get_mrope_cos_sin_for_apply_raises_when_unprepared():
    _reset_mrope_globals()
    emb = _build_mrope_embedding()
    with pytest.raises(RuntimeError, match="not prepared"):
        emb._get_mrope_cos_sin_for_apply()


def test_get_mrope_cos_sin_for_apply_after_prepare():
    _reset_mrope_globals()
    emb = _build_mrope_embedding()
    positions = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 4), dtype=torch.long)
    set_mrope_apply_rotary_slices(emb, positions, torch.float32, torch.device("cpu"))
    cos, sin = emb._get_mrope_cos_sin_for_apply()
    assert cos is rotary_310._mrope_cos_slice
    assert sin is rotary_310._mrope_sin_slice
