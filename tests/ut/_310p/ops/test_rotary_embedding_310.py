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

from vllm_ascend._310p.ops import rotary_embedding as rotary_310
from vllm_ascend._310p.ops.rotary_embedding import (
    AscendMRotaryEmbedding310,
    begin_mrope_forward_310,
)


def _reset_mrope_globals():
    rotary_310._mrope_cos_slice = None
    rotary_310._mrope_sin_slice = None
    rotary_310._mrope_forward_id = 0
    rotary_310._prepared_mrope_forward_id = -1
    rotary_310._prepared_mrope_dtype = None
    rotary_310._prepared_mrope_device = None
    rotary_310._prepared_mrope_interleaved = None
    rotary_310._prepared_mrope_section = None


def _build_mrope_embedding() -> AscendMRotaryEmbedding310:
    emb = AscendMRotaryEmbedding310.__new__(AscendMRotaryEmbedding310)
    emb.mrope_section = [2, 2, 2]
    emb.mrope_interleaved = False
    emb.cos_sin_cache = torch.randn(64, 12, dtype=torch.float32)
    return emb


def test_begin_mrope_forward_310_increases_forward_id():
    _reset_mrope_globals()
    before = rotary_310._mrope_forward_id
    begin_mrope_forward_310()
    assert rotary_310._mrope_forward_id == before + 1


def test_get_or_prepare_mrope_cos_sin_reuses_cache_in_same_forward():
    _reset_mrope_globals()
    emb = _build_mrope_embedding()
    positions = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 4), dtype=torch.long)
    query = torch.randn(4, 16, dtype=torch.float32)

    cos_1, sin_1 = emb._get_or_prepare_mrope_cos_sin(positions, query)
    cos_2, sin_2 = emb._get_or_prepare_mrope_cos_sin(positions, query)

    assert cos_1 is cos_2
    assert sin_1 is sin_2


def test_get_or_prepare_mrope_cos_sin_refresh_after_new_forward():
    _reset_mrope_globals()
    emb = _build_mrope_embedding()
    positions = torch.randint(0, emb.cos_sin_cache.shape[0], (3, 4), dtype=torch.long)
    query = torch.randn(4, 16, dtype=torch.float32)

    cos_1, sin_1 = emb._get_or_prepare_mrope_cos_sin(positions, query)
    begin_mrope_forward_310()
    cos_2, sin_2 = emb._get_or_prepare_mrope_cos_sin(positions, query)

    assert cos_1 is not cos_2
    assert sin_1 is not sin_2
