# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from vllm_ascend.ops.triton.rope import _get_block_size_head


def test_rope_head_tile_uses_actual_tp_shard_size():
    assert _get_block_size_head(n_q_head=8, n_kv_head=4, is_neox_style=True) == 8


def test_rope_head_tile_preserves_default_caps():
    assert _get_block_size_head(n_q_head=96, n_kv_head=8, is_neox_style=True) == 64
    assert _get_block_size_head(n_q_head=40, n_kv_head=8, is_neox_style=False) == 32
