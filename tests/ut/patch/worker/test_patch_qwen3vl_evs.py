#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from unittest.mock import MagicMock

from vllm_ascend.patch.worker import patch_qwen3vl


def test_evs_recompute_mrope_positions_moves_positions_to_embedding_device():
    original = MagicMock(return_value="positions")
    moved_positions = object()
    mrope_positions = MagicMock()
    mrope_positions.to.return_value = moved_positions
    embedding = MagicMock(device="npu:0")
    wrapper = patch_qwen3vl._evs_recompute_mrope_positions_wrap(original)

    result = wrapper(object(), [1, 2], [embedding], mrope_positions, 7)

    assert result == "positions"
    mrope_positions.to.assert_called_once_with("npu:0")
    original.assert_called_once()
    assert original.call_args.args[3] is moved_positions


def test_evs_recompute_mrope_positions_preserves_cpu_positions_without_embeddings():
    original = MagicMock(return_value="positions")
    mrope_positions = MagicMock()
    wrapper = patch_qwen3vl._evs_recompute_mrope_positions_wrap(original)

    wrapper(object(), [1], [], mrope_positions, 0)

    mrope_positions.to.assert_not_called()
    assert original.call_args.args[3] is mrope_positions
