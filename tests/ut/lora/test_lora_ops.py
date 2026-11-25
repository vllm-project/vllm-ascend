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
# This file is a part of the vllm-ascend project.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.lora import lora_ops


def _patch_c_ascend(**ops):
    fake = SimpleNamespace(**ops)
    return patch.object(torch.ops, "_C_ascend", fake, create=True)


def test_bgmv_shrink_forwards_scale_in_kernel_order() -> None:
    shrink = Mock(return_value="shrink")
    inputs = torch.ones(2, 4)
    weights = torch.ones(3, 8, 4)
    output = torch.zeros(2, 8)
    indices = torch.tensor([0, -1])
    with _patch_c_ascend(bgmv_shrink=shrink):
        assert lora_ops.bgmv_shrink(inputs, weights, output, indices, scaling=0.5) == "shrink"
    shrink.assert_called_once_with(inputs, weights, indices, output, 0.5)


def test_bgmv_expand_uses_full_output_width_and_drops_add_inputs() -> None:
    expand = Mock(return_value="expand")
    inputs = torch.ones(2, 8)
    weights = torch.ones(3, 16, 8)
    output = torch.zeros(2, 16)
    indices = torch.tensor([1, 0])
    with _patch_c_ascend(bgmv_expand=expand):
        assert lora_ops.bgmv_expand(inputs, weights, output, indices, add_inputs=False) == "expand"
    expand.assert_called_once_with(inputs, weights, indices, output, 0, 16)


def test_bgmv_expand_slice_forwards_offset_and_size() -> None:
    expand = Mock(return_value="slice")
    inputs = torch.ones(2, 8)
    weights = torch.ones(3, 4, 8)
    output = torch.zeros(2, 12)
    indices = torch.tensor([0, 1])
    with _patch_c_ascend(bgmv_expand=expand):
        assert (
            lora_ops.bgmv_expand_slice(inputs, weights, output, indices, slice_offset=4, slice_size=4, add_inputs=False)
            == "slice"
        )
    expand.assert_called_once_with(inputs, weights, indices, output, 4, 4)


def test_sgmv_wrappers_drop_unused_batch_metadata() -> None:
    shrink = Mock(return_value="sgmv_shrink")
    expand = Mock(return_value="sgmv_expand")
    inputs = torch.ones(4, 8)
    a_weights = torch.ones(2, 16, 8)
    b_weights = torch.ones(2, 32, 16)
    output = torch.zeros(4, 16)
    expand_out = torch.zeros(4, 32)
    seq_len = torch.tensor([2, 2])
    indices = torch.tensor([0, 1])
    unused = torch.tensor([0, 2])
    with _patch_c_ascend(sgmv_shrink=shrink, sgmv_expand=expand):
        assert lora_ops.sgmv_shrink(inputs, a_weights, output, unused, seq_len, indices, 2, 2, 4, 0.25) == "sgmv_shrink"
        assert (
            lora_ops.sgmv_expand(
                inputs,
                b_weights,
                expand_out,
                unused,
                seq_len,
                indices,
                2,
                2,
                4,
                add_inputs=True,
            )
            == "sgmv_expand"
        )
        assert (
            lora_ops.sgmv_expand_slice(
                inputs,
                b_weights,
                expand_out,
                unused,
                seq_len,
                indices,
                2,
                2,
                4,
                slice_offset=8,
                slice_size=16,
                add_inputs=True,
            )
            == "sgmv_expand"
        )
    shrink.assert_called_once_with(inputs, a_weights, indices, seq_len, output, 0.25)
    assert expand.call_args_list[0].args == (inputs, b_weights, indices, seq_len, expand_out, 0, 32)
    assert expand.call_args_list[1].args == (inputs, b_weights, indices, seq_len, expand_out, 8, 16)


def test_bgmv_torch_shrink_and_expand_slice_accuracy_and_masks() -> None:
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [2.0, 0.0, 1.0, 0.0]])
    a = torch.tensor(
        [
            [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0]]],
        ]
    )
    indices = torch.tensor([0, -1, 1])
    shrink_out = torch.full((3, 2), 99.0)
    lora_ops.bgmv_shrink_torch(inputs, a, shrink_out, indices, scaling=0.5)
    torch.testing.assert_close(shrink_out, torch.tensor([[0.5, 1.0], [0.0, 0.0], [1.0, 0.0]]))

    b = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[2.0, 0.0], [0.0, 2.0]]],
        ]
    )
    out_add = torch.ones(3, 4)
    lora_ops.bgmv_expand_slice_torch(shrink_out, b, out_add, indices, slice_offset=1, slice_size=2, add_inputs=True)
    expected_add = torch.tensor(
        [
            [1.0, 3.5, 6.5, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 3.0, 1.0, 1.0],
        ]
    )
    torch.testing.assert_close(out_add, expected_add)

    out_overwrite = torch.full((3, 2), 7.0)
    lora_ops.bgmv_expand_torch(shrink_out, b, out_overwrite, indices, add_inputs=False)
    expected_overwrite = torch.tensor([[2.5, 5.5], [7.0, 7.0], [2.0, 0.0]])
    torch.testing.assert_close(out_overwrite, expected_overwrite)


def test_bgmv_torch_handles_empty_and_mismatched_index_lengths() -> None:
    empty_in = torch.zeros(0, 4)
    empty_out = torch.full((0, 2), 1.0)
    lora_ops.bgmv_shrink_torch(empty_in, torch.ones(1, 2, 4), empty_out, torch.zeros(0, dtype=torch.long))
    lora_ops.bgmv_expand_slice_torch(empty_in, torch.ones(1, 4, 2), torch.ones(2, 4), torch.tensor([0]), 0, 4)

    inputs = torch.ones(3, 4)
    a = torch.ones(1, 2, 4)
    short_out = torch.zeros(3, 2)
    lora_ops.bgmv_shrink_torch(inputs, a, short_out, torch.tensor([0]), scaling=1.0)
    torch.testing.assert_close(short_out, torch.tensor([[4.0, 4.0], [0.0, 0.0], [0.0, 0.0]]))

    long_out = torch.zeros(3, 2)
    lora_ops.bgmv_shrink_torch(inputs, a, long_out, torch.tensor([0, -1, 0, 0]), scaling=0.25)
    torch.testing.assert_close(long_out, torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]))


def test_sgmv_torch_wrappers_match_bgmv_torch() -> None:
    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    a = torch.ones(2, 2, 4)
    b = torch.full((2, 4, 2), 0.5)
    seq_lens = torch.tensor([2, 1])
    batch_indices = torch.tensor([0, 1])
    dummy_loc = torch.tensor([0, 2])

    sgmv_buf = torch.zeros(3, 2)
    bgmv_buf = torch.zeros(3, 2)
    lora_ops.sgmv_shrink_torch(inputs, a, sgmv_buf, dummy_loc, seq_lens, batch_indices, 2, 2, 3, 0.5)
    lora_ops.bgmv_shrink_torch(inputs, a, bgmv_buf, torch.tensor([0, 0, 1]), scaling=0.5)
    torch.testing.assert_close(sgmv_buf, bgmv_buf)

    sgmv_out = torch.ones(3, 4)
    bgmv_out = torch.ones(3, 4)
    lora_ops.sgmv_expand_torch(sgmv_buf, b, sgmv_out, dummy_loc, seq_lens, batch_indices, 2, 2, 3, add_inputs=True)
    lora_ops.bgmv_expand_torch(bgmv_buf, b, bgmv_out, torch.tensor([0, 0, 1]), add_inputs=True)
    torch.testing.assert_close(sgmv_out, bgmv_out)

    sgmv_slice_out = torch.zeros(3, 6)
    bgmv_slice_out = torch.zeros(3, 6)
    lora_ops.sgmv_expand_slice_torch(
        sgmv_buf, b, sgmv_slice_out, dummy_loc, seq_lens, batch_indices, 2, 2, 3, 2, 4, add_inputs=True
    )
    lora_ops.bgmv_expand_slice_torch(bgmv_buf, b, bgmv_slice_out, torch.tensor([0, 0, 1]), 2, 4, add_inputs=True)
    torch.testing.assert_close(sgmv_slice_out, bgmv_slice_out)
