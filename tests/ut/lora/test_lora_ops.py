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
