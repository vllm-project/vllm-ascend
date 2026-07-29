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
# This file is a part of the vllm-ascend project.

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.kimi_kda import (
    _load_a_log,
    _zero_padded_spec_output,
)


def test_load_a_log_slices_padded_1d_checkpoint_by_tp_rank():
    param = torch.empty(1, 1, 2, 1)
    loaded_weight = torch.arange(6, dtype=torch.float32)

    with patch("vllm_ascend.ops.kimi_kda.get_tensor_model_parallel_rank", return_value=1):
        _load_a_log(param, loaded_weight, num_heads=4)

    torch.testing.assert_close(param, torch.tensor([[[[2.0], [3.0]]]]))


def test_load_a_log_preserves_exact_local_4d_checkpoint():
    param = torch.empty(1, 1, 2, 1)
    loaded_weight = torch.tensor([[[[4.0], [5.0]]]])

    _load_a_log(param, loaded_weight, num_heads=4)

    torch.testing.assert_close(param, loaded_weight)


def test_load_a_log_rejects_unsupported_layout():
    with pytest.raises(ValueError, match="must be 1-D or 4-D"):
        _load_a_log(torch.empty(1, 1, 2, 1), torch.empty(2, 2), num_heads=4)


def test_zero_padded_spec_output_clears_uninitialized_tail():
    output = torch.arange(16 * 2 * 3, dtype=torch.float32).reshape(1, 16, 2, 3)
    output[:, 8:] = torch.nan
    query_start_loc = torch.tensor([0, 8, 8], dtype=torch.int32)

    masked = _zero_padded_spec_output(output, query_start_loc)

    torch.testing.assert_close(masked[:, :8], output[:, :8])
    assert torch.equal(masked[:, 8:], torch.zeros_like(masked[:, 8:]))
    assert torch.isfinite(masked).all()


def test_zero_padded_spec_output_preserves_fully_covered_output():
    output = torch.randn(1, 16, 2, 3)
    query_start_loc = torch.tensor([0, 8, 16], dtype=torch.int32)

    masked = _zero_padded_spec_output(output, query_start_loc)

    torch.testing.assert_close(masked, output)


def test_zero_padded_spec_output_supports_multiple_real_and_dummy_rows():
    output = torch.randn(1, 32, 2, 3)
    expected = output[:, :16].clone()
    output[:, 16:] = torch.nan
    query_start_loc = torch.tensor([0, 8, 16, 16, 16], dtype=torch.int32)

    masked = _zero_padded_spec_output(output, query_start_loc)

    torch.testing.assert_close(masked[:, :16], expected)
    assert torch.equal(masked[:, 16:], torch.zeros_like(masked[:, 16:]))
    assert masked.shape == output.shape
    assert masked.dtype == output.dtype
    assert masked.device == output.device
