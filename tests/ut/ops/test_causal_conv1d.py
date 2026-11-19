#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.causal_conv1d import _causal_conv1d_npu


@pytest.mark.parametrize("has_initial_state", [False, True])
@pytest.mark.parametrize("seqlen", [1, 7])
def test_causal_conv1d_npu_fallback_matches_grouped_conv(monkeypatch, has_initial_state, seqlen):
    batch, dim, width = 2, 3, 4
    x = torch.randn(batch, dim, seqlen)
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)

    if has_initial_state:
        initial_states = torch.randn(batch, dim, width - 1)
        conv_input = torch.cat([initial_states, x], dim=-1)
        expected = F.conv1d(conv_input, weight.unsqueeze(1), bias, groups=dim)
    else:
        conv_input = F.pad(x, (width - 1, 0))
        expected = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)[..., :seqlen]

    def fail_conv1d(*args, **kwargs):
        pytest.fail("The NPU fallback must not dispatch to grouped Conv1D/Conv2D")

    monkeypatch.setattr(F, "conv1d", fail_conv1d)
    actual = _causal_conv1d_npu(conv_input, weight, bias, seqlen)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

