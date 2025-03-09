#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/blob/main/tests/models/utils.py
# Copyright 2023 The vLLM team.
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
from vllm_ascend.patch.patch_linear_method import linear_without_trans, transpose_linear_weights


@pytest.fixture
def linear_layer(request):
    input_dim, output_dim, device = request.param
    layer = torch.nn.Linear(input_dim, output_dim).to(device)
    original_weight = layer.weight.data.clone()
    return (layer, original_weight)


@pytest.mark.parametrize(
    "linear_layer", 
    [(5, 3, "cpu"),
     (12, 7, "cpu"),
     (6, 8, "npu:0"),
     (12, 9, "npu:1")],
    indirect=True)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_transpose_linear_weights(linear_layer, dtype):
    layer, original_weight = linear_layer
    layer = layer.to(dtype)
    original_weight = original_weight.to(dtype)

    transpose_linear_weights(None, layer)

    assert layer.weight.data.shape == torch.Size([layer.in_features, layer.out_features])
    assert torch.allclose(layer.weight.data, original_weight.transpose(0, 1))


@pytest.mark.parametrize(
    "linear_layer", 
    [(2, 5, "cpu"),
     (3, 2, "cpu"),
     (6, 7, "npu:0"),
     (11, 6, "npu:1")],
    indirect=True)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_linear_without_trans(linear_layer, dtype):
    layer, original_weight = linear_layer
    layer = layer.to(dtype)
    original_weight = original_weight.to(dtype)
    
    transpose_linear_weights(None, layer)

    x = torch.randn(2, layer.in_features).to(dtype).to(layer.weight.device)
    output_no_bias = linear_without_trans(None, layer, x)
    expected_output_no_bias = F.linear(x, original_weight)
    assert torch.allclose(output_no_bias, expected_output_no_bias)

    bias = torch.randn(layer.out_features).to(dtype).to(layer.weight.device)
    output_with_bias = linear_without_trans(None, layer, x, bias)
    expected_output_with_bias = F.linear(x, original_weight, bias)
    assert torch.allclose(output_with_bias, expected_output_with_bias)