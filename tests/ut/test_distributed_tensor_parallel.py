#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
from unittest.mock import MagicMock

import pytest
import torch
from vllm_ascend.distributed.tensor_parallel import (
    _gather_along_first_dim, _gather_along_last_dim,
    _reduce_scatter_along_first_dim, _reduce_scatter_along_last_dim,
    all_to_all_hp2sp, all_to_all_sp2hp)


@pytest.fixture
def test_tensor():
    return torch.randn(8, 16)


@pytest.fixture
def test_tensor_last_dim():
    return torch.randn(8, 16, 32)


@pytest.fixture
def mock_group(monkeypatch):
    group = MagicMock()

    # 模拟 torch.distributed 函数
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *_: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *_: 0)

    return group


@pytest.fixture(autouse=True)
def mock_npu_device(monkeypatch):
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)


def test_gather_along_first_dim(test_tensor, mock_group):
    result = _gather_along_first_dim(test_tensor, mock_group)
    assert result.shape == (32, 16)  # 8*4=32

    output_split_sizes = [5, 10, 15, 2]
    result = _gather_along_first_dim(test_tensor, mock_group, output_split_sizes)
    assert result.shape == (32, 16)  # 5+10+15+2=32


def test_gather_along_last_dim(test_tensor_last_dim, mock_group):
    result = _gather_along_last_dim(test_tensor_last_dim, mock_group)
    assert result.shape == (8, 16, 128)  # 32*4=128


@pytest.mark.parametrize("input_shape,expected_shape", [
    ((32, 16), (8, 16)),
    ((40, 10), (10, 10)),
])
def test_reduce_scatter_along_first_dim(mock_group, input_shape, expected_shape):
    input_tensor = torch.randn(*input_shape)
    result = _reduce_scatter_along_first_dim(input_tensor, mock_group)
    assert result.shape == expected_shape


def test_reduce_scatter_along_last_dim(mock_group):
    input_tensor = torch.randn(8, 16, 32)
    result = _reduce_scatter_along_last_dim(input_tensor, mock_group)
    assert result.shape == (8, 16, 8)  # 32/4=8


@pytest.mark.parametrize("func,input_shape,expected_shape", [
    ("all_gather_last_dim_from_tensor_parallel_region", (8, 16, 32),
     (8, 16, 128)),
    ("reduce_scatter_to_sequence_parallel_region", (32, 16), (8, 16)),
    ("reduce_scatter_last_dim_to_tensor_parallel_region", (8, 16, 32),
     (8, 16, 8)),
    ("gather_from_sequence_parallel_region", (8, 16), (32, 16)),
])
def test_wrapper_functions(mock_group, func, input_shape, expected_shape):
    from vllm_ascend.distributed import tensor_parallel as tp
    test_func = getattr(tp, func)

    input_tensor = torch.randn(*input_shape)
    result = test_func(input_tensor, mock_group)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [((8, 16), (32, 4))]  # [num_tokens/TP, H] -> [num_tokens, H/TP]
)
def test_all_to_all_sp2hp(mock_group, input_shape, output_shape):
    input_tensor = torch.randn(*input_shape)
    result = all_to_all_sp2hp(input_tensor, mock_group)
    assert result.shape == output_shape


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [((32, 4), (8, 16))]  # [num_tokens, H/TP] -> [num_tokens/TP, H]
)
def test_all_to_all_hp2sp(mock_group, input_shape, output_shape):
    input_tensor = torch.randn(*input_shape)
    result = all_to_all_hp2sp(input_tensor, mock_group)
    assert result.shape == output_shape