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

import importlib
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import PytestBase
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
def mock_group():
    return MagicMock()


@pytest.fixture(autouse=True)
def mock_dist():
    with patch("torch.distributed") as mock:
        mock.get_world_size.return_value = 4
        mock.get_rank.return_value = 0
        yield mock


class TestDistributedCommunication(PytestBase):

    @pytest.mark.parametrize("world_size", [1, 4])
    def test_gather_along_first_dim(self, test_tensor, mock_group, mock_dist,
                                    world_size):
        """test _gather_along_first_dim"""
        mock_dist.get_world_size.return_value = world_size

        result = _gather_along_first_dim(test_tensor, mock_group)

        if world_size == 1:
            assert result.shape == (8, 16)
        else:
            assert result.shape == (32, 16)  # 8*4=32

    def test_gather_along_first_dim_unequal_split(self, test_tensor,
                                                  mock_group):
        """test unequal split"""
        output_split_sizes = [5, 10, 15, 2]
        result = _gather_along_first_dim(test_tensor, mock_group,
                                         output_split_sizes)
        assert result.shape == (32, 16)  # 5+10+15+2=32

    @pytest.mark.parametrize("world_size", [1, 4])
    def test_gather_along_last_dim(self, test_tensor_last_dim, mock_group,
                                   mock_dist, world_size):
        """test _gather_along_last_dim"""
        mock_dist.get_world_size.return_value = world_size

        result = _gather_along_last_dim(test_tensor_last_dim, mock_group)

        assert result.shape == (8, 16, 32 * world_size)

    @pytest.mark.parametrize("input_shape,expected_shape", [
        ((32, 16), (8, 16)),
        ((40, 10), (10, 10)),
    ])
    def test_reduce_scatter_along_first_dim(self, mock_group, input_shape,
                                            expected_shape):
        input_tensor = torch.randn(*input_shape)
        result = _reduce_scatter_along_first_dim(input_tensor, mock_group)
        assert result.shape == expected_shape

    def test_reduce_scatter_along_last_dim(self, mock_group):
        input_tensor = torch.randn(8, 16, 32)
        result = _reduce_scatter_along_last_dim(input_tensor, mock_group)
        assert result.shape == (8, 16, 8)

    @pytest.mark.parametrize("func,input_shape,expected_shape", [
        ("all_gather_last_dim_from_tensor_parallel_region", (8, 16, 32),
         (8, 16, 128)),
        ("reduce_scatter_to_sequence_parallel_region", (32, 16), (8, 16)),
        ("reduce_scatter_last_dim_to_tensor_parallel_region", (8, 16, 32),
         (8, 16, 8)),
        ("gather_from_sequence_parallel_region", (8, 16), (32, 16)),
    ])
    def test_wrapper_functions(self, mock_group, func, input_shape,
                               expected_shape):
        """test wrapper funcs"""
        mod = importlib.import_module(
            'vllm_ascend.distributed.tensor_parallel')
        globals = mod.__dict__
        test_func = globals[func]
        input_tensor = torch.randn(*input_shape)
        result = test_func(input_tensor, mock_group)
        assert result.shape == expected_shape

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [
            ((8, 16), (32, 4)),  # [num_tokens/TP, H] -> [num_tokens, H/TP]
        ])
    def test_all_to_all_sp2hp(self, mock_group, input_shape, output_shape):
        input_tensor = torch.randn(*input_shape)
        result = all_to_all_sp2hp(input_tensor, mock_group)
        assert result.shape == output_shape

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [
            ((32, 4), (8, 16)),  # [num_tokens, H/TP] -> [num_tokens/TP, H]
        ])
    def test_all_to_all_hp2sp(self, mock_group, input_shape, output_shape):
        input_tensor = torch.randn(*input_shape)
        result = all_to_all_hp2sp(input_tensor, mock_group)
        assert result.shape == output_shape
