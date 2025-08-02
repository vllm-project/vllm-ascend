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

import pytest
import torch
from pytest_mock import MockerFixture

from tests.ut.base import PytestBase
from vllm_ascend.distributed.tensor_parallel import (
    _gather_along_first_dim, _gather_along_last_dim,
    _reduce_scatter_along_first_dim, _reduce_scatter_along_last_dim,
    all_to_all_hp2sp, all_to_all_sp2hp)


class TestDistributedCommunication(PytestBase):

    @pytest.fixture(autouse=True)
    def context(self, mocker: MockerFixture):
        mocker.patch("torch.npu.current_device", return_value="cpu")
        mocker.patch("torch.distributed.get_world_size", return_value=4)

        mocker.patch("torch.distributed.get_rank", return_value=0)

    @pytest.mark.parametrize("world_size, test_tensor, expected",
                             [(1, torch.randn(8, 16), (8, 16)),
                              (4, torch.randn(8, 16), (32, 16))])
    def test_gather_along_first_dim(self, test_tensor, expected, world_size,
                                    mocker: MockerFixture):
        """test _gather_along_first_dim"""
        mocker.patch("torch.distributed.get_world_size",
                     return_value=world_size)

        result = _gather_along_first_dim(test_tensor, mocker.MagicMock())

        assert result.shape == expected

    @pytest.mark.parametrize("test_tensor, output_split_sizes, expected", [
        (torch.randn(8, 16), [5, 10, 15, 2], (32, 16)),
    ])
    def test_gather_along_first_dim_unequal_split(self, test_tensor, expected,
                                                  output_split_sizes,
                                                  mocker: MockerFixture):
        """test _gather_along_first_dim"""

        result = _gather_along_first_dim(test_tensor, mocker.MagicMock(),
                                         output_split_sizes)

        assert result.shape == expected

    @pytest.mark.parametrize("world_size, test_tensor, expected",
                             [(1, torch.randn(8, 16, 32), (8, 16, 32)),
                              (4, torch.randn(8, 16, 32), (8, 16, 32 * 4))])
    def test_gather_along_last_dim(self, test_tensor, expected, world_size,
                                   mocker: MockerFixture):
        """test _gather_along_last_dim"""
        mocker.patch("torch.distributed.get_world_size",
                     return_value=world_size)

        result = _gather_along_last_dim(test_tensor, mocker.MagicMock())

        assert result.shape == expected

    @pytest.mark.parametrize("input_shape,expected_shape", [
        ((32, 16), (8, 16)),
        ((40, 10), (10, 10)),
    ])
    def test_reduce_scatter_along_first_dim(self, input_shape, expected_shape,
                                            mocker: MockerFixture):
        input_tensor = torch.randn(*input_shape)
        result = _reduce_scatter_along_first_dim(input_tensor,
                                                 mocker.MagicMock())
        assert result.shape == expected_shape

    @pytest.mark.parametrize("input_shape,expected_shape", [
        ((8, 16, 32), (8, 16, 8)),
    ])
    def test_reduce_scatter_along_last_dim(self, input_shape, expected_shape,
                                           mocker: MockerFixture):
        input_tensor = torch.randn(*input_shape)
        result = _reduce_scatter_along_last_dim(input_tensor,
                                                mocker.MagicMock())
        assert result.shape == expected_shape

    @pytest.mark.parametrize("func,input_shape,expected_shape", [
        ("all_gather_last_dim_from_tensor_parallel_region", (8, 16, 32),
         (8, 16, 128)),
        ("reduce_scatter_to_sequence_parallel_region", (32, 16), (8, 16)),
        ("reduce_scatter_last_dim_to_tensor_parallel_region", (8, 16, 32),
         (8, 16, 8)),
        ("gather_from_sequence_parallel_region", (8, 16), (32, 16)),
    ])
    def test_wrapper_functions(self, func, input_shape, expected_shape,
                               mocker: MockerFixture):
        """test wrapper funcs"""
        mod = importlib.import_module(
            'vllm_ascend.distributed.tensor_parallel')
        globals = mod.__dict__
        test_func = globals[func]
        input_tensor = torch.randn(*input_shape)
        result = test_func(input_tensor, mocker.MagicMock())
        assert result.shape == expected_shape

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [
            ((8, 16), (32, 4)),  # [num_tokens/TP, H] -> [num_tokens, H/TP]
        ])
    def test_all_to_all_sp2hp(self, input_shape, output_shape,
                              mocker: MockerFixture):
        input_tensor = torch.randn(*input_shape)
        result = all_to_all_sp2hp(input_tensor, mocker.MagicMock())
        assert result.shape == output_shape

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [
            ((32, 4), (8, 16)),  # [num_tokens, H/TP] -> [num_tokens/TP, H]
        ])
    def test_all_to_all_hp2sp(self, input_shape, output_shape,
                              mocker: MockerFixture):
        input_tensor = torch.randn(*input_shape)
        result = all_to_all_hp2sp(input_tensor, mocker.MagicMock())
        assert result.shape == output_shape
