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

import pytest
import torch
from pytest_mock import MockerFixture

from tests.ut.base import PytestBase
from vllm_ascend.ops.moe.comm_utils import (
    _gather_along_first_dim, async_all_to_all,
    gather_from_sequence_parallel_region)


class TestDistributedCommunication(PytestBase):

    @pytest.fixture(autouse=True)
    def context(self, mocker: MockerFixture):
        mocker.patch("torch.npu.current_device", return_value="cpu")
        mocker.patch("torch.distributed.get_world_size", return_value=4)

        mocker.patch("torch.distributed.get_rank", return_value=0)

    @pytest.mark.parametrize(
        "input_tensor, output_split_sizes, input_split_sizes",
        [(torch.randn(8, 16), [2, 2, 2, 2], [2, 2, 2, 2]),
         (torch.randn(16, 32), None, None)])
    def test_async_all_to_all(self, input_tensor, output_split_sizes,
                              input_split_sizes, mocker: MockerFixture):
        """Test async_all_to_all"""
        mock_group = mocker.MagicMock()
        mocker.patch("torch.distributed.all_to_all_single",
                     return_value=mocker.MagicMock())

        _, a2a_out, handle = async_all_to_all(input_tensor, output_split_sizes,
                                              input_split_sizes, mock_group)

        # Check if the output tensor is created properly
        if output_split_sizes is None:
            assert a2a_out.shape == input_tensor.shape
        else:
            total_output_size = sum(output_split_sizes)
            expected_shape = [total_output_size] + list(
                input_tensor.size())[1:]
            assert a2a_out.shape == torch.Size(expected_shape)

        # Ensure handle is returned from async operation
        assert handle is not None
        assert isinstance(handle, mocker.MagicMock)

    @pytest.mark.parametrize("world_size, test_tensor, expected",
                             [(1, torch.randn(8, 16), (8, 16)),
                              (4, torch.randn(8, 16), (32, 16))])
    def test_gather_along_first_dim(self, test_tensor, expected, world_size,
                                    mocker: MockerFixture):
        """Test _gather_along_first_dim"""
        mocker.patch("torch.distributed.get_world_size",
                     return_value=world_size)

        result = _gather_along_first_dim(test_tensor, mocker.MagicMock())

        assert result.shape == expected

    @pytest.mark.parametrize("input_tensor, output_split_sizes",
                             [(torch.randn(8, 16), None),
                              (torch.randn(8, 16), [2, 2, 2, 2])])
    def test_gather_from_sequence_parallel_region(self, input_tensor,
                                                  output_split_sizes,
                                                  mocker: MockerFixture):
        """Test gather_from_sequence_parallel_region"""
        mock_group = mocker.MagicMock()

        result = gather_from_sequence_parallel_region(input_tensor, mock_group,
                                                      output_split_sizes)

        # If output_split_sizes is not provided, result should have expanded first dimension by world size
        if output_split_sizes is None:
            expected_shape = [input_tensor.shape[0] * 4] + list(
                input_tensor.shape[1:])
            assert result.shape == torch.Size(expected_shape)
        else:
            # If output_split_sizes is provided, result shape is dictated by sum of output_split_sizes
            expected_shape = [sum(output_split_sizes)] + list(
                input_tensor.shape[1:])
            assert result.shape == torch.Size(expected_shape)
