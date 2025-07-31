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
from pytest_mock import MockerFixture

from tests.ut.base import PytestBase
from vllm_ascend.ops.moe_dispatcher.token_dispatcher import (
    MoEAlltoAllSeqOverLapDispatcher, MoEDispatcherConfig)
from vllm_ascend.utils import adapt_patch  # noqa E402


class TestMoEAlltoAllSeqOverLapDispatcher(PytestBase):

    @pytest.fixture
    def config(self):
        config = MoEDispatcherConfig()
        config.set_num_local_experts(2)
        config.set_num_moe_experts(4)
        config.set_moe_pad_expert_input_to_capacity(False)
        config.set_moe_expert_capacity_factor(None)
        config.set_moe_router_topk(2)
        config.set_moe_grouped_gemm(False)
        config.set_group_topk(0)
        config.set_num_groups(1)
        config.set_is_fused(False)
        return config.build()

    def mock_ep_group(self, mocker):
        mock_group = mocker.MagicMock()
        mock_group.rank_in_group = 0
        mock_group.world_size = 2
        mock_group.device_group = "mock_group"
        return mock_group

    @pytest.fixture
    def dispatcher(self, config, mocker: MockerFixture):
        mocker.patch(
            "vllm_ascend.ops.moe_dispatcher.token_dispatcher.get_ep_group",
            return_value=self.mock_ep_group(mocker))
        mocker.patch("torch.npu.current_device", return_value="cpu")
        mocker.patch("torch.npu.Stream", return_value=mocker.MagicMock)
        return MoEAlltoAllSeqOverLapDispatcher(config)

    def test_initialization(self, dispatcher, config):
        assert dispatcher.num_local_experts == config.num_local_experts
        assert dispatcher.num_experts == config.num_moe_experts
        assert dispatcher.local_expert_indices == [0, 1]
        assert dispatcher.ep_rank == 0
        assert dispatcher.ep_size == 2
        assert dispatcher.overlap_stream is not None
