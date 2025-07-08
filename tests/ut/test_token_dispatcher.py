# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import pytest
from pytest_mock import MockerFixture
import vllm_ascend.patch.worker.patch_common.patch_utils
from vllm_ascend.utils import adapt_patch # noqa E402

from vllm_ascend.ops.moe_dispatcher.token_dispatcher import MoeDispatcherConfig, MoEAlltoAllSeqOverLapDispatcher

adapt_patch(True)

class TestMoEAlltoAllSeqOverLapDispatcher:

    @pytest.fixture
    def config(self):
        config = MoeDispatcherConfig()
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
        mocker.patch("vllm_ascend.ops.moe_dispatcher.token_dispatcher.get_ep_group",
                     return_value=self.mock_ep_group(mocker))
        return MoEAlltoAllSeqOverLapDispatcher(config)

    def test_initialization(self, dispatcher, config):
        assert dispatcher.num_local_experts == config.num_local_experts
        assert dispatcher.num_experts == config.num_moe_experts
        assert dispatcher.local_expert_indices == [0, 1]
        assert dispatcher.ep_rank == 0
        assert dispatcher.ep_size == 2
        assert dispatcher.overlap_stream is not None

    def test_routing(self, dispatcher):
        probs = torch.randn(4, 4)  # 4 tokens, 4 experts
        scores, routing_map = dispatcher.routing(probs)
        assert scores.shape == (4, 4)  # topk=2
        assert routing_map.shape == (4, 4)