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

from tests.ut.base import PytestBase, TestBase
from unittest.mock import patch, MagicMock, PropertyMock
import torch
from vllm_ascend.ops.moe_dispatcher.token_dispatcher import (
    MoEAlltoAllSeqOverLapDispatcher, MoEDispatcherConfig)
from vllm_ascend.utils import adapt_patch  # noqa E402
from vllm_ascend.ops.moe_dispatcher.token_dispatcher import UnquantizedTokenDispatcherWithAll2AllV


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


class TestUnquantizedTokenDispatcherWithAll2AllV(TestBase):

    def setUp(self):
        self.dispatcher = UnquantizedTokenDispatcherWithAll2AllV()

        # Mock distributed properties
        self.mock_ep_group = MagicMock()
        self.mock_ep_rank = 0
        self.mock_ep_size = 2
        self.mock_tp_ep_group = None
        self.mock_tp_ep_size = 1

        # Patch properties
        patcher1 = patch.object(UnquantizedTokenDispatcherWithAll2AllV, 'ep_group', new_callable=PropertyMock)
        patcher2 = patch.object(UnquantizedTokenDispatcherWithAll2AllV, 'ep_rank', new_callable=PropertyMock)
        patcher3 = patch.object(UnquantizedTokenDispatcherWithAll2AllV, 'ep_size', new_callable=PropertyMock)
        patcher4 = patch.object(UnquantizedTokenDispatcherWithAll2AllV, 'tp_ep_group', new_callable=PropertyMock)
        patcher5 = patch.object(UnquantizedTokenDispatcherWithAll2AllV, 'tp_ep_size', new_callable=PropertyMock)

        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.addCleanup(patcher3.stop)
        self.addCleanup(patcher4.stop)
        self.addCleanup(patcher5.stop)

        self.mock_ep_group_prop = patcher1.start()
        self.mock_ep_rank_prop = patcher2.start()
        self.mock_ep_size_prop = patcher3.start()
        self.mock_tp_ep_group_prop = patcher4.start()
        self.mock_tp_ep_size_prop = patcher5.start()

        self.mock_ep_group_prop.return_value = self.mock_ep_group
        self.mock_ep_rank_prop.return_value = self.mock_ep_rank
        self.mock_ep_size_prop.return_value = self.mock_ep_size
        self.mock_tp_ep_group_prop.return_value = self.mock_tp_ep_group
        self.mock_tp_ep_size_prop.return_value = self.mock_tp_ep_size

        # Mock async_all_to_all 返回值 (3 个值)
        patcher6 = patch('vllm_ascend.ops.moe_dispatcher.token_dispatcher.async_all_to_all')
        self.mock_async_all_to_all = patcher6.start()
        self.addCleanup(patcher6.stop)
        self.mock_async_all_to_all.return_value = (None, torch.randn(16, 16), MagicMock())

        # Mock torch_npu.npu_moe_token_permute 返回值 (2 个值)
        patcher7 = patch('vllm_ascend.ops.moe_dispatcher.token_dispatcher.torch_npu.npu_moe_token_permute')
        self.mock_npu_moe_token_permute = patcher7.start()
        self.addCleanup(patcher7.stop)
        self.mock_npu_moe_token_permute.return_value = (torch.randn(16, 16), torch.arange(16))

        # Mock torch_npu.npu_moe_token_unpermute
        patcher8 = patch('vllm_ascend.ops.moe_dispatcher.token_dispatcher.torch_npu.npu_moe_token_unpermute')
        self.mock_npu_moe_token_unpermute = patcher8.start()
        self.addCleanup(patcher8.stop)
        self.mock_npu_moe_token_unpermute.return_value = torch.randn(8, 16)

        # Mock gather_from_sequence_parallel_region
        patcher9 = patch('vllm_ascend.ops.moe_dispatcher.token_dispatcher.gather_from_sequence_parallel_region')
        self.mock_gather_from_sequence_parallel_region = patcher9.start()
        self.addCleanup(patcher9.stop)
        self.mock_gather_from_sequence_parallel_region.return_value = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2])

        # Mock torch.histc
        patcher10 = patch('vllm_ascend.ops.moe_dispatcher.token_dispatcher.torch.histc')
        self.mock_histc = patcher10.start()
        self.addCleanup(patcher10.stop)
        self.mock_histc.return_value = torch.tensor([2, 2, 2, 2], dtype=torch.float32)

        # Mock torch.npu.stream 上下文管理器
        patcher11 = patch('vllm_ascend.ops.moe_dispatcher.token_dispatcher.torch.npu.stream')
        self.mock_npu_stream = patcher11.start()
        self.addCleanup(patcher11.stop)
        self.mock_npu_stream.return_value.__enter__ = MagicMock()
        self.mock_npu_stream.return_value.__exit__ = MagicMock()

        # Mock stream and event
        patcher12 = patch('vllm_ascend.ops.moe_dispatcher.token_dispatcher.torch.npu.Stream')
        patcher13 = patch('vllm_ascend.ops.moe_dispatcher.token_dispatcher.torch.npu.Event')
        self.mock_stream = patcher12.start()
        self.mock_event = patcher13.start()
        self.addCleanup(patcher12.stop)
        self.addCleanup(patcher13.stop)

        # Mock torch.npu.current_device()
        patcher14 = patch('vllm_ascend.ops.moe_dispatcher.token_dispatcher.torch.npu.current_device')
        self.mock_current_device = patcher14.start()
        self.addCleanup(patcher14.stop)
        self.mock_current_device.return_value = 'cpu'

        # Mock torch.npu.current_stream()
        patcher15 = patch('vllm_ascend.ops.moe_dispatcher.token_dispatcher.torch.npu.current_stream')
        self.mock_current_stream = patcher15.start()
        self.addCleanup(patcher15.stop)
        mock_stream_obj = MagicMock()
        mock_stream_obj.record_event.return_value = MagicMock()
        mock_stream_obj.wait_event = MagicMock()
        self.mock_current_stream.return_value = mock_stream_obj

        # Mock shared experts
        self.mock_shared_experts = MagicMock()

    def test_token_permutation(self):
        top_k = 2
        num_experts = 4
        hidden_states = torch.randn(8, 16)
        topk_weights = torch.rand(8, 4)
        topk_ids = torch.randint(0, 4, (8, 2)).long()

        shared_out, global_input_tokens, tokens_per_expert = self.dispatcher.token_permutation(
            top_k=top_k,
            num_experts=num_experts,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids
        )

        self.assertIsNotNone(global_input_tokens)
        self.assertIsNotNone(tokens_per_expert)

    def test_token_unpermutation(self):
        self.dispatcher.hidden_shape = (8, 16)
        self.dispatcher.hidden_shape_before_permute = (8, 16)
        self.dispatcher.reversed_local_input_permutation_mapping = torch.arange(8)
        self.dispatcher.topk_weights = torch.rand(8, 4)
        self.dispatcher.input_splits = [4, 4]
        self.dispatcher.output_splits = [4, 4]
        self.dispatcher.reversed_global_input_permutation_mapping = torch.arange(8)

        expert_output = torch.randn(8, 16)
        output = self.dispatcher.token_unpermutation(expert_output)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (8, 16))

    def test_preprocess_and_permtute1(self):
        self.dispatcher.num_experts = 4
        hidden_states = torch.randn(8, 16)
        topk_weights = torch.rand(8, 4)
        topk_ids = torch.randint(0, 4, (8, 2)).long()

        self.dispatcher.preprocess_and_permtute1(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=self.mock_shared_experts,
            shared_experts_input=hidden_states
        )

        self.assertIsNotNone(self.dispatcher.cached_permutated_local_input_tokens)
        self.assertIsNotNone(self.dispatcher.tokens_per_expert)

    def test_dispatch_alltoall(self):
        self.dispatcher.cached_permutated_local_input_tokens = torch.randn(16, 16)
        self.dispatcher.output_splits = [4, 4]
        self.dispatcher.input_splits = [4, 4]

        self.dispatcher.dispatch_alltoall()

        self.mock_async_all_to_all.assert_called()
        self.assertIsNone(self.dispatcher.cached_permutated_local_input_tokens)

    def test_permute2(self):
        self.dispatcher.cached_global_input_tokens = torch.randn(16, 16)
        self.dispatcher.num_local_experts = 2
        self.dispatcher.expert_ids_per_ep_rank = torch.tensor([0, 1], dtype=torch.int32, device='cpu')
        self.dispatcher.num_global_tokens_per_local_expert = torch.tensor([[2, 2], [2, 2]], device='cpu')
        self.dispatcher.global_input_tokens_local_experts_indices = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])

        global_input_tokens, tokens_per_expert = self.dispatcher.permute2()

        self.assertIsNotNone(global_input_tokens)

    def test_unpermute1(self):
        hidden_states = torch.randn(16, 16)
        self.dispatcher.reversed_global_input_permutation_mapping = torch.arange(16)
        self.dispatcher.num_local_experts = 2

        self.dispatcher.unpermute1(hidden_states)

        self.assertIsNotNone(self.dispatcher.cached_global_output_tokens)

    def test_combine_alltoall(self):
        self.dispatcher.cached_global_output_tokens = torch.randn(16, 16)
        self.dispatcher.input_splits = [4, 4]
        self.dispatcher.output_splits = [4, 4]

        self.dispatcher.combine_alltoall()

        self.mock_async_all_to_all.assert_called()
        self.assertIsNone(self.dispatcher.cached_global_output_tokens)

    def test_unpermute2(self):
        self.dispatcher.cached_local_output_tokens = torch.randn(8, 16)
        self.dispatcher.reversed_local_input_permutation_mapping = torch.arange(8)
        self.dispatcher.topk_weights = torch.rand(8, 4)
        self.dispatcher.hidden_shape_before_permute = (8, 16)
        self.dispatcher.hidden_shape = (8, 16)

        output = self.dispatcher.unpermute2()

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (8, 16))
