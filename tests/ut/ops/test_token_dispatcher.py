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

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from pytest_mock import MockerFixture

from tests.ut.base import PytestBase, TestBase
from vllm_ascend.ops.moe_dispatcher.token_dispatcher import (
    MoEAlltoAllSeqOverLapDispatcher, MoEDispatcherConfig,
    QuantizedTokenDispatcherWithAll2All,
    UnquantizedTokenDispatcherWithAll2AllV)
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


class TestUnquantizedTokenDispatcherWithAll2AllV(TestBase):

    def setUp(self):
        need_param = {"top_k": 2, "num_experts": 4}

        # Mock distributed properties
        self.mock_ep_group = MagicMock()
        self.mock_ep_rank = 0
        self.mock_ep_size = 2
        self.mock_tp_ep_group = None
        self.mock_tp_ep_size = 1

        # Patch properties
        patcher1 = patch.object(UnquantizedTokenDispatcherWithAll2AllV,
                                'ep_group',
                                new_callable=PropertyMock,
                                return_value=self.mock_ep_group)
        patcher2 = patch.object(UnquantizedTokenDispatcherWithAll2AllV,
                                'ep_rank',
                                new_callable=PropertyMock,
                                return_value=self.mock_ep_rank)
        patcher3 = patch.object(UnquantizedTokenDispatcherWithAll2AllV,
                                'ep_size',
                                new_callable=PropertyMock,
                                return_value=self.mock_ep_size)
        patcher4 = patch.object(UnquantizedTokenDispatcherWithAll2AllV,
                                'tp_ep_group',
                                new_callable=PropertyMock,
                                return_value=self.mock_tp_ep_group)
        patcher5 = patch.object(UnquantizedTokenDispatcherWithAll2AllV,
                                'tp_ep_size',
                                new_callable=PropertyMock,
                                return_value=self.mock_tp_ep_size)

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

        # Mock async_all_to_all
        patcher6 = patch('vllm_ascend.ops.comm_utils.async_all_to_all')
        self.mock_async_all_to_all = patcher6.start()
        self.addCleanup(patcher6.stop)
        self.mock_async_all_to_all.return_value = (None, torch.randn(16, 16),
                                                   MagicMock())

        # Mock torch_npu.npu_moe_token_permute
        patcher7 = patch('torch_npu.npu_moe_token_permute')
        self.mock_npu_moe_token_permute = patcher7.start()
        self.addCleanup(patcher7.stop)
        self.mock_npu_moe_token_permute.return_value = (torch.randn(16, 16),
                                                        torch.arange(16))

        # Mock torch_npu.npu_moe_token_unpermute
        patcher8 = patch('torch_npu.npu_moe_token_unpermute')
        self.mock_npu_moe_token_unpermute = patcher8.start()
        self.addCleanup(patcher8.stop)
        self.mock_npu_moe_token_unpermute.return_value = torch.randn(8, 16)

        # Mock gather_from_sequence_parallel_region
        patcher9 = patch(
            'vllm_ascend.ops.moe_dispatcher.token_dispatcher.gather_from_sequence_parallel_region'
        )
        self.mock_gather_from_sequence_parallel_region = patcher9.start()
        self.addCleanup(patcher9.stop)
        self.mock_gather_from_sequence_parallel_region.return_value = torch.tensor(
            [[2, 2, 2, 2], [2, 2, 2, 2]], dtype=torch.int64)

        # Mock torch.histc
        patcher10 = patch('torch.histc')
        self.mock_histc = patcher10.start()
        self.addCleanup(patcher10.stop)
        self.mock_histc.return_value = torch.tensor([2, 2, 2, 2],
                                                    dtype=torch.int64)

        # Mock torch.npu.stream
        patcher11 = patch('torch.npu.stream')
        self.mock_npu_stream = patcher11.start()
        self.addCleanup(patcher11.stop)
        self.mock_npu_stream.return_value.__enter__ = MagicMock()
        self.mock_npu_stream.return_value.__exit__ = MagicMock()

        # Mock stream and event
        patcher12 = patch('torch.npu.Stream')
        patcher13 = patch('torch.npu.Event')
        self.mock_stream = patcher12.start()
        self.mock_event = patcher13.start()
        self.addCleanup(patcher12.stop)
        self.addCleanup(patcher13.stop)

        # Mock torch.npu.current_device()
        patcher14 = patch('torch.npu.current_device')
        self.mock_current_device = patcher14.start()
        self.addCleanup(patcher14.stop)
        self.mock_current_device.return_value = 'cpu'

        # Mock torch.npu.current_stream()
        patcher15 = patch('torch.npu.current_stream')
        self.mock_current_stream = patcher15.start()
        self.addCleanup(patcher15.stop)
        mock_stream_obj = MagicMock()
        mock_stream_obj.record_event.return_value = MagicMock()
        mock_stream_obj.wait_event = MagicMock()
        self.mock_current_stream.return_value = mock_stream_obj

        # Mock shared experts
        self.mock_shared_experts = MagicMock()

        self.dispatcher = UnquantizedTokenDispatcherWithAll2AllV(need_param)

    def test_token_permutation(self):
        hidden_states = torch.randn(8, 16)
        topk_weights = torch.rand(8, 4)
        topk_ids = torch.randint(0, 4, (8, 2)).long()

        # num_experts=4, ep_size=2, 所以 num_local_experts=2
        # expert_ids_per_ep_rank = [i % 2 for i in range(4)] = [0, 1, 0, 1] (长度4)
        self.dispatcher.expert_ids_per_ep_rank = torch.tensor(
            [0, 1, 0, 1], dtype=torch.int32)

        # local_expert_indices = [ep_rank * num_local_experts + i for i in range(num_local_experts)]
        # ep_rank=0, num_local_experts=2, 所以 local_expert_indices=[0, 1]
        self.dispatcher.local_expert_indices = [0, 1]

        result = self.dispatcher.token_permutation(hidden_states=hidden_states,
                                                   topk_weights=topk_weights,
                                                   topk_ids=topk_ids)

        self.assertIsNotNone(result["global_input_tokens"])
        self.assertIsNotNone(result["tokens_per_expert"])

    def test_token_unpermutation(self):
        self.dispatcher.hidden_shape = (8, 16)
        self.dispatcher.hidden_shape_before_permute = (8, 16)
        self.dispatcher.reversed_local_input_permutation_mapping = torch.arange(
            8)
        self.dispatcher.topk_weights = torch.rand(8, 4)
        self.dispatcher.input_splits = [4, 4]
        self.dispatcher.output_splits = [4, 4]
        self.dispatcher.reversed_global_input_permutation_mapping = torch.arange(
            16)

        self.dispatcher.expert_ids_per_ep_rank = torch.tensor(
            [0, 1, 0, 1], dtype=torch.int32)
        self.dispatcher.local_expert_indices = [0, 1]

        self.dispatcher.num_global_tokens_per_local_expert = torch.tensor(
            [[2, 2], [2, 2]], dtype=torch.int64)

        expert_output = torch.randn(16, 16)
        output = self.dispatcher.token_unpermutation(expert_output)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (8, 16))


class TestQuantizedTokenDispatcherWithAll2All(TestBase):

    def setUp(self):
        need_param = {"top_k": 2, "num_experts": 4}

        # Mock distributed properties
        self.mock_ep_group = MagicMock()
        self.mock_ep_rank = 0
        self.mock_ep_size = 2
        self.mock_tp_ep_group = None
        self.mock_tp_ep_size = 1

        # Patch properties
        patcher1 = patch.object(QuantizedTokenDispatcherWithAll2All,
                                'ep_group',
                                new_callable=PropertyMock,
                                return_value=self.mock_ep_group)
        patcher2 = patch.object(QuantizedTokenDispatcherWithAll2All,
                                'ep_rank',
                                new_callable=PropertyMock,
                                return_value=self.mock_ep_rank)
        patcher3 = patch.object(QuantizedTokenDispatcherWithAll2All,
                                'ep_size',
                                new_callable=PropertyMock,
                                return_value=self.mock_ep_size)
        patcher4 = patch.object(QuantizedTokenDispatcherWithAll2All,
                                'tp_ep_group',
                                new_callable=PropertyMock,
                                return_value=self.mock_tp_ep_group)
        patcher5 = patch.object(QuantizedTokenDispatcherWithAll2All,
                                'tp_ep_size',
                                new_callable=PropertyMock,
                                return_value=self.mock_tp_ep_size)

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

        # Mock torch.distributed.all_to_all_single
        patcher6 = patch('torch.distributed.all_to_all_single')
        self.mock_all_to_all_single = patcher6.start()
        self.addCleanup(patcher6.stop)

        # Mock torch_npu functions (common ones)
        patcher7 = patch('torch_npu.npu_moe_init_routing')
        self.mock_npu_moe_init_routing = patcher7.start()
        self.addCleanup(patcher7.stop)
        self.mock_npu_moe_init_routing.return_value = (
            torch.randn(16, 32),  # hidden_states
            torch.arange(16),  # expanded_row_idx
            torch.randint(0, 4, (16, ))  # expanded_expert_idx
        )

        patcher8 = patch('torch_npu.npu_moe_compute_expert_tokens')
        self.mock_npu_moe_compute_expert_tokens = patcher8.start()
        self.addCleanup(patcher8.stop)
        self.mock_npu_moe_compute_expert_tokens.return_value = torch.tensor(
            [4, 4, 4, 4])

        patcher9 = patch('torch_npu.npu_moe_finalize_routing')
        self.mock_npu_moe_finalize_routing = patcher9.start()
        self.addCleanup(patcher9.stop)
        self.mock_npu_moe_finalize_routing.return_value = torch.randn(8, 32)

        patcher10 = patch('torch_npu.npu_moe_re_routing')
        self.mock_npu_moe_re_routing = patcher10.start()
        self.addCleanup(patcher10.stop)
        self.mock_npu_moe_re_routing.return_value = (
            torch.randn(16, 32),  # hidden_states
            torch.randn(16),  # dynamic_scale
            torch.arange(16),  # inverse_indices
            torch.tensor([4, 4, 4, 4])  # expert_tokens
        )

        patcher11 = patch('torch_npu.npu_dynamic_quant')
        self.mock_npu_dynamic_quant = patcher11.start()
        self.addCleanup(patcher11.stop)
        self.mock_npu_dynamic_quant.return_value = (
            torch.randn(16, 32),  # quantized_tokens
            torch.randn(16)  # token_scales
        )

        # Mock torch.index_select
        patcher12 = patch('torch.index_select')
        self.mock_index_select = patcher12.start()
        self.addCleanup(patcher12.stop)
        self.mock_index_select.return_value = torch.randn(16, 32)

        # Mock hasattr for torch_npu
        patcher13 = patch(
            'vllm_ascend.ops.moe_dispatcher.token_dispatcher.hasattr')
        self.mock_hasattr = patcher13.start()
        self.addCleanup(patcher13.stop)

        # Mock torch.stack to return correct shape
        patcher14 = patch('torch.stack')
        self.mock_stack = patcher14.start()
        self.addCleanup(patcher14.stop)
        # Return tensor of shape [2, 4] to match ep_size=2, num_experts=4
        self.mock_stack.return_value = torch.tensor([[4, 4, 4, 4],
                                                     [4, 4, 4, 4]])

        # Mock tensor.view and sum to return proper values
        patcher15 = patch('torch.Tensor.view')
        self.mock_tensor_view = patcher15.start()
        self.addCleanup(patcher15.stop)
        mock_view_result = MagicMock()
        mock_view_result.sum.return_value = MagicMock()
        mock_view_result.sum.return_value.to.return_value.numpy.return_value = [
            [8, 8], [8, 8]
        ]
        self.mock_tensor_view.return_value = mock_view_result

        # Mock tensor.sum to return proper value
        patcher16 = patch('torch.Tensor.sum')
        self.mock_tensor_sum = patcher16.start()
        self.addCleanup(patcher16.stop)
        mock_sum_result = MagicMock()
        mock_sum_result.item.return_value = 16
        self.mock_tensor_sum.return_value = mock_sum_result

        # Mock tensor.new_empty to avoid negative dimension error
        patcher17 = patch('torch.Tensor.new_empty')
        self.mock_new_empty = patcher17.start()
        self.addCleanup(patcher17.stop)
        self.mock_new_empty.return_value = torch.randn(16, 32)

        self.dispatcher = QuantizedTokenDispatcherWithAll2All(need_param)

    def test_token_permutation_without_expert_map(self):
        # Mock hasattr to return False (simulate no npu_moe_init_routing_quant)
        self.mock_hasattr.return_value = False

        hidden_states = torch.randn(8, 32)
        topk_weights = torch.rand(8, 4)
        topk_ids = torch.randint(0, 4, (8, 2))

        with patch.object(self.dispatcher, '_save_meta') as mock_save_meta:
            result = self.dispatcher.token_permutation(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids)

            mock_save_meta.assert_called()

            call_args = mock_save_meta.call_args[1]

            actual_keys = set(call_args.keys())

            self.assertFalse('quantized_tokens_shape' in actual_keys)
            self.assertFalse('inverse_indices' in actual_keys)
            self.assertFalse('scatter_size_list' in actual_keys)
            self.assertFalse('gather_size_list' in actual_keys)

        self.assertEqual(len(result), 4)
        self.assertIsNotNone(result["hidden_states"])
        self.assertIsNotNone(result["expert_tokens"])
        self.assertEqual(result["group_list_type"], 0)
        self.assertIsNone(result["dynamic_scale"])

    def test_token_permutation_with_expert_map_hasattr_false(self):
        self.mock_hasattr.return_value = False

        hidden_states = torch.randn(8, 32)
        topk_weights = torch.rand(8, 4)
        topk_ids = torch.randint(0, 4, (8, 2))
        expert_map = torch.tensor([0, 1, 2, 3])

        result = self.dispatcher.token_permutation(hidden_states=hidden_states,
                                                   topk_weights=topk_weights,
                                                   topk_ids=topk_ids,
                                                   expert_map=expert_map)

        self.assertEqual(len(result), 4)
        self.assertIsNotNone(result["hidden_states"])
        self.assertIsNotNone(result["expert_tokens"])
        self.assertEqual(result["group_list_type"], 1)
        self.assertIsNotNone(result["dynamic_scale"])

    def test_token_permutation_with_expert_map_hasattr_true(self):
        self.mock_hasattr.return_value = True

        with patch(
                'vllm_ascend.ops.moe_dispatcher.token_dispatcher.torch_npu.npu_moe_init_routing_quant',
                create=True) as mock_init_routing_quant:
            mock_init_routing_quant.return_value = (torch.randn(16, 32),
                                                    torch.arange(16),
                                                    torch.tensor([4, 4, 4,
                                                                  4]), None,
                                                    torch.randn(16))

            hidden_states = torch.randn(8, 32)
            topk_weights = torch.rand(8, 4)
            topk_ids = torch.randint(0, 4, (8, 2))
            expert_map = torch.tensor([0, 1, 2, 3])

            result = self.dispatcher.token_permutation(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map)

            self.assertEqual(len(result), 4)
            self.assertIsNotNone(result["hidden_states"])
            self.assertIsNotNone(result["expert_tokens"])
            self.assertEqual(result["group_list_type"], 1)
            self.assertIsNotNone(result["dynamic_scale"])

    def test_token_unpermutation_without_expert_map(self):
        # Setup meta data
        self.dispatcher._meta = {
            "expert_map": None,
            "topk_weights": torch.rand(8, 4),
            "expanded_row_idx": torch.arange(16),
            "topk_ids": torch.randint(0, 4, (8, 2)),
            "original_shape": (8, 32)
        }

        expert_output = torch.randn(16, 32)
        result = self.dispatcher.token_unpermutation(expert_output)

        self.assertIsNotNone(result)
        self.mock_npu_moe_finalize_routing.assert_called()

    def test_token_unpermutation_with_expert_map(self):
        # Setup meta data
        self.dispatcher._meta = {
            "expert_map": torch.tensor([0, 1, 2, 3]),
            "inverse_indices": torch.arange(16),
            "quantized_tokens_shape": (16, 32),
            "gather_size_list": [8, 8],
            "scatter_size_list": [8, 8],
            "topk_weights": torch.rand(8, 4),
            "expanded_row_idx": torch.arange(16),
            "original_shape": (8, 32)
        }

        expert_output = torch.randn(16, 32)
        result = self.dispatcher.token_unpermutation(expert_output)

        self.assertIsNotNone(result)
        self.mock_index_select.assert_called()
        self.mock_all_to_all_single.assert_called()
