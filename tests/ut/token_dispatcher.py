# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import pytest
from unittest.mock import MagicMock

from vllm_ascend.ops.moe_dispatcher.moe_utils import get_capacity
from vllm_ascend.ops.moe_dispatcher.token_dispatcher import MoeDispatcherConfig, MoEAlltoAllSeqOverLapDispatcher


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
        config.set_is_fused(True)
        return config.build()

    @pytest.fixture
    def mock_ep_group(self, mocker):
        mock_group = MagicMock()
        mock_group.rank_in_group = 0
        mock_group.world_size = 2
        mock_group.device_group = "mock_group"
        mocker.patch('vllm_ascend.distributed.tensor_parallel.get_ep_group', return_value=mock_group)
        return mock_group

    @pytest.fixture
    def dispatcher(self, config, mock_ep_group):
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
        assert scores.shape == (4, 2)  # topk=2
        assert routing_map.shape == (4, 2)

    def test_preprocess(self, dispatcher):
        routing_map = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 1]], dtype=torch.long)
        num_tokens_per_local_expert = dispatcher.preprocess(routing_map)
        assert num_tokens_per_local_expert.shape == (2,)

    def test_token_permutation(self, dispatcher):
        hidden_states = torch.randn(4, 8)  # 4 tokens, hidden_size=8
        probs = torch.randn(4, 4)  # 4 tokens, 4 experts
        routing_map = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 1]], dtype=torch.long)

        shared_output, global_input, tokens_per_expert = dispatcher.token_permutation(
            hidden_states, probs, routing_map
        )

        assert shared_output is None
        assert global_input.shape[1] == 8  # hidden size preserved
        assert tokens_per_expert.shape == (2,)

    def test_token_unpermutation(self, dispatcher):
        # First do permutation to setup state
        hidden_states = torch.randn(4, 8)
        probs = torch.randn(4, 4)
        routing_map = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 1]], dtype=torch.long)
        _, global_input, _ = dispatcher.token_permutation(hidden_states, probs, routing_map)

        # Now test unpermutation
        expert_output = torch.randn_like(global_input)
        output, bias = dispatcher.token_unpermutation(expert_output)

        assert output.shape == hidden_states.shape
        assert bias is None

    def test_preprocess_and_permute1(self, dispatcher):
        hidden_states = torch.randn(4, 8)
        probs = torch.randn(4, 4)
        routing_map = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 1]], dtype=torch.long)

        dispatcher.preprocess_and_permtute1(hidden_states, probs, routing_map)

        assert dispatcher.cached_permutated_local_input_tokens is not None
        assert dispatcher.tokens_per_expert is not None

    def test_dispatch_alltoall(self, dispatcher):
        # Setup with preprocess_and_permute1
        hidden_states = torch.randn(4, 8)
        probs = torch.randn(4, 4)
        routing_map = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 1]], dtype=torch.long)
        dispatcher.preprocess_and_permtute1(hidden_states, probs, routing_map)

        dispatcher.dispatch_alltoall()

        assert dispatcher.cached_global_input_tokens is not None
        assert dispatcher.cached_permutated_local_input_tokens is None

    def test_permute2(self, dispatcher):
        # Setup chain
        hidden_states = torch.randn(4, 8)
        probs = torch.randn(4, 4)
        routing_map = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 1]], dtype=torch.long)
        dispatcher.preprocess_and_permtute1(hidden_states, probs, routing_map)
        dispatcher.dispatch_alltoall()

        global_input, tokens_per_expert = dispatcher.permute2()

        assert global_input is not None
        assert tokens_per_expert.shape == (2,)

    def test_unpermute1(self, dispatcher):
        # Setup chain
        hidden_states = torch.randn(4, 8)
        probs = torch.randn(4, 4)
        routing_map = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 1]], dtype=torch.long)
        dispatcher.preprocess_and_permtute1(hidden_states, probs, routing_map)
        dispatcher.dispatch_alltoall()
        global_input, _ = dispatcher.permute2()

        dispatcher.unpermute1(global_input)

        assert dispatcher.cached_global_output_tokens is not None

    def test_combine_alltoall(self, dispatcher):
        # Setup chain
        hidden_states = torch.randn(4, 8)
        probs = torch.randn(4, 4)
        routing_map = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 1]], dtype=torch.long)
        dispatcher.preprocess_and_permtute1(hidden_states, probs, routing_map)
        dispatcher.dispatch_alltoall()
        global_input, _ = dispatcher.permute2()
        dispatcher.unpermute1(global_input)

        dispatcher.combine_alltoall()

        assert dispatcher.cached_local_output_tokens is not None
        assert dispatcher.cached_global_output_tokens is None

    def test_unpermute2(self, dispatcher):
        # Setup chain
        hidden_states = torch.randn(4, 8)
        probs = torch.randn(4, 4)
        routing_map = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 1]], dtype=torch.long)
        dispatcher.preprocess_and_permtute1(hidden_states, probs, routing_map)
        dispatcher.dispatch_alltoall()
        global_input, _ = dispatcher.permute2()
        dispatcher.unpermute1(global_input)
        dispatcher.combine_alltoall()

        output = dispatcher.unpermute2()

        assert output.shape == hidden_states.shape
        assert dispatcher.cached_local_output_tokens is None

    @pytest.mark.parametrize("capacity_factor", [1.0, 1.5, 2.0])
    def test_with_capacity_factor(self, config, capacity_factor):
        config.set_moe_pad_expert_input_to_capacity(True)
        config.set_moe_expert_capacity_factor(capacity_factor)
        dispatcher = MoEAlltoAllSeqOverLapDispatcher(config)

        hidden_states = torch.randn(4, 8)
        probs = torch.randn(4, 4)
        routing_map = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 1]], dtype=torch.long)

        shared_output, global_input, tokens_per_expert = dispatcher.token_permutation(
            hidden_states, probs, routing_map
        )

        # Check capacity was calculated correctly
        num_tokens = hidden_states.shape[0]
        expected_capacity = get_capacity(
            num_tokens=num_tokens,
            num_experts=dispatcher.num_experts,
            capacity_factor=capacity_factor,
        )
        assert dispatcher.capacity == expected_capacity

    def test_shared_experts(self, dispatcher):
        mock_shared_experts = MagicMock()
        mock_shared_experts.return_value = (torch.randn(4, 8),)
        dispatcher.set_shared_experts(mock_shared_experts)

        hidden_states = torch.randn(4, 8)
        probs = torch.randn(4, 4)
        routing_map = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 1]], dtype=torch.long)

        shared_output, _, _ = dispatcher.token_permutation(
            hidden_states, probs, routing_map
        )

        assert shared_output is not None
        assert shared_output.shape == hidden_states.shape
        mock_shared_experts.assert_called_once()
