# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import torch
import pytest
import math

from vllm_ascend.ops.moe_dispatcher.moe_utils import permute, get_capacity, topk_softmax_with_capacity, \
    group_limited_topk, unpermute, sort_chunks_by_idxs


class TestMoeUtils:

    @pytest.fixture
    def setup(self):
        self.num_tokens = 16
        self.num_experts = 4
        self.hidden_size = 8
        self.topk = 2
        self.capacity_factor = 1.0
        self.group_topk = 2
        self.num_groups = 2
        self.scaling_factor = 1.0

    def test_group_limited_topk(self, setup):
        # Test group-limited topk routing
        scores = torch.randn(self.num_tokens, self.num_experts)
        probs, indices = group_limited_topk(
            scores,
            topk=self.topk,
            num_tokens=self.num_tokens,
            num_experts=self.num_experts,
            num_groups=self.num_groups,
            group_topk=self.group_topk
        )

        assert probs.shape == (self.num_tokens, self.topk)
        assert indices.shape == (self.num_tokens, self.topk)
        assert torch.all(indices < self.num_experts)

    def test_topk_softmax_with_capacity(self, setup):
        # Test topk softmax with capacity
        logits = torch.randn(self.num_tokens, self.num_experts)

        # Test without capacity
        probs, routing_map, tokens_per_expert, top_indices = topk_softmax_with_capacity(
            logits,
            topk=self.topk
        )
        assert probs.shape == (self.num_tokens, self.num_experts)
        assert routing_map.shape == (self.num_tokens, self.num_experts)
        assert tokens_per_expert.shape == (self.num_experts,)

        # Test with capacity
        probs, routing_map, tokens_per_expert, top_indices = topk_softmax_with_capacity(
            logits,
            topk=self.topk,
            capacity_factor=self.capacity_factor,
            pad_to_capacity=True
        )
        expert_capacity = get_capacity(
            num_tokens=self.num_tokens * self.topk,
            num_experts=self.num_experts,
            capacity_factor=self.capacity_factor
        )
        assert tokens_per_expert.max() <= expert_capacity

        # Test with group routing
        probs, routing_map, tokens_per_expert, top_indices = topk_softmax_with_capacity(
            logits,
            topk=self.topk,
            num_groups=self.num_groups,
            group_topk=self.group_topk
        )
        assert probs.shape == (self.num_tokens, self.num_experts)

    def test_get_capacity(self, setup):
        # Test capacity calculation
        capacity = get_capacity(
            num_tokens=self.num_tokens,
            num_experts=self.num_experts,
            capacity_factor=self.capacity_factor
        )
        expected = math.ceil((self.num_tokens / self.num_experts) * self.capacity_factor)
        assert capacity == expected

        # Test with min capacity
        min_capacity = 5
        capacity = get_capacity(
            num_tokens=self.num_tokens,
            num_experts=self.num_experts,
            capacity_factor=self.capacity_factor,
            min_capacity=min_capacity
        )
        assert capacity == min_capacity

    def test_permute(self, setup):
        # Test token permutation
        tokens = torch.randn(self.num_tokens, self.hidden_size)
        routing_map = torch.randint(0, 2, (self.num_tokens, self.num_experts)).bool()

        # Basic permutation
        permuted_tokens, sorted_indices = permute(tokens, routing_map)
        assert permuted_tokens.shape[0] == routing_map.sum()
        assert sorted_indices.shape[0] == routing_map.sum()

        # With drop and pad
        capacity = get_capacity(
            num_tokens=self.num_tokens * self.topk,
            num_experts=self.num_experts,
            capacity_factor=self.capacity_factor
        )
        num_out_tokens = capacity * self.num_experts
        permuted_tokens, sorted_indices = permute(
            tokens,
            routing_map,
            num_out_tokens=num_out_tokens,
            drop_and_pad=True
        )
        assert permuted_tokens.shape[0] == num_out_tokens
        assert sorted_indices.shape[0] == num_out_tokens

    def test_unpermute(self, setup):
        # Test token unpermutation
        tokens = torch.randn(self.num_tokens, self.hidden_size)
        routing_map = torch.randint(0, 2, (self.num_tokens, self.num_experts)).bool()
        probs = torch.rand(self.num_tokens, self.num_experts)

        # First permute
        permuted_tokens, sorted_indices = permute(tokens, routing_map)

        # Then unpermute
        restored_tokens = unpermute(
            permuted_tokens,
            sorted_indices,
            tokens.shape,
            probs=probs,
            routing_map=routing_map
        )
        assert restored_tokens.shape == tokens.shape

        # With drop and pad
        capacity = get_capacity(
            num_tokens=self.num_tokens * self.topk,
            num_experts=self.num_experts,
            capacity_factor=self.capacity_factor
        )
        num_out_tokens = capacity * self.num_experts
        permuted_tokens, sorted_indices = permute(
            tokens,
            routing_map,
            num_out_tokens=num_out_tokens,
            drop_and_pad=True
        )
        restored_tokens = unpermute(
            permuted_tokens,
            sorted_indices,
            tokens.shape,
            probs=probs,
            routing_map=routing_map,
            drop_and_pad=True
        )
        assert restored_tokens.shape == tokens.shape

    def test_sort_chunks_by_idxs(self, setup):
        # Test chunk sorting
        input_tensor = torch.randn(10, self.hidden_size)
        split_sizes = torch.tensor([3, 2, 5])
        sorted_idxs = torch.tensor([2, 0, 1])

        output = sort_chunks_by_idxs(input_tensor, split_sizes, sorted_idxs)
        assert output.shape == input_tensor.shape

        # Verify the order is correct
        expected = torch.cat([input_tensor[5:], input_tensor[0: 3], input_tensor[3: 5]])
        assert torch.allclose(output, expected) \
 \
               @ pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])

    def test_score_functions(self, setup, score_function):
        # Test different score functions
        logits = torch.randn(self.num_tokens, self.num_experts)
        expert_bias = torch.randn(self.num_experts)

        probs, routing_map, tokens_per_expert, top_indices = topk_softmax_with_capacity(
            logits,
            topk=self.topk,
            score_function=score_function,
            expert_bias=expert_bias
        )
        assert probs.shape == (self.num_tokens, self.num_experts)
        assert routing_map.shape == (self.num_tokens, self.num_experts)
        assert tokens_per_expert.shape == (self.num_experts,)

    def test_edge_cases(self, setup):
        # Test empty input
        empty_logits = torch.randn(0, self.num_experts)
        with pytest.raises(AssertionError):
            topk_softmax_with_capacity(empty_logits, topk=self.topk)

        # Test invalid score function
        logits = torch.randn(self.num_tokens, self.num_experts)
        with pytest.raises(ValueError):
            topk_softmax_with_capacity(
                logits,
                topk=self.topk,
                score_function="invalid"
            )

        # Test invalid drop policy
        with pytest.raises(ValueError):
            topk_softmax_with_capacity(
                logits,
                topk=self.topk,
                capacity_factor=1.0,
                drop_policy="invalid"
            )