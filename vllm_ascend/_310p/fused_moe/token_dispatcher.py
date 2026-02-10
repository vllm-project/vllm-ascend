# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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


import torch
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather, TokenDispatchResult


class TokenDispatcherWithAllGather310(TokenDispatcherWithAllGather):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def token_dispatch(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
    ):
        self.original_shape = hidden_states.shape

        num_tokens = hidden_states.shape[:-1].numel()
        self.apply_router_weight_on_input = apply_router_weight_on_input
        if self.apply_router_weight_on_input:
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        if expert_map is not None:
            mask = expert_map[topk_ids] != -1
            topk_weights = topk_weights * mask
            first_expert_idx = get_ep_group().rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
        else:
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local

        sorted_hidden_states, expanded_row_idx, expert_tokens = self.moe_init_routing(
            hidden_states,
            topk_ids,
            active_num=num_tokens * self.top_k,
            active_expert_range=[first_expert_idx, last_expert_idx],
        )
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1  # `count` mode
        context_metadata = {"topk_weights": topk_weights, "expanded_row_idx": expanded_row_idx}

        return TokenDispatchResult(
            hidden_states=sorted_hidden_states,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            context_metadata=context_metadata,
        )

    def moe_init_routing(self, x, expert_idx, active_num, active_expert_range):
        """
        Initialize routing for Mixture of Experts (MoE) model by organizing tokens
        according to their assigned experts and preparing data structures for
        efficient expert computation.

        Args:
            x (torch.Tensor): Input tensor containing token representations
            expert_idx (torch.Tensor): Tensor containing expert indices for each token
            active_num (int): Number of active experts or None
            active_expert_range (tuple): Range (start, end) of active experts

        Returns:
            tuple: A tuple containing:
                   - expanded_x: Subset of input tensor for active experts
                   - expanded_row_idx: Mapping indices for token positions
                   - expert_tokens_count: Count of tokens assigned to each expert
        """
        MAX_INT32 = torch.iinfo(torch.int32).max
        expert_start, expert_end = active_expert_range
        num_rows = x.shape[0]
        k = expert_idx.shape[-1]
        expert_idx_flat = expert_idx.flatten()
        mask = (expert_idx_flat >= expert_start) & (expert_idx_flat < expert_end)
        actual_expert_total_num = mask.sum().item()
        expert_idx_flat = torch.where(
            ~mask, torch.full_like(expert_idx_flat, MAX_INT32, dtype=torch.int32), expert_idx_flat
        )
        sorted_idx = torch.argsort(expert_idx_flat, stable=True)
        sorted_expert_idx = expert_idx_flat[sorted_idx]
        expanded_row_idx = torch.full((num_rows * k,), -1, dtype=torch.int32, device=expert_idx.device)
        expanded_row_idx[sorted_idx[:actual_expert_total_num]] = torch.arange(
            actual_expert_total_num, dtype=torch.int32, device=expert_idx.device
        )
        expert_tokens_count = torch.bincount(
            sorted_expert_idx[:actual_expert_total_num] - expert_start, minlength=expert_end - expert_start
        )
        active_num = min(active_num or actual_expert_total_num, actual_expert_total_num)
        expanded_x = x[sorted_idx[:active_num] // k]

        return expanded_x, expanded_row_idx, expert_tokens_count
