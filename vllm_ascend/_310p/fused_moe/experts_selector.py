#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
from typing import Callable, Optional

import torch

from vllm_ascend.utils import get_weight_prefetch_method
from vllm_ascend.ops.fused_moe.experts_selector import _native_select_experts


def select_experts(hidden_states: torch.Tensor,
                   router_logits: torch.Tensor,
                   top_k: int,
                   use_grouped_topk: bool,
                   renormalize: bool,
                   topk_group: Optional[int] = None,
                   num_expert_group: Optional[int] = None,
                   custom_routing_function: Optional[Callable] = None,
                   scoring_func: str = "softmax",
                   routed_scaling_factor=1.0,
                   e_score_correction_bias: Optional[torch.Tensor] = None,
                   indices_type: Optional[torch.dtype] = None,
                   global_num_experts: int = -1):
    """
    Fused experts with select experts.

    Args:
        router_logits: router logits of shape (num_tokens, hidden_size).
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        top_k: number of top k experts.
        use_grouped_topk: Whether to group experts before selecting top-k.
        renormalize: Whether to renormalize the routing weights.
        topk_group: Number of expert groups to select from.
        num_expert_group: Number of experts in each group.
        custom_routing_function: Custom routing function.
        scoring_func: Scoring function to use.
        e_score_correction_bias: Correction bias to apply to expert scores.
        indices_type: dtype of indices
        global_num_experts: Global number of experts.

    Returns:
        topk_weights: router weights of shape (num_tokens, top_k).
        topk_ids: selected expert IDs of shape (num_tokens, top_k).
    """
    # prefetch w1_w3_proj.weight preprocess
    weight_prefetch_method = get_weight_prefetch_method()
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_preprocess(
            hidden_states, "gate_up")
    topk_weights, topk_ids = _native_select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias,
        global_num_experts=global_num_experts,
    )
    return topk_weights, topk_ids


def zero_experts_compute(
    expert_indices: torch.Tensor,
    expert_scales: torch.Tensor,
    num_experts: int,
    zero_expert_type: str,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if zero_expert_type == "identity":
        zero_expert_mask = expert_indices < num_experts
        zero_expert_scales = expert_scales.clone()
        zero_expert_scales = torch.where(zero_expert_mask, 0.0,
                                         zero_expert_scales)

        hidden_states = hidden_states.unsqueeze(1)
        zero_expert_scales = zero_expert_scales.unsqueeze(2)
        result = hidden_states * zero_expert_scales
        result = result.sum(dim=1)

    normal_expert_mask = expert_indices >= num_experts
    expert_indices = torch.where(normal_expert_mask, 0, expert_indices)
    expert_scales = torch.where(normal_expert_mask, 0.0, expert_scales)

    return expert_indices, expert_scales, result
