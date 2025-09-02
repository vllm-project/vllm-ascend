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
import torch_npu


def return_row_idx(hidden_states, top_k):
    num_tokens = hidden_states.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (torch.arange(0,
                            row_idx_len,
                            dtype=torch.int32,
                            device=hidden_states.device).view(
                                top_k, -1).permute(1, 0).contiguous())
    return row_idx


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

    topk_weights, topk_ids, row_idx = _select_experts_with_fusion_ops(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        topk_group=topk_group,
        renormalize=renormalize,
        e_score_correction_bias=e_score_correction_bias,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        global_num_experts=global_num_experts)

    if topk_weights is None:
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
    if row_idx is None:
        row_idx = return_row_idx(hidden_states, top_k)
    return topk_weights, topk_ids, row_idx


def _native_grouped_topk(
    topk_weights: torch.Tensor,
    num_expert_group: Optional[int],
    topk_group: Optional[int],
):
    topk_group = 0 if topk_group is None else topk_group
    num_expert_group = 0 if num_expert_group is None else num_expert_group

    num_token = topk_weights.shape[0]
    grouped_weights = topk_weights.view(num_token, num_expert_group,
                                        -1).max(dim=-1).values
    topk_group_indices = torch.topk(grouped_weights.to(torch.float32),
                                    k=topk_group,
                                    dim=-1,
                                    sorted=False)[1]
    topk_group_mask = torch.zeros_like(grouped_weights)
    topk_group_mask.scatter_(1, topk_group_indices, 1)
    topk_weight_mask = (topk_group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        topk_weights.shape[-1] // num_expert_group).reshape(num_token, -1))
    topk_weights = topk_weights.masked_fill(~topk_weight_mask.bool(), 0.0)

    return topk_weights


def _renormalize_topk_weights(
    topk_weights: torch.Tensor,
    renormalize: bool,
):
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights


def _select_expert_use_group_topk(
        topk_weights: torch.Tensor, topk_group: Optional[int],
        renormalize: bool, top_k: int, num_expert_group: Optional[int],
        e_score_correction_bias: Optional[torch.Tensor]):
    assert topk_group is not None
    assert num_expert_group is not None

    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_weights = topk_weights
        topk_weights = topk_weights + e_score_correction_bias.unsqueeze(0)

    # TODO: Change to npu_group_topk when the latest CANN and NNAL is available
    # >>> torch_npu._npu_group_topk(topk_weights, group_num=num_expert_group, k=topk_group)
    topk_weights = _native_grouped_topk(topk_weights, num_expert_group,
                                        topk_group)
    # TODO bfloat16 is not supported in torch.topk with ge graph.
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(topk_weights.to(torch.float32),
                              k=top_k,
                              dim=-1,
                              sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_weights.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(topk_weights.to(torch.float32),
                                            k=top_k,
                                            dim=-1,
                                            sorted=False)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = _renormalize_topk_weights(topk_weights, renormalize)
    return topk_weights, topk_ids


def _select_experts_with_fusion_ops(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        e_score_correction_bias: Optional[torch.Tensor],
        topk_group: Optional[int],
        num_expert_group: Optional[int],
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor=1.0,
        global_num_experts: int = -1):

    topk_weights, topk_ids, row_idx = None, None, None
    # NOTE: now npu_moe_gating_top_k can only support 'group_count=256' pattern
    is_deepseek_v3_r1 = global_num_experts == 256
    if is_deepseek_v3_r1:
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
            router_logits,
            k=top_k,  # topk currently 8
            bias=e_score_correction_bias,
            k_group=topk_group,  # fix: 4
            group_count=num_expert_group,  # fix 8
            group_select_mode=
            1,  # 0: the maximum in the group; 1: topk2.sum(fix)
            renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
            norm_type=1,  # 0: softmax; 1: sigmoid(fix)
            # out_flag=False, # todo new api; should the third output be output
            # y2_flag=False, # old api; should the third output be output
            routed_scaling_factor=1,
            eps=float(1e-20))
        row_idx = return_row_idx(hidden_states, top_k)
    if not use_grouped_topk and custom_routing_function is None and scoring_func == "softmax":
        topk_weights, topk_ids, row_idx = torch_npu.npu_moe_gating_top_k_softmax(
            x=router_logits, finished=None, k=top_k)
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = _renormalize_topk_weights(topk_weights, renormalize)

    return topk_weights, topk_ids, row_idx


def _native_select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    global_num_experts: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k experts based on router logits.

    Args:
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        router_logits: Router logits of shape (num_tokens, num_experts).
        top_k: Number of experts to select.
        use_grouped_topk: Whether to group experts before selecting top-k.
        renormalize: Whether to renormalize the routing weights.
        topk_group: Number of expert groups to select from.
        num_expert_group: Number of experts in each group.
        custom_routing_function: Custom routing function.
        scoring_func: Scoring function to use.
        e_score_correction_bias: Correction bias to apply to expert scores.

    Returns:
        topk_weights: Routing weights of shape (num_tokens, top_k).
        topk_ids: Selected expert IDs of shape (num_tokens, top_k).

    Raises:
        ValueError: If an unsupported scoring function is provided.
    """

    if scoring_func == "softmax":
        topk_weights = router_logits.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        topk_weights = router_logits.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if use_grouped_topk:
        return _select_expert_use_group_topk(
            topk_weights=topk_weights,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias)

    if custom_routing_function is not None:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            global_num_experts=global_num_experts)
        # Required by npu_moe_init_routing
        topk_ids = topk_ids.to(torch.int32)
        return topk_weights, topk_ids

    topk_weights, topk_ids = topk_weights.topk(top_k, dim=-1)
    topk_weights = topk_weights.to(hidden_states.dtype)

    # Required by npu_moe_init_routing
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = _renormalize_topk_weights(topk_weights, renormalize)

    return topk_weights, topk_ids
