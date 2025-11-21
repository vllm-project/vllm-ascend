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
# Adapted from the vllm-ascend project to reuse its model components
#  for omni-infer integration.
# Adapted from vllm/tests/kernels/test_moe.py

from typing import Callable, Optional

import torch
import torch_npu
from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import (
    UnquantizedFusedMoEMethod)

from .device import is_310p

logger = init_logger(__name__)


def fused_experts_moge(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    expert_map: torch.Tensor = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """

    Args:
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        w1: Expert weights1 of shape (num_experts, intermediate_size * 2, hidden_size).
        w2: Expert weights2 of shape (num_experts, hidden_size, intermediate_size).
        topk_weights: Routing weights of shape (num_tokens, top_k).
        topk_ids: Selected expert IDs of shape (num_tokens, top_k).
        top_k: Number of experts to select.
        expert_map: Expert mapping of shape (num_experts,).

    Returns:
        hidden_states: Hidden states after routing.
    """
    ep_size = get_ep_group().world_size
    local_num_experts = global_num_experts // ep_size
    local_num_group = top_k // ep_size

    if apply_router_weight_on_input:
        assert (topk_weights.dim() == 2
                ), "`topk_weights` should be in shape (num_tokens, topk)"
        _, topk = topk_weights.shape
        assert (
            topk == 1
        ), "Only support topk=1 when `apply_router_weight_on_input` is True"
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

    bsz, _ = hidden_states.shape
    flatten_topk_ids = topk_ids.view(-1)
    sorted_topk_ids = torch.argsort(flatten_topk_ids.float())
    sorted_topk_ids = sorted_topk_ids.to(torch.int32)
    sorted_hidden_states = hidden_states.index_select(
        0, sorted_topk_ids // local_num_group)

    experts_id = torch.arange(0,
                              local_num_experts,
                              dtype=topk_ids.dtype,
                              device=topk_ids.device)
    num_tokens_per_expert = (flatten_topk_ids.unsqueeze(-1) == experts_id).to(
        torch.float32).sum(0)
    topk_scales = topk_weights.view(-1).index_select(
        0, sorted_topk_ids).unsqueeze(-1)
    group_list = num_tokens_per_expert.cumsum(dim=0).to(torch.int64)

    w1 = w1.transpose(1, 2)

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[sorted_hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
    )[0]

    if is_310p():
        gate_up_out = torch_npu.npu_swiglu(gate_up_out.to(torch.float32)).to(
            torch.float16)
    else:
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)
        gate_up_out *= topk_scales

    w2 = w2.transpose(1, 2)
    down_out_list = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
    )[0]

    unsorted_topk_ids = torch.argsort(sorted_topk_ids.float()).to(torch.int32)
    unsorted_hidden_states = down_out_list.index_select(0, unsorted_topk_ids)
    final_hidden_states = unsorted_hidden_states.reshape(
        bsz, top_k // ep_size, -1).sum(1)

    return final_hidden_states


def native_grouped_topk(
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


def select_experts(
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
        # NOTE: vLLM use dtype=torch.float here
        topk_weights = router_logits.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        topk_weights = router_logits.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None

        if e_score_correction_bias is not None:
            # Store original scores before applying correction bias. We use biased
            # scores for expert selection but original scores for routing weights
            original_weights = topk_weights
            topk_weights = topk_weights + e_score_correction_bias.unsqueeze(0)

        topk_weights = native_grouped_topk(topk_weights, num_expert_group,
                                           topk_group)
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
    elif custom_routing_function is None:
        topk_weights, topk_ids = topk_weights.topk(top_k, dim=-1)
        topk_weights = topk_weights.to(hidden_states.dtype)
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            global_num_experts=global_num_experts)
        # Required by npu_moe_init_routing
        topk_ids = topk_ids.to(torch.int32)
        return topk_weights, topk_ids

    # Required by npu_moe_init_routing
    topk_ids = topk_ids.to(torch.int32)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


# This function will be patched to UnquantizedFusedMoEMethod.forward_oot
def forward_oot(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    global_num_experts: Optional[int] = None,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
) -> torch.Tensor:
    topk_weights, topk_ids = select_experts(
        hidden_states=x,
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

    assert global_num_experts is not None

    return fused_experts_moge(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        top_k=top_k,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input)


def patch_fused_moe_ops():
    UnquantizedFusedMoEMethod.forward_oot = forward_oot
    logger.info("UnquantizedFusedMoEMethod.forward_oot is replaced.")
