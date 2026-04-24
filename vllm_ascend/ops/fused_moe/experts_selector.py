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
import torch.nn.functional as F
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.distributed.utils import split_tensor_along_first_dim
from vllm_ascend.utils import get_weight_prefetch_method


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
    routed_scaling_factor=1.0,
    e_score_correction_bias: Optional[torch.Tensor] = None,
    indices_type: Optional[torch.dtype] = None,
    mix_placement: Optional[bool] = False,
    num_logical_experts: int = -1,
    num_shared_experts: int = 0,
    global_num_experts: int = -1,
    input_ids: Optional[torch.Tensor] = None,
    tid2eid: Optional[torch.Tensor] = None,
):
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
    is_support_npu_moe_gating_top_k = check_npu_moe_gating_top_k(
        hidden_states=hidden_states,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        scoring_func=scoring_func,
        custom_routing_function=custom_routing_function)

    if is_support_npu_moe_gating_top_k:
        topk_weights, topk_ids = _select_experts_with_fusion_ops(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            global_num_experts=global_num_experts,
            tid2eid=tid2eid,
            input_ids=input_ids,
        )
    else:
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
            tid2eid=None,
            input_ids=None,
        )
    if mix_placement:
        if scoring_func == "sqrtsoftplus":
            shared_expert_routing_factor = 1.0 / 1.5  # hardcode for new model
        else:
            shared_expert_routing_factor = 0.4
        batch_size = topk_ids.shape[0]
        pad_shared_expert_ids = torch.arange(
            num_logical_experts,
            num_logical_experts + num_shared_experts,
            dtype=topk_ids.dtype,
            device=topk_ids.device).repeat(batch_size, 1)

        pad_shared_expert_weights = torch.full(
            (topk_weights.shape[0], num_shared_experts),
            shared_expert_routing_factor,
            dtype=topk_weights.dtype,
            device=topk_weights.device)

        topk_ids = torch.cat([topk_ids, pad_shared_expert_ids], dim=1)
        topk_weights = torch.cat([topk_weights, pad_shared_expert_weights],
                                 dim=1)
    return topk_weights, topk_ids


def check_npu_moe_gating_top_k(
        hidden_states: torch.Tensor,
        top_k: int,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        scoring_func: str = "softmax",
        custom_routing_function: Optional[Callable] = None):
    if scoring_func == "sigmoid" and not renormalize:  #sigmoid + renorm=0 is not supported in current branch
        return False
    if custom_routing_function is not None:
        return False
    if scoring_func != "softmax" and scoring_func != "sigmoid" and scoring_func != 'sqrtsoftplus':
        return False
    topk_group = topk_group if topk_group is not None else 1
    num_expert_group = num_expert_group if num_expert_group is not None else 1
    if not (num_expert_group > 0 and hidden_states.shape[-1] % num_expert_group
            == 0 and hidden_states.shape[-1] // num_expert_group > 2):
        return False
    if topk_group < 1 or topk_group > num_expert_group:
        return False
    if top_k < 1 or \
        top_k > (hidden_states.shape[-1] / (num_expert_group * topk_group)):
        return False
    if topk_group * hidden_states.shape[-1] / num_expert_group < top_k:
        return False
    return True


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
        scoring_func: str = "softmax",
        routed_scaling_factor=1.0,
        global_num_experts: int = -1,
        tid2eid=None,
        input_ids=None):

    topk_group = topk_group if topk_group is not None else 1
    num_expert_group = num_expert_group if num_expert_group is not None else 1
    renorm = int(renormalize)
    if scoring_func == "sqrtsoftplus":
        if tid2eid is not None:
            forward_context = get_forward_context()
            input_ids = forward_context.input_ids.to(torch.int64)
            # tid2eid_ones = torch.ones(tid2eid.shape[0],tid2eid.shape[1],device=router_logits.device,dtype=torch.int32)
            tid2eid_ones = tid2eid.to(torch.int32)
            if forward_context.moe_comm_type == MoECommType.ALLGATHER:
                prepare_finalize = forward_context.moe_comm_method.prepare_finalize
                input_ids = prepare_finalize.all_gather_input_id_with_dp_group(
                    input_ids)
            else:
                input_ids = forward_context.moe_comm_method.pad_and_split_input_ids(
                    input_ids)

            if forward_context.sp_enabled:
                # Process for Flash Comm V1
                tp_size = get_tp_group().world_size
                tp_rank = get_tp_group().rank_in_group
                splitted_input = split_tensor_along_first_dim(
                    input_ids, num_partitions=tp_size)
                input_ids = splitted_input[tp_rank].contiguous()
            input_ids = torch.where(input_ids == -1, 0, input_ids)
        else:
            input_ids = None
            tid2eid_ones = None
        topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
            x=router_logits,  # 输入张量
            k=top_k,  # 选取的专家数量
            bias=e_score_correction_bias,  # 偏置张量（可选）
            input_ids=input_ids,  # 输入词表（可选）
            tid2eid=tid2eid_ones,  # 词表到专家id的映射关系表（可选）
            k_group=topk_group,  # 选取的组数量（可选）
            group_count=num_expert_group,  # 总组数（可选）
            routed_scaling_factor=routed_scaling_factor,  # 路由缩放因子（可选）
            eps=float(1e-20),  # 数值稳定性参数（可选）
            group_select_mode=1,  # 组选择模式（可选）
            renorm=0,  # 重归一化标志（可选）
            norm_type=2,  # 归一化类型（可选）
            out_flag=False  # 是否输出归一化结果（可选）
        )
        return topk_weights, topk_ids

    elif scoring_func == "softmax":
        norm_type = 0
    else:
        norm_type = 1
    if e_score_correction_bias is not None and \
        e_score_correction_bias.dtype != router_logits.dtype:
        e_score_correction_bias = e_score_correction_bias.to(
            router_logits.dtype)
    topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k(
        router_logits,
        k=top_k,
        k_group=topk_group,
        group_count=num_expert_group,
        group_select_mode=1,
        renorm=renorm,
        norm_type=norm_type,  # 0: softmax; 1: sigmoid
        out_flag=False,
        routed_scaling_factor=routed_scaling_factor,
        eps=float(1e-20),
        bias_opt=e_score_correction_bias)
    if scoring_func == "softmax":
        topk_weights = _renormalize_topk_weights(topk_weights, renormalize)

    return topk_weights, topk_ids


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
        global_num_experts: Optional[torch.Tensor] = None,
        use_hash: bool = False,
        tid2eid=None,
        input_ids=None) -> tuple[torch.Tensor, torch.Tensor]:
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
    elif scoring_func == "softplus":
        topk_weights = F.softplus(topk_weights).sqrt()
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

    if e_score_correction_bias is not None:
        topk_weights = topk_weights + e_score_correction_bias
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
