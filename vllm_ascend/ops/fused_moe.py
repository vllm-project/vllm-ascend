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
from vllm.model_executor.layers.fused_moe.layer import \
    UnquantizedFusedMoEMethod


def group_topk(hidden_states: torch.Tensor,
               gating_output: torch.Tensor,
               topk: int,
               renormalize: bool,
               num_expert_group: Optional[int] = 0,
               topk_group: Optional[int] = 0,
               scoring_func: str = "softmax",
               e_score_correction_bias: Optional[torch.Tensor] = None):

    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)

    topk_group = 0 if topk_group is None else topk_group
    num_expert_group = 0 if num_expert_group is None else num_expert_group

    # TODO: Replace this piece of code to npu_group_topk when CANN and NNAL version is update
    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values
    group_idx = torch.topk(group_scores.to(torch.float32),
                           k=topk_group,
                           dim=-1,
                           sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)
    scores = scores.masked_fill(~score_mask.bool(), 0.0)

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids.to(torch.int32)


def fused_experts(hidden_states: torch.Tensor, w1: torch.Tensor,
                  w2: torch.Tensor, topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor, top_k: int):
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    ori_shape = hidden_states.shape
    if len(ori_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape

    row_idx_len = num_tokens * top_k
    row_idx = torch.arange(0,
                           row_idx_len,
                           dtype=torch.int32,
                           device=topk_weights.device).view(top_k, -1).permute(
                               1, 0).contiguous()
    expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
        hidden_states,
        row_idx=row_idx,
        expert_idx=topk_ids,
        active_num=num_tokens)

    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, E)
    expert_tokens = expert_tokens.to(torch.int64)

    w1 = w1.transpose(1, 2)
    gate_up_out_list = torch_npu.npu_grouped_matmul(x=[expanded_x],
                                                    weight=[w1],
                                                    split_item=2,
                                                    group_list_type=0,
                                                    group_type=0,
                                                    group_list=expert_tokens)

    # TODO: Remove this in the future.
    gate_up_out = torch.cat(gate_up_out_list, dim=0)
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    w2 = w2.transpose(1, 2)
    down_out_list = torch_npu.npu_grouped_matmul(x=[gate_up_out],
                                                 weight=[w2],
                                                 split_item=2,
                                                 group_list_type=0,
                                                 group_type=0,
                                                 group_list=expert_tokens)

    down_out_list = torch.cat(down_out_list, dim=0)
    # TODO: Reorder device memory 2 times here, replace the current
    # implementation here when suitable operators become available.
    hidden_states = torch_npu.npu_moe_finalize_routing(
        down_out_list,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids)
    if len(ori_shape) == 3:
        hidden_states = hidden_states.view(ori_shape)
    return hidden_states


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
        e_score_correction_bias: Optional[torch.Tensor] = None
) -> torch.Tensor:

    topk_weights, topk_ids = group_topk(
        hidden_states=x,
        gating_output=router_logits,
        topk=top_k,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        e_score_correction_bias=e_score_correction_bias)

    return fused_experts(hidden_states=x,
                         w1=layer.w13_weight,
                         w2=layer.w2_weight,
                         topk_weights=topk_weights,
                         topk_ids=topk_ids,
                         top_k=top_k)


UnquantizedFusedMoEMethod.forward_oot = forward_oot
