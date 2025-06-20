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
from vllm.model_executor.layers.fused_moe.layer import \
    UnquantizedFusedMoEMethod

from vllm_ascend.ops.fused_moe import fused_experts, select_experts


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
        global_num_experts=global_num_experts,
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
    )

    return fused_experts(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        top_k=top_k,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input)


UnquantizedFusedMoEMethod.forward_oot = forward_oot
