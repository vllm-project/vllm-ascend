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
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.model_executor.layers.fused_moe.layer import \
    UnquantizedFusedMoEMethod

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ops.fused_moe import (fused_experts, fused_experts_moge,
                                       select_experts,
                                       select_gating_top_k_softmax_experts)
from vllm_ascend.utils import is_310p

SELECT_GATING_TOPK_SOTFMAX_EXPERTS: bool = envs_ascend.SELECT_GATING_TOPK_SOTFMAX_EXPERTS
original_unquantized_fused_moe_init_func = UnquantizedFusedMoEMethod.__init__


def unquantized_fused_moe_init_func(self, *args, **kwargs):
    original_unquantized_fused_moe_init_func(self, *args, **kwargs)
    vllm_config = get_current_vllm_config()
    self.max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    self.use_aclgraph = vllm_config.compilation_config.level == CompilationLevel.PIECEWISE and not vllm_config.model_config.enforce_eager


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

    if SELECT_GATING_TOPK_SOTFMAX_EXPERTS:
        topk_weights, topk_ids = select_gating_top_k_softmax_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize)
    else:
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

    if topk_ids.shape[1] < top_k or is_310p():
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

    # If use aclgraph, we need to set max_num_tokens to make
    # the input shape of `npu_moe_init_routing` fixed
    max_num_tokens = self.max_num_batched_tokens if self.use_aclgraph else None

    return fused_experts(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        top_k=top_k,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
        max_num_tokens=max_num_tokens)


UnquantizedFusedMoEMethod.__init__ = unquantized_fused_moe_init_func
UnquantizedFusedMoEMethod.forward_oot = forward_oot
