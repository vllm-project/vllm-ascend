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
import torch.distributed as dist
from torch import nn
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.moe_comm_method import MC2CommImpl
from vllm_ascend.ops.fused_moe import fused_experts_moge, unified_fused_experts
from vllm_ascend.ops.layers.experts_selector import select_experts
from vllm_ascend.utils import is_310p

original_unquantized_fused_moe_init_func = UnquantizedFusedMoEMethod.__init__


def unquantized_fused_moe_init_func(self, *args, **kwargs):
    original_unquantized_fused_moe_init_func(self, *args, **kwargs)
    vllm_config = get_current_vllm_config()
    self.max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

    ascend_config = get_ascend_config()

    if ascend_config.torchair_graph_config.enabled:
        self.use_aclgraph = False
    else:
        self.use_aclgraph = (vllm_config.compilation_config.level
                             == CompilationLevel.PIECEWISE
                             and not vllm_config.model_config.enforce_eager)


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
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None) -> torch.Tensor:

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
        global_num_experts=global_num_experts)

    if topk_ids.shape[1] < top_k or is_310p():
        assert global_num_experts is not None
        return fused_experts_moge(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            moe_parallel_config=self.moe.moe_parallel_config,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=top_k,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input)

    moe_comm_method = get_forward_context().moe_comm_method

    return unified_fused_experts(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        moe_comm_method=moe_comm_method,
    )


class AscendFusedMoE(FusedMoE):

    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        params_dtype=None,
        reduce_results=False,
        renormalize=True,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        quant_config=None,
        tp_size=None,
        ep_size=None,
        dp_size=None,
        prefix="",
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        apply_router_weight_on_input=False,
        activation="silu",
        enable_eplb=False,
        num_redundant_experts=0,
        has_bias=False,
    ):
        super().__init__(
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            params_dtype,
            reduce_results,
            renormalize,
            use_grouped_topk,
            num_expert_group,
            topk_group,
            quant_config,
            tp_size,
            ep_size,
            dp_size,
            prefix,
            custom_routing_function,
            scoring_func,
            e_score_correction_bias,
            apply_router_weight_on_input,
            activation,
            enable_eplb,
            num_redundant_experts,
            has_bias,
        )

        self.tp_group = get_tp_group().device_group

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        assert self.quant_method is not None

        num_tokens, _ = hidden_states.shape
        forward_context = get_forward_context()

        moe_comm_method = forward_context.moe_comm_method
        if type(moe_comm_method) is MC2CommImpl:
            # NOTE: Pad tensors to make sure they can be evenly split.
            if num_tokens % self.ep_size != 0:
                pad_size = self.ep_size - (num_tokens % self.ep_size)
                hidden_states = nn.functional.pad(hidden_states,
                                                  (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits,
                                                  (0, 0, 0, pad_size))

            split_hidden_states = torch.tensor_split(hidden_states,
                                                     self.ep_size,
                                                     dim=0)
            split_router_logits = torch.tensor_split(router_logits,
                                                     self.ep_size,
                                                     dim=0)
            hidden_states = split_hidden_states[self.ep_rank]
            router_logits = split_router_logits[self.ep_rank]

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            enable_eplb=self.enable_eplb,
            expert_load_view=self.expert_load_view,
            logical_to_physical_map=self.logical_to_physical_map,
            logical_replica_count=self.logical_replica_count,
        )

        if type(moe_comm_method) is MC2CommImpl:
            dist.all_gather(list(split_hidden_states), final_hidden_states,
                            self.tp_group)
            final_hidden_states = torch.cat(split_hidden_states, dim=0)
            if num_tokens % self.ep_size != 0:
                final_hidden_states = final_hidden_states[:num_tokens]
        elif self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            final_hidden_states = self.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states)

        return final_hidden_states


UnquantizedFusedMoEMethod.__init__ = unquantized_fused_moe_init_func
UnquantizedFusedMoEMethod.forward_oot = forward_oot
