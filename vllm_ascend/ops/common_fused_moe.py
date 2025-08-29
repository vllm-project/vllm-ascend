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

from typing import Any, Callable, Optional

import torch
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.moe_comm_method import (AllGatherCommImpl,
                                                     MC2CommImpl,
                                                     MoECommMethod)
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe import apply_mlp, fused_experts_moge
from vllm_ascend.ops.layers.experts_selector import select_experts
from vllm_ascend.utils import is_310p

original_unquantized_fused_moe_init_func = UnquantizedFusedMoEMethod.__init__


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_int8_w8a8: bool = False,
    use_int4_w4a8: bool = False,
    global_num_experts: Optional[int] = None,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale_bias: torch.Tensor = None,
    w2_scale_bias: torch.Tensor = None,
    moe_comm_method: Optional[MoECommMethod] = None,
    # For TorchAir graph
    is_torchair: bool = False,
    # For Cube/Vector parallel
    shared_experts: Optional[Any] = None,
    quantized_x_for_share: Optional[Any] = None,
    dynamic_scale_for_share: Optional[Any] = None,
    # For load balance
    log2phy: torch.Tensor = None,
    global_redundant_expert_num: int = 0,
) -> torch.Tensor:
    # Check constraints
    assert hidden_states.shape[1] == w1.shape[2], (
        f"Hidden size mismatch {hidden_states.shape[1]} != {w1.shape[2]}")

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    assert moe_comm_method is not None, "Missing communication context"

    num_experts = w1.shape[0]

    permuted_hidden_states, expert_tokens, group_list_type = moe_comm_method.permute(
        hidden_states, topk_ids, topk_weights, expert_map, num_experts)
    mlp_output = apply_mlp(
        permuted_hidden_states,
        w1,
        w2,
        expert_tokens,
        group_list_type=group_list_type,
    )
    moe_comm_method.unpermute(mlp_output, hidden_states)

    return hidden_states


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

    topk_weights, topk_ids, _ = select_experts(
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

    return fused_experts(
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

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.mc2_group = get_mc2_group()

        for method in {AllGatherCommImpl, MC2CommImpl}:
            setattr(
                self, method.__name__.lower(),
                method(moe_config=self.moe_config))  # type: ignore[abstract]

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        assert self.quant_method is not None

        forward_context = get_forward_context()
        moe_comm_method_name = forward_context.moe_comm_method_name

        # TODO: Can we refactor this logic to model_runner?
        # TODO: Adjusted logic to differentiate between A2 and A3, we check ep_size here since mc2 only support ep_size >= 16 on A3 now
        if self.moe_config.ep_size < 16:
            moe_comm_method_name = "allgathercommimpl"

        forward_context.moe_comm_method = getattr(self, moe_comm_method_name)

        hidden_states, router_logits = forward_context.moe_comm_method.prepare(
            hidden_states=hidden_states, router_logits=router_logits)

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

        final_hidden_states = forward_context.moe_comm_method.finalize(
            hidden_states=final_hidden_states,
            reduce_results=self.reduce_results)

        return final_hidden_states


UnquantizedFusedMoEMethod.__init__ = unquantized_fused_moe_init_func
UnquantizedFusedMoEMethod.forward_oot = forward_oot
