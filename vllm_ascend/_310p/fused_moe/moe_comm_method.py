# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
from __future__ import annotations

import torch
from vllm.forward_context import get_forward_context

from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult

from .moe_mlp import unified_apply_mlp
from .token_dispatcher import TokenDispatcherWithAllGather310


class AllGatherCommImpl310(AllGatherCommImpl):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.
    """

    def fused_experts(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None = None,
        use_int8_w8a8: bool = False,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
    ) -> FusedExpertsResult:
        # This method is overridden to use the 310p-specific unified_apply_mlp
        # which provides optimized MLP computation for the 310p platform
        moe_comm_method = get_forward_context().moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        dispatch_results = self.token_dispatcher.token_dispatch(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        mlp_output = unified_apply_mlp(
            hidden_states=dispatch_results.hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            group_list=dispatch_results.group_list,
            group_list_type=dispatch_results.group_list_type,
            with_quant=use_int8_w8a8,
        )

        combine_results = self.token_dispatcher.token_combine(
            hidden_states=mlp_output, context_metadata=dispatch_results.context_metadata
        )

        return FusedExpertsResult(
            routed_out=combine_results.routed_out,
            group_list_type=dispatch_results.group_list_type,
            expert_tokens=dispatch_results.group_list,
        )

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather310(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )
