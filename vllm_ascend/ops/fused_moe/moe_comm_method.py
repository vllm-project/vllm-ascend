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
# This file is a part of the vllm-ascend project.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalizeWithAll2All, PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2, QuantType)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    TokenDispatcherWithAll2AllV, TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2, TokenDispatcherWithMoge)

_MoECommMethods: Dict[Optional[MoECommType], MoECommMethod] = {}


def get_moe_comm_method(
        moe_comm_type: Optional[MoECommType]) -> Optional[MoECommMethod]:
    return _MoECommMethods.get(moe_comm_type, None)


def setup_moe_comm_method(moe_config):
    _MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
    _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)
    _MoECommMethods[MoECommType.MC2] = MC2CommImpl(moe_config)


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.model_type = get_current_vllm_config(
        ).model_config.hf_config.model_type
        self.moe_config = moe_config

        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        hidden_states, router_logits, mc2_mask, context_metadata = self.prepare_finalize.prepare(
            hidden_states, router_logits, enable_shared_expert_dp,
            replace_allreduce, quant_type)
        return hidden_states, router_logits, mc2_mask, context_metadata

    def finalize(self,
                 hidden_states: torch.Tensor,
                 reduce_results: bool,
                 context_metadata: Optional[dict] = None) -> torch.Tensor:
        hidden_states = self.prepare_finalize.finalize(hidden_states,
                                                       reduce_results,
                                                       context_metadata)
        return hidden_states

    def fused_experts(
            self,
            hidden_states: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            activation: str = "silu",
            w1_bias: Optional[torch.Tensor] = None,
            w2_bias: Optional[torch.Tensor] = None,
            apply_router_weight_on_input: bool = False,
            use_int8_w8a8: bool = False,
            use_int4_w4a8: bool = False,
            global_num_experts: Optional[int] = None,
            expert_map: Optional[torch.Tensor] = None,
            w1_scale: Optional[torch.Tensor] = None,
            w2_scale: Optional[torch.Tensor] = None,
            w1_scale_bias: torch.Tensor = None,
            w2_scale_bias: torch.Tensor = None,
            # For TorchAir graph
            is_torchair: bool = False,
            # For Cube/Vector parallel
            shared_experts: Optional[Any] = None,
            quantized_x_for_share: Optional[Any] = None,
            dynamic_scale_for_share: Optional[Any] = None,
            # For load balance
            log2phy: torch.Tensor = None,
            global_redundant_expert_num: int = 0,
            need_trans: bool = False,
            dynamic_eplb: bool = False,
            mc2_mask: torch.Tensor = None,
            pertoken_scale: Optional[torch.Tensor] = None):
        # Check constraints
        assert hidden_states.dtype in [
            torch.float32, torch.float16, torch.bfloat16, torch.int8
        ]

        moe_comm_method = get_forward_context().moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        results = self.token_dispatcher.token_dispatch(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            shared_experts=shared_experts,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            with_quant=use_int8_w8a8 or use_int4_w4a8,
            dynamic_eplb=dynamic_eplb,
            pertoken_scale=pertoken_scale)

        permuted_hidden_states, expert_tokens, dynamic_scale, group_list_type, topk_scales, context_metadata = \
            results["hidden_states"], results["group_list"], results.get("dynamic_scale"), results["group_list_type"], results.get("topk_scales"), results.get("context_metadata")

        mlp_output = unified_apply_mlp(hidden_states=permuted_hidden_states,
                                       w1=w1,
                                       w1_scale=w1_scale,
                                       w2=w2,
                                       w2_scale=w2_scale,
                                       w1_bias=w1_bias,
                                       w2_bias=w2_bias,
                                       activation=activation,
                                       group_list=expert_tokens,
                                       dynamic_scale=dynamic_scale,
                                       group_list_type=group_list_type,
                                       w1_scale_bias=w1_scale_bias,
                                       w2_scale_bias=w2_scale_bias,
                                       topk_scales=topk_scales,
                                       with_quant=use_int8_w8a8
                                       or use_int4_w4a8,
                                       fusion=use_int8_w8a8,
                                       need_trans=need_trans,
                                       dynamic_eplb=dynamic_eplb)

        final_hidden_states = self.token_dispatcher.token_combine(
            hidden_states=mlp_output, context_metadata=context_metadata)

        if dynamic_eplb:
            return (final_hidden_states, group_list_type, expert_tokens)

        return final_hidden_states

    @abstractmethod
    def _get_token_dispatcher(self):
        raise NotImplementedError(
            "_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_prepare_finalize(self):
        raise NotImplementedError(
            "_get_prepare_finalize function not implemented.")


class AllGatherCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def _get_token_dispatcher(self):
        if self.model_type == "PanguProMoE":
            return TokenDispatcherWithMoge(
                top_k=self.moe_config.experts_per_token,
                num_experts=self.moe_config.num_experts,
                num_local_experts=self.moe_config.num_local_experts)
        else:
            return TokenDispatcherWithAllGather(
                top_k=self.moe_config.experts_per_token,
                num_experts=self.moe_config.num_experts,
                num_local_experts=self.moe_config.num_local_experts)

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAllGather(self.moe_config)


class MC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.
    
    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)


class AlltoAllCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_grouped_matmul` is available.

    This implementation uses all-to-all communication to exchange tokens
    between data parallel ranks before and after the MLP computation. It should
    have better performance than AllGatherCommImpl when DP size > 1.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts)

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)
