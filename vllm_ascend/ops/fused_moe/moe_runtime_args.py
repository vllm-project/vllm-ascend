#
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
#
"""Typed runtime contracts and builders for fused MoE execution.

This module is the single entry point for the runtime payloads used across the
fused MoE pipeline.

Relationship overview:

    stage params: reusable sub-payloads
      - MoERoutingParams
      - MoEQuantParams
          - internal MXFP leaf: MoEMxfpParams

    stage contracts: stage input/output payloads
      prepare
        -> MoEPrepareOutput

      fused_experts input
        -> MoEFusedExpertsInput
            |- weights: MoEWeights
            |- routing: MoERoutingParams
            |- quant: MoEQuantParams

      dispatch
        input  -> MoETokenDispatchInput
        output -> MoETokenDispatchOutput[TMoECombineMetadata]
                    TMoECombineMetadata is one of:
                      - MoEAllGatherCombineMetadata
                      - MoEAllToAllCombineMetadata
                      - MoEMC2CombineMetadata

      mlp
        input -> MoEMlpComputeInput

      combine
        output -> torch.Tensor

The helper builders below adapt legacy call sites into these typed contracts.
Only the fused_moe package should need to know about the internal MXFP leaf
dataclass directly.
"""

from __future__ import annotations

import torch

import vllm_ascend.ops.fused_moe.moe_stage_params as _stage_params
from vllm_ascend.ops.fused_moe.moe_stage_contracts import (
    MoEAllGatherCombineMetadata,
    MoEAllToAllCombineMetadata,
    MoEFusedExpertsInput,
    MoEMC2CombineMetadata,
    MoEMlpComputeInput,
    MoEPrepareOutput,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
    MoEWeights,
    TMoECombineMetadata,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import (
    MoEQuantParams,
    MoERoutingParams,
)
from vllm_ascend.quantization.quant_type import QuantType


def _build_mxfp_params(
    *,
    quant_type: QuantType,
    mxfp_act_quant_type: torch.dtype | None = None,
    mxfp_weight_quant_type: torch.dtype | None = None,
    mxfp_scale_dtype: torch.dtype | None = None,
    mxfp_per_token_scale_dtype: torch.dtype | None = None,
    mxfp_use_bf16: bool | None = None,
) -> _stage_params.MoEMxfpParams | None:
    if quant_type != QuantType.MXFP8:
        return None

    has_explicit_mxfp_args = any(
        value is not None
        for value in (
            mxfp_act_quant_type,
            mxfp_weight_quant_type,
            mxfp_scale_dtype,
            mxfp_per_token_scale_dtype,
            mxfp_use_bf16,
        )
    )
    if not has_explicit_mxfp_args:
        raise ValueError("primitive MXFP params are required when quant_type is QuantType.MXFP8.")

    return _stage_params.MoEMxfpParams(
        act_quant_type=mxfp_act_quant_type,
        weight_quant_type=mxfp_weight_quant_type,
        scale_dtype=mxfp_scale_dtype,
        per_token_scale_dtype=mxfp_per_token_scale_dtype,
        use_bf16=True if mxfp_use_bf16 is None else mxfp_use_bf16,
    )


def build_fused_experts_input(
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor | list[torch.Tensor],
    w2: torch.Tensor | list[torch.Tensor],
    quant_type: QuantType,
    dynamic_eplb: bool,
    expert_map: torch.Tensor | None = None,
    global_redundant_expert_num: int = 0,
    mc2_mask: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    log2phy: torch.Tensor | None = None,
    pertoken_scale: torch.Tensor | None = None,
    activation: str = "silu",
    need_trans: bool = False,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    comm_quant_mode: int | None = None,
    mxfp_act_quant_type: torch.dtype | None = None,
    mxfp_weight_quant_type: torch.dtype | None = None,
    mxfp_scale_dtype: torch.dtype | None = None,
    mxfp_per_token_scale_dtype: torch.dtype | None = None,
    mxfp_use_bf16: bool | None = None,
    w1_scale: list[torch.Tensor] | torch.Tensor | None = None,
    w2_scale: list[torch.Tensor] | torch.Tensor | None = None,
    w1_scale_bias: torch.Tensor | None = None,
    w2_scale_bias: torch.Tensor | None = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
) -> MoEFusedExpertsInput:
    return MoEFusedExpertsInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        weights=MoEWeights(
            w1=w1,
            w2=w2,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
        ),
        routing=MoERoutingParams(
            expert_map=expert_map,
            global_redundant_expert_num=global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            log2phy=log2phy,
            pertoken_scale=pertoken_scale,
        ),
        activation=activation,
        need_trans=need_trans,
        dynamic_eplb=dynamic_eplb,
        quant=MoEQuantParams(
            quant_type=quant_type,
            comm_quant_mode=comm_quant_mode,
            mxfp=_build_mxfp_params(
                quant_type=quant_type,
                mxfp_act_quant_type=mxfp_act_quant_type,
                mxfp_weight_quant_type=mxfp_weight_quant_type,
                mxfp_scale_dtype=mxfp_scale_dtype,
                mxfp_per_token_scale_dtype=mxfp_per_token_scale_dtype,
                mxfp_use_bf16=mxfp_use_bf16,
            ),
        ),
    )


def build_token_dispatch_input(
    *,
    fused_experts_input: MoEFusedExpertsInput,
    topk_ids: torch.Tensor | None = None,
) -> MoETokenDispatchInput:
    return MoETokenDispatchInput(
        hidden_states=fused_experts_input.hidden_states,
        topk_weights=fused_experts_input.topk_weights,
        topk_ids=fused_experts_input.topk_ids if topk_ids is None else topk_ids,
        routing=fused_experts_input.routing,
        quant=fused_experts_input.quant,
    )


def build_mlp_compute_input(
    *,
    fused_experts_input: MoEFusedExpertsInput,
    token_dispatch_output: MoETokenDispatchOutput[TMoECombineMetadata],
    use_fusion_ops: bool,
) -> MoEMlpComputeInput:
    if fused_experts_input.quant.is_mxfp and fused_experts_input.quant.mxfp is None:
        raise ValueError("fused_experts_input.quant.mxfp is required when quant_type is QuantType.MXFP8.")

    return MoEMlpComputeInput(
        hidden_states=token_dispatch_output.hidden_states,
        group_list=token_dispatch_output.group_list,
        group_list_type=token_dispatch_output.group_list_type,
        dynamic_scale=token_dispatch_output.dynamic_scale,
        topk_scales=token_dispatch_output.topk_scales,
        weights=fused_experts_input.weights,
        quant=fused_experts_input.quant,
        fusion=fused_experts_input.quant.quant_type in (QuantType.W8A8, QuantType.MXFP8) and use_fusion_ops,
        activation=fused_experts_input.activation,
        need_trans=fused_experts_input.need_trans,
        dynamic_eplb=fused_experts_input.dynamic_eplb,
    )


__all__ = [
    "MoEAllGatherCombineMetadata",
    "MoEAllToAllCombineMetadata",
    "MoEFusedExpertsInput",
    "MoEMC2CombineMetadata",
    "MoEMlpComputeInput",
    "MoEPrepareOutput",
    "MoEQuantParams",
    "MoERoutingParams",
    "MoETokenDispatchInput",
    "MoETokenDispatchOutput",
    "MoEWeights",
    "TMoECombineMetadata",
    "build_fused_experts_input",
    "build_token_dispatch_input",
    "build_mlp_compute_input",
]
