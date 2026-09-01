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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams, build_quant_params
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.quantization.quant_type import QuantType


@dataclass(frozen=True, slots=True)
class MoEWeights:
    """Dense and quantized weight payloads consumed by MoE execution."""

    w1: torch.Tensor | list[torch.Tensor]
    w2: torch.Tensor | list[torch.Tensor]
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None
    w1_scale: torch.Tensor | list[torch.Tensor] | None = None
    w2_scale: torch.Tensor | list[torch.Tensor] | None = None
    w1_scale_bias: torch.Tensor | list[torch.Tensor] | None = None
    w2_scale_bias: torch.Tensor | list[torch.Tensor] | None = None
    w1_offset: torch.Tensor | None = None
    w2_offset: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MoEFusedExpertsInput:
    """Top-level input for the routed experts pipeline."""

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    weights: MoEWeights
    routing: MoeRouterInput
    quant: MoEQuantParams
    activation: MoEActivation | str = MoEActivation.SILU
    need_trans: bool = False
    dynamic_eplb: bool = False
    # Optional per-layer MoE LoRA state (vllm_ascend.lora MoELoRAContext).
    # ``Any`` avoids coupling the core contracts to the LoRA module; only the
    # unquant MLP path reads it, and only when a LoRA adapter is active.
    lora_context: Any = None


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
    pertoken_scale: torch.Tensor | None = None,
    activation: MoEActivation | str = MoEActivation.SILU,
    need_trans: bool = False,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    comm_quant_mode: int | None = None,
    mxfp_act_quant_type: torch.dtype | None = None,
    mxfp_weight_quant_type: torch.dtype | None = None,
    mxfp_scale_dtype: torch.dtype | None = None,
    mxfp_per_token_scale_dtype: torch.dtype | None = None,
    mxfp_use_bf16: bool | None = None,
    is_per_channel_weight: bool = False,
    w1_scale: list[torch.Tensor] | torch.Tensor | None = None,
    w2_scale: list[torch.Tensor] | torch.Tensor | None = None,
    w1_scale_bias: list[torch.Tensor] | torch.Tensor | None = None,
    w2_scale_bias: list[torch.Tensor] | torch.Tensor | None = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    lora_context=None,
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
        routing=MoeRouterInput(
            expert_map=expert_map,
            global_redundant_expert_num=global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            pertoken_scale=pertoken_scale,
        ),
        activation=activation,
        need_trans=need_trans,
        dynamic_eplb=dynamic_eplb,
        quant=build_quant_params(
            quant_type=quant_type,
            comm_quant_mode=comm_quant_mode,
            mxfp_act_quant_type=mxfp_act_quant_type,
            mxfp_weight_quant_type=mxfp_weight_quant_type,
            mxfp_scale_dtype=mxfp_scale_dtype,
            mxfp_per_token_scale_dtype=mxfp_per_token_scale_dtype,
            mxfp_use_bf16=mxfp_use_bf16,
            is_per_channel_weight=is_per_channel_weight,
        ),
        lora_context=lora_context,
    )
