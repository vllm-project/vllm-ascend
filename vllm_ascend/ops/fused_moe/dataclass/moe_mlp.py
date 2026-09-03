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
from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEFusedExpertsInput, MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.moe_utils import enable_fusion_gmmswigluquant
from vllm_ascend.quantization.quant_type import QuantType

if TYPE_CHECKING:
    from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchOutput, TMoECombineMetadata


@dataclass(frozen=True, slots=True)
class MoEMlpComputeInput:
    """Input to MLP compute."""

    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    dynamic_scale: torch.Tensor | None
    topk_scales: torch.Tensor | None
    weights: MoEWeights
    quant: MoEQuantParams
    fusion: bool
    activation: MoEActivation = MoEActivation.SILU
    need_trans: bool = False
    dynamic_eplb: bool = False
    record_before_gmm2: bool = False
    activation_situ_beta: float | None = None
    activation_situ_linear_beta: float | None = None
    swiglu_limit: float = 0.0
    swiglu_alpha: float = 1.0
    swiglu_beta: float = 0.0
    expanded_row_idx: torch.Tensor | None = None
    topk_ids: torch.Tensor | None = None
    # Optional per-layer MoE LoRA state, propagated from MoEFusedExpertsInput.
    lora_context: Any = None


def build_mlp_compute_input(
    *,
    fused_experts_input: MoEFusedExpertsInput,
    token_dispatch_output: MoETokenDispatchOutput[TMoECombineMetadata],
    moe_config: FusedMoEConfig | None = None,
    use_fusion_ops: bool | None = None,
) -> MoEMlpComputeInput:
    if fused_experts_input.quant.is_mxfp and fused_experts_input.quant.mxfp is None:
        raise ValueError("fused_experts_input.quant.mxfp is required for MXFP quant types.")

    expanded_row_idx = getattr(token_dispatch_output.combine_metadata, "expanded_row_idx", None)
    activation = (
        fused_experts_input.activation
        if moe_config is None
        else getattr(moe_config, "activation", fused_experts_input.activation)
    )
    activation_situ_beta = None if moe_config is None else moe_config.activation_situ_beta
    activation_situ_linear_beta = None if moe_config is None else moe_config.activation_situ_linear_beta
    swiglu_limit = 0.0 if moe_config is None else getattr(moe_config, "swiglu_limit", 0.0) or 0.0
    swiglu_alpha = 1.0 if moe_config is None else getattr(moe_config, "swiglu_alpha", 1.0) or 1.0
    swiglu_beta = 0.0 if moe_config is None else getattr(moe_config, "swiglu_beta", 0.0) or 0.0
    fusion_enabled = enable_fusion_gmmswigluquant() if use_fusion_ops is None else use_fusion_ops

    return MoEMlpComputeInput(
        hidden_states=token_dispatch_output.hidden_states,
        group_list=token_dispatch_output.group_list,
        group_list_type=token_dispatch_output.group_list_type,
        dynamic_scale=token_dispatch_output.dynamic_scale,
        topk_scales=token_dispatch_output.topk_scales,
        weights=fused_experts_input.weights,
        quant=fused_experts_input.quant,
        fusion=fused_experts_input.quant.quant_type
        in (
            QuantType.W8A8,
            QuantType.W8A8MXFP,
            QuantType.W4A4MXFP,
            QuantType.W4A8MXFP,
            QuantType.W8A8FP,
            QuantType.W4A16MXFP,
        )
        and fusion_enabled,
        activation=activation,
        need_trans=fused_experts_input.need_trans,
        dynamic_eplb=fused_experts_input.dynamic_eplb,
        record_before_gmm2=fused_experts_input.record_before_gmm2,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        expanded_row_idx=expanded_row_idx,
        topk_ids=fused_experts_input.topk_ids,
        lora_context=fused_experts_input.lora_context,
    )
