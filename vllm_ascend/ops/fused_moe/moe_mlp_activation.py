# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from enum import Enum

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul


class MoEMLPActivationKind(Enum):
    SWIGLU = "swiglu"
    SWIGLUSTEP = "swiglustep"
    SWIGLUOAI = "swigluoai"
    SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"
    GELU = "gelu"
    GELU_TANH = "gelu_tanh"


def resolve_mlp_activation(activation: str | MoEActivation | None) -> MoEMLPActivationKind:
    act_name = getattr(activation, "value", activation)
    if activation == MoEActivation.SWIGLUSTEP or act_name == "swiglustep":
        return MoEMLPActivationKind.SWIGLUSTEP
    if activation == MoEActivation.SWIGLUOAI or act_name == "swigluoai":
        return MoEMLPActivationKind.SWIGLUOAI
    if act_name == "swigluoai_uninterleave":
        return MoEMLPActivationKind.SWIGLUOAI_UNINTERLEAVE
    if activation == MoEActivation.GELU or act_name == "gelu":
        return MoEMLPActivationKind.GELU
    if activation == MoEActivation.GELU_TANH or act_name == "gelu_tanh":
        return MoEMLPActivationKind.GELU_TANH
    return MoEMLPActivationKind.SWIGLU


def supports_fused_swiglu(activation: MoEMLPActivationKind) -> bool:
    return activation == MoEMLPActivationKind.SWIGLU


def apply_unquantized_activation(
    hidden_states: torch.Tensor,
    activation: MoEMLPActivationKind,
    *,
    hidden_size: int,
    swiglu_limit: float,
    swiglu_alpha: float,
    swiglu_beta: float,
) -> torch.Tensor:
    if activation == MoEMLPActivationKind.SWIGLUOAI:
        return AscendSwigluOAIAndMul.swiglu_oai_forward(
            hidden_states.view(-1, hidden_size),
            alpha=swiglu_alpha,
            limit=swiglu_limit or 7.0,
        )
    if activation == MoEMLPActivationKind.SWIGLUOAI_UNINTERLEAVE:
        return torch_npu.npu_clipped_swiglu(
            hidden_states,
            interleaved=False,
            alpha=swiglu_alpha,
            limit=swiglu_limit,
            bias=swiglu_beta,
        )
    if activation == MoEMLPActivationKind.SWIGLUSTEP:
        return AscendSwigluStepAndMul.swiglustep_forward(hidden_states, limit=swiglu_limit or 7.0)
    if activation in (MoEMLPActivationKind.GELU, MoEMLPActivationKind.GELU_TANH):
        gate, up = hidden_states.chunk(2, dim=-1)
        approximate = "tanh" if activation == MoEMLPActivationKind.GELU_TANH else "none"
        return torch.nn.functional.gelu(gate, approximate=approximate) * up
    if swiglu_limit > 0:
        gate, up = hidden_states.chunk(2, dim=-1)
        gate.clamp_(max=swiglu_limit)
        up.clamp_(min=-swiglu_limit, max=swiglu_limit)
    return torch_npu.npu_swiglu(hidden_states)


def apply_quantized_activation(
    hidden_states: torch.Tensor,
    activation: MoEMLPActivationKind,
    *,
    swiglu_limit: float,
    swiglu_alpha: float,
    swiglu_beta: float,
    act_quant_type: torch.dtype,
    use_mxfp_quant: bool,
    group_list: torch.Tensor,
    group_list_type: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if activation == MoEMLPActivationKind.SWIGLUSTEP:
        hidden_states = AscendSwigluStepAndMul.swiglustep_forward(hidden_states, limit=swiglu_limit or 7.0)
        return DeviceOperator.npu_dynamic_quant(
            hidden_states,
            act_quant_type=act_quant_type,
            use_mxfp_quant=use_mxfp_quant,
        )
    if activation in (MoEMLPActivationKind.GELU, MoEMLPActivationKind.GELU_TANH):
        gate, up = hidden_states.chunk(2, dim=-1)
        approximate = "tanh" if activation == MoEMLPActivationKind.GELU_TANH else "none"
        hidden_states = torch.nn.functional.gelu(gate, approximate=approximate) * up
        return torch_npu.npu_dynamic_quant(hidden_states)
    if activation == MoEMLPActivationKind.SWIGLUOAI:
        hidden_states = AscendSwigluOAIAndMul.swiglu_oai_forward(
            hidden_states,
            alpha=swiglu_alpha,
            limit=swiglu_limit or 7.0,
        )
        return DeviceOperator.npu_dynamic_quant(
            hidden_states,
            act_quant_type=act_quant_type,
            use_mxfp_quant=use_mxfp_quant,
        )
    if activation == MoEMLPActivationKind.SWIGLUOAI_UNINTERLEAVE:
        hidden_states = torch_npu.npu_clipped_swiglu(
            hidden_states,
            interleaved=False,
            alpha=swiglu_alpha,
            limit=swiglu_limit,
            bias=swiglu_beta,
        )
        return DeviceOperator.npu_dynamic_quant(
            hidden_states,
            act_quant_type=act_quant_type,
            use_mxfp_quant=use_mxfp_quant,
        )
    if HAS_TRITON:
        from vllm_ascend.ops.triton.activation.swiglu_quant import swiglu_quant

        return swiglu_quant(hidden_states, group_list=group_list, group_list_type=group_list_type)
    hidden_states = torch_npu.npu_swiglu(hidden_states)
    return torch_npu.npu_dynamic_quant(hidden_states)


__all__ = [
    "MoEMLPActivationKind",
    "apply_quantized_activation",
    "apply_unquantized_activation",
    "resolve_mlp_activation",
    "supports_fused_swiglu",
]
