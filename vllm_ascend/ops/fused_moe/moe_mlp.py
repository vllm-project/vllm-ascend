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


import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput


def apply_moe_mlp(
    mlp_compute_input: MoEMlpComputeInput,
    quant_method,
) -> tuple[torch.Tensor, torch.npu.Event]:
    """
    Unified MoE MLP entry.
    Quant path is dispatched by each FusedMoEMethod with explicit typed kernel flags.
    """

    # When LoRA adapter is used in quantized weight, use individual lora impl.
    if mlp_compute_input.quant.is_quant and mlp_compute_input.lora_context is not None:
        from vllm_ascend.lora.fused_moe import has_lora

        if has_lora(mlp_compute_input.lora_context):
            from vllm_ascend.lora.quant_moe import quant_apply_mlp_with_moe_lora

            return quant_apply_mlp_with_moe_lora(
                mlp_compute_input=mlp_compute_input,
                quant_method=quant_method,
            )

    if quant_method.supports_fused_activation(mlp_compute_input.activation):
        hidden_states, act_out_scale = quant_method.apply_gmm1_act_quant(mlp_compute_input)
    else:
        hidden_states = quant_method.apply_gmm1(mlp_compute_input)
        hidden_states = _unified_apply_activation(mlp_compute_input, hidden_states, quant_method)
        hidden_states, act_out_scale = quant_method.apply_act_quant(mlp_compute_input, hidden_states)

    before_gmm2_evt = torch.npu.current_stream().record_event()
    hidden_states = quant_method.apply_gmm2(mlp_compute_input, hidden_states, act_out_scale)
    return hidden_states, before_gmm2_evt


def _unified_apply_activation(
    mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor, quant_method
) -> torch.Tensor:
    activation = mlp_compute_input.activation

    if activation == MoEActivation.SITU:
        hidden_states = _apply_situ(
            hidden_states,
            beta=1.0 if mlp_compute_input.activation_situ_beta is None else mlp_compute_input.activation_situ_beta,
            linear_beta=mlp_compute_input.activation_situ_linear_beta,
        )
    elif activation == MoEActivation.SWIGLUOAI:
        w1, _ = quant_method.get_mlp_weights(mlp_compute_input.layer)
        _, _, hidden_size = w1.shape
        hidden_states = AscendSwigluOAIAndMul.swiglu_oai_forward(hidden_states.view(-1, hidden_size))
    elif activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        hidden_states = DeviceOperator.clipped_swiglu(
            hidden_states,
            swiglu_limit=mlp_compute_input.swiglu_limit,
            swiglu_alpha=mlp_compute_input.swiglu_alpha,
            swiglu_beta=mlp_compute_input.swiglu_beta,
        )
    elif activation == MoEActivation.SWIGLUSTEP:
        hidden_states = AscendSwigluStepAndMul.swiglustep_forward(hidden_states, limit=7.0)
    elif activation == MoEActivation.GELU:
        gate, up = hidden_states.chunk(2, dim=-1)
        hidden_states = torch.nn.functional.gelu(gate) * up
    elif activation == MoEActivation.GELU_TANH:
        gate, up = hidden_states.chunk(2, dim=-1)
        hidden_states = torch.nn.functional.gelu(gate, approximate="tanh") * up
    else:
        if mlp_compute_input.swiglu_limit > 0:
            gate, up = hidden_states.chunk(2, dim=-1)
            gate.clamp_(max=mlp_compute_input.swiglu_limit)
            up.clamp_(min=-mlp_compute_input.swiglu_limit, max=mlp_compute_input.swiglu_limit)
        hidden_states = torch_npu.npu_swiglu(hidden_states)

    return hidden_states


def _apply_situ(
    hidden_states: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Match SituAndMul.forward_native without constructing a CustomOp in forward."""
    gate, up = hidden_states.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(hidden_states.dtype)
