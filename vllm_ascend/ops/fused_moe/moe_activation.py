#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Activation-aware MoE MLP orchestration.

Each ``MoeActionMethod`` owns the orchestration of one activation family:
it decides whether the quant method provides a fused gmm1+act+quant kernel
and otherwise runs gmm1 -> activation -> (re)quant -> gmm2, delegating every
quant-specific kernel to the ``quant_method`` (an ``AscendMoEScheme``
subclass) passed at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput


class MoeActionMethod(ABC):
    """Base orchestrator for one MoE activation family."""

    #: activation string value handled by this action method
    activation: str = MoEActivation.SILU.value

    def apply_mlp(
        self,
        mlp_compute_input: MoEMlpComputeInput,
        quant_method,
    ) -> tuple[torch.Tensor, torch.npu.Event]:
        """Run the full MLP compute for this activation family."""
        if quant_method.supports_fused_activation(self.activation):
            hidden_states, act_out_scale = quant_method.apply_gmm1_act_quant(mlp_compute_input)
        else:
            hidden_states = quant_method.apply_gmm1(mlp_compute_input)
            hidden_states = self.apply_activation(mlp_compute_input, hidden_states)
            hidden_states, act_out_scale = quant_method.apply_act_quant(mlp_compute_input, hidden_states)

        before_gmm2_evt = torch.npu.current_stream().record_event()
        hidden_states = quant_method.apply_gmm2(mlp_compute_input, hidden_states, act_out_scale)
        return hidden_states, before_gmm2_evt

    @abstractmethod
    def apply_activation(
        self,
        mlp_compute_input: MoEMlpComputeInput,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the activation function to the gmm1 output."""
        ...


class SiluMoeActionMethod(MoeActionMethod):
    """Default/other activations: silu family (incl. clamped silu and the
    unquant-only interleaved swigluoai)."""

    activation: str = MoEActivation.SILU.value

    def apply_activation(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor) -> torch.Tensor:
        act = getattr(mlp_compute_input.activation, "value", mlp_compute_input.activation)
        if act == MoEActivation.SWIGLUOAI.value:
            # gpt-oss style interleaved gate/up layout, unquant path only.
            layer = mlp_compute_input.layer
            hidden_size = layer.w13_weight.shape[-1]
            return AscendSwigluOAIAndMul.swiglu_oai_forward(hidden_states.view(-1, hidden_size))
        if mlp_compute_input.swiglu_limit > 0:
            gate, up = hidden_states.chunk(2, dim=-1)
            gate.clamp_(max=mlp_compute_input.swiglu_limit)
            up.clamp_(min=-mlp_compute_input.swiglu_limit, max=mlp_compute_input.swiglu_limit)
        return torch_npu.npu_swiglu(hidden_states)


class GeluMoeActionMethod(MoeActionMethod):
    activation: str = MoEActivation.GELU.value

    def apply_activation(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = hidden_states.chunk(2, dim=-1)
        return torch.nn.functional.gelu(gate) * up


class GeluTanhMoeActionMethod(GeluMoeActionMethod):
    activation: str = MoEActivation.GELU_TANH.value

    def apply_activation(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = hidden_states.chunk(2, dim=-1)
        return torch.nn.functional.gelu(gate, approximate="tanh") * up


class SwigluStepMoeActionMethod(MoeActionMethod):
    activation: str = MoEActivation.SWIGLUSTEP.value

    def apply_activation(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor) -> torch.Tensor:
        return AscendSwigluStepAndMul.swiglustep_forward(
            hidden_states,
            limit=mlp_compute_input.swiglu_limit or 7.0,
        )


class SwigluOaiUninterleaveMoeActionMethod(MoeActionMethod):
    activation: str = MoEActivation.SWIGLUOAI_UNINTERLEAVE.value

    def apply_activation(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_clipped_swiglu(
            hidden_states,
            interleaved=False,
            alpha=mlp_compute_input.swiglu_alpha,
            limit=mlp_compute_input.swiglu_limit,
            bias=mlp_compute_input.swiglu_beta,
        )


_MOE_ACTIVATION_METHODS: dict[str, MoeActionMethod] = {
    MoEActivation.SILU.value: SiluMoeActionMethod(),
    MoEActivation.GELU.value: GeluMoeActionMethod(),
    MoEActivation.GELU_TANH.value: GeluTanhMoeActionMethod(),
    MoEActivation.SWIGLUSTEP.value: SwigluStepMoeActionMethod(),
    MoEActivation.SWIGLUOAI_UNINTERLEAVE.value: SwigluOaiUninterleaveMoeActionMethod(),
}


def get_moe_activation_method(activation) -> MoeActionMethod:
    """Resolve the orchestration method for an MoE activation (enum or str)."""
    act = getattr(activation, "value", activation)
    try:
        act = MoEActivation.from_str(act).value
    except ValueError:
        # Unknown/legacy activation strings fall back to the default silu path.
        return SiluMoeActionMethod()
    return _MOE_ACTIVATION_METHODS.get(act, SiluMoeActionMethod())
