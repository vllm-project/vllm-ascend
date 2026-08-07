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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.moe_mlp_activation import (
    MoEMLPActivationKind,
    resolve_mlp_activation,
    supports_fused_swiglu,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import enable_custom_op

MXFP_QUANT_TYPES = (
    QuantType.W8A8MXFP,
    QuantType.W4A4MXFP,
    QuantType.W4A8MXFP,
    QuantType.W4A16MXFP,
)


class GateUpKernel(Enum):
    UNQUANTIZED = "unquantized"
    ANTIQUANT = "antiquant"
    CUSTOM_NZ = "custom_nz"
    FUSED = "fused"
    DEQUANT_SWIGLU = "dequant_swiglu"
    W4A8_PER_CHANNEL = "w4a8_per_channel"
    DECOMPOSED = "decomposed"


@dataclass(frozen=True, slots=True)
class MoEMLPPlanKey:
    """Static inputs that affect MLP kernel selection.

    FusedMC2 bypasses this executor. Regular MC2 remains a plan dimension
    because it enables the GMM + dequant-SwiGLU-quant kernel sequence.
    """

    quant_type: QuantType
    activation: MoEMLPActivationKind
    fusion: bool
    dynamic_eplb: bool
    group_list_type: int
    custom_op_enabled: bool
    moe_comm_type: MoECommType = MoECommType.ALLGATHER


@dataclass(frozen=True, slots=True)
class MoEMLPPlan:
    """Immutable kernel choices for one MoE MLP configuration."""

    key: MoEMLPPlanKey
    gate_up_kernel: GateUpKernel

    def execute(self, mlp_compute_input: MoEMlpComputeInput) -> tuple[torch.Tensor, torch.npu.Event | None]:
        # Local import keeps the plan definition independent from the kernel
        # implementation and avoids a module import cycle.
        from vllm_ascend.ops.fused_moe.moe_mlp import execute_mlp_plan

        return execute_mlp_plan(self, mlp_compute_input)


def build_mlp_plan(key: MoEMLPPlanKey) -> MoEMLPPlan:
    if key.quant_type == QuantType.NONE:
        return MoEMLPPlan(key, GateUpKernel.UNQUANTIZED)
    if key.quant_type == QuantType.W4A16:
        return MoEMLPPlan(key, GateUpKernel.ANTIQUANT)
    if key.quant_type == QuantType.W4A8:
        kernel = (
            GateUpKernel.W4A8_PER_CHANNEL
            if key.custom_op_enabled and supports_fused_swiglu(key.activation)
            else GateUpKernel.DECOMPOSED
        )
        return MoEMLPPlan(key, kernel)

    use_mxfp_quant = key.quant_type in MXFP_QUANT_TYPES
    is_mc2 = key.moe_comm_type == MoECommType.MC2
    if supports_fused_swiglu(key.activation):
        if key.fusion and key.dynamic_eplb and key.custom_op_enabled and not use_mxfp_quant:
            kernel = GateUpKernel.CUSTOM_NZ
        elif use_mxfp_quant or (key.fusion and not key.dynamic_eplb):
            kernel = GateUpKernel.FUSED
        elif is_mc2:
            kernel = GateUpKernel.DEQUANT_SWIGLU
        else:
            kernel = GateUpKernel.DECOMPOSED
    elif key.activation == MoEMLPActivationKind.SWIGLUOAI_UNINTERLEAVE:
        kernel = GateUpKernel.DEQUANT_SWIGLU if is_mc2 else GateUpKernel.DECOMPOSED
    else:
        kernel = GateUpKernel.DECOMPOSED
    return MoEMLPPlan(key, kernel)


class MoEMLPPlanner:
    """Caches plans so static kernel decisions stay out of the hot path."""

    def __init__(self, custom_op_enabled: bool | None = None) -> None:
        self._plans: dict[MoEMLPPlanKey, MoEMLPPlan] = {}
        self._custom_op_enabled = custom_op_enabled

    def _resolve_custom_op_enabled(self) -> bool:
        if self._custom_op_enabled is None:
            self._custom_op_enabled = enable_custom_op()
        return self._custom_op_enabled

    def get_plan(self, mlp_compute_input: MoEMlpComputeInput) -> MoEMLPPlan:
        quant_type = mlp_compute_input.quant.quant_type
        activation = resolve_mlp_activation(mlp_compute_input.activation)
        custom_op_relevant = supports_fused_swiglu(activation) and (
            quant_type == QuantType.W4A8
            or (
                mlp_compute_input.fusion
                and mlp_compute_input.dynamic_eplb
                and quant_type not in (QuantType.NONE, QuantType.W4A16, *MXFP_QUANT_TYPES)
            )
        )
        key = MoEMLPPlanKey(
            quant_type=quant_type,
            activation=activation,
            fusion=mlp_compute_input.fusion,
            dynamic_eplb=mlp_compute_input.dynamic_eplb,
            group_list_type=mlp_compute_input.group_list_type,
            custom_op_enabled=self._resolve_custom_op_enabled() if custom_op_relevant else False,
            moe_comm_type=mlp_compute_input.moe_comm_type,
        )
        plan = self._plans.get(key)
        if plan is None:
            plan = build_mlp_plan(key)
            self._plans[key] = plan
        return plan

    def execute(self, mlp_compute_input: MoEMlpComputeInput) -> tuple[torch.Tensor, torch.npu.Event | None]:
        return self.get_plan(mlp_compute_input).execute(mlp_compute_input)


__all__ = [
    "GateUpKernel",
    "MoEMLPPlan",
    "MoEMLPPlanKey",
    "MoEMLPPlanner",
    "build_mlp_plan",
]
