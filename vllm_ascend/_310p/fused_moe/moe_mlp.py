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

import torch

from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput


def apply_moe_mlp(
    mlp_compute_input: MoEMlpComputeInput,
    quant_method,
) -> tuple[torch.Tensor, torch.npu.Event]:
    """
    Unified MoE MLP entry (310P).

    310P MoE only supports the swiglu (silu) activation, so every method
    implements the fused ``gmm1 + swiglu (+ quant)`` path and the separate
    ``apply_gmm1`` / ``apply_act_quant`` hooks are not used.
    """

    if not quant_method.supports_fused_activation(mlp_compute_input.activation):
        activation = mlp_compute_input.activation
        act_name = getattr(activation, "value", activation)
        raise NotImplementedError(f"310P MoE only supports the swiglu (silu) activation, but got {act_name}.")

    hidden_states, act_out_scale = quant_method.apply_gmm1_act_quant(mlp_compute_input)

    before_gmm2_evt = torch.npu.current_stream().record_event()
    hidden_states = quant_method.apply_gmm2(mlp_compute_input, hidden_states, act_out_scale)
    return hidden_states, before_gmm2_evt
