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
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput


def clipped_swiglu_310p(
    gate_up: torch.Tensor,
    *,
    limit: float = 0.0,
    alpha: float = 1.0,
    bias: float = 0.0,
) -> torch.Tensor:
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(f"Gate/up width must be even, got {gate_up.shape[-1]}.")
    if alpha != 1.0 or bias != 0.0:
        raise NotImplementedError("The 310P composed routed-expert SwiGLU path supports only alpha=1.0 and bias=0.0.")
    original_dtype = gate_up.dtype
    gate, up = gate_up.float().chunk(2, dim=-1)
    if limit > 0.0:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    return (F.silu(gate) * up).to(original_dtype)


def zero_inactive_grouped_matmul_rows(
    hidden_states: torch.Tensor,
    cumulative_group_list: torch.Tensor,
) -> torch.Tensor:
    """Clear rows that 310P grouped matmul leaves unwritten.

    ``npu_quant_grouped_matmul_dequant`` preserves the input row count even
    when the cumulative group list covers fewer rows. On 310P, the uncovered
    tail is not initialized and may contain NaNs or stale values. Keep the
    operation device-side so it also works with dynamic local expert loads.
    """
    if cumulative_group_list.numel() == 0:
        return torch.zeros_like(hidden_states)
    row_ids = torch.arange(
        hidden_states.shape[0],
        dtype=cumulative_group_list.dtype,
        device=hidden_states.device,
    )
    valid_rows = row_ids < cumulative_group_list[-1]
    return torch.where(valid_rows.unsqueeze(-1), hidden_states, torch.zeros_like(hidden_states))


def quant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_beta: float = 0.0,
) -> torch.Tensor:
    if group_list_type == 1:
        # Convert expert row counts to the cumulative format required by GMM.
        group_list = torch.cumsum(group_list, dim=0)

    hidden_states = torch_npu.npu_quant_grouped_matmul_dequant(
        x=hidden_states,
        quantized_weight=w1,
        weight_scale=w1_scale,
        group_list=group_list,
        quant_mode="pertoken",
    )
    hidden_states = zero_inactive_grouped_matmul_rows(hidden_states, group_list)
    hidden_states = clipped_swiglu_310p(
        hidden_states,
        limit=swiglu_limit,
        alpha=swiglu_alpha,
        bias=swiglu_beta,
    )
    hidden_states = torch_npu.npu_quant_grouped_matmul_dequant(
        x=hidden_states,
        quantized_weight=w2,
        weight_scale=w2_scale,
        group_list=group_list,
        quant_mode="pertoken",
    )
    return zero_inactive_grouped_matmul_rows(hidden_states, group_list)


def unquant_apply_mlp(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, group_list: torch.Tensor, group_list_type: int = 1
) -> torch.Tensor:
    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    act_out = torch_npu.npu_swiglu(gate_up_out)

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[act_out],
        weight=[w2],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    return hidden_states


def unified_apply_mlp(*, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
    hidden_states = mlp_compute_input.hidden_states
    w1 = mlp_compute_input.weights.w1
    w2 = mlp_compute_input.weights.w2
    w1_scale = mlp_compute_input.weights.w1_scale
    w2_scale = mlp_compute_input.weights.w2_scale
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    assert isinstance(w1, torch.Tensor)
    assert isinstance(w2, torch.Tensor)

    if mlp_compute_input.quant.is_quant:
        assert isinstance(w1_scale, torch.Tensor)
        assert isinstance(w2_scale, torch.Tensor)
        assert w1_scale is not None and w2_scale is not None
        return quant_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            group_list=group_list,
            group_list_type=group_list_type,
            swiglu_limit=mlp_compute_input.swiglu_limit,
            swiglu_alpha=mlp_compute_input.swiglu_alpha,
            swiglu_beta=mlp_compute_input.swiglu_beta,
        )

    return unquant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        group_list=group_list,
        group_list_type=group_list_type,
    )
