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
import torch_npu


def quant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
) -> torch.Tensor:
    if group_list_type == 1:
        # Convert group_list to cumulative sum format if group_list is count format
        group_list = torch.cumsum(group_list, dim=0)

    hidden_states = torch_npu.npu_quant_grouped_matmul_dequant(
        x=hidden_states, quantized_weight=w1, weight_scale=w1_scale, group_list=group_list, quant_mode="pertoken"
    )
    hidden_states = torch_npu.npu_swiglu(hidden_states)
    hidden_states = torch_npu.npu_quant_grouped_matmul_dequant(
        x=hidden_states, quantized_weight=w2, weight_scale=w2_scale, group_list=group_list, quant_mode="pertoken"
    )
    return hidden_states


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


def unified_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    group_list_type: int = 1,
    with_quant: bool = False,
) -> torch.Tensor:
    if with_quant:
        assert w1_scale is not None and w2_scale is not None
        return quant_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            group_list=group_list,
            group_list_type=group_list_type,
        )
    else:
        return unquant_apply_mlp(
            hidden_states=hidden_states, w1=w1, w2=w2, group_list=group_list, group_list_type=group_list_type
        )
