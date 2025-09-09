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

from typing import Optional

import torch
import torch_npu
from torch.nn.functional import pad
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import FusedMoEState
from vllm_ascend.utils import dispose_tensor, is_310p


def cumsum_group_list(group_list: torch.Tensor,
                      group_list_type: int,
                      active_num: int = 0,
                      expert_num: int = 0) -> torch.Tensor:
    if group_list_type not in [0, 1, 2]:
        raise ValueError(
            f"group_list_type should be in [0, 1, 2], but received {group_list_type}"
        )

    if group_list_type == 0:
        return group_list
    if group_list_type == 1:
        return group_list.cumsum(dim=0)

    experts = pad(group_list[:, 0], (1, 0))
    tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
    cumsum_group_list = torch.full(size=(expert_num, ),
                                   fill_value=active_num,
                                   dtype=group_list.dtype,
                                   device=group_list.device)

    for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
        if end > start:
            cumsum_group_list[start:end] = tokens[i]

    return cumsum_group_list


def quant_apply_mlp(hidden_states: torch.Tensor,
                    w1: torch.Tensor,
                    w1_scale: torch.Tensor,
                    w2: torch.Tensor,
                    w2_scale: torch.Tensor,
                    group_list: torch.Tensor,
                    group_list_type: int = 1,
                    dynamic_scale: torch.Tensor = None,
                    w1_scale_bias: torch.Tensor = None,
                    w2_scale_bias: torch.Tensor = None,
                    fusion: bool = False) -> torch.Tensor:
    if dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(
            hidden_states)
        # Dispose the original unquantized hidden states
        # to save npu memory because they're no longer used.
        dispose_tensor(unquantized_hidden_states)
    else:
        pertoken_scale = dynamic_scale

    bias1, bias2 = None, None
    _output_dtype = w2_scale.dtype

    is_mc2 = get_forward_context().fused_moe_state == FusedMoEState.MC2
    if w1_scale_bias is None and is_mc2:
        if w1_scale.dtype != torch.float32:
            w1_scale = w1_scale.to(torch.float32)
        if fusion:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w1,
                group_list=cumsum_group_list(group_list, group_list_type),
                weight_scale=w1_scale,
                x_scale=pertoken_scale)
        else:
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[w1],
                split_item=3,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.int32)[0]
            # act_fn: swiglu
            hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
                x=hidden_states,
                weight_scale=w1_scale,
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=group_list,
                activate_left=True,
                quant_mode=1,
            )
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            scale=[w2_scale],
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=w2_scale.dtype)[0]
    else:
        if w1_scale_bias is not None:
            if group_list_type == 0:
                group_list = torch.cat(
                    [group_list[:1],
                     torch.diff(group_list, dim=0)])
                group_list_type = 1
            bias1 = [w1_scale_bias] if not fusion else w1_scale_bias
            bias2 = [w2_scale_bias]
            # TODO w4a8 scene: dynamic acquisition of dtype in the future
            _output_dtype = torch.bfloat16

        if fusion:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w1,
                bias=bias1,
                group_list=cumsum_group_list(group_list, group_list_type),
                weight_scale=w1_scale,
                x_scale=pertoken_scale)
        else:
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[w1],
                scale=[w1_scale.to(w2_scale.dtype)],
                bias=bias1,
                per_token_scale=[pertoken_scale],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=_output_dtype)[0]
            # act_fn: swiglu
            hidden_states = torch_npu.npu_swiglu(hidden_states)
            hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(
                hidden_states)
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            scale=[w2_scale],
            bias=bias2,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]

    return hidden_states


def unquant_apply_mlp(hidden_states: torch.Tensor,
                      w1: torch.Tensor,
                      w2: torch.Tensor,
                      group_list: torch.Tensor,
                      group_list_type: int = 1,
                      topk_scales: Optional[torch.Tensor] = None,
                      need_trans: bool = True) -> torch.Tensor:

    if need_trans:
        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2)

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    if is_310p():
        gate_up_out = torch_npu.npu_swiglu(gate_up_out.to(torch.float32)).to(
            torch.float16)
    else:
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    if topk_scales is not None:
        gate_up_out *= topk_scales

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    return hidden_states


def unified_apply_mlp(hidden_states: torch.Tensor,
                      w1: torch.Tensor,
                      w1_scale: torch.Tensor,
                      w2: torch.Tensor,
                      w2_scale: torch.Tensor,
                      group_list: torch.Tensor,
                      dynamic_scale: torch.Tensor = None,
                      group_list_type: int = 1,
                      w1_scale_bias: torch.Tensor = None,
                      w2_scale_bias: torch.Tensor = None,
                      topk_scales: Optional[torch.Tensor] = None,
                      with_quant: bool = False,
                      fusion: bool = False,
                      need_trans: bool = True) -> torch.Tensor:
    if with_quant:
        return quant_apply_mlp(hidden_states=hidden_states,
                               w1=w1,
                               w1_scale=w1_scale,
                               w2=w2,
                               w2_scale=w2_scale,
                               group_list=group_list,
                               dynamic_scale=dynamic_scale,
                               group_list_type=group_list_type,
                               w1_scale_bias=w1_scale_bias,
                               w2_scale_bias=w2_scale_bias,
                               fusion=fusion)
    else:
        return unquant_apply_mlp(hidden_states=hidden_states,
                                 w1=w1,
                                 w2=w2,
                                 group_list=group_list,
                                 group_list_type=group_list_type,
                                 topk_scales=topk_scales,
                                 need_trans=need_trans)
