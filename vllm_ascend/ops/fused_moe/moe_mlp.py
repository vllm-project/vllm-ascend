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
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import (AscendDeviceType, dispose_tensor,
                               enable_custom_op, get_ascend_device_type,
                               get_weight_prefetch_method)


def _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
    return fusion and dynamic_eplb and enable_custom_op()


def cumsum_group_list(group_list: torch.Tensor,
                      src_list_type: int,
                      dst_list_type: int,
                      active_num: int = 0,
                      expert_num: int = 0) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(
            f"group_list_type should be in [0, 1, 2], but received {src_list_type}"
        )

    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        group_diff = torch.diff(group_list)
        new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
        return new_group
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumsum_group_list = torch.full(size=(expert_num,),
                                       fill_value=active_num,
                                       dtype=group_list.dtype,
                                       device=group_list.device)

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumsum_group_list[start:end] = tokens[i]

        return cumsum_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
        "This feature is under development.")


def quant_apply_mlp(hidden_states: torch.Tensor,
                    w1: list[torch.Tensor],
                    w1_scale: list[torch.Tensor],
                    w2: list[torch.Tensor],
                    w2_scale: list[torch.Tensor],
                    group_list: torch.Tensor,
                    group_list_type: int = 1,
                    dynamic_scale: torch.Tensor = None,
                    w1_scale_bias: torch.Tensor = None,
                    w2_scale_bias: torch.Tensor = None,
                    w1_offset: Optional[torch.Tensor] = None,
                    w2_offset: Optional[torch.Tensor] = None,
                    fusion: bool = False,
                    dynamic_eplb: bool = False) -> torch.Tensor:
    if w1_offset is not None:
        unquantized_hidden_states = hidden_states
        quantized_hidden_states = None
        pertoken_scale = None
    elif dynamic_scale is None:
        print(f" dynamic_scale is None")
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(
            hidden_states)
        # Dispose the original unquantized hidden states
        # to save npu memory because they're no longer used.
        dispose_tensor(unquantized_hidden_states)
        quantized_hidden_states = None
    else:
        unquantized_hidden_states = None
        pertoken_scale = dynamic_scale
        quantized_hidden_states = hidden_states

    bias1, bias2 = None, None
    _output_dtype = w2_scale[0].dtype

    weight_prefetch_method = get_weight_prefetch_method()
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_postprocess(
            hidden_states)
    is_mc2 = get_forward_context().moe_comm_type == MoECommType.MC2
    if w1_scale_bias is None and w1_offset is None and is_mc2:
        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = (
                torch.ops._C_ascend.
                    grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                    x=hidden_states,
                    weight=w1,
                    weight_scale=w1_scale,
                    x_scale=pertoken_scale,
                    group_list=cumsum_group_list(group_list, group_list_type,
                                                 0),
                ))
        elif fusion and not dynamic_eplb:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w1[0],
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=w1_scale[0],
                x_scale=pertoken_scale)
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            if w1_scale[0].dtype != torch.float32:
                w1_scale[0] = w1_scale[0].to(torch.float32)
            if hidden_states.dtype != w1[0].dtype:
                hidden_states, _ = torch_npu.npu_dynamic_quant(
                    hidden_states, dst_dtype=w1[0].dtype)
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                split_item=3,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.int32)[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
                x=hidden_states,
                weight_scale=w1_scale[0],
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=cumsum_group_list(group_list, group_list_type, 1),
                activate_left=True,
                quant_mode=1,
            )
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            scale=w2_scale,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=w2_scale[0].dtype)[0]
    elif w1_offset is not None:
        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[unquantized_hidden_states],
            weight=w1,
            antiquant_scale=w1_scale,
            antiquant_offset=[w1_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]
        dispose_tensor(unquantized_hidden_states)
        # act_fn: swiglu
        hidden_states = torch_npu.npu_swiglu(hidden_states)
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            antiquant_scale=w2_scale,
            antiquant_offset=[w2_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]
    else:
        if w1_scale_bias is not None:
            if group_list_type == 0:
                group_list = torch.cat(
                    [group_list[:1],
                     torch.diff(group_list, dim=0)])
                group_list_type = 1
            # w1_scale_bias = w1_scale_bias.to(torch.float32)
            # w2_scale_bias = w2_scale_bias.to(torch.float32)
            bias1 = [w1_scale_bias] if not fusion else w1_scale_bias
            bias2 = [w2_scale_bias]
            # TODO w4a8 scene: dynamic acquisition of dtype in the future
            _output_dtype = torch.bfloat16

        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = (
                torch.ops._C_ascend.
                    grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                    x=hidden_states,
                    weight=w1,
                    weight_scale=w1_scale,
                    x_scale=pertoken_scale,
                    group_list=cumsum_group_list(group_list, group_list_type,
                                                 0),
                    bias=bias1,
                ))
        elif fusion and not dynamic_eplb:
            # gmm1: gate_up_proj & act_fn: swiglu
            print(f"##DEBUG before npu_grouped_matmul_swiglu_quant -----")
            hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w1[0],
                bias=bias1,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=w1_scale[0],
                x_scale=pertoken_scale)
            print(f"##DEBUG after npu_grouped_matmul_swiglu_quant -----")
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            w1_scale[0] = w1_scale[0].to(w2_scale[0].dtype)
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                scale=w1_scale,
                bias=bias1,
                per_token_scale=[pertoken_scale],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=_output_dtype)[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            if HAS_TRITON:
                from vllm_ascend.ops.triton.activation.swiglu_quant import \
                    swiglu_quant
                hidden_states, swiglu_out_scale = swiglu_quant(
                    hidden_states,
                    group_list=group_list,
                    group_list_type=group_list_type)
            else:
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(
                    hidden_states)
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            scale=w2_scale,
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
    if get_ascend_device_type() == AscendDeviceType._310P:
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
                      w1: torch.Tensor | list[torch.Tensor],
                      w2: torch.Tensor | list[torch.Tensor],
                      group_list: torch.Tensor,
                      w1_scale: Optional[list[torch.Tensor]] = None,
                      w2_scale: Optional[list[torch.Tensor]] = None,
                      dynamic_scale: torch.Tensor = None,
                      group_list_type: int = 1,
                      w1_scale_bias: torch.Tensor = None,
                      w2_scale_bias: torch.Tensor = None,
                      w1_offset: Optional[torch.Tensor] = None,
                      w2_offset: Optional[torch.Tensor] = None,
                      topk_scales: Optional[torch.Tensor] = None,
                      with_quant: bool = False,
                      fusion: bool = False,
                      need_trans: bool = True,
                      dynamic_eplb: bool = False) -> torch.Tensor:
    if with_quant:
        if isinstance(w1, (torch.Tensor, torch.nn.Parameter)):
            w1 = [w1]
        if isinstance(w2, (torch.Tensor, torch.nn.Parameter)):
            w2 = [w2]
        if isinstance(w1_scale, (torch.Tensor, torch.nn.Parameter)):
            w1_scale = [w1_scale]
        if isinstance(w2_scale, (torch.Tensor, torch.nn.Parameter)):
            w2_scale = [w2_scale]

        assert w1_scale is not None and w2_scale is not None
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
                               w1_offset=w1_offset,
                               w2_offset=w2_offset,
                               fusion=fusion,
                               dynamic_eplb=dynamic_eplb)
    else:
        return unquant_apply_mlp(hidden_states=hidden_states,
                                 w1=w1,
                                 w2=w2,
                                 group_list=group_list,
                                 group_list_type=group_list_type,
                                 topk_scales=topk_scales,
                                 need_trans=need_trans)


def fused_experts(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        x_active_mask: torch.Tensor,
        group_ep: str,
        ep_rank_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
):
    output, _ = torch.ops.umdk_cam_op_lib.dispatch_gmm_combine_decode(
        x=hidden_states,
        expert_ids=topk_ids,
        gmm1_permuted_weight=w1,
        gmm1_permuted_weight_scale=w1_scale,
        gmm2_weight=w2,
        gmm2_weight_scale=w2_scale,
        expert_scales=topk_weights,
        expert_smooth_scales=None,
        x_active_mask=x_active_mask,
        group_ep=group_ep,
        ep_rank_size=ep_rank_size,
        ep_rank_id=ep_rank_id,
        moe_expert_num=moe_expert_num,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        quant_mode=0,
        global_bs=0)
    return output


def dispatch_experts(
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        x_active_mask: torch.Tensor,
        group_ep: str,
        ep_rank_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
):
    """
    Dispatch阶段：将token分发到对应的专家
    返回值：dispatch_output的所有输出
    """
    print(f"##DEBUG## group_ep {group_ep}")

    dispatch_kwargs = {
        "x": hidden_states,
        "expert_ids": topk_ids,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": moe_expert_num,
        "global_bs": 0,
        "expert_token_nums_type": 0,
        "scales": None,
        "quant_mode": 0,
        "group_ep": group_ep,
        "ep_world_size": ep_rank_size,
        "ep_rank_id": ep_rank_id,
        "x_active_mask": x_active_mask,
        "group_tp": group_ep,
        "tp_world_size": 1,
        "tp_rank_id": 0,
    }

    dispatch_output = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)

    # 返回所有输出，保持与原函数完全相同的解析方式
    return dispatch_output


def combine_experts(
        gmm2_output: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        x_active_mask: torch.Tensor,
        assist_info_for_combine: torch.Tensor,
        ep_recv_counts: torch.Tensor,
        tp_recv_counts: torch.Tensor,
        expand_scales: torch.Tensor,
        group_ep: str,
        ep_rank_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
):
    """
    Combine阶段：将计算结果重新组合
    返回值：combined_output
    """
    combine_kwargs = {
        "expand_x": gmm2_output,
        "expert_ids": topk_ids,
        "expert_scales": topk_weights.to(torch.float32) if topk_weights is not None else None,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": moe_expert_num,
        "global_bs": 0,
        "ep_send_counts": ep_recv_counts,
        "group_ep": group_ep,
        "ep_world_size": ep_rank_size,
        "ep_rank_id": ep_rank_id,
        "expand_scales": expand_scales,
        "x_active_mask": x_active_mask,
        "tp_send_counts": tp_recv_counts,
        "group_tp": group_ep,
        "tp_world_size": 1,
        "tp_rank_id": 0,
    }

    # 添加assist_info_for_combine
    if hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
        combine_kwargs["assist_info_for_combine"] = assist_info_for_combine
    else:
        combine_kwargs["expand_idx"] = assist_info_for_combine

    # 调用combine算子
    if hasattr(torch_npu, "npu_moe_distribute_combine_v2"):
        combined_output = torch_npu.npu_moe_distribute_combine_v2(**combine_kwargs)
    else:
        combined_output = torch_npu.npu_moe_distribute_combine(**combine_kwargs)

    return combined_output
