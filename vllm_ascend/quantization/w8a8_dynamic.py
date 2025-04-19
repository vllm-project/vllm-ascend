# Adapted from
# https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe.py

# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from typing import Callable, Optional, Dict, Any

import torch
import torch_npu


def group_topk(hidden_states: torch.Tensor,
               gating_output: torch.Tensor,
               topk: int,
               renormalize: bool,
               num_expert_group: Optional[int] = 0,
               topk_group: Optional[int] = 0,
               scoring_func: str = "softmax",
               e_score_correction_bias: Optional[torch.Tensor] = None):
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)

    topk_group = 0 if topk_group is None else topk_group
    num_expert_group = 0 if num_expert_group is None else num_expert_group

    num_token = scores.shape[0]
    group_scores = scores.view(num_token, num_expert_group,
                               -1).max(dim=-1).values
    group_idx = torch.topk(group_scores.to(torch.float32),
                           k=topk_group,
                           dim=-1,
                           sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)
    scores = scores.masked_fill(~score_mask.bool(), 0.0)

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids.to(torch.int32)


def fused_experts(hidden_states: torch.Tensor, w1: torch.Tensor, w1_scale: torch.Tensor,
                  w2: torch.Tensor, w2_scale: torch.Tensor, topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor, top_k: int):
    ori_shape = hidden_states.shape
    if len(ori_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape

    row_idx_len = num_tokens * top_k
    row_idx = torch.arange(0,
                           row_idx_len,
                           dtype=torch.int32,
                           device=topk_weights.device).view(top_k, -1).permute(
                               1, 0).contiguous()
    expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
        hidden_states,
        row_idx=row_idx,
        expert_idx=topk_ids,
        active_num=num_tokens)
    del hidden_states

    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, E)
    expert_tokens = expert_tokens.to(torch.int64)

    quant_x, x_dynamic_scale = torch_npu.npu_dynamic_quant(expanded_x)
    output_dtype = torch.bfloat16 if w1_scale.dtype == torch.bfloat16 else torch.float16

    gate_up_out_list = torch_npu.npu_grouped_matmul(x=[quant_x],
                                                    weight=[w1],
                                                    scale=[w1_scale],
                                                    per_token_scale=[x_dynamic_scale],
                                                    split_item=2,
                                                    group_list_type=0,
                                                    group_type=0,
                                                    group_list=expert_tokens,
                                                    output_dtype=output_dtype)
    del quant_x

    gate_up_out = gate_up_out_list[0] if len(gate_up_out_list) == 1 else torch.cat(gate_up_out_list, dim=0)
    del gate_up_out_list
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    quant_gate_up_out, gate_up_out_dynamic_scale = torch_npu.npu_dynamic_quant(gate_up_out)
    del gate_up_out

    down_out_list = torch_npu.npu_grouped_matmul(x=[quant_gate_up_out],
                                                 weight=[w2],
                                                 scale=[w2_scale],
                                                 per_token_scale=[gate_up_out_dynamic_scale],
                                                 split_item=2,
                                                 group_list_type=0,
                                                 group_type=0,
                                                 group_list=expert_tokens,
                                                 output_dtype=output_dtype)
    del quantized_gate_up_out

    down_out = down_out_list[0] if len(down_out_list) == 1 else torch.cat(down_out_list, dim=0)

    hidden_states = torch_npu.npu_moe_finalize_routing(
        down_out,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids)
    del down_out
    if len(ori_shape) == 3:
        hidden_states = hidden_states.view(ori_shape)
    return hidden_states


class AscendW8A8DynamicLinearMethod:
    """Linear method for Ascend W8A8_DYNAMIC.
    """

    def __init__(self):
        self.transpose_weight = True

    @staticmethod
    def get_weight(input_size: int,
                   output_size: int,
                   params_dtype: torch.dtype
    ) -> Dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict
    
    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        # use ATB quantize
        quant_out, dynamic_scale = torch_npu.npu_dynamic_quant(x)
        return torch_npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale,
            bias=bias,
            output_dtype=original_dtype,
        )
    
    def process_weights_after_loading(self, layer):
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = layer.weight_offset.data.flatten()


class AscendW8A8DynamicFusedMoEMethod:
    """FusedMoe method for Ascend W8A8_DYNAMIC.
    """

    def __init__(self):
        self.transpose_weight = True

    @staticmethod
    def get_weight(num_experts: int, intermediate_size_per_partition: int,
                   hidden_sizes: int, params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(num_experts, 2 * intermediate_size_per_partition,
                                               hidden_sizes, dtype=torch.int8)
        param_dict["w2_weight"] = torch.empty(num_experts, hidden_sizes,
                                              intermediate_size_per_partition, dtype=torch.int8)
        return param_dict

    @staticmethod
    def get_dynamic_quant_param(num_experts: int, intermediate_size_per_partition: int,
                                hidden_sizes: int, params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(num_experts, 2 * intermediate_size_per_partition, 1,
                                                     dtype=params_dtype)
        param_dict["w13_weight_offset"] = torch.empty(num_experts, 2 * intermediate_size_per_partition, 1,
                                                      dtype=params_dtype)
        param_dict["w2_weight_scale"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
        param_dict["w2_weight_offset"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
        return param_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        topk_weights, topk_ids = group_topk(
            hidden_states=x,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias)

        return fused_experts(hidden_states=x,
                             w1=layer.w13_weight,
                             w1_scale=layer.w13_weight_scale,
                             w2=layer.w2_weight,
                             w2_scale=layer.w2_weight_scale,
                             topk_weights=topk_weights,
                             topk_ids=topk_ids,
                             top_k=top_k)

    def process_weights_after_loading(self, layer):
        if self.transpose_weight:
            layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
            layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(layer.w13_weight_scale.data.shape[0], -1)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(layer.w13_weight_offset.data.shape[0], -1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(layer.w2_weight_scale.data.shape[0], -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(layer.w2_weight_offset.data.shape[0], -1)
