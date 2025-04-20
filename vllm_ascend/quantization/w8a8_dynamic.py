#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

from typing import Any, Callable, Dict, Optional

import torch
import torch_npu

from vllm_ascend.ops.fused_moe import select_experts


def fused_experts(hidden_states: torch.Tensor,
                  w1: torch.Tensor,
                  w1_scale: torch.Tensor,
                  w2: torch.Tensor,
                  w2_scale: torch.Tensor,
                  topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  top_k: int,
                  expert_map: torch.Tensor = None):
    original_shape = hidden_states.shape
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    num_tokens, _ = hidden_states.shape
    num_experts = w1.shape[0]
    dtype = hidden_states.dtype
    device = hidden_states.device

    if expert_map is not None:
        # Generate token indices and flatten
        token_indices = (torch.arange(num_tokens,
                                      device=device,
                                      dtype=torch.int64).unsqueeze(1).expand(
                                          -1, top_k).reshape(-1))

        # Flatten token-to-expert mappings and map to local experts
        weights_flat = topk_weights.view(-1)
        experts_flat = topk_ids.view(-1)
        local_experts_flat = expert_map[experts_flat]

        # Filter valid token-expert pairs
        mask = local_experts_flat != -1
        filtered_weights = torch.where(
            mask, weights_flat, torch.zeros_like(weights_flat)).to(dtype)
        filtered_experts = torch.where(
            mask, local_experts_flat,
            torch.full_like(local_experts_flat,
                            num_experts)).to(topk_ids.dtype)

        # Sort by local expert IDs
        sort_indices = torch.argsort(filtered_experts)
        sorted_token_indices = token_indices[sort_indices]
        sorted_weights = filtered_weights[sort_indices]

        # Compute token counts with minlength of num_experts
        # This is equivalent to but faster than:
        # >>> token_counts = torch.bincount(filtered_experts, minlength=num_experts)[:-1]
        token_counts = torch.zeros(num_experts + 1,
                                   device=device,
                                   dtype=torch.int64)
        ones = torch.ones_like(filtered_experts, dtype=torch.int64)
        token_counts.scatter_add_(0, filtered_experts.to(torch.int64), ones)
        token_counts = token_counts[:num_experts]
        expert_tokens = torch.cumsum(token_counts, dim=0, dtype=torch.int64)

        # Rearrange hidden_states
        sorted_hidden_states = hidden_states[sorted_token_indices]
    else:
        row_idx_len = num_tokens * top_k
        row_idx = torch.arange(0,
                               row_idx_len,
                               dtype=torch.int32,
                               device=topk_weights.device).view(
                                   top_k, -1).permute(1, 0).contiguous()
        sorted_hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens)
        del hidden_states

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts)
        expert_tokens = expert_tokens.to(torch.int64)

    quant_x, x_dynamic_scale = torch_npu.npu_dynamic_quant(
        sorted_hidden_states)
    del sorted_hidden_states
    output_dtype = torch.bfloat16 if w1_scale.dtype == torch.bfloat16 else torch.float16

    gate_up_out_list = torch_npu.npu_grouped_matmul(
        x=[quant_x],
        weight=[w1],
        scale=[w1_scale],
        per_token_scale=[x_dynamic_scale],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=output_dtype)
    del quant_x

    gate_up_out_list = gate_up_out_list[0] if len(
        gate_up_out_list) == 1 else torch.cat(gate_up_out_list, dim=0)
    gate_up_out_list = torch_npu.npu_swiglu(gate_up_out_list)

    quant_gate_up_out_list, gate_up_out_dynamic_scale = torch_npu.npu_dynamic_quant(
        gate_up_out_list)
    del gate_up_out_list

    down_out_list = torch_npu.npu_grouped_matmul(
        x=[quant_gate_up_out_list],
        weight=[w2],
        scale=[w2_scale],
        per_token_scale=[gate_up_out_dynamic_scale],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=output_dtype)
    del quant_gate_up_out_list

    down_out_list = down_out_list[0] if len(down_out_list) == 1 else torch.cat(
        down_out_list, dim=0)

    if expert_map is not None:
        weighted_down_out = down_out_list * sorted_weights.unsqueeze(1)

        final_hidden_states = torch.zeros(*original_shape,
                                          device=hidden_states.device,
                                          dtype=dtype)
        final_hidden_states.index_add_(0, sorted_token_indices,
                                       weighted_down_out)
        # TODO: This should not happen! Look into it!
        # fill nan with 0.0
        final_hidden_states[torch.isnan(final_hidden_states)] = 0.0
    else:
        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            down_out_list,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids)
    del down_out_list
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


class AscendW8A8DynamicLinearMethod:
    """Linear method for Ascend W8A8_DYNAMIC.
    """

    def __init__(self):
        self.transpose_weight = True

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
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
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=params_dtype)
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
                   hidden_sizes: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(num_experts,
                                               2 *
                                               intermediate_size_per_partition,
                                               hidden_sizes,
                                               dtype=torch.int8)
        param_dict["w2_weight"] = torch.empty(num_experts,
                                              hidden_sizes,
                                              intermediate_size_per_partition,
                                              dtype=torch.int8)
        return param_dict

    @staticmethod
    def get_dynamic_quant_param(num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=params_dtype)
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=params_dtype)
        param_dict["w2_weight_scale"] = torch.empty(num_experts,
                                                    hidden_sizes,
                                                    1,
                                                    dtype=params_dtype)
        param_dict["w2_weight_offset"] = torch.empty(num_experts,
                                                     hidden_sizes,
                                                     1,
                                                     dtype=params_dtype)
        return param_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert router_logits.shape[
            1] == global_num_experts, "Number of global experts mismatch"

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        return fused_experts(hidden_states=x,
                             w1=layer.w13_weight,
                             w1_scale=layer.w13_weight_scale,
                             w2=layer.w2_weight,
                             w2_scale=layer.w2_weight_scale,
                             topk_weights=topk_weights,
                             topk_ids=topk_ids,
                             top_k=top_k,
                             expert_map=expert_map)

    def process_weights_after_loading(self, layer):
        if self.transpose_weight:
            layer.w13_weight.data = layer.w13_weight.data.transpose(
                1, 2).contiguous()
            layer.w2_weight.data = layer.w2_weight.data.transpose(
                1, 2).contiguous()
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(
            layer.w13_weight_scale.data.shape[0], -1)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(
            layer.w13_weight_offset.data.shape[0], -1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(
            layer.w2_weight_scale.data.shape[0], -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(
            layer.w2_weight_offset.data.shape[0], -1)
