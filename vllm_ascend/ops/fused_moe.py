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
# Adapted from vllm/tests/kernels/test_moe.py

import os
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.distributed import (GroupCoordinator, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEParallelConfig, MoEConfig, UnquantizedFusedMoEMethod,
    determine_expert_map)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_ep_group, get_etp_group
from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer
from vllm_ascend.utils import (FusedMoEState, dispose_tensor,
                               get_fused_moe_state, npu_stream_switch,
                               npu_wait_tensor)

MOE_ALL2ALL_BUFFER: bool = envs_ascend.MOE_ALL2ALL_BUFFER


def process_topk_ids(topk_ids: torch.Tensor, expert_num: int, ep_size: int,
                     max_row_per_ep_rank: int, num_tokens: int,
                     top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    original_total_elements = num_tokens * top_k
    device = topk_ids.device
    original_dtype = topk_ids.dtype

    if original_total_elements == 0:
        output_len = ep_size * max_row_per_ep_rank
        topk_ids_pad = torch.full((output_len, ),
                                  expert_num,
                                  dtype=original_dtype,
                                  device=device)
        unpad_indices = torch.full((original_total_elements, ),
                                   -1,
                                   dtype=torch.long,
                                   device=device)
        return topk_ids_pad, unpad_indices

    experts_per_ep_rank_val = expert_num // ep_size
    if experts_per_ep_rank_val == 0:
        raise ValueError(
            "expert_num // ep_size is 0, which leads to division by zero in ep_rank calculation. "
            "Ensure expert_num >= ep_size.")

    assigned_ep_rank = (topk_ids.float() /
                        experts_per_ep_rank_val).to(original_dtype)
    indices_arange = torch.arange(topk_ids.shape[0], device=device)

    is_new_segment = torch.cat((torch.tensor([True], device=device),
                                assigned_ep_rank[1:] != assigned_ep_rank[:-1]))
    temp_start_markers = torch.full_like(indices_arange,
                                         -1,
                                         dtype=indices_arange.dtype)
    temp_start_markers[is_new_segment] = indices_arange[is_new_segment]
    start_offset_for_each_token = torch.cummax(temp_start_markers, dim=0)[0]
    token_intra_ep_rank_idx = indices_arange - start_offset_for_each_token
    is_kept_mask = token_intra_ep_rank_idx < max_row_per_ep_rank
    cumsum_kept = torch.cumsum(is_kept_mask.float(), dim=0).to(torch.long)
    indices_in_rec_cond_list_for_all = cumsum_kept - 1
    unpad_indices = torch.where(
        is_kept_mask, indices_in_rec_cond_list_for_all,
        torch.tensor(-1, device=device, dtype=torch.long))
    output_len = ep_size * max_row_per_ep_rank
    topk_ids_pad = torch.full((output_len, ),
                              expert_num,
                              dtype=original_dtype,
                              device=device)
    if topk_ids.shape[0] > 0:
        all_destination_indices = assigned_ep_rank * max_row_per_ep_rank + token_intra_ep_rank_idx
        temp_pad_buffer = torch.full((output_len + 1, ),
                                     expert_num,
                                     dtype=original_dtype,
                                     device=device)
        output_len_tensor = torch.tensor(output_len,
                                         dtype=torch.long,
                                         device=device)
        scatter_indices = torch.where(is_kept_mask, all_destination_indices,
                                      output_len_tensor)
        temp_pad_buffer.scatter_(0, scatter_indices, topk_ids)
        topk_ids_pad = temp_pad_buffer[:output_len]
    return topk_ids_pad, unpad_indices


def fused_experts_with_mc2(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    expert_map: torch.Tensor = None,
    moe_all_to_all_group_name: Optional[str] = None,
    shared_experts: Optional[Any] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    global_bs = 0
    moe_expert_num = len(expert_map)
    kwargs_mc2 = {
        "x": hidden_states,
        "expert_ids": topk_ids,
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": moe_expert_num,
        "global_bs": global_bs,
    }

    rank = torch.distributed.get_rank()

    quant_mode = 0
    ep_group = get_ep_group().device_group
    local_rank = torch.distributed.get_rank(group=ep_group)
    all_to_all_group_size = torch.distributed.get_world_size(ep_group)

    tp_size = get_etp_group().world_size
    tp_rank = rank % tp_size

    stage1_kwargs = {
        "scales": None,
        "quant_mode": quant_mode,
        "group_ep": moe_all_to_all_group_name,
        "ep_world_size": all_to_all_group_size,
        "ep_rank_id": local_rank,
        # "group_tp": self.moe_rs_group_name,
        "group_tp": moe_all_to_all_group_name,
        "tp_world_size": tp_size,
        "tp_rank_id": tp_rank,
    }
    kwargs_mc2.update(stage1_kwargs)

    output = torch_npu.npu_moe_distribute_dispatch(**kwargs_mc2)
    # comm_stream.wait_stream(torch.npu.current_stream())
    expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts = output[
        0:5]

    if shared_experts is not None:
        with npu_stream_switch("moe_secondary", 0):
            npu_wait_tensor(hidden_states, topk_weights)
            shared_gate_up, _ = shared_experts.gate_up_proj(hidden_states)
            npu_wait_tensor(shared_gate_up, expand_x)
            shared_act = shared_experts.act_fn(shared_gate_up)

    w1 = w1.transpose(1, 2)

    group_list = expert_token_nums.to(torch.int64)
    gate_up_out_list = torch_npu.npu_grouped_matmul(
        x=[expand_x],
        weight=[w1],
        split_item=2,
        # 1 means count mode, to avoid cumulative operation of the group list
        group_list_type=1,
        group_type=0,
        group_list=group_list,
    )

    # TODO: Remove this in the future.
    gate_up_out = torch.cat(gate_up_out_list, dim=0)
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    w2 = w2.transpose(1, 2)
    down_out_list = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
    )

    down_out_list = torch.cat(down_out_list, dim=0)

    # moeCombine
    kwargs_mc2 = {
        "expand_x": down_out_list,
        "expert_ids": topk_ids,
        "expand_idx": expand_idx,
        "expert_scales": topk_weights.to(torch.float32),
        "expert_shard_type": 0,
        "shared_expert_rank_num": 0,
        "moe_expert_num": moe_expert_num,
        "global_bs": 0,
    }
    tp_recv_counts = output[5]
    stage3_kwargs = {
        "ep_send_counts": ep_recv_counts,
        "group_ep": moe_all_to_all_group_name,
        "ep_world_size": all_to_all_group_size,
        "ep_rank_id": local_rank,
        "tp_send_counts": tp_recv_counts,
        # "group_tp": self.moe_rs_group_name,
        "group_tp": moe_all_to_all_group_name,
        "tp_world_size": tp_size,
        "tp_rank_id": tp_rank,
    }
    kwargs_mc2.update(stage3_kwargs)

    hidden_states = torch_npu.npu_moe_distribute_combine(**kwargs_mc2)

    if shared_experts is None:
        return hidden_states
    else:
        with npu_stream_switch("moe_secondary", 0):
            npu_wait_tensor(shared_act, down_out_list)
            shared_hidden_states, _ = shared_experts.down_proj(shared_act)
        return hidden_states, shared_hidden_states


def apply_mlp(hidden_states_wrapper: List[torch.Tensor],
              w1: torch.Tensor,
              w2: torch.Tensor,
              group_list: torch.Tensor,
              group_list_type: int = 1) -> torch.Tensor:
    """
    apply MLP: gate_up_proj -> swiglu -> down_proj

    Args:
        hidden_states_wrapper: wrapper of input hidden states with shape (num_tokens, hidden_size).
        w1: expert weights1 with shape
            (num_experts, hidden_size, intermediate_size * 2)
        w2: expert weights2 with shape
            (num_experts, intermediate_size, hidden_size)
        group_list: number of tokens for each expert, follow cumsum mode, and
            with shape (num_experts).
        transpose_weight:
            w1: (num_experts, intermediate_size * 2, hidden_size) ->
                    (num_experts, hidden_size, intermediate_size * 2)
            w2: (num_experts, hidden_size, intermediate_size) ->
                    (num_experts, intermediate_size, hidden_size)

    Returns:
        hidden_states: output hidden states after MLP.
    """

    assert len(hidden_states_wrapper) == 1
    hidden_states = hidden_states_wrapper.pop()

    w1 = w1.transpose(1, 2)
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )

    hidden_states = torch.cat(hidden_states, dim=0)
    hidden_states = torch_npu.npu_swiglu(hidden_states)

    w2 = w2.transpose(1, 2)
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )

    hidden_states = torch.cat(hidden_states, dim=0)
    return hidden_states


def fused_experts_with_all2all(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    expert_map: torch.Tensor = None,
    ep_group: GroupCoordinator = None,
):
    original_shape = hidden_states.shape
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    num_tokens, _ = hidden_states.shape
    num_experts = w1.shape[0]
    device = hidden_states.device

    if expert_map is not None:
        global_num_experts = len(expert_map)
        local_num_experts = global_num_experts // ep_group.world_size
        row_idx_len = num_tokens * top_k
        row_idx = (torch.arange(0,
                                row_idx_len,
                                dtype=torch.int32,
                                device=device).view(top_k, -1).permute(
                                    1, 0).contiguous())
        hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens)

        global_expert_tokens = torch.bincount(expanded_expert_idx,
                                              minlength=global_num_experts)
        scatter_sizes = global_expert_tokens.view(ep_group.world_size,
                                                  -1).sum(-1)

        gather_sizes = torch.empty_like(scatter_sizes)
        dist.all_to_all_single(gather_sizes,
                               scatter_sizes,
                               group=ep_group.device_group)
        scatter_size_list = scatter_sizes.cpu().tolist()
        gather_size_list = gather_sizes.cpu().tolist()

        expanded_expert_idx = expanded_expert_idx % local_num_experts
        hidden_states = ep_group.all_to_all(hidden_states, 0, 0,
                                            scatter_size_list,
                                            gather_size_list)
        local_expert_idx = ep_group.all_to_all(expanded_expert_idx, 0, 0,
                                               scatter_size_list,
                                               gather_size_list)

        sorted_local_expert_idx, sorted_idx = torch.sort(local_expert_idx)

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            sorted_local_expert_idx, local_num_experts).to(torch.int64)

        hidden_states = hidden_states[sorted_idx]
    else:
        row_idx_len = num_tokens * top_k
        row_idx = torch.arange(0,
                               row_idx_len,
                               dtype=torch.int32,
                               device=topk_weights.device).view(
                                   top_k, -1).permute(1, 0).contiguous()
        hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens)

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts)
        expert_tokens = expert_tokens.to(torch.int64)

    w1 = w1.transpose(1, 2)
    gate_up_out_list = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    # TODO: Remove this in the future.
    hidden_states = torch.cat(gate_up_out_list, dim=0)
    hidden_states = torch_npu.npu_swiglu(hidden_states)

    w2 = w2.transpose(1, 2)
    down_out_list = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    hidden_states = torch.cat(down_out_list, dim=0)

    if expert_map is not None:
        resorted_idx = torch.argsort(sorted_idx)
        hidden_states = hidden_states[resorted_idx]
        hidden_states = ep_group.all_to_all(hidden_states, 0, 0,
                                            gather_size_list,
                                            scatter_size_list)

        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )
    else:
        # TODO: Reorder device memory 2 times here, replace the current
        # implementation here when suitable operators become available.
        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )
    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


# currently expert parallelism implemented with all2all
# is under-optimized.
def fused_experts_with_all2all_buffer(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    max_model_len: int,
    global_batch_size: int,
    expert_map: torch.Tensor = None,
    ep_group: GroupCoordinator = None,
):
    original_shape = hidden_states.shape
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    num_tokens, _ = hidden_states.shape
    device = hidden_states.device

    global_num_experts = len(expert_map)
    local_num_experts = global_num_experts // ep_group.world_size
    row_idx_len = num_tokens * top_k
    row_idx = (torch.arange(0, row_idx_len, dtype=torch.int32,
                            device=device).view(top_k,
                                                -1).permute(1, 0).contiguous())
    hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
        hidden_states,
        row_idx=row_idx,
        expert_idx=topk_ids,
        active_num=num_tokens)

    max_row_per_ep_rank = (-(-global_batch_size // ep_group.world_size) *
                           max_model_len // ep_group.world_size +
                           1) * top_k * 2
    expert_idx_buffer_scatter, unpad_indices = process_topk_ids(
        expanded_expert_idx, global_num_experts, ep_group.world_size,
        max_row_per_ep_rank, num_tokens, top_k)
    hidden_states_pad_idx = torch.zeros(
        expert_idx_buffer_scatter.shape,
        dtype=expert_idx_buffer_scatter.dtype,
        device=expert_idx_buffer_scatter.device)
    non_pad_len = torch.sum(
        (expert_idx_buffer_scatter != global_num_experts).to(torch.int32))
    hidden_states_pad_idx[
        expert_idx_buffer_scatter != global_num_experts] = torch.arange(
            non_pad_len,
            dtype=expert_idx_buffer_scatter.dtype,
            device=hidden_states.device)

    hidden_states_buffer_scatter = hidden_states[hidden_states_pad_idx]
    expert_idx_buffer_gather = torch.empty_like(
        expert_idx_buffer_scatter,
        dtype=expert_idx_buffer_scatter.dtype,
        device=expert_idx_buffer_scatter.device)
    hidden_states_buffer_gather = torch.empty_like(
        hidden_states_buffer_scatter,
        dtype=hidden_states_buffer_scatter.dtype,
        device=hidden_states_buffer_scatter.device)
    dist.all_to_all_single(expert_idx_buffer_gather,
                           expert_idx_buffer_scatter,
                           group=ep_group.device_group)
    dist.all_to_all_single(hidden_states_buffer_gather,
                           hidden_states_buffer_scatter,
                           group=ep_group.device_group)
    mask = expert_idx_buffer_gather != global_num_experts
    local_expert_idx = expert_idx_buffer_gather[mask] - ep_group.rank * (
        global_num_experts // ep_group.world_size)
    hidden_states = hidden_states_buffer_gather[mask]
    idx_type = local_expert_idx.dtype
    sorted_local_expert_idx, sorted_idx = torch.sort(local_expert_idx.float())
    sorted_local_expert_idx = sorted_local_expert_idx.to(idx_type)

    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        sorted_local_expert_idx, local_num_experts).to(torch.int64)
    hidden_states = hidden_states[sorted_idx]
    group_list_type = 0

    hidden_states_wrapper = [hidden_states]
    del hidden_states

    hidden_states = apply_mlp(hidden_states_wrapper,
                              w1,
                              w2,
                              expert_tokens,
                              group_list_type=group_list_type)

    resorted_idx = torch.argsort(sorted_idx.float()).to(sorted_idx.dtype)
    hidden_states = hidden_states[resorted_idx]
    hidden_states_scatter = torch.zeros(
        (mask.shape[0], hidden_states.shape[1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device)
    hidden_states_scatter[mask] = hidden_states
    hidden_states_gatter = torch.empty_like(
        hidden_states_scatter,
        dtype=hidden_states_scatter.dtype,
        device=hidden_states_scatter.device)
    dist.all_to_all_single(hidden_states_gatter,
                           hidden_states_scatter,
                           group=ep_group.device_group)
    hidden_states_gatter = hidden_states_gatter[
        expert_idx_buffer_scatter != global_num_experts]
    if hidden_states_gatter.shape[0] != row_idx_len:
        hidden_states = torch.zeros((row_idx_len, hidden_states.shape[1]),
                                    dtype=hidden_states.dtype,
                                    device=hidden_states.device)
        hidden_states[unpad_indices != -1] = hidden_states_gatter
    else:
        # TODO: Reorder device memory 2 times here, replace the current
        hidden_states = hidden_states_gatter
    final_hidden_states = torch_npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )

    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    expert_map: torch.Tensor = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """
    Fused experts with top-k routing.

    Args:
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        w1: Expert weights1 of shape (num_experts, intermediate_size * 2, hidden_size).
        w2: Expert weights2 of shape (num_experts, hidden_size, intermediate_size).
        topk_weights: Routing weights of shape (num_tokens, top_k).
        topk_ids: Selected expert IDs of shape (num_tokens, top_k).
        top_k: Number of experts to select.
        expert_map: Expert mapping of shape (num_experts,).

    Returns:
        hidden_states: Hidden states after routing.
    """
    """
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    """
    # if torch.distributed.get_rank() == 0:
    #     print(w1.shape)
    #     print(hidden_states.shape)

    original_shape = hidden_states.shape
    # assert len(original_shape) == 2

    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w1.shape[0]
    dtype = hidden_states.dtype
    device = hidden_states.device
    # assert dtype in [torch.float32, torch.float16, torch.bfloat16
    #                  ], "Only float32, float16, and bfloat16 are supported"

    if apply_router_weight_on_input:
        assert (topk_weights.dim() == 2
                ), "`topk_weights` should be in shape (num_tokens, topk)"
        _, topk = topk_weights.shape
        assert (
            topk == 1
        ), "Only support topk=1 when `apply_router_weight_on_input` is True"
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

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
        sort_indices = torch.argsort(filtered_experts.view(torch.float32))
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
        row_idx = (torch.arange(0,
                                row_idx_len,
                                dtype=torch.int32,
                                device=device).view(top_k, -1).permute(
                                    1, 0).contiguous())
        sorted_hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens)

        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts)
        expert_tokens = expert_tokens.to(torch.int64)

    w1 = w1.transpose(1, 2)
    gate_up_out_list = torch_npu.npu_grouped_matmul(
        x=[sorted_hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    # TODO: Remove this in the future.
    gate_up_out = torch.cat(gate_up_out_list, dim=0)
    gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    w2 = w2.transpose(1, 2)
    down_out_list = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
    )

    down_out_list = torch.cat(down_out_list, dim=0)

    if expert_map is not None:
        weighted_down_out = down_out_list * sorted_weights.unsqueeze(1)

        final_hidden_states = torch.zeros(*original_shape,
                                          device=hidden_states.device,
                                          dtype=dtype)

        # TODO: npu_grouped_matmul output random values at [num_valid_tokens:, ...]
        # This created multiple NaN and index_add_ will mix them up which harms accuracy
        # remove this mask and filter after it being fixed
        num_valid_tokens = mask.sum()
        valid_token_mask = torch.arange(
            0, sorted_token_indices.shape[0],
            device=device).unsqueeze(1) < num_valid_tokens
        valid_output = torch.where(
            valid_token_mask, weighted_down_out,
            torch.zeros_like(weighted_down_out)).to(dtype)
        final_hidden_states.index_add_(0, sorted_token_indices, valid_output)
    else:
        scales = torch.ones_like(
            topk_weights) if apply_router_weight_on_input else topk_weights
        # TODO: Reorder device memory 2 times here, replace the current
        # implementation here when suitable operators become available.
        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            down_out_list,
            skip1=None,
            skip2=None,
            bias=None,
            scales=scales,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )

    return final_hidden_states


def native_grouped_topk(
    topk_weights: torch.Tensor,
    num_expert_group: Optional[int],
    topk_group: Optional[int],
):
    topk_group = 0 if topk_group is None else topk_group
    num_expert_group = 0 if num_expert_group is None else num_expert_group

    num_token = topk_weights.shape[0]
    grouped_weights = topk_weights.view(num_token, num_expert_group,
                                        -1).max(dim=-1).values
    topk_group_indices = torch.topk(grouped_weights.to(torch.float32),
                                    k=topk_group,
                                    dim=-1,
                                    sorted=False)[1]
    topk_group_mask = torch.zeros_like(grouped_weights)
    topk_group_mask.scatter_(1, topk_group_indices, 1)
    topk_weight_mask = (topk_group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        topk_weights.shape[-1] // num_expert_group).reshape(num_token, -1))
    topk_weights = topk_weights.masked_fill(~topk_weight_mask.bool(), 0.0)

    return topk_weights


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    global_num_experts: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k experts based on router logits.

    Args:
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        router_logits: Router logits of shape (num_tokens, num_experts).
        top_k: Number of experts to select.
        use_grouped_topk: Whether to group experts before selecting top-k.
        renormalize: Whether to renormalize the routing weights.
        topk_group: Number of expert groups to select from.
        num_expert_group: Number of experts in each group.
        custom_routing_function: Custom routing function.
        scoring_func: Scoring function to use.
        e_score_correction_bias: Correction bias to apply to expert scores.

    Returns:
        topk_weights: Routing weights of shape (num_tokens, top_k).
        topk_ids: Selected expert IDs of shape (num_tokens, top_k).

    Raises:
        ValueError: If an unsupported scoring function is provided.
    """

    if scoring_func == "softmax":
        # NOTE: vLLM use dtype=torch.float here
        topk_weights = router_logits.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        topk_weights = router_logits.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None

        if e_score_correction_bias is not None:
            # Store original scores before applying correction bias. We use biased
            # scores for expert selection but original scores for routing weights
            original_weights = topk_weights
            topk_weights = topk_weights + e_score_correction_bias.unsqueeze(0)

        # TODO: Change to npu_group_topk when the latest CANN and NNAL is available
        # >>> torch_npu._npu_group_topk(topk_weights, group_num=num_expert_group, k=topk_group)
        topk_weights = native_grouped_topk(topk_weights, num_expert_group,
                                           topk_group)
        # TODO bfloat16 is not supported in torch.topk with ge graph.
        if e_score_correction_bias is not None:
            topk_ids = torch.topk(topk_weights.to(torch.float32),
                                  k=top_k,
                                  dim=-1,
                                  sorted=False)[1]
            # Use original unbiased scores for the routing weights
            topk_weights = original_weights.gather(1, topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(topk_weights.to(torch.float32),
                                                k=top_k,
                                                dim=-1,
                                                sorted=False)
    elif custom_routing_function is None:
        topk_weights, topk_ids = topk_weights.topk(top_k, dim=-1)
        topk_weights = topk_weights.to(hidden_states.dtype)
    else:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            global_num_experts=global_num_experts)
        # Required by npu_moe_init_routing
        topk_ids = topk_ids.to(torch.int32)
        return topk_weights, topk_ids

    # Required by npu_moe_init_routing
    topk_ids = topk_ids.to(torch.int32)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self, moe: MoEConfig = None):

        super().__init__(moe=moe)
        vllm_config = get_current_vllm_config()

        self.ep_group = get_ep_group()
        self.ep_size = self.ep_group.world_size
        self.global_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.local_batch_size = self.global_batch_size // self.ep_size
        self.max_model_len = vllm_config.model_config.max_model_len

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled

        try:
            device_group = self.ep_group.device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = None

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod,
              self).process_weights_after_loading(layer)
        layer.w13_weight = torch.nn.Parameter(self._maybe_pad_weight(
            layer.w13_weight.data),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(self._maybe_pad_weight(
            layer.w2_weight.data),
                                             requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
        enable_force_load_balance: bool = False,
        shared_experts: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:

        # NOTE: now npu_moe_gating_top_k can only support `group_count=256` pattern
        if global_num_experts == 256:
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits,
                k=top_k,  # topk当前写8
                bias=e_score_correction_bias,
                k_group=topk_group,  # fix: 4
                group_count=num_expert_group,  # fix 8
                group_select_mode=1,  # 0: group中的最大; 1: topk2.sum(fix)
                renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
                norm_type=1,  # 0: softmax; 1: sigmoid(fix)
                # out_flag=False, # todo new api; 第三个输出是否输出
                # y2_flag=False, # old api; 第三个输出是否输出
                routed_scaling_factor=1,
                eps=float(1e-20))
        else:
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

        topk_weights = topk_weights.to(x.dtype)
        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            topk_ids = torch.randint_like(topk_ids, 0, global_num_experts)

        fused_moe_state = get_fused_moe_state(self.ep_group.world_size,
                                              is_prefill)
        if fused_moe_state == FusedMoEState.MC2:
            return fused_experts_with_mc2(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=top_k,
                expert_map=expert_map,
                moe_all_to_all_group_name=self.moe_all_to_all_group_name,
                shared_experts=shared_experts)
        elif fused_moe_state == FusedMoEState.AllGather:
            return fused_experts(hidden_states=x,
                                 w1=layer.w13_weight,
                                 w2=layer.w2_weight,
                                 topk_weights=topk_weights,
                                 topk_ids=topk_ids,
                                 top_k=top_k,
                                 expert_map=expert_map)
        elif MOE_ALL2ALL_BUFFER:
            return fused_experts_with_all2all_buffer(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                top_k=top_k,
                max_model_len=self.max_model_len,
                global_batch_size=self.global_batch_size,
                expert_map=expert_map,
                ep_group=get_ep_group())
        else:
            return fused_experts_with_all2all(hidden_states=x,
                                              w1=layer.w13_weight,
                                              w2=layer.w2_weight,
                                              topk_weights=topk_weights,
                                              topk_ids=topk_ids,
                                              top_k=top_k,
                                              expert_map=expert_map,
                                              ep_group=get_ep_group())


class AscendFusedMoE(FusedMoE):

    # The moe_counter parameter is required during the initialization of EPLB
    # to identify the current layer index within the MOE model.
    moe_counter = -1

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        # TODO: This could not initialize FusedMoE baseclass,
        # fixme and make __init__() of AscendFusedMoE more clear
        super(FusedMoE, self).__init__()

        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        vllm_config = get_current_vllm_config()

        self.moe_parallel_config: FusedMoEParallelConfig = (
            FusedMoEParallelConfig.make(
                tp_size_=(tp_size if tp_size is not None else
                          get_tensor_model_parallel_world_size()),
                dp_size_=(dp_size if dp_size is not None else
                          get_dp_group().world_size),
                vllm_parallel_config=vllm_config.parallel_config))

        self.top_k = top_k
        self.num_experts = num_experts
        self.global_num_experts = num_experts
        assert intermediate_size % self.tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.expert_map = None
        self.activation = activation
        self.log2phy = None
        self.global_redundant_expert_num = 0

        ascend_config = get_ascend_config()
        expert_map_path = ascend_config.expert_map_path
        if expert_map_path and os.path.exists(expert_map_path):
            # moe expert load balance
            expert_load_balancer = ExpertLoadBalancer(expert_map_path,
                                                      self.global_num_experts)
            self.local_num_experts, self.expert_map = \
                                expert_load_balancer.get_rank_placement_map(
                                                self.moe_instance_id,
                                                get_ep_group().rank_in_group)
            self.log2phy = expert_load_balancer.get_rank_log2phy_map(
                self.moe_instance_id,
                get_ep_group().rank_in_group)
            self.global_redundant_expert_num = \
                        expert_load_balancer.get_global_redundant_expert_num()
        else:
            # Create a tensor of size num_experts filled with -1
            self.local_num_experts, self.expert_map = determine_expert_map(
                self.ep_size,
                get_ep_group().rank_in_group, self.global_num_experts)

        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.enable_multistream_moe = \
            ascend_config.torchair_graph_config.enable_multistream_moe

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        moe = MoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            # TODO (bnell): this needs to be fixed for quantized types.
            in_dtype=params_dtype,
        )

        if quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod(moe)
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)

        assert self.quant_method is not None

        local_num_experts = torch.sum(self.expert_map != -1) \
            if self.expert_map is not None else num_experts

        moe_quant_params = {
            "num_experts": local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.ep_group = get_ep_group()
        # NOTE: self.tp_group is not expert_tp_group
        self.tp_group = get_tp_group().device_group
        self.quant_method.create_weights(layer=self, **moe_quant_params)

    def forward(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                is_prefill: bool,
                enable_force_load_balance: bool = False,
                top_k: Optional[int] = None,
                shared_experts: Optional[Any] = None):
        assert self.quant_method is not None

        if top_k:
            real_top_k = top_k
        else:
            real_top_k = self.top_k

        num_tokens, hidden_size = hidden_states.shape

        fused_moe_state = get_fused_moe_state(self.moe_parallel_config.ep_size,
                                              is_prefill)
        if shared_experts:
            if not self.enable_multistream_moe or fused_moe_state != FusedMoEState.MC2:
                shared_hidden_states = shared_experts(hidden_states)

        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1 and fused_moe_state != FusedMoEState.AllGather:
            if num_tokens < tp_size:
                hidden_states = nn.functional.pad(
                    hidden_states, (0, 0, 0, tp_size - num_tokens))
                router_logits = nn.functional.pad(
                    router_logits, (0, 0, 0, tp_size - num_tokens))
            chunk_hidden_states = torch.tensor_split(hidden_states,
                                                     tp_size,
                                                     dim=0)
            chunk_router_logits = torch.tensor_split(router_logits,
                                                     tp_size,
                                                     dim=0)
            tp_rank = get_tensor_model_parallel_rank()
            hidden_states = chunk_hidden_states[tp_rank]
            router_logits = chunk_router_logits[tp_rank]
        if self.dp_size > 1 and fused_moe_state == FusedMoEState.AllGather:
            # NOTE: When in torchair graph, it has been padded in model_runner_v1
            if not self.torchair_graph_enabled or is_prefill:
                attn_metadata = get_forward_context().attn_metadata
                if attn_metadata is not None:
                    max_num_tokens_across_dp = attn_metadata.max_num_tokens_across_dp
                    if num_tokens < max_num_tokens_across_dp:
                        hidden_states = nn.functional.pad(
                            hidden_states,
                            (0, 0, 0, max_num_tokens_across_dp - num_tokens))
                        router_logits = nn.functional.pad(
                            router_logits,
                            (0, 0, 0, max_num_tokens_across_dp - num_tokens))
            hidden_states = get_dp_group().all_gather(hidden_states, 0)
            router_logits = get_dp_group().all_gather(router_logits, 0)

        # Matrix multiply.
        e_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=real_top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            is_prefill=is_prefill,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            shared_experts=shared_experts if self.torchair_graph_enabled
            and self.enable_multistream_moe and not is_prefill else None,
        )

        if shared_experts:
            if isinstance(e_hidden_states, tuple):
                e_hidden_states, shared_hidden_states = e_hidden_states

        if tp_size > 1 and fused_moe_state != FusedMoEState.AllGather:
            dist.all_gather(list(chunk_hidden_states), e_hidden_states,
                            self.tp_group)
            final_hidden_states = torch.cat(chunk_hidden_states, dim=0)
            if num_tokens < tp_size:
                final_hidden_states = final_hidden_states[:num_tokens]
            dispose_tensor(e_hidden_states)
        elif self.dp_size > 1 and fused_moe_state == FusedMoEState.AllGather:
            final_hidden_states = dist._functional_collectives.reduce_scatter_tensor(
                e_hidden_states,
                "sum",
                scatter_dim=0,
                group=get_dp_group().device_group)
            final_hidden_states = final_hidden_states[:num_tokens]
            dispose_tensor(e_hidden_states)
        else:
            final_hidden_states = e_hidden_states

        if tp_size > 1 and fused_moe_state == FusedMoEState.AllGather:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        if shared_experts:
            return final_hidden_states, shared_hidden_states
        else:
            return final_hidden_states

    # ----------------------------------------- TBO-related --------------------------------------------

    def _forward_ms_fused_moe_comp(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_prefill: bool,
        real_top_k,
        enable_force_load_balance: bool = False,
    ):
        hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=real_top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            is_prefill=is_prefill,
            enable_force_load_balance=enable_force_load_balance)

        return hidden_states
