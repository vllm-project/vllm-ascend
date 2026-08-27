#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch


def moe_lora_prefill_route_allgather(
    x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    routed_topk_ids: torch.Tensor,
    token_lora_indices: torch.Tensor,
    adapter_enabled: torch.Tensor,
    workspaces: tuple[torch.Tensor, ...],
    top_k: int,
    num_experts: int,
    first_expert_idx: int,
):
    return torch.ops._C_ascend.moe_lora_prefill_route_allgather(
        x,
        expanded_row_idx,
        routed_topk_ids,
        token_lora_indices,
        adapter_enabled,
        *workspaces,
        top_k,
        num_experts,
        first_expert_idx,
    )


def moe_lora_prefill_route_alltoall(
    x: torch.Tensor,
    expert_count: torch.Tensor,
    exchanged_lora_indices: torch.Tensor,
    adapter_enabled: torch.Tensor,
    workspaces: tuple[torch.Tensor, ...],
):
    return torch.ops._C_ascend.moe_lora_prefill_route_alltoall(
        x,
        expert_count,
        exchanged_lora_indices,
        adapter_enabled,
        *workspaces,
    )


def moe_lora_prefill_gather_by_perm(
    source: torch.Tensor,
    perm_record: torch.Tensor,
    grouped_x: torch.Tensor,
):
    return torch.ops._C_ascend.moe_lora_prefill_gather_by_perm(
        source, perm_record, grouped_x
    )


def moe_lora_prefill_scatter_add(
    delta: torch.Tensor,
    perm_record: torch.Tensor,
    y: torch.Tensor,
    output_offset: int,
):
    return torch.ops._C_ascend.moe_lora_prefill_scatter_add(
        delta, perm_record, y, output_offset
    )


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
    return torch.ops._C_ascend.bgmv_shrink(
        inputs,
        lora_a_weights,
        lora_indices_tensor,
        output_tensor,
        scaling,
    )


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    slice_offset = 0
    slice_size = output_tensor.size(1)
    return torch.ops._C_ascend.bgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        output_tensor,
        slice_offset,
        slice_size,
    )


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    return torch.ops._C_ascend.bgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, output_tensor, slice_offset, slice_size
    )


def bgmv_moe_w13(
    inputs: torch.Tensor,
    lora_a0_weights: torch.Tensor,
    lora_a1_weights: torch.Tensor,
    lora_b0_weights: torch.Tensor,
    lora_b1_weights: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    workspace: torch.Tensor,
    output_tensor: torch.Tensor,
    slice_offset: int = 0,
    scaling: float = 1.0,
):
    """Apply two contiguous rank-16 MoE LoRA slices with two launches."""
    return torch.ops._C_ascend.bgmv_moe_w13(
        inputs,
        lora_a0_weights,
        lora_a1_weights,
        lora_b0_weights,
        lora_b1_weights,
        lora_indices_tensor,
        workspace,
        output_tensor,
        slice_offset,
        scaling,
    )


def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    return torch.ops._C_ascend.sgmv_shrink(
        inputs, lora_a_weights, lora_indices_tensor, seq_len_tensor, output_tensor, scaling
    )


def sgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
):
    slice_offset = 0
    slice_size = output_tensor.size(1)
    return torch.ops._C_ascend.sgmv_expand(
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        seq_len_tensor,
        output_tensor,
        slice_offset,
        slice_size,
    )


def sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False,
):
    return torch.ops._C_ascend.sgmv_expand(
        inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, output_tensor, slice_offset, slice_size
    )
