# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

MAPPING_ASSIGNMENTS_PER_TILE = 512


@triton.jit(do_not_specialize=["tokens", "num_grids"])
def _map_to_physical_kernel(
    topk_ids_ptr,
    routing_table_ptr,
    physical_ids_ptr,
    num_logical_experts,
    tokens,
    num_grids,
    routing_table_rows,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_tokens = tokens // num_grids
    extra_tokens = tokens % num_grids
    token_start = pid * base_tokens + tl.minimum(pid, extra_tokens)
    token_end = token_start + base_tokens + (pid < extra_tokens)

    assignment_end = token_end * TOPK
    for base in range(token_start * TOPK, assignment_end, BLOCK_SIZE):
        offsets = base + tl.arange(0, BLOCK_SIZE)
        mask = offsets < assignment_end
        logical_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=-1)
        valid_logical_id = (logical_id >= 0) & (logical_id < num_logical_experts)
        safe_logical_id = tl.where(valid_logical_id, logical_id, 0)
        routing_row = (offsets // TOPK) % routing_table_rows
        routing_index = routing_row * num_logical_experts + safe_logical_id
        physical_id = tl.load(
            routing_table_ptr + routing_index,
            mask=mask & valid_logical_id,
            other=-1,
        )
        tl.store(physical_ids_ptr + offsets, physical_id, mask=mask)


@triton.jit
def _record_expert_tokens_kernel(
    expert_tokens_ptr,
    expert_load_ptr,
    record_enabled_ptr,
    num_local_experts,
    local_expert_start,
    group_list_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if tl.load(record_enabled_ptr) != 0:
        mask = offsets < num_local_experts
        current = tl.load(expert_tokens_ptr + offsets, mask=mask, other=0)
        # Narrow before arithmetic when the destination is narrower: integer
        # add/subtract has the same result modulo the destination width.
        if current.dtype.primitive_bitwidth > expert_load_ptr.dtype.element_ty.primitive_bitwidth:
            current = current.to(expert_load_ptr.dtype.element_ty)
        if group_list_type == 1:
            local_load = current
        else:
            previous_offset = tl.maximum(offsets - 1, 0)
            previous = tl.load(expert_tokens_ptr + previous_offset, mask=mask & (offsets > 0), other=0)
            if previous.dtype.primitive_bitwidth > expert_load_ptr.dtype.element_ty.primitive_bitwidth:
                previous = previous.to(expert_load_ptr.dtype.element_ty)
            local_load = current - previous
        load_offsets = local_expert_start + offsets
        previous_load = tl.load(expert_load_ptr + load_offsets, mask=mask, other=0)
        tl.store(expert_load_ptr + load_offsets, previous_load + local_load, mask=mask)


def map_to_physical_triton(
    topk_ids: torch.Tensor,
    expert_replica_routing_table: torch.Tensor,
) -> torch.Tensor:
    """Map logical IDs to physical IDs without collecting expert load."""
    if topk_ids.numel() == 0:
        return topk_ids

    physical_ids = torch.empty_like(topk_ids)
    init_device_properties_triton()
    tokens, topk = topk_ids.shape
    num_grids = min(tokens, get_vectorcore_num())
    _map_to_physical_kernel[(num_grids,)](
        topk_ids,
        expert_replica_routing_table,
        physical_ids,
        expert_replica_routing_table.shape[1],
        tokens,
        num_grids,
        expert_replica_routing_table.shape[0],
        TOPK=topk,
        # Keep the constexpr tile fixed across token counts; mask each range's tail.
        BLOCK_SIZE=MAPPING_ASSIGNMENTS_PER_TILE,
    )
    return physical_ids


def record_expert_tokens_triton(
    expert_tokens: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    group_list_type: int,
    local_expert_start: int,
) -> None:
    """Accumulate operator-provided local counts when collection is enabled."""
    num_local_experts = expert_tokens.numel()
    if num_local_experts == 0:
        return
    block_size = min(256, triton.next_power_of_2(num_local_experts))
    _record_expert_tokens_kernel[(triton.cdiv(num_local_experts, block_size),)](
        expert_tokens,
        expert_load_view,
        record_enabled,
        num_local_experts,
        local_expert_start,
        group_list_type=group_list_type,
        BLOCK_SIZE=block_size,
    )
