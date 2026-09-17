# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.triton_utils import tl, triton


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
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_local_experts
    current = tl.load(expert_tokens_ptr + offsets, mask=mask, other=0)
    if group_list_type == 1:
        local_load = current
    else:
        previous = tl.load(
            expert_tokens_ptr + offsets - 1,
            mask=mask & (offsets > 0),
            other=0,
        )
        local_load = current - previous

    record_enabled = tl.load(record_enabled_ptr) != 0
    expert_load_offsets = local_expert_start + offsets
    previous_load = tl.load(expert_load_ptr + expert_load_offsets, mask=mask)
    tl.store(
        expert_load_ptr + expert_load_offsets,
        previous_load + local_load,
        mask=mask & record_enabled,
    )


@triton.jit
def _map_to_physical_kernel(
    topk_ids_ptr,
    routing_table_ptr,
    physical_ids_ptr,
    num_logical_experts,
    numel,
    topk,
    routing_table_rows,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    logical_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=-1).to(tl.int64)
    valid_logical_id = (logical_id >= 0) & (logical_id < num_logical_experts)
    safe_logical_id = tl.where(valid_logical_id, logical_id, 0)

    token_idx = offsets // topk
    routing_row = token_idx % routing_table_rows
    routing_index = routing_row * num_logical_experts + safe_logical_id
    physical_id = tl.load(
        routing_table_ptr + routing_index,
        mask=mask & valid_logical_id,
        other=-1,
    )
    tl.store(physical_ids_ptr + offsets, physical_id, mask=mask)


@triton.jit
def _record_physical_expert_load_kernel(
    physical_ids_ptr,
    expert_load_ptr,
    record_enabled_ptr,
    num_unpadded_tokens_ptr,
    num_physical_experts,
    numel,
    topk,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    physical_id = tl.load(physical_ids_ptr + offsets, mask=mask, other=-1)
    token_idx = offsets // topk
    record_enabled = tl.load(record_enabled_ptr) != 0
    num_unpadded_tokens = tl.load(num_unpadded_tokens_ptr)
    valid_physical_id = (physical_id >= 0) & (physical_id < num_physical_experts)
    should_record = mask & valid_physical_id & record_enabled & (token_idx < num_unpadded_tokens)
    safe_physical_id = tl.where(valid_physical_id, physical_id, 0)
    tl.atomic_add(expert_load_ptr + safe_physical_id, 1, mask=should_record)


@triton.jit
def _map_to_physical_and_record_kernel(
    topk_ids_ptr,
    routing_table_ptr,
    physical_ids_ptr,
    expert_load_ptr,
    record_enabled_ptr,
    num_unpadded_tokens_ptr,
    num_logical_experts,
    num_physical_experts,
    numel,
    topk,
    routing_table_rows,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    logical_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=-1).to(tl.int64)
    valid_logical_id = (logical_id >= 0) & (logical_id < num_logical_experts)
    safe_logical_id = tl.where(valid_logical_id, logical_id, 0)

    token_idx = offsets // topk
    routing_row = token_idx % routing_table_rows
    routing_index = routing_row * num_logical_experts + safe_logical_id
    physical_id = tl.load(
        routing_table_ptr + routing_index,
        mask=mask & valid_logical_id,
        other=-1,
    )
    tl.store(physical_ids_ptr + offsets, physical_id, mask=mask)

    record_enabled = tl.load(record_enabled_ptr) != 0
    num_unpadded_tokens = tl.load(num_unpadded_tokens_ptr)
    valid_physical_id = (physical_id >= 0) & (physical_id < num_physical_experts)
    should_record = mask & valid_logical_id & valid_physical_id & record_enabled & (token_idx < num_unpadded_tokens)
    safe_physical_id = tl.where(valid_physical_id, physical_id, 0)
    tl.atomic_add(expert_load_ptr + safe_physical_id, 1, mask=should_record)


def record_expert_tokens_triton(
    expert_tokens: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    group_list_type: int,
    local_expert_start: int,
) -> None:
    """Accumulate operator-provided expert counts with one small kernel."""
    num_local_experts = expert_tokens.numel()
    if num_local_experts == 0:
        return

    block_size = triton.next_power_of_2(num_local_experts)
    _record_expert_tokens_kernel[(1,)](
        expert_tokens,
        expert_load_view,
        record_enabled,
        num_local_experts,
        local_expert_start,
        group_list_type=group_list_type,
        BLOCK_SIZE=block_size,
    )


def map_to_physical_triton(
    topk_ids: torch.Tensor,
    expert_replica_routing_table: torch.Tensor,
) -> torch.Tensor:
    """Map logical IDs to physical IDs without collecting expert load."""
    if topk_ids.numel() == 0:
        return topk_ids

    physical_ids = torch.empty_like(topk_ids)
    numel = topk_ids.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _map_to_physical_kernel[grid](
        topk_ids,
        expert_replica_routing_table,
        physical_ids,
        expert_replica_routing_table.shape[1],
        numel,
        topk_ids.shape[1],
        expert_replica_routing_table.shape[0],
        BLOCK_SIZE=256,
    )
    return physical_ids


def record_physical_expert_load_triton(
    physical_ids: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    num_unpadded_tokens: torch.Tensor,
) -> None:
    """Fallback load collection when a MoE path has no expert-token counts."""
    if physical_ids.numel() == 0:
        return

    numel = physical_ids.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _record_physical_expert_load_kernel[grid](
        physical_ids,
        expert_load_view,
        record_enabled,
        num_unpadded_tokens,
        expert_load_view.numel(),
        numel,
        physical_ids.shape[1],
        BLOCK_SIZE=256,
    )


def map_to_physical_and_record_triton(
    topk_ids: torch.Tensor,
    expert_replica_routing_table: torch.Tensor,
    expert_load_view: torch.Tensor,
    record_enabled: torch.Tensor,
    num_unpadded_tokens: torch.Tensor,
) -> torch.Tensor:
    """Map logical IDs and optionally record physical-expert load."""
    if topk_ids.numel() == 0:
        return topk_ids

    physical_ids = torch.empty_like(topk_ids)
    numel = topk_ids.numel()
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _map_to_physical_and_record_kernel[grid](
        topk_ids,
        expert_replica_routing_table,
        physical_ids,
        expert_load_view,
        record_enabled,
        num_unpadded_tokens,
        expert_replica_routing_table.shape[1],
        expert_load_view.numel(),
        numel,
        topk_ids.shape[1],
        expert_replica_routing_table.shape[0],
        BLOCK_SIZE=256,
    )
    return physical_ids
