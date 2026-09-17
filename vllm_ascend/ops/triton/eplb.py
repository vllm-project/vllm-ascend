# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.triton_utils import tl, triton


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
