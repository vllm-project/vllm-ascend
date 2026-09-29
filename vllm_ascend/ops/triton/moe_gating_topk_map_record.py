# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CANN gating TopK followed by grid-owned EPLB mapping and recording.

The first Triton kernel maps CANN's logical IDs and writes one physical
expert-count row per grid. A single-writer second kernel adds their sum to
the cumulative load. Neither Triton kernel uses a global atomic.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

# Bound the live [assignment, local-expert] comparison tensor. The token
# ownership range is independent of this inner processing tile.
MAX_ASSIGNMENTS_PER_TILE = 512
MAX_COMPARISON_ELEMENTS = 8192


@triton.jit
def _map_grid_record_kernel(
    logical_ids_ptr,
    table_ptr,
    record_enabled_ptr,
    valid_tokens_ptr,
    physical_ids_ptr,
    grid_records_ptr,
    tokens,
    experts,
    table_rows,
    local_expert_start,
    local_expert_count,
    K: tl.constexpr,
    TOKENS_PER_GRID: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_P: tl.constexpr,
    VALID_IS_TENSOR: tl.constexpr,
):
    grid_id = tl.program_id(0)
    assignment_start = grid_id * TOKENS_PER_GRID * K
    assignment_end = tl.minimum((grid_id + 1) * TOKENS_PER_GRID, tokens) * K
    physical = tl.arange(0, BLOCK_P)
    grid_record = tl.full((BLOCK_P,), 0, tl.int32)
    recording = tl.load(record_enabled_ptr) != 0
    if VALID_IS_TENSOR:
        valid_end = tl.load(valid_tokens_ptr) * K
    else:
        valid_end = valid_tokens_ptr * K

    for base in range(assignment_start, assignment_end, BLOCK):
        assignment = base + tl.arange(0, BLOCK)
        assignment_mask = assignment < assignment_end
        logical_id = tl.load(logical_ids_ptr + assignment, mask=assignment_mask, other=0).to(tl.int32)
        logical_valid = (logical_id >= 0) & (logical_id < experts)
        safe_logical_id = tl.where(logical_valid, logical_id, 0)
        table_index = ((assignment // K) % table_rows) * experts + safe_logical_id
        physical_id = tl.load(
            table_ptr + table_index,
            mask=assignment_mask & logical_valid,
            other=-1,
        )
        tl.store(physical_ids_ptr + assignment, physical_id, mask=assignment_mask)
        if recording:
            valid_assignment = assignment_mask & (assignment < valid_end)
            hits = (physical_id[:, None] - local_expert_start == physical[None, :]) & valid_assignment[:, None]
            grid_record += tl.sum(hits.to(tl.int32), axis=0)

    # Grid rows have disjoint addresses, so this is an ordinary store.
    tl.store(
        grid_records_ptr + grid_id * local_expert_count + physical,
        grid_record,
        mask=physical < local_expert_count,
    )


@triton.jit
def _reduce_grid_records_kernel(
    grid_records_ptr,
    load_ptr,
    record_enabled_ptr,
    num_grids,
    local_expert_start,
    local_expert_count,
    BLOCK_GRID: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    grid = tl.arange(0, BLOCK_GRID)
    physical = tl.arange(0, BLOCK_P)
    mask = (grid[:, None] < num_grids) & (physical[None, :] < local_expert_count)
    partial = tl.load(
        grid_records_ptr + grid[:, None] * local_expert_count + physical[None, :],
        mask=mask,
        other=0,
    )
    record = tl.sum(partial, 0)
    if tl.load(record_enabled_ptr) != 0:
        load_ptrs = load_ptr + local_expert_start + physical
        old_load = tl.load(load_ptrs, mask=physical < local_expert_count, other=0)
        tl.store(load_ptrs, old_load + record, mask=physical < local_expert_count)


def moe_gating_topk_map_record(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    routing_table: torch.Tensor,
    expert_load: torch.Tensor,
    record_enabled: torch.Tensor,
    valid_tokens: torch.Tensor | int,
    *,
    k: int,
    scoring: str,
    routed_scaling_factor: float = 1.0,
    local_expert_start: int = 0,
    local_expert_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CANN routing weights and mapped IDs, accumulating valid load.

    Only the ungrouped, renormalized route is accepted by its router guard.
    Device-side routing-table, recording-flag and valid-count updates remain
    visible under NPU graph replay.
    """
    if logits.ndim != 2 or not logits.is_contiguous():
        raise ValueError("logits must be contiguous [tokens, experts]")
    tokens, experts = logits.shape
    if not 1 <= k <= experts:
        raise ValueError("k must be between 1 and the expert count")
    if scoring not in ("softmax", "sigmoid"):
        raise ValueError("scoring must be softmax or sigmoid")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("logits must be float16, bfloat16 or float32")
    if bias is not None and (bias.shape != (experts,) or not bias.is_contiguous() or bias.dtype != logits.dtype):
        raise ValueError("bias must be contiguous [experts] with logits dtype")
    if routing_table.ndim != 2 or routing_table.shape[1] != experts or not routing_table.is_contiguous():
        raise ValueError("routing_table must be contiguous [rows, experts]")
    if routing_table.dtype != torch.int32 or expert_load.dtype != torch.int32:
        raise ValueError("routing_table and expert_load must be int32")
    if expert_load.ndim != 1 or expert_load.stride(0) != 1:
        raise ValueError("expert_load must be a contiguous one-dimensional view")
    if record_enabled.numel() != 1:
        raise ValueError("record_enabled must contain one device value")
    if isinstance(valid_tokens, torch.Tensor) and valid_tokens.numel() != 1:
        raise ValueError("valid_tokens must contain one device value")
    tensors = (
        routing_table,
        expert_load,
        record_enabled,
        bias,
        valid_tokens if isinstance(valid_tokens, torch.Tensor) else None,
    )
    if any(tensor is not None and tensor.device != logits.device for tensor in tensors):
        raise ValueError("all tensors must be on the same device")
    if routing_table.shape[0] == 0:
        raise ValueError("routing_table must have at least one row")
    if isinstance(valid_tokens, int) and not 0 <= valid_tokens <= tokens:
        raise ValueError("valid_tokens must be between zero and tokens")
    if local_expert_count is None:
        local_expert_count = expert_load.numel() - local_expert_start
    if (
        local_expert_start < 0
        or local_expert_count <= 0
        or local_expert_start + local_expert_count > expert_load.numel()
    ):
        raise ValueError("local expert range exceeds expert_load")
    if tokens == 0:
        return (
            torch.empty((0, k), dtype=logits.dtype, device=logits.device),
            torch.empty((0, k), dtype=torch.int32, device=logits.device),
        )
    weights, logical_ids, _ = torch.ops._C_ascend.moe_gating_top_k(
        logits,
        k=k,
        k_group=1,
        group_count=1,
        group_select_mode=1,
        renorm=1,
        norm_type=0 if scoring == "softmax" else 1,
        out_flag=False,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-20,
        bias_opt=bias,
    )
    logical_ids = logical_ids.to(torch.int32)
    physical_ids = torch.empty_like(logical_ids)

    block_p = triton.next_power_of_2(local_expert_count)
    block = min(MAX_ASSIGNMENTS_PER_TILE, max(triton.next_power_of_2(k), MAX_COMPARISON_ELEMENTS // block_p))
    init_device_properties_triton()
    num_grids = min(triton.cdiv(tokens * k, block), get_vectorcore_num())
    tokens_per_grid = triton.cdiv(tokens, num_grids)
    grid_records = torch.empty((num_grids, local_expert_count), dtype=torch.int32, device=logits.device)
    _map_grid_record_kernel[(num_grids,)](
        logical_ids,
        routing_table,
        record_enabled,
        valid_tokens,
        physical_ids,
        grid_records,
        tokens,
        experts,
        routing_table.shape[0],
        local_expert_start,
        local_expert_count,
        K=k,
        TOKENS_PER_GRID=tokens_per_grid,
        BLOCK=block,
        BLOCK_P=block_p,
        VALID_IS_TENSOR=isinstance(valid_tokens, torch.Tensor),
    )
    _reduce_grid_records_kernel[(1,)](
        grid_records,
        expert_load,
        record_enabled,
        num_grids,
        local_expert_start,
        local_expert_count,
        BLOCK_GRID=triton.next_power_of_2(num_grids),
        BLOCK_P=block_p,
    )
    return weights, physical_ids
