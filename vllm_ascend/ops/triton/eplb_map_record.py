# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Map logical expert IDs and record valid-token EPLB load.

Routing and TopK are owned by the caller. Each first-kernel program maps a
contiguous token range and writes one private physical-expert count row. The
second kernel adds those rows to the cumulative load without global atomics.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

# Bound the live [token, local-expert] comparison for one TopK slot. The
# program count and contiguous token ownership are independent of this tile.
MAX_TOKEN_TILE = 512
MAX_COMPARISON_ELEMENTS = 8192


@triton.jit
def _eplb_map_grid_record_kernel(
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
    num_grids,
    K: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    VALID_IS_TENSOR: tl.constexpr,
):
    grid_id = tl.program_id(0)
    base_tokens = tokens // num_grids
    extra_tokens = tokens % num_grids
    token_start = grid_id * base_tokens + tl.minimum(grid_id, extra_tokens)
    token_end = token_start + base_tokens + (grid_id < extra_tokens)
    physical = tl.arange(0, BLOCK_P)
    grid_record = tl.full((BLOCK_P,), 0, tl.int32)
    recording = tl.load(record_enabled_ptr) != 0
    if VALID_IS_TENSOR:
        valid_end = tl.minimum(tl.maximum(tl.load(valid_tokens_ptr), 0), tokens)
    else:
        valid_end = valid_tokens_ptr

    for token_base in range(token_start, token_end, TOKEN_TILE):
        token = token_base + tl.arange(0, TOKEN_TILE)
        token_mask = token < token_end
        valid_token = token_mask & (token < valid_end)
        for slot in tl.static_range(K):
            assignment = token * K + slot
            logical_id = tl.load(logical_ids_ptr + assignment, mask=token_mask, other=0).to(tl.int64)
            logical_valid = (logical_id >= 0) & (logical_id < experts)
            safe_logical_id = tl.where(logical_valid, logical_id, 0)
            table_index = (token % table_rows) * experts + safe_logical_id
            physical_id = tl.load(
                table_ptr + table_index,
                mask=token_mask & logical_valid,
                other=-1,
            )
            tl.store(physical_ids_ptr + assignment, physical_id, mask=token_mask)
            if recording:
                hits = (physical_id[:, None] - local_expert_start == physical[None, :]) & valid_token[:, None]
                grid_record += tl.sum(hits.to(tl.int32), axis=0)

    # Grid rows have disjoint addresses, so this is an ordinary store.
    tl.store(
        grid_records_ptr + grid_id * local_expert_count + physical,
        grid_record,
        mask=physical < local_expert_count,
    )


def _select_tiling(tokens: int, local_expert_count: int, vector_core_num: int) -> tuple[int, int, int]:
    """Select independent program parallelism and per-slot comparison tile."""
    block_p = triton.next_power_of_2(local_expert_count)
    if block_p > MAX_COMPARISON_ELEMENTS:
        raise ValueError("local physical expert count exceeds the comparison resource budget")
    num_grids = min(tokens, vector_core_num)
    max_owned_tokens = triton.cdiv(tokens, num_grids)
    token_tile = min(
        MAX_TOKEN_TILE,
        MAX_COMPARISON_ELEMENTS // block_p,
        triton.next_power_of_2(max_owned_tokens),
    )
    return num_grids, block_p, token_tile


@triton.jit
def _eplb_reduce_grid_records_kernel(
    grid_records_ptr,
    load_ptr,
    record_enabled_ptr,
    num_grids: tl.constexpr,
    local_expert_start,
    local_expert_count,
    BLOCK_GRID: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    physical = tl.arange(0, BLOCK_P)
    record = tl.full((BLOCK_P,), 0, tl.int32)
    # Bound the reduce live set without introducing another writer.
    for base in range(0, num_grids, BLOCK_GRID):
        grid = base + tl.arange(0, BLOCK_GRID)
        mask = (grid[:, None] < num_grids) & (physical[None, :] < local_expert_count)
        partial = tl.load(
            grid_records_ptr + grid[:, None] * local_expert_count + physical[None, :],
            mask=mask,
            other=0,
        )
        record += tl.sum(partial, 0)
    if tl.load(record_enabled_ptr) != 0:
        load_ptrs = load_ptr + local_expert_start + physical
        old_load = tl.load(load_ptrs, mask=physical < local_expert_count, other=0)
        tl.store(load_ptrs, old_load + record, mask=physical < local_expert_count)


def eplb_map_and_record(
    logical_ids: torch.Tensor,
    routing_table: torch.Tensor,
    expert_load: torch.Tensor,
    record_enabled: torch.Tensor,
    valid_tokens: torch.Tensor | int,
    *,
    local_expert_start: int = 0,
    local_expert_count: int | None = None,
) -> torch.Tensor:
    """Return physical IDs and add valid-token assignments to expert_load.

    The routing table, record flag and valid count remain device-side mutable,
    including during NPU graph replay. The local range names a slice of the
    global physical-expert domain, not a subset of logical IDs.
    """
    if logical_ids.ndim != 2:
        raise ValueError("logical_ids must have shape [tokens, top_k]")
    if logical_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("logical_ids must be int32 or int64")
    tokens, k = logical_ids.shape
    if k < 1:
        raise ValueError("logical_ids must have at least one top-k slot")
    logical_ids = logical_ids.contiguous()
    experts = routing_table.shape[1] if routing_table.ndim == 2 else 0
    if routing_table.ndim != 2 or not routing_table.is_contiguous():
        raise ValueError("routing_table must be contiguous [rows, experts]")
    if routing_table.dtype != torch.int32 or expert_load.dtype != torch.int32:
        raise ValueError("routing_table and expert_load must be int32")
    if expert_load.ndim != 1 or expert_load.stride(0) != 1:
        raise ValueError("expert_load must be a contiguous one-dimensional view")
    if record_enabled.numel() != 1:
        raise ValueError("record_enabled must contain one device value")
    if isinstance(valid_tokens, torch.Tensor) and (
        valid_tokens.ndim != 0 or valid_tokens.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("valid_tokens must be a scalar int32 or int64 device tensor")
    tensors = (
        routing_table,
        expert_load,
        record_enabled,
        valid_tokens if isinstance(valid_tokens, torch.Tensor) else None,
    )
    if any(tensor is not None and tensor.device != logical_ids.device for tensor in tensors):
        raise ValueError("all tensors must be on the same device")
    if routing_table.shape[0] == 0 or experts == 0:
        raise ValueError("routing_table must have nonempty rows and expert columns")
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
        return torch.empty_like(logical_ids)
    physical_ids = torch.empty_like(logical_ids)

    init_device_properties_triton()
    num_grids, block_p, token_tile = _select_tiling(tokens, local_expert_count, get_vectorcore_num())
    grid_records = torch.empty((num_grids, local_expert_count), dtype=torch.int32, device=logical_ids.device)
    _eplb_map_grid_record_kernel[(num_grids,)](
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
        num_grids,
        K=k,
        TOKEN_TILE=token_tile,
        BLOCK_P=block_p,
        VALID_IS_TENSOR=isinstance(valid_tokens, torch.Tensor),
    )
    _eplb_reduce_grid_records_kernel[(1,)](
        grid_records,
        expert_load,
        record_enabled,
        num_grids,
        local_expert_start,
        local_expert_count,
        BLOCK_GRID=min(triton.next_power_of_2(num_grids), max(1, MAX_COMPARISON_ELEMENTS // block_p)),
        BLOCK_P=block_p,
    )
    return physical_ids
