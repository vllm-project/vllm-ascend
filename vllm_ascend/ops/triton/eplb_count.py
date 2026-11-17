# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

ASSIGNMENTS_PER_TILE = 512
# 64 KiB of INT64 partial counts, leaving room for reduction temporaries.
MAX_REDUCE_ELEMENTS = 8192


@triton.jit(do_not_specialize=["num_assignments", "local_start", "local_count"])
def _count_allgather_kernel(
    ids_ptr,
    mask_ptr,
    partial_counts_ptr,
    counts_ptr,
    num_assignments,
    local_start,
    local_count,
    MASK_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    SINGLE_REDUCER: tl.constexpr,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    tiles = tl.cdiv(num_assignments, BLOCK_SIZE)
    quotient = tiles // programs
    remainder = tiles % programs
    first = pid * quotient + tl.minimum(pid, remainder)
    count = quotient + (pid < remainder)
    local_counts = tl.full((NUM_BINS,), 0, tl.int64)
    for tile in range(first, first + count):
        offsets = tile.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        rows = offsets // TOPK
        in_bounds = offsets < num_assignments
        physical_ids = tl.load(ids_ptr + offsets, mask=in_bounds, other=-1).to(tl.int64)
        valid_token = tl.load(mask_ptr + rows * MASK_STRIDE, mask=in_bounds, other=0).to(tl.int1)
        local_ids = physical_ids - local_start.to(tl.int64)
        eligible = in_bounds & valid_token & (local_ids >= 0) & (local_ids < local_count)
        safe_ids = tl.where(eligible, local_ids, 0).to(tl.int32)
        local_counts += tl.histogram(safe_ids, NUM_BINS, mask=eligible).to(tl.int64)
    bins = tl.arange(0, NUM_BINS)
    tl.store(partial_counts_ptr + pid * NUM_BINS + bins, local_counts)
    # Multiple row tiles merge atomically; stream order places initialization
    # before every merge. A single reducer overwrites the complete output.
    if not SINGLE_REDUCER:
        if pid == 0:
            tl.store(counts_ptr + bins, tl.full((NUM_BINS,), 0, tl.int64), mask=bins < local_count)


@triton.jit
def _reduce_allgather_counts_kernel(
    partial_counts_ptr,
    counts_ptr,
    NUM_PARTIAL_ROWS: tl.constexpr,
    NUM_BINS: tl.constexpr,
    LOCAL_COUNT: tl.constexpr,
    ROWS_PER_TILE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    SINGLE_REDUCER: tl.constexpr,
):
    tile_rows = tl.arange(0, BLOCK_ROWS)
    rows = tl.program_id(0) * ROWS_PER_TILE + tile_rows
    bins = tl.arange(0, NUM_BINS)
    partial_counts = tl.load(
        partial_counts_ptr + rows[:, None] * NUM_BINS + bins[None, :],
        mask=(tile_rows[:, None] < ROWS_PER_TILE) & (rows[:, None] < NUM_PARTIAL_ROWS),
        other=0,
    )
    counts = tl.sum(partial_counts, axis=0)
    if SINGLE_REDUCER:
        tl.store(counts_ptr + bins, counts, mask=bins < LOCAL_COUNT)
    else:
        tl.atomic_add(counts_ptr + bins, counts, mask=bins < LOCAL_COUNT, sem="relaxed")


def count_allgather_tokens_triton(
    topk_ids: torch.Tensor,
    token_mask: torch.Tensor,
    local_expert_start: int,
    local_expert_count: int,
) -> torch.Tensor:
    """Count valid physical routes into INT64 local bins without host reads.

    The output is fresh per call. Persistent load accumulation remains in the
    existing EPLB record operator. Repeated routes each contribute one.
    """
    if topk_ids.ndim != 2 or token_mask.ndim != 1 or token_mask.shape[0] != topk_ids.shape[0]:
        raise ValueError("Expected physical IDs [T,K] and aligned token mask [T]")
    if topk_ids.dtype not in (torch.int32, torch.int64) or token_mask.dtype != torch.bool:
        raise TypeError("Expected INT32/INT64 IDs and BOOL mask")
    if topk_ids.device.type != "npu" or token_mask.device != topk_ids.device:
        raise ValueError("Inputs must be on the same NPU")
    if not topk_ids.is_contiguous():
        raise ValueError("Expected contiguous physical IDs [T,K]")
    if local_expert_start < 0 or local_expert_count <= 0:
        raise ValueError("Expected nonnegative local start and positive local expert count")
    num_assignments = topk_ids.numel()
    if num_assignments == 0:
        return torch.zeros(local_expert_count, dtype=torch.int64, device=topk_ids.device)
    init_device_properties_triton()
    vector_cores = get_vectorcore_num()
    programs = min(vector_cores, triton.cdiv(num_assignments, ASSIGNMENTS_PER_TILE))
    # Keep a vector accumulator: the backend cannot lower a one-element loop-carried tensor.
    num_bins = max(2, triton.next_power_of_2(local_expert_count))
    partial_counts = torch.empty((programs, num_bins), dtype=torch.int64, device=topk_ids.device)
    counts = torch.empty(local_expert_count, dtype=torch.int64, device=topk_ids.device)
    reduce_rows = min(programs, max(1, MAX_REDUCE_ELEMENTS // num_bins))
    reduce_tiles = triton.cdiv(programs, reduce_rows)
    _count_allgather_kernel[(programs,)](
        topk_ids,
        token_mask,
        partial_counts,
        counts,
        num_assignments,
        local_expert_start,
        local_expert_count,
        MASK_STRIDE=token_mask.stride(0),
        TOPK=topk_ids.shape[1],
        BLOCK_SIZE=ASSIGNMENTS_PER_TILE,
        NUM_BINS=num_bins,
        SINGLE_REDUCER=reduce_tiles == 1,
        compile_mode="simd",
    )
    _reduce_allgather_counts_kernel[(reduce_tiles,)](
        partial_counts,
        counts,
        NUM_PARTIAL_ROWS=programs,
        NUM_BINS=num_bins,
        LOCAL_COUNT=local_expert_count,
        ROWS_PER_TILE=reduce_rows,
        BLOCK_ROWS=triton.next_power_of_2(reduce_rows),
        SINGLE_REDUCER=reduce_tiles == 1,
        compile_mode="simd",
    )
    return counts
