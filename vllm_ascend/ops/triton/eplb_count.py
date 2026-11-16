# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

ASSIGNMENTS_PER_TILE = 512


@triton.jit(do_not_specialize=["num_assignments", "local_start", "local_count"])
def _count_allgather_kernel(
    ids_ptr,
    mask_ptr,
    counts_ptr,
    num_assignments,
    local_start,
    local_count,
    ID_ROW_STRIDE: tl.constexpr,
    ID_COL_STRIDE: tl.constexpr,
    MASK_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
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
        cols = offsets % TOPK
        in_bounds = offsets < num_assignments
        physical_ids = tl.load(ids_ptr + rows * ID_ROW_STRIDE + cols * ID_COL_STRIDE, mask=in_bounds, other=-1).to(
            tl.int64
        )
        valid_token = tl.load(mask_ptr + rows * MASK_STRIDE, mask=in_bounds, other=0).to(tl.int1)
        local_ids = physical_ids - local_start.to(tl.int64)
        eligible = in_bounds & valid_token & (local_ids >= 0) & (local_ids < local_count)
        safe_ids = tl.where(eligible, local_ids, 0).to(tl.int32)
        if NUM_BINS == 1:
            # A single bin is a reduction; avoid the backend's 1-bin histogram conversion.
            local_counts += tl.sum(eligible.to(tl.int32), 0).to(tl.int64)
        else:
            local_counts += tl.histogram(safe_ids, NUM_BINS, mask=eligible).to(tl.int64)
    bins = tl.arange(0, NUM_BINS)
    tl.atomic_add(counts_ptr + bins, local_counts, mask=(bins < local_count) & (local_counts != 0), sem="relaxed")


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
    if local_expert_start < 0 or local_expert_count <= 0:
        raise ValueError("Expected nonnegative local start and positive local expert count")
    counts = torch.zeros(local_expert_count, dtype=torch.int64, device=topk_ids.device)
    num_assignments = topk_ids.numel()
    if num_assignments == 0:
        return counts
    init_device_properties_triton()
    vector_cores = get_vectorcore_num()
    programs = min(vector_cores, triton.cdiv(num_assignments, ASSIGNMENTS_PER_TILE))
    _count_allgather_kernel[(programs,)](
        topk_ids,
        token_mask,
        counts,
        num_assignments,
        local_expert_start,
        local_expert_count,
        ID_ROW_STRIDE=topk_ids.stride(0),
        ID_COL_STRIDE=topk_ids.stride(1),
        MASK_STRIDE=token_mask.stride(0),
        TOPK=topk_ids.shape[1],
        BLOCK_SIZE=ASSIGNMENTS_PER_TILE,
        NUM_BINS=triton.next_power_of_2(local_expert_count),
    )
    return counts
