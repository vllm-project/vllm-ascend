# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch


def profile_dsv4_dcp_topk_memory(
    num_queries: int,
    topk: int,
    dcp_size: int,
    block_size: int,
    device: torch.device,
) -> int:
    """Measure candidate merge allocations, including stable-sort workspace.

    No collective is issued: repeated candidates have the same allocation
    geometry as the two AllGather outputs. Run after saving the ordinary
    activation profile; this independent peak is added to its memory budget.
    """
    from vllm_ascend.attention.context_parallel.dsv4_indexer import (
        local_to_global_indices,
        merge_dcp_topk,
    )

    torch.npu.synchronize(device)
    baseline = torch.npu.memory_allocated(device)
    torch.npu.reset_peak_memory_stats(device)
    local_indices = torch.zeros((num_queries, 1, topk), dtype=torch.int32, device=device)
    local_scores = torch.zeros_like(local_indices, dtype=torch.float32)
    global_indices = local_to_global_indices(local_indices, block_size, dcp_size, 0)
    gathered_indices = global_indices.repeat(dcp_size, 1, 1)
    gathered_scores = local_scores.repeat(dcp_size, 1, 1)
    merged = merge_dcp_topk(gathered_indices, gathered_scores, dcp_size, topk)
    torch.npu.synchronize(device)
    peak = torch.npu.max_memory_allocated(device) - baseline
    del merged, gathered_scores, gathered_indices, global_indices, local_scores, local_indices
    return peak


def plan_dsv4_dcp_memory(
    available_memory: int,
    pool_bytes_per_block: int,
    gather_bytes_per_block: int,
    max_gather_blocks: int,
    decode_workspace: int,
) -> tuple[int, int]:
    """Fit the shared pool and one layer's peak DCP workspace together.

    The compact gather includes at most one copy of each pool ID, including
    the reserved null block. All layers reuse the same temporary allocation
    capacity. Pool size and reachable gather size must therefore be solved
    together rather than reserving max_num_seqs * max_model_len unconditionally.
    """
    low, high = 0, max(0, available_memory // pool_bytes_per_block)
    while low < high:
        num_blocks = (low + high + 1) // 2
        gather_workspace = min(num_blocks, max_gather_blocks) * gather_bytes_per_block
        workspace = max(decode_workspace, gather_workspace)
        if num_blocks * pool_bytes_per_block + workspace <= available_memory:
            low = num_blocks
        else:
            high = num_blocks - 1
    workspace = max(decode_workspace, min(low, max_gather_blocks) * gather_bytes_per_block)
    return low * pool_bytes_per_block, workspace
