# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""310P implementations of pluggable vLLM Ascend V2 kernels."""

from typing import Any

import torch
from vllm.model_executor.triton_dispatcher import register_kernel


@register_kernel("vllm_ascend.worker.v2.block_table._compute_slot_mappings_kernel")
def compute_slot_mappings(
    max_num_tokens: int,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_table_stride: int,
    block_size: int,
    slot_mapping: torch.Tensor,
    cp_rank: int,
    *,
    CP_SIZE: int,
    CP_INTERLEAVE: int,
    PAD_ID: int,
    TRITON_BLOCK_SIZE: int,
    TOTAL_BLOCK_SIZE: int,
    grid: Any = None,
) -> None:
    """Compute one KV-cache group's slot mapping without Triton.

    The 310P block-table owner passes CPU tensors here, then performs one H2D
    copy after all cache groups have been processed. Keeping the signature
    aligned with the default Triton kernel lets both implementations share the
    existing ``kernel[grid](...)`` call site.
    """
    del block_table_stride, TRITON_BLOCK_SIZE, TOTAL_BLOCK_SIZE, grid
    if any(tensor.device.type != "cpu" for tensor in (idx_mapping, query_start_loc, positions, block_table)):
        raise TypeError("310P slot-mapping metadata must be backed by CPU tensors.")

    slot_mapping[:max_num_tokens].fill_(PAD_ID)
    num_reqs = idx_mapping.numel()
    for batch_idx in range(num_reqs):
        req_idx = int(idx_mapping[batch_idx])
        start = int(query_start_loc[batch_idx])
        end = int(query_start_loc[batch_idx + 1])
        if end <= start:
            continue

        token_positions = positions[start:end].to(torch.int64)
        block_indices = torch.div(token_positions, block_size * CP_SIZE, rounding_mode="floor")
        block_offsets = token_positions - block_indices * (block_size * CP_SIZE)
        block_numbers = block_table[req_idx].index_select(0, block_indices)

        if CP_SIZE == 1:
            slot_ids = block_numbers * block_size + block_offsets
        else:
            is_local = torch.div(block_offsets, CP_INTERLEAVE, rounding_mode="floor") % CP_SIZE == cp_rank
            rounds = torch.div(block_offsets, CP_INTERLEAVE * CP_SIZE, rounding_mode="floor")
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = torch.where(is_local, slot_ids, PAD_ID)

        slot_mapping[start:end].copy_(slot_ids.to(slot_mapping.dtype))
