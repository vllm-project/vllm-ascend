# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact candidate exchange for token-sharded lightning indexer caches."""

import torch

MAX_EXACT_FLOAT32_INTEGER = 1 << 24
MAX_GATHERED_CANDIDATE_BYTES = 256 << 20
PACKED_PAIR_BYTES = 8  # INT32 global ID and FP32 score bits.
MIN_SINGLE_TILE_ROWS = 256


def use_local_topk_output(sharded_indexer: bool, pcp_size: int, dsa_cp: bool, *, turboquant: bool = False) -> bool:
    """Keep one output coordinate system for producing and shared-index layers."""
    return sharded_indexer and pcp_size == 1 and not dsa_cp and not turboquant


def gathered_candidate_row_tile(rows: int, world_size: int, local_k: int) -> int:
    """Budget candidate exchange for long batches using shared host shapes.

    The 256-row guarantee may exceed the nominal budget for large world*K.
    Only shared host-side shapes determine collective boundaries.
    """
    if rows == 0:
        return 0
    row_bytes = world_size * local_k * PACKED_PAIR_BYTES
    return max(MIN_SINGLE_TILE_ROWS, MAX_GATHERED_CANDIDATE_BYTES // row_bytes)


def distributed_topk(indices, scores, group, topk: int, interleave: int, *, return_local: bool = False):
    from vllm_ascend.ops.triton.sharded_indexer import MAX_FUSED_TOPK, merge_packed_candidates, pack_local_candidates

    # Exchange candidates only. Never gather an indexer's token-history cache.
    rows, local_k = indices.shape[0], indices.shape[-1]
    output = torch.empty((rows, 1, topk), dtype=torch.int32, device=indices.device)
    if rows == 0:
        return output
    # The merge may omit local sorts only when pack established their order.
    # This contract depends on shared shapes, never rank-local candidate data.
    presorted = topk == MAX_FUSED_TOPK and local_k <= MAX_FUSED_TOPK
    query_tile = gathered_candidate_row_tile(rows, group.world_size, local_k)
    for start in range(0, rows, query_tile):
        end = min(start + query_tile, rows)
        packed = pack_local_candidates(
            indices[start:end],
            scores[start:end],
            group.world_size,
            group.rank_in_group,
            interleave,
            sort_candidates=presorted,
            fuse_sort=return_local and presorted,
        )
        gathered = group.all_gather(packed, dim=0)
        merge_packed_candidates(
            gathered,
            packed.shape[0],
            group.world_size,
            topk,
            local_rank=group.rank_in_group if return_local else None,
            interleave=interleave,
            out=output[start:end],
            presorted=presorted,
            owner_sharded=return_local and presorted,
        )
    return output
