# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact candidate exchange for token-sharded lightning indexer caches."""

import torch

MAX_EXACT_FLOAT32_INTEGER = 1 << 24


def use_local_topk_output(sharded_indexer: bool, pcp_size: int, dsa_cp: bool, *, turboquant: bool = False) -> bool:
    """Keep one output coordinate system for producing and shared-index layers."""
    return sharded_indexer and pcp_size == 1 and not dsa_cp and not turboquant


def query_visible_lengths(seq_lens, cum_query_lens, rows, requests):
    """Global causal prefix length per varlen query, with padded rows empty."""
    visible = (seq_lens[requests] - cum_query_lens[requests] + rows + 1).clamp_min(0)
    return torch.where(rows < cum_query_lens[-1], visible, 0)


def local_visible_lengths(visible, world_size: int, rank: int, interleave: int):
    """Count owner-local tokens in the global prefix [0, visible)."""
    cycles = visible // (world_size * interleave)
    tail = (visible % (world_size * interleave) - rank * interleave).clamp(0, interleave)
    return cycles * interleave + tail


def local_to_global(indices, world_size: int, rank: int, interleave: int):
    mapped = (indices // interleave) * (world_size * interleave) + rank * interleave + indices % interleave
    return torch.where(indices >= 0, mapped, -1).to(torch.int32)


def merge_topk_candidates(
    indices, scores, topk: int, *, local_rank: int | None = None, world_size: int = 1, interleave: int = 1
):
    """Use the QLI comparator score and global-ID ascending tie order.

    Scores are FP32 transport values of the native QLI comparator. Casting
    them does not introduce more precision than the local kernel provides.
    Invalid candidates never enter the selected valid prefix.
    """
    valid = (indices >= 0) & torch.isfinite(scores)
    safe_ids = torch.where(valid, indices, torch.iinfo(torch.int32).max)
    # Ascend's INT32 argsort dispatches to AiCPU. Token IDs up to 2**24 are
    # exactly representable as FP32 (validated by the owning backend), so use
    # a temporary AiCore sort key while storing and exchanging actual INT32 IDs.
    id_keys = safe_ids.to(torch.float32) if getattr(indices, "is_npu", False) else safe_ids
    id_order = torch.argsort(id_keys, dim=-1, stable=True)
    ids = torch.gather(indices, -1, id_order)
    values = torch.gather(torch.where(valid, scores, -torch.inf), -1, id_order)
    score_order = torch.argsort(values, dim=-1, descending=True, stable=True)[..., :topk]
    selected_ids = torch.gather(ids, -1, score_order)
    selected_scores = torch.gather(values, -1, score_order)
    if local_rank is not None:
        from vllm_ascend.ops.triton.sharded_indexer import select_local_candidates

        return select_local_candidates(selected_ids, selected_scores, world_size, local_rank, interleave)
    return torch.where(torch.isfinite(selected_scores), selected_ids, -1).to(torch.int32)


def distributed_topk(indices, scores, group, topk: int, interleave: int, *, return_local: bool = False):
    from vllm_ascend.ops.triton.sharded_indexer import merge_packed_candidates, pack_local_candidates

    # Exchange candidates only. Never gather an indexer's token-history cache.
    # Bound prefill candidate workspace independently of prompt history and
    # max_num_batched_tokens. Every rank executes the same fixed row tiles.
    query_tile = 64
    output = torch.empty((indices.shape[0], 1, topk), dtype=torch.int32, device=indices.device)
    for start in range(0, indices.shape[0], query_tile):
        end = start + query_tile
        packed = pack_local_candidates(
            indices[start:end], scores[start:end], group.world_size, group.rank_in_group, interleave
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
        )
    return output
