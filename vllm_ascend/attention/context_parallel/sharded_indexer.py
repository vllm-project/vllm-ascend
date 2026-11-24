# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact candidate exchange for token-sharded lightning indexer caches."""

import torch

MAX_EXACT_FLOAT32_INTEGER = 1 << 24


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


def merge_topk_candidates(indices, scores, topk: int):
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
    return torch.where(torch.isfinite(selected_scores), selected_ids, -1).to(torch.int32)


def distributed_topk(indices, scores, group, topk: int, interleave: int):
    global_ids = local_to_global(indices, group.world_size, group.rank_in_group, interleave)
    # Exchange candidates only. Never gather an indexer's token-history cache.
    # Bound prefill candidate workspace independently of prompt history and
    # max_num_batched_tokens. Every rank executes the same fixed row tiles.
    query_tile = 64
    result = []
    for start in range(0, indices.shape[0], query_tile):
        end = start + query_tile
        all_ids = group.all_gather(global_ids[start:end].contiguous(), dim=-1)
        all_scores = group.all_gather(scores[start:end].contiguous(), dim=-1)
        result.append(merge_topk_candidates(all_ids, all_scores, topk))
    return torch.cat(result, dim=0) if result else global_ids
