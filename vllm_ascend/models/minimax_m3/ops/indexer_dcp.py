# SPDX-License-Identifier: Apache-2.0
"""Context sharding of the replicated M3 index-key cache.

Each 128-token logical page stores offsets rank, rank + size, ... . Main
attention remains head-parallel. MAX over the local block maxima reconstructs
exact global block scores; summing local maxima would be incorrect.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

INDEX_BLOCK_SIZE = tl.constexpr(128)
INDEX_HEAD_DIM = tl.constexpr(128)


@triton.jit
def _scatter_index_shard(
    Updates,
    Slots,
    Cache,
    NUM_TOKENS: tl.constexpr,
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
    UPDATE_STRIDE: tl.constexpr,
    UPDATE_DIM_STRIDE: tl.constexpr,
    CACHE_STRIDE_P: tl.constexpr,
    CACHE_STRIDE_T: tl.constexpr,
    CACHE_STRIDE_D: tl.constexpr,
    TILE: tl.constexpr,
):
    token = tl.program_id(0) * TILE + tl.arange(0, TILE)
    dim = tl.arange(0, INDEX_HEAD_DIM)
    slot = tl.load(Slots + token, token < NUM_TOKENS, other=-1)
    owned = (slot >= 0) & (slot % SIZE == RANK) & (token < NUM_TOKENS)
    cache_offset = slot // INDEX_BLOCK_SIZE * CACHE_STRIDE_P + slot % INDEX_BLOCK_SIZE // SIZE * CACHE_STRIDE_T
    value = tl.load(
        Updates + token[:, None] * UPDATE_STRIDE + dim[None, :] * UPDATE_DIM_STRIDE, owned[:, None], other=0
    )
    tl.store(Cache + cache_offset[:, None] + dim[None, :] * CACHE_STRIDE_D, value, owned[:, None])


def scatter_index_shard(cache: torch.Tensor, updates: torch.Tensor, slots: torch.Tensor, size: int, rank: int) -> None:
    """Skip graph-padding slots (-1) without dynamic tensor indexing."""
    if slots.numel() == 0:
        return
    updates = updates.reshape(slots.numel(), 128)
    _scatter_index_shard[(triton.cdiv(slots.numel(), 16),)](
        updates,
        slots,
        cache,
        slots.numel(),
        size,
        rank,
        updates.stride(0),
        updates.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        16,
    )


def gather_index_shards(cache: torch.Tensor, tp_group, pages: torch.Tensor | None = None) -> torch.Tensor:
    """Materialize one layer's full index cache for the existing prefill op."""
    if pages is not None:
        cache = cache.index_select(0, pages)
    size = tp_group.world_size
    blocks, local_tokens, dim = cache.shape
    gathered = tp_group.all_gather(cache, dim=1)
    return gathered.view(blocks, size, local_tokens, dim).transpose(1, 2).reshape(blocks, 128, dim).contiguous()


def index_shard_scores_cann(
    query: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor,
    size: int,
    rank: int,
) -> torch.Tensor:
    """Use CANN's Cube score kernel with one layer of temporary local keys.

    Repeating each owned key fills the fixed 128-token CANN page. Shifting
    the causal frontier by rank makes key group j visible precisely when
    the original interleaved position size * j + rank becomes visible.
    No other rank's index keys are materialized or communicated here.
    """
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _minimax_m3_index_score

    expanded_cache = cache.repeat_interleave(size, dim=1)
    return _minimax_m3_index_score(
        query,
        expanded_cache,
        block_table,
        cu_seqlens_q,
        (seq_lens - rank).clamp_min(0),
        start_loc,
        causal_mask,
    )
