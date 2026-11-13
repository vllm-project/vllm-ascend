# SPDX-License-Identifier: Apache-2.0
"""Context sharding of the replicated M3 index-key cache.

Each 128-token logical page stores offsets rank, rank + size, ... . Main
attention remains head-parallel. MAX over the local block maxima reconstructs
exact global block scores; summing local maxima would be incorrect.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton

INDEX_BLOCK_SIZE = tl.constexpr(128)
INDEX_HEAD_DIM = tl.constexpr(128)
SCORE_ALIGNMENT = 16
MAX_QUERY_TILE_ROWS = 32
SCORE_FILL_VALUE = -3.4028234663852886e38


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


def plan_prefill_index_gather(
    block_table_cpu: torch.Tensor, seq_lens_cpu: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exclude stale/padded columns; the caller builds this CPU plan once per batch."""
    columns = torch.arange(block_table_cpu.shape[1])
    valid = (columns[None, :] * block_size < seq_lens_cpu[:, None]) & (block_table_cpu >= 0)
    # Retain a safe page for empty rows and invalid entries. Unused columns
    # must not cause a gather of stale physical pages from an earlier batch.
    pages = torch.unique(torch.cat((block_table_cpu.new_zeros(1), block_table_cpu[valid])), sorted=True)
    remapped = torch.zeros_like(block_table_cpu)
    remapped[valid] = torch.searchsorted(pages, block_table_cpu[valid]).to(remapped.dtype)
    return pages, remapped


@triton.jit
def _score_index_shard(
    Query,
    Cache,
    BlockTable,
    SeqLens,
    Scores,
    QUERY_STRIDE_T: tl.constexpr,
    QUERY_STRIDE_H: tl.constexpr,
    QUERY_STRIDE_D: tl.constexpr,
    CACHE_STRIDE_P: tl.constexpr,
    CACHE_STRIDE_T: tl.constexpr,
    CACHE_STRIDE_D: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    SCORE_WIDTH: tl.constexpr,
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    CAUSAL: tl.constexpr,
    FILL: tl.constexpr,
):
    # Combine SIZE compact pages into one 128-key Cube tile. Query heads
    # share the same page table and can occupy rows of the same matrix.
    request = tl.program_id(0)
    query_row = tl.program_id(1) * QUERY_TILE + tl.arange(0, QUERY_TILE)
    query_offset = query_row // NUM_HEADS
    head = query_row % NUM_HEADS
    token = request * QUERY_LEN + query_offset
    block_start = tl.program_id(2) * SIZE
    columns = block_start + tl.arange(0, SIZE)
    output = Scores + (head[:, None] * NUM_TOKENS + token[:, None]) * SCORE_WIDTH + columns[None, :]
    output_mask = (query_offset[:, None] < QUERY_LEN) & (columns[None, :] < SCORE_WIDTH)
    length = tl.load(SeqLens + request)
    # Graph padding and table-capacity tails must still initialize scores, but
    # do not need key/query loads or a Cube operation.
    if block_start * INDEX_BLOCK_SIZE + RANK >= length:
        tl.store(output, FILL, output_mask)
        return
    dim = tl.arange(0, INDEX_HEAD_DIM)
    query = tl.load(
        Query + token[:, None] * QUERY_STRIDE_T + head[:, None] * QUERY_STRIDE_H + dim[None, :] * QUERY_STRIDE_D,
        query_offset[:, None] < QUERY_LEN,
        other=0,
    )

    local_tokens: tl.constexpr = INDEX_BLOCK_SIZE // SIZE
    lane = tl.arange(0, INDEX_BLOCK_SIZE)
    block = block_start + lane // local_tokens
    local_offset = lane % local_tokens
    page = tl.load(BlockTable + request * TABLE_STRIDE + block, block < TABLE_WIDTH, other=0).to(tl.int64)
    key_position = block * INDEX_BLOCK_SIZE + local_offset * SIZE + RANK
    valid_key = (block < TABLE_WIDTH) & (page >= 0) & (key_position < length)
    key = tl.load(
        Cache + page[:, None] * CACHE_STRIDE_P + local_offset[:, None] * CACHE_STRIDE_T + dim[None, :] * CACHE_STRIDE_D,
        valid_key[:, None],
        other=0,
    )
    products = tl.dot(query, tl.trans(key))
    visible = valid_key[None, :]
    if CAUSAL:
        frontier = length - QUERY_LEN + query_offset + 1
        visible = visible & (key_position[None, :] < frontier[:, None])
    products = tl.where(visible, products, FILL)
    maxima = tl.max(tl.reshape(products, (QUERY_TILE, SIZE, local_tokens)), axis=2)
    tl.store(output, maxima, output_mask)


def index_shard_scores(
    query: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_len: int,
    size: int,
    rank: int,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Score compact pages directly; decode metadata requires uniform query lengths."""
    if size not in (2, 4) or not 0 <= rank < size:
        raise ValueError("M3 index scores require TP2/TP4 and a valid shard rank")
    if query.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise ValueError("M3 compact index scores require BF16 queries and keys")
    if cache.shape[1:] != (128 // size, 128) or query.shape[-1] != 128:
        raise ValueError("M3 compact index scores require 128-wide keys and compact 128-token pages")
    if query_len <= 0 or query.shape[0] != seq_lens.numel() * query_len:
        raise ValueError("M3 compact index scores require uniform decode query lengths")
    if block_table.shape[0] != seq_lens.numel():
        raise ValueError("M3 compact index scores require one block table per sequence")
    width = triton.cdiv(block_table.shape[1], SCORE_ALIGNMENT) * SCORE_ALIGNMENT
    scores = torch.empty((query.shape[1], query.shape[0], width), dtype=torch.float32, device=query.device)
    if scores.numel() == 0:
        return scores
    query_tile = max(16, triton.next_power_of_2(min(query_len * query.shape[1], MAX_QUERY_TILE_ROWS)))
    _score_index_shard[
        (seq_lens.numel(), triton.cdiv(query_len * query.shape[1], query_tile), triton.cdiv(width, size))
    ](
        query,
        cache,
        block_table,
        seq_lens,
        scores,
        *query.stride(),
        *cache.stride(),
        block_table.stride(0),
        block_table.shape[1],
        query.shape[0],
        query.shape[1],
        query_len,
        width,
        size,
        rank,
        query_tile,
        causal,
        SCORE_FILL_VALUE,
    )
    return scores


def reduce_scatter_index_scores(scores: torch.Tensor, tp_group) -> torch.Tensor:
    """Reduce block maxima directly to the TP rank that consumes each head."""
    if scores.shape[0] % tp_group.world_size:
        raise ValueError("Index heads must divide evenly across the TP group")
    local_heads = scores.shape[0] // tp_group.world_size
    output = torch.empty((local_heads, *scores.shape[1:]), dtype=scores.dtype, device=scores.device)
    dist.reduce_scatter_tensor(output, scores.contiguous(), op=dist.ReduceOp.MAX, group=tp_group.device_group)
    return output
