# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner-local pool scoring and exact candidate exchange for GLM DCP."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.attention.context_parallel.sharded_indexer import distributed_topk
from vllm_ascend.ops.triton.glm5_next_lightning_indexer import (
    NEG_INF_SENTINEL,
    QUERY_END_SENTINEL,
    TRITON_INDEX_BLOCK,
    TRITON_MAX_PROGRAMS,
    _expand_pool_indices,
)

# Sorting also allocates values and INT64 permutations. Budget the score
# tensor separately so long prefill batches do not materialize all Q x K.
DCP_SCORE_BYTES = 32 << 20
DCP_POOL_TILE = 32


@triton.jit(do_not_specialize=["rows", "token_offset", "capacity", "requests"])
def _score_local_pools(
    qbar,
    cache,
    scales,
    ends,
    lengths,
    table,
    positions,
    scores,
    rows,
    token_offset,
    capacity,
    requests,
    CACHE_SB: tl.constexpr,
    CACHE_ST: tl.constexpr,
    CACHE_SD: tl.constexpr,
    SCALE_SB: tl.constexpr,
    SCALE_ST: tl.constexpr,
    TABLE_SR: tl.constexpr,
    TABLE_SC: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    DIM: tl.constexpr,
    POOL: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    QUANTIZED: tl.constexpr,
    REQ_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    LOWEST: tl.constexpr,
    END_SENTINEL: tl.constexpr,
):
    tiles = tl.cdiv(capacity, BLOCK)
    for work in range(tl.program_id(0), rows * tiles, tl.num_programs(0)):
        row, tile = work // tiles, work % tiles
        token = token_offset + row
        reqs = tl.arange(0, REQ_BLOCK)
        query_ends = tl.load(ends + reqs, reqs < requests, other=END_SENTINEL)
        request = tl.minimum(tl.sum((token >= query_ends).to(tl.int32)), requests - 1)
        live = token < tl.load(ends + requests - 1)
        pos = tl.load(positions + token).to(tl.int64)
        visible = tl.minimum((pos + 1) // POOL, tl.load(lengths + request))
        local = tile * BLOCK + tl.arange(0, BLOCK)
        global_pool = (local // INTERLEAVE * WORLD + RANK) * INTERLEAVE + local % INTERLEAVE
        valid = live & (local < capacity) & (global_pool < visible)
        # Block IDs are local physical pages. Only the visibility comparison
        # uses global pool coordinates; addresses never use global_pool.
        page = tl.load(table + request * TABLE_SR + local // CACHE_BLOCK * TABLE_SC, valid, other=0)
        dims = tl.arange(0, DIM)
        address = page.to(tl.int64)[:, None] * CACHE_SB + (local % CACHE_BLOCK)[:, None] * CACHE_ST
        key = tl.load(cache + address + dims[None, :] * CACHE_SD, valid[:, None], other=0).to(tl.float32)
        if QUANTIZED:
            scale = tl.load(scales + page.to(tl.int64) * SCALE_SB + local % CACHE_BLOCK * SCALE_ST, valid, other=1)
            key *= scale[:, None]
        query = tl.load(qbar + row * DIM + dims)
        score = tl.sum(key * query[None, :], 1)
        score = tl.where(valid & (score == score), tl.minimum(tl.maximum(score, LOWEST), -LOWEST), LOWEST)
        tl.store(scores + row * capacity + local, score, local < capacity)


def select_local_pool_candidates(
    query,
    cache,
    weights,
    metadata,
    *,
    world_size,
    rank,
    interleave,
    pool_size,
    pool_topk,
    capacity,
    token_offset=0,
    scale_cache=None,
):
    """Return fixed-width local pool IDs/FP32 scores, masking before Top-K."""
    rows, _, dim = query.shape
    ids = torch.full((rows, 1, pool_topk), -1, dtype=torch.int32, device=query.device)
    values = torch.full((rows, 1, pool_topk), -torch.inf, dtype=torch.float32, device=query.device)
    if rows == 0 or capacity == 0 or metadata.cum_query_lens.numel() == 0:
        return ids, values
    qbar = (query.float() * weights.float().unsqueeze(-1)).sum(1).contiguous()
    scores = torch.empty((rows, capacity), dtype=torch.float32, device=query.device)
    _score_local_pools[(min(rows * triton.cdiv(capacity, DCP_POOL_TILE), TRITON_MAX_PROGRAMS),)](
        qbar,
        cache,
        scale_cache,
        metadata.cum_query_lens,
        metadata.seq_lens,
        metadata.block_table,
        metadata.positions,
        scores,
        rows,
        token_offset,
        capacity,
        metadata.cum_query_lens.numel(),
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        scale_cache.stride(0) if scale_cache is not None else 0,
        scale_cache.stride(1) if scale_cache is not None else 0,
        *metadata.block_table.stride(),
        cache.shape[1],
        dim,
        pool_size,
        world_size,
        rank,
        interleave,
        scale_cache is not None,
        triton.next_power_of_2(metadata.cum_query_lens.numel()),
        DCP_POOL_TILE,
        NEG_INF_SENTINEL,
        QUERY_END_SENTINEL,
    )
    # Local ID order is also global ID order within one owner. Stable sorting
    # keeps the lowest global IDs at a tied cutoff before candidates are lost.
    ordered_scores, ordered_ids = torch.sort(scores, dim=-1, descending=True, stable=True)
    count = min(pool_topk, capacity)
    selected_scores = ordered_scores[:, :count]
    valid = selected_scores > NEG_INF_SENTINEL
    ids[:, 0, :count] = torch.where(valid, ordered_ids[:, :count], -1).to(torch.int32)
    values[:, 0, :count] = torch.where(valid, selected_scores, -torch.inf)
    return ids, values


def glm5_next_dcp_indexer(
    query,
    cache,
    weights,
    metadata,
    group,
    *,
    topk,
    pool_size,
    interleave,
    max_model_len,
    output_buffer=None,
    scale_cache=None,
):
    """Merge pool candidates globally, then expand tokens and append the tail.

    All ranks must have identical query rows (PCP=1). Main MLA and compressed
    pools are token-sharded; the small compressor ring remains replicated.
    Tensor contents may change on replay, including request lengths/pages.
    """
    if pool_size <= 1 or topk <= 0 or topk % pool_size or interleave <= 0 or interleave % pool_size:
        raise ValueError("KPool DCP requires pool-aligned topk and positive token interleave.")
    if query.shape[-1] != triton.next_power_of_2(query.shape[-1]):
        raise ValueError("KPool DCP requires a power-of-two indexer head dimension.")
    if cache.dtype == torch.float8_e4m3fn and scale_cache is None:
        raise ValueError("FP8 KPool DCP requires the per-pool scale cache.")
    tokens = query.shape[0]
    width = topk + pool_size - 1
    if output_buffer is None:
        output = torch.empty((tokens, width), dtype=torch.int32, device=query.device)
    else:
        if (
            output_buffer.dtype != torch.int32
            or output_buffer.device != query.device
            or output_buffer.ndim != 2
            or output_buffer.shape[0] < tokens
            or output_buffer.shape[1] < width
            or output_buffer.stride(1) != 1
            or output_buffer.stride(0) < output_buffer.shape[1]
        ):
            raise ValueError("KPool DCP output must cover query rows and the pooled topk plus tail.")
        output = output_buffer[:tokens]
    if tokens == 0:
        return output[:, :width].unsqueeze(1)
    if metadata.cum_query_lens.numel() == 0:
        output.fill_(-1)
        return output[:, :width].unsqueeze(1)
    pool_interleave = interleave // pool_size
    # Same bound on every rank, including owners with no visible pools.
    capacity = min(
        triton.cdiv(max_model_len, group.world_size * interleave) * pool_interleave,
        metadata.block_table.shape[1] * cache.shape[1],
    )
    query_tile = max(1, DCP_SCORE_BYTES // max(4, capacity * 4))
    pool_topk = topk // pool_size
    for start in range(0, tokens, query_tile):
        end = min(start + query_tile, tokens)
        local_ids, scores = select_local_pool_candidates(
            query[start:end],
            cache,
            weights[start:end],
            metadata,
            world_size=group.world_size,
            rank=group.rank_in_group,
            interleave=pool_interleave,
            pool_size=pool_size,
            pool_topk=pool_topk,
            capacity=capacity,
            token_offset=start,
            scale_cache=scale_cache,
        )
        pools = distributed_topk(local_ids, scores, group, pool_topk, pool_interleave)
        # Expansion only needs validity; global scores have already served
        # their purpose. Keep global token coordinates in shared layer buffers.
        validity = torch.where(pools >= 0, 0.0, NEG_INF_SENTINEL)
        tiles = triton.cdiv(output.shape[1], TRITON_INDEX_BLOCK)
        _expand_pool_indices[(min((end - start) * tiles, TRITON_MAX_PROGRAMS),)](
            pools,
            validity,
            metadata.positions,
            metadata.cum_query_lens,
            output,
            end - start,
            start,
            metadata.cum_query_lens.numel() - 1,
            pool_topk,
            True,
            topk,
            pool_size,
            output.shape[1],
            output.stride(0),
            True,
            NEG_INF_SENTINEL,
            TRITON_INDEX_BLOCK,
        )
    return output[:, :width].unsqueeze(1)
