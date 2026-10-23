# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 QK scoring for GLM KPool prefill on supported CANN QLI v2 kernels.

The persistent cache contains INT8 key bytes followed by an FP32 scale. Gather
live rows into the separate key/FP16-scale layout required by QLI v2, quantize
queries per head, and expand selected pools back into causal token indices.
All scratch state belongs to one eager forward and is refilled between layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_ascend.attention.indexer_kpool import (
        AscendIndexerKPoolMetadata,
        AscendIndexerKPoolQueryMetadata,
    )
import torch
import torch_npu
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["requests", "pools_count"])
def _gather_int8_pool_rows(
    cache,
    keys,
    scales,
    table,
    lengths,
    requests,
    pools_count,
    blocks_count,
    HALF_DIM: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_SB: tl.constexpr,
    CACHE_ST: tl.constexpr,
    TABLE_SR: tl.constexpr,
    TABLE_SP: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chunks = tl.cdiv(pools_count, BLOCK)
    for tile in range(tl.program_id(0), requests * chunks, tl.num_programs(0)):
        request, chunk = tile // chunks, tile % chunks
        pools = chunk * BLOCK + tl.arange(0, BLOCK)
        valid = (pools < pools_count) & (pools < tl.load(lengths + request)) & (pools // CACHE_BLOCK < TABLE_WIDTH)
        blocks = tl.load(table + request * TABLE_SR + (pools // CACHE_BLOCK) * TABLE_SP, valid, other=0)
        blocks = tl.minimum(tl.maximum(blocks, 0), blocks_count - 1)
        row = blocks.to(tl.int64) * CACHE_SB + (pools % CACHE_BLOCK) * CACHE_ST
        lanes = tl.arange(0, HALF_DIM)
        words = tl.load(cache + row[:, None] + lanes[None, :], valid[:, None], other=0)
        lo = tl.load(cache + row + HALF_DIM, valid, other=0).to(tl.int32) & 0xFFFF
        hi = tl.load(cache + row + HALF_DIM + 1, valid, other=0).to(tl.int32) & 0xFFFF
        scale = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        outrow = request * pools_count + pools
        tl.store(keys + outrow[:, None] * HALF_DIM + lanes[None, :], words, (pools < pools_count)[:, None])
        tl.store(scales + outrow, scale, pools < pools_count)


@triton.jit(do_not_specialize=["tokens"])
def _expand_quant_pool_indices(
    ids,
    positions,
    ends,
    output,
    tokens,
    LAST: tl.constexpr,
    SELECTED: tl.constexpr,
    POOL: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tiles = tl.cdiv(OUT_WIDTH, BLOCK)
    for tile in range(tl.program_id(0), tokens * tiles, tl.num_programs(0)):
        token, part = tile // tiles, tile % tiles
        cols = part * BLOCK + tl.arange(0, BLOCK)
        position = tl.load(positions + token).to(tl.int64)
        live = token < tl.load(ends + LAST)
        poolcols = part * (BLOCK // POOL) + tl.arange(0, BLOCK // POOL)
        unique_ids = tl.load(ids + token * SELECTED + poolcols, poolcols < SELECTED, other=-1)
        poolids = tl.gather(unique_ids.to(tl.float32, bitcast=True), tl.arange(0, BLOCK) // POOL, axis=0).to(
            tl.int32, bitcast=True
        )
        value = tl.where((poolids >= 0) & (cols < SELECTED * POOL), poolids * POOL + cols % POOL, -1)
        tail_start = (position + 1) // POOL * POOL
        tail_col = tl.minimum(tail_start, SELECTED * POOL).to(tl.int32)
        off = cols - tail_col
        count = (position + 1 - tail_start).to(tl.int32)
        value = tl.where((off >= 0) & (off < POOL - 1), tl.where(off < count, tail_start.to(tl.int32) + off, -1), value)
        value = tl.where(live, value, -1)
        tl.store(output + token * OUT_STRIDE + cols, value, cols < OUT_WIDTH)


def _prepare_quant_keys(
    cache: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    max_pools: int,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Copy INT8 bytes without dequantizing; bound scratch to 256 MiB."""
    requests = lengths.numel()
    if requests == 0:
        return None
    pages = max(1, triton.cdiv(min(max_pools, table.shape[1] * cache.shape[1]), 128))
    pools = pages * 128
    dim = cache.shape[-1] - 4
    shape = (requests * pages, 128, 1, dim)
    if requests * pools * (dim + 2) > 256 * 1024 * 1024:
        return None
    if workspace is None:
        keys = torch.empty(shape, device=cache.device, dtype=torch.int8)
        scales = torch.empty(shape[:-1], device=cache.device, dtype=torch.float16)
        packed_table = torch.arange(requests * pages, device=cache.device, dtype=torch.int32).view(requests, pages)
        workspace = keys, scales, packed_table
    keys, scales, packed_table = workspace
    if keys.shape != shape or keys.device != cache.device:
        raise ValueError("GLM KPool INT8 workspace does not match this forward.")
    _gather_int8_pool_rows[(min(requests * pages, 256),)](
        cache.view(torch.int16),
        keys.view(torch.int16),
        scales,
        table,
        lengths,
        requests,
        pools,
        cache.shape[0],
        dim // 2,
        cache.shape[1],
        cache.stride(0) // 2,
        cache.stride(1) // 2,
        *table.stride(),
        table.shape[1],
        128,
    )
    return workspace


def glm5_next_c8_quant_indexer(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    meta: AscendIndexerKPoolMetadata,
    query_meta: AscendIndexerKPoolQueryMetadata | AscendIndexerKPoolMetadata,
    max_pools: int,
    topk_tokens: int,
    pool_size: int,
    output: torch.Tensor | None,
) -> torch.Tensor | None:
    """Score with INT8 dot/INT32 accumulation, then scaled per-head ReLU.

    Positive query scales can be folded into head weights after ReLU; the key
    scale applies after the head reduction. Query quantization adds rounding
    relative to the BF16 query path. This helper is for eager prefill only.
    Return None when the bounded key workspace cannot fit.
    """
    # A2/A3 QLI v2 requires a power-of-two compression ratio and <=2048 pools.
    if pool_size <= 0 or pool_size > 128 or pool_size & (pool_size - 1) or topk_tokens // pool_size > 2048:
        return None
    assert query_meta.cum_query_lens is not None
    assert query_meta.pool_tail is not None
    workspace = _prepare_quant_keys(cache, meta.block_table, meta.seq_lens, max_pools, meta.c8_quant_workspace)
    if workspace is None:
        return None
    meta.c8_quant_workspace = workspace
    keys, scales, table = workspace
    state = meta.c8_quant_metadata
    if state is None:
        ends = query_meta.cum_query_lens.to(torch.int32)
        cu = torch.cat((torch.zeros(1, device=q.device, dtype=torch.int32), ends))
        lengths = query_meta.seq_lens.to(torch.int32)
        tails = query_meta.pool_tail.to(torch.int32)
        tiling = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            num_heads_q=q.shape[1],
            num_heads_k=1,
            head_dim=q.shape[2],
            topk=topk_tokens // pool_size,
            quant_mode=2,
            cu_seqlens_q=cu,
            seqused_k=lengths,
            cmp_residual_k=tails,
            batch_size=lengths.numel(),
            max_seqlen_q=q.shape[0],
            max_seqlen_k=max_pools * pool_size + pool_size - 1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=pool_size,
        )
        state = cu, lengths, tails, tiling
        meta.c8_quant_metadata = state
    cu, lengths, tails, tiling = state
    q8, sq = torch_npu.npu_dynamic_quant(q.contiguous())
    ids, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query=q8,
        key=keys,
        weights=weights.half(),
        query_dequant_scale=sq.half(),
        key_dequant_scale=scales,
        topk=topk_tokens // pool_size,
        quant_mode=2,
        cu_seqlens_q=cu,
        seqused_k=lengths,
        cmp_residual_k=tails,
        block_table=table,
        metadata=tiling,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=pool_size,
        return_value=0,
    )
    width = topk_tokens + pool_size - 1
    if output is None:
        output = torch.empty((q.shape[0], width), dtype=torch.int32, device=q.device)
    result = output[: q.shape[0]]
    block = 4096 if q.shape[0] >= 512 else 256
    _expand_quant_pool_indices[(min(q.shape[0] * triton.cdiv(result.shape[1], block), 256),)](
        ids,
        query_meta.positions,
        query_meta.cum_query_lens,
        result,
        q.shape[0],
        lengths.numel() - 1,
        topk_tokens // pool_size,
        pool_size,
        result.shape[1],
        result.stride(0),
        block,
    )
    return result[:, :width].unsqueeze(1)
