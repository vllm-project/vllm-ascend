# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged INT8 QK scoring for all GLM KPool queries, including graph replay.

Each cache page holds contiguous INT8 keys followed by FP16 scales. QLI v2
reads these views directly through the original block table; no history-sized
scratch or floating QK fallback is needed.
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
    POOL_SHIFT: tl.constexpr,
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
        # Pool sizes are powers of two and column indices are nonnegative.
        # Explicit bit operations avoid expensive vector division on Ascend.
        poolids = tl.gather(unique_ids.to(tl.float32, bitcast=True), tl.arange(0, BLOCK) >> POOL_SHIFT, axis=0).to(
            tl.int32, bitcast=True
        )
        value = tl.where((poolids >= 0) & (cols < SELECTED * POOL), poolids * POOL + (cols & (POOL - 1)), -1)
        tail_start = (position + 1) // POOL * POOL
        tail_col = tl.minimum(tail_start, SELECTED * POOL).to(tl.int32)
        off = cols - tail_col
        count = (position + 1 - tail_start).to(tl.int32)
        value = tl.where((off >= 0) & (off < POOL - 1), tl.where(off < count, tail_start.to(tl.int32) + off, -1), value)
        value = tl.where(live, value, -1)
        tl.store(output + token * OUT_STRIDE + cols, value, cols < OUT_WIDTH)


def view_quant_kpool_cache(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """View page-local keys/scales without copying, preserving physical strides.

    The allocation retains 132 bytes per row. Within each page, B*128 key
    bytes precede B FP16 scales; the remaining 2*B bytes are reserved.
    """
    if cache.dtype != torch.int8 or cache.ndim != 4 or cache.shape[2:] != (1, 132):
        raise ValueError("GLM KPool C8 requires an INT8 [pages, rows, 1, 132] allocation.")
    pages, rows = cache.shape[:2]
    if (
        rows % 16
        or not 16 <= rows <= 1024
        or cache.stride(0) < rows * 132
        or cache.stride(0) % 2
        or cache.storage_offset() % 2
    ):
        raise ValueError("GLM KPool C8 requires 16-aligned page rows and halfword-aligned storage.")
    keys = cache.as_strided((pages, rows, 1, 128), (cache.stride(0), 128, 128, 1))
    scales = cache.view(torch.float16).as_strided(
        (pages, rows, 1),
        (cache.stride(0) // 2, 1, 1),
        (cache.storage_offset() + rows * 128) // 2,
    )
    return keys, scales


def glm5_next_c8_quant_indexer(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    meta: AscendIndexerKPoolMetadata,
    query_meta: AscendIndexerKPoolQueryMetadata | AscendIndexerKPoolMetadata,
    topk_tokens: int,
    pool_size: int,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """Quantize Q, multiply INT8 Q/K with INT32 accumulation, then scale scores."""
    if (
        pool_size <= 0
        or pool_size > 128
        or pool_size & (pool_size - 1)
        or topk_tokens <= 0
        or topk_tokens % pool_size
        or topk_tokens // pool_size > 2048
    ):
        raise ValueError("GLM KPool C8 requires a power-of-two pool size <=128 and <=2048 selected pools.")
    if q.shape[1] not in (32, 64) or q.shape[2] != 128:
        raise ValueError("GLM KPool C8 integer scoring requires 32/64 query heads of dimension 128.")
    if not all(
        hasattr(torch.ops._C_ascend, op)
        for op in ("npu_quant_lightning_indexer_v2", "npu_quant_lightning_indexer_v2_metadata")
    ):
        raise RuntimeError("GLM KPool C8 requires the QLI v2 integer operators; rebuild the Ascend custom operators.")
    width = topk_tokens + pool_size - 1
    if output is not None and (
        output.ndim != 2
        or output.dtype != torch.int32
        or output.device != q.device
        or output.stride(1) != 1
        or output.shape[0] < q.shape[0]
        or output.shape[1] < width
        or (q.shape[0] > 1 and output.stride(0) < output.shape[1])
    ):
        raise ValueError("GLM KPool output requires int32 nonoverlapping rows covering queries and top-k width.")
    assert query_meta.cum_query_lens is not None
    assert query_meta.pool_tail is not None
    keys, scales = view_quant_kpool_cache(cache)
    table = meta.block_table
    state = meta.c8_quant_metadata
    if state is None:
        ends = query_meta.cum_query_lens.to(torch.int32)
        cu = torch.cat((torch.zeros(1, device=q.device, dtype=torch.int32), ends))
        lengths = query_meta.seq_lens.to(torch.int32)
        tails = query_meta.pool_tail.to(torch.int32)
        # PA_BBND reads actual K lengths from seqused_k, including graph replay.
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
    if output is None:
        output = torch.empty((q.shape[0], width), dtype=torch.int32, device=q.device)
    result = output[: q.shape[0]]
    block = 4096
    _expand_quant_pool_indices[(min(q.shape[0] * triton.cdiv(result.shape[1], block), 256),)](
        ids,
        query_meta.positions,
        query_meta.cum_query_lens,
        result,
        q.shape[0],
        lengths.numel() - 1,
        topk_tokens // pool_size,
        pool_size,
        pool_size.bit_length() - 1,
        result.shape[1],
        result.stride(0),
        block,
    )
    return result[:, :width].unsqueeze(1)
