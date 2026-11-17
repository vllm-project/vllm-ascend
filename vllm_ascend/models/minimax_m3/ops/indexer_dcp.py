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
MAX_GEMM_QUERY_ROWS = 128
GATHER_BLOCKS_PER_TILE = 64
SCORE_FILL_VALUE = tl.constexpr(float("-inf"))


@triton.jit(do_not_specialize=["num_tokens"])
def _scatter_index_shard(
    updates_ptr,
    slots_ptr,
    cache_ptr,
    num_tokens,
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
    slot = tl.load(slots_ptr + token, token < num_tokens, other=-1).to(tl.int64)
    owned = (slot >= 0) & (slot % SIZE == RANK) & (token < num_tokens)
    cache_offset = slot // INDEX_BLOCK_SIZE * CACHE_STRIDE_P + slot % INDEX_BLOCK_SIZE // SIZE * CACHE_STRIDE_T
    value = tl.load(
        updates_ptr + token[:, None] * UPDATE_STRIDE + dim[None, :] * UPDATE_DIM_STRIDE, owned[:, None], other=0
    )
    tl.store(cache_ptr + cache_offset[:, None] + dim[None, :] * CACHE_STRIDE_D, value, owned[:, None])


def scatter_index_shard(cache: torch.Tensor, updates: torch.Tensor, slots: torch.Tensor, size: int, rank: int) -> None:
    """Write BF16 keys owned by rank in place; leave other cache entries unchanged."""
    if size not in (2, 4) or not 0 <= rank < size:
        raise ValueError("M3 index scatter requires TP2/TP4 and a valid shard rank")
    if cache.ndim != 3 or cache.shape[1:] != (128 // size, 128):
        raise ValueError("M3 index scatter requires compact 128-token pages with 128-wide keys")
    if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64) or not slots.is_contiguous():
        raise ValueError("M3 index scatter requires contiguous int32/int64 slots")
    if cache.dtype != torch.bfloat16 or updates.dtype != cache.dtype:
        raise ValueError("M3 index scatter requires BF16 keys and cache")
    if cache.device != updates.device or cache.device != slots.device:
        raise ValueError("M3 index scatter inputs must be on the same device")
    if updates.numel() != slots.numel() * 128:
        raise ValueError("M3 index scatter requires one 128-wide update per slot")
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


def gather_index_shards(cache: torch.Tensor, tp_group, pages: torch.Tensor) -> torch.Tensor:
    """Materialize one layer's full index cache for the existing prefill op."""
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
    pages, inverse = torch.unique(
        torch.cat((block_table_cpu.new_zeros(1), block_table_cpu[valid])), sorted=True, return_inverse=True
    )
    remapped = torch.zeros_like(block_table_cpu)
    remapped[valid] = inverse[1:].to(remapped.dtype)
    return pages, remapped


# Request sizes and pool-dependent GEMM row strides are runtime indexing values.
# Keep only fixed geometry, tiling and causal policy as compile-time parameters.
@triton.jit(
    do_not_specialize=[
        "row_start",
        "raw_stride",
        "table_stride",
        "table_width",
        "tokens",
        "query_len",
        "width",
        "block_tiles",
    ]
)
def _gather_index_shard_scores(
    raw_ptr,
    table_ptr,
    lengths_ptr,
    scores_ptr,
    row_start,
    raw_stride,
    table_stride,
    table_width,
    tokens,
    HEADS: tl.constexpr,
    query_len,
    width,
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
    block_tiles,
    TILE: tl.constexpr,
    CAUSAL: tl.constexpr,
    GATHERED_QUERY: tl.constexpr = False,
):
    # GEMM rows flatten (token, head). Select only this request's physical
    # pages, then mask interleaved positions before taking the block maximum.
    program = tl.program_id(0)
    row = program // block_tiles
    absolute_row = row_start + row
    if GATHERED_QUERY:
        local_heads: tl.constexpr = HEADS // SIZE
        token = absolute_row // local_heads % tokens
        head = absolute_row // (tokens * local_heads) * local_heads + absolute_row % local_heads
    else:
        token = absolute_row // HEADS
        head = absolute_row % HEADS
    request = token // query_len
    columns = program % block_tiles * TILE + tl.arange(0, TILE)
    lane = tl.arange(0, INDEX_BLOCK_SIZE // SIZE)
    page = tl.load(table_ptr + request * table_stride + columns, columns < table_width, other=-1).to(tl.int64)
    length = tl.load(lengths_ptr + request)
    position = columns[:, None] * INDEX_BLOCK_SIZE + lane[None, :] * SIZE + RANK
    valid = (columns[:, None] < table_width) & (page[:, None] >= 0) & (position < length)
    if CAUSAL:
        valid = valid & (position < length - query_len + token % query_len + 1)
    dots = tl.load(
        raw_ptr + row * raw_stride + page[:, None] * (INDEX_BLOCK_SIZE // SIZE) + lane[None, :],
        valid,
        other=SCORE_FILL_VALUE,
    )
    maxima = tl.max(dots, axis=1)
    tl.store(scores_ptr + (head * tokens + token) * width + columns, maxima, columns < width)


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
    gathered_query: bool = False,
) -> torch.Tensor:
    """Score the compact pool with GEMM, then gather each request's block maxima.

    FP32 GEMM tiles are bounded to MAX_GEMM_QUERY_ROWS; keys are never expanded
    to logical 128-token pages. Dense storage is a view; padded/strided storage
    may need a compact copy. FP16 score storage matches CANN's BF16 QK precision
    and stays FP16 through ReduceScatter and into the FP32 TopK preparation.
    ``gathered_query`` consumes contiguous [TP, token, local head, dim] storage
    directly, avoiding the AllGather's otherwise necessary layout conversion.
    """
    if size not in (2, 4) or not 0 <= rank < size:
        raise ValueError("M3 index scores require TP2/TP4 and a valid shard rank")
    if query.ndim != (4 if gathered_query else 3) or cache.ndim != 3 or block_table.ndim != 2 or seq_lens.ndim != 1:
        raise ValueError("M3 index scores require rank-3 queries/cache, rank-2 table and rank-1 lengths")
    if block_table.dtype not in (torch.int32, torch.int64) or seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("M3 index scores require int32/int64 block tables and lengths")
    if block_table.stride(1) != 1 or not seq_lens.is_contiguous():
        raise ValueError("M3 index scores require unit-stride table columns and contiguous lengths")
    if any(t.device != query.device for t in (cache, block_table, seq_lens)):
        raise ValueError("M3 index scores inputs must be on the same device")
    if query.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise ValueError("M3 compact index scores require BF16 queries and keys")
    if cache.shape[1:] != (128 // size, 128) or query.shape[-1] != 128:
        raise ValueError("M3 compact index scores require 128-wide keys and compact 128-token pages")
    tokens = query.shape[1] if gathered_query else query.shape[0]
    heads = query.shape[0] * query.shape[2] if gathered_query else query.shape[1]
    if gathered_query and (query.shape[0] != size or not query.is_contiguous()):
        raise ValueError("M3 gathered index queries require contiguous [TP, token, local head, dim] storage")
    if query_len <= 0 or tokens != seq_lens.numel() * query_len:
        raise ValueError("M3 compact index scores require uniform decode query lengths")
    if block_table.shape[0] != seq_lens.numel():
        raise ValueError("M3 compact index scores require one block table per sequence")
    width = triton.cdiv(block_table.shape[1], SCORE_ALIGNMENT) * SCORE_ALIGNMENT
    scores = torch.empty((heads, tokens, width), dtype=torch.float16, device=query.device)
    if scores.numel() == 0:
        return scores
    queries = query.reshape(-1, 128).contiguous()
    keys = cache.reshape(-1, 128).contiguous()
    block_tiles = triton.cdiv(width, GATHER_BLOCKS_PER_TILE)
    # Reuse keys across token/head rows to fill Cube tiles. Unreferenced pages
    # cost extra arithmetic, but are excluded by the gather's validity mask.
    for start in range(0, queries.shape[0], MAX_GEMM_QUERY_ROWS):
        stop = min(start + MAX_GEMM_QUERY_ROWS, queries.shape[0])
        raw = torch.mm(queries[start:stop], keys.T, out_dtype=torch.float32)
        _gather_index_shard_scores[((stop - start) * block_tiles,)](
            raw,
            block_table,
            seq_lens,
            scores,
            start,
            raw.stride(0),
            block_table.stride(0),
            block_table.shape[1],
            tokens,
            heads,
            query_len,
            width,
            size,
            rank,
            block_tiles,
            GATHER_BLOCKS_PER_TILE,
            causal,
            gathered_query,
        )
        # Avoid retaining two large GEMM outputs when the next tile allocates.
        del raw
    return scores


def reduce_scatter_index_scores(scores: torch.Tensor, tp_group) -> torch.Tensor:
    """Reduce block maxima directly to the TP rank that consumes each head."""
    if scores.shape[0] % tp_group.world_size:
        raise ValueError("Index heads must divide evenly across the TP group")
    local_heads = scores.shape[0] // tp_group.world_size
    output = torch.empty((local_heads, *scores.shape[1:]), dtype=scores.dtype, device=scores.device)
    dist.reduce_scatter_tensor(output, scores.contiguous(), op=dist.ReduceOp.MAX, group=tp_group.device_group)
    return output


def gather_index_queries_async(query: torch.Tensor, tp_group):
    """Start Q AllGather without waiting on the compute stream.

    The caller retains the contiguous input until ``work.wait()`` and orders
    all readers after that wait. Keep rank-major storage for the score GEMM;
    converting it to token-major would introduce an extra device copy.
    """
    if query.ndim != 3 or not query.is_contiguous():
        raise ValueError("M3 asynchronous query gather requires contiguous [token, head, dim] input")
    gathered = torch.empty((tp_group.world_size, *query.shape), dtype=query.dtype, device=query.device)
    work = dist.all_gather_into_tensor(gathered.flatten(0, 1), query, group=tp_group.device_group, async_op=True)
    return gathered, work
