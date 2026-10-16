# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expand referenced TQ4 pages into a bounded, ephemeral BF16 layer workspace."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

LATENT_DIM = 512
ROPE_DIM = 64
BF16_PAGE = 128
TILE_ROWS = 8
MIN_QUERY_TOKENS = 32
# Measured on A3 with 64 query heads and Top-K 2048. Do not expand a long
# prefix for a few new queries; the original TQ operator is faster there.
MAX_EXPANDED_ROWS_PER_QUERY = 128
WORKSPACE_ALIGNMENT = 256


def workspace_shapes(num_requests: int, max_seq_len: int, query_tokens: int, budget_bytes: int):
    """Return bounded rectangular page views, or None when expansion is costly."""
    if num_requests <= 0 or max_seq_len <= 0 or query_tokens < MIN_QUERY_TOKENS or budget_bytes <= 0:
        return None
    pages_per_request = triton.cdiv(max_seq_len, BF16_PAGE)
    pages = num_requests * pages_per_request
    rows = pages * BF16_PAGE
    if rows > query_tokens * MAX_EXPANDED_ROWS_PER_QUERY:
        return None
    shapes = (
        ((pages, BF16_PAGE, 1, LATENT_DIM), torch.bfloat16),
        ((pages, BF16_PAGE, 1, ROPE_DIM), torch.bfloat16),
        ((num_requests, pages_per_request), torch.int32),
    )
    sizes = (rows * LATENT_DIM * 2, rows * ROPE_DIM * 2, pages * 4)
    required = sum(triton.cdiv(size, WORKSPACE_ALIGNMENT) * WORKSPACE_ALIGNMENT for size in sizes)
    return shapes if required <= budget_bytes else None


@triton.jit(do_not_specialize=["rows", "rows_per_request"])
def _expand_pages(
    packed,
    packed16,
    block_table,
    seq_lens,
    centroids,
    nope,
    rope,
    output_table,
    rows,
    rows_per_request,
    TABLE_STRIDE: tl.constexpr,
    SOURCE_PAGE: tl.constexpr,
    TILE: tl.constexpr,
):
    code_col = tl.arange(0, 256)
    latent_col = tl.arange(0, 512)
    rope_col = tl.arange(0, 64)
    lut = tl.load(centroids + tl.arange(0, 16)).to(tl.bfloat16).to(tl.float32)
    for tile in range(tl.program_id(0), tl.cdiv(rows, TILE), tl.num_programs(0)):
        base = tile * TILE
        request = base // rows_per_request
        position = base % rows_per_request
        length = tl.load(seq_lens + request)
        # TILE divides both supported source page sizes and the output page.
        # No tile crosses a request or page boundary. Invalid tail pages must
        # never be read: prefix caches may leave these table entries stale.
        source_page = tl.load(
            block_table + request * TABLE_STRIDE + position // SOURCE_PAGE, position < length, other=0
        )
        offset = tl.arange(0, TILE)
        valid = position + offset < length
        source_row = source_page * SOURCE_PAGE + position % SOURCE_PAGE + offset
        codes_byte = tl.load(
            packed + source_row[:, None] * 386 + code_col[None, :], valid[:, None], other=0
        ).to(tl.int32)
        codes = tl.interleave(codes_byte & 15, codes_byte >> 4)
        decoded = tl.gather(lut, codes.reshape((TILE * 512,)), 0).reshape((TILE, 512))
        scale_bits = tl.load(packed16 + source_row * 193 + 192, valid, other=0).to(tl.uint16)
        scale = scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
        out_row = base + offset
        tl.store(nope + out_row[:, None] * 512 + latent_col[None, :], decoded * scale[:, None])
        rope_bits = tl.load(
            packed16 + source_row[:, None] * 193 + 128 + rope_col[None, :], valid[:, None], other=0
        ).to(tl.uint16)
        rope_value = rope_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        tl.store(rope + out_row[:, None] * 64 + rope_col[None, :], rope_value * scale[:, None])
        if position % 128 == 0:
            tl.store(output_table + base // 128, base // 128)


def expand_turboquant_pages(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    nope: torch.Tensor,
    rope: torch.Tensor,
    output_table: torch.Tensor,
) -> None:
    """Read packed cache without modifying it; populate caller-owned workspace.

    Shared prefix pages may appear in several request rows. They are read-only
    and expanded independently; the rectangular budget includes this cost.
    Source tables must reference valid pages up to each device sequence length.
    """
    rows = nope.shape[0] * BF16_PAGE
    rows_per_request = output_table.shape[1] * BF16_PAGE
    _expand_pages[(min(get_vectorcore_num(), triton.cdiv(rows, TILE_ROWS)),)](
        cache.view(torch.uint8),
        cache.view(torch.int16),
        block_table,
        seq_lens,
        centroids,
        nope,
        rope,
        output_table,
        rows,
        rows_per_request,
        block_table.stride(0),
        cache.shape[1],
        TILE_ROWS,
    )
