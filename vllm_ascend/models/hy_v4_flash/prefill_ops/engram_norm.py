# SPDX-License-Identifier: Apache-2.0
"""Fuse Engram's elementwise operations around the original RMS and dot reductions."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["rows"])
def _engram_norm_squares(
    key,
    query,
    key_square,
    query_square,
    rows,
    QUERY_STRIDE: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    row = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    col = tl.program_id(1) * TILE_COLS + tl.arange(0, TILE_COLS)
    idx = row[:, None] * 3072 + col[None, :]
    qidx = row[:, None] * QUERY_STRIDE + col[None, :]
    valid = row[:, None] < rows
    k = tl.load(key + idx, valid, other=0).to(tl.float32)
    q = tl.load(query + qidx, valid, other=0).to(tl.float32)
    tl.store(key_square + idx, k * k, valid)
    tl.store(query_square + idx, q * q, valid)


@triton.jit(do_not_specialize=["rows"])
def _engram_norm_products(
    key,
    query,
    kw,
    qw,
    ki,
    qi,
    output,
    rows,
    QUERY_STRIDE: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    row = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    col = tl.program_id(1) * TILE_COLS + tl.arange(0, TILE_COLS)
    idx = row[:, None] * 3072 + col[None, :]
    qidx = row[:, None] * QUERY_STRIDE + col[None, :]
    valid = row[:, None] < rows
    k = tl.load(key + idx, valid, other=0).to(tl.float32)
    q = tl.load(query + qidx, valid, other=0).to(tl.float32)
    k_inv = tl.load(ki + row, row < rows, other=0)
    q_inv = tl.load(qi + row, row < rows, other=0)
    k_weight = tl.load(kw + col).to(tl.float32)
    q_weight = tl.load(qw + col).to(tl.float32)
    # Match the native BF16 outputs of both RMSNorms before multiplying.
    kn = ((k * k_inv[:, None]) * k_weight[None, :]).to(tl.bfloat16).to(tl.float32)
    qn = ((q * q_inv[:, None]) * q_weight[None, :]).to(tl.bfloat16).to(tl.float32)
    tl.store(output + idx, (kn * qn).to(tl.bfloat16), valid)


def engram_normalized_dot(key, query, key_weight, query_weight):
    rows = key.shape[0]
    eps = torch.finfo(torch.float32).eps
    tile_rows = 8
    squares = torch.empty((2, rows, 3072), device=key.device, dtype=torch.float32)
    grid = (triton.cdiv(rows, tile_rows), 3)
    _engram_norm_squares[grid](key, query, squares[0], squares[1], rows, query.stride(0), tile_rows, 1024)
    inv = (squares.view(-1, 3072).mean(-1) + eps).rsqrt().view(2, rows)
    output = torch.empty_like(key)
    _engram_norm_products[grid](
        key,
        query,
        key_weight,
        query_weight,
        inv[0],
        inv[1],
        output,
        rows,
        query.stride(0),
        tile_rows,
        1024,
        enable_fp_fusion=False,
    )
    # Use the original reduction order for the dot product.
    return output.sum(-1)
