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


@triton.jit(do_not_specialize=["count"])
def _engram_value_squares(x, square, count, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    value = tl.load(x + i, i < count, other=0).to(tl.float32)
    tl.store(square + i, value * value, i < count)


def engram_value_inverse(value, epsilon):
    square = torch.empty_like(value, dtype=torch.float32)
    tile = 4096
    _engram_value_squares[(triton.cdiv(value.numel(), tile),)](value, square, value.numel(), tile)
    return (square.mean(-1, keepdim=True) + epsilon).rsqrt()

@triton.jit(do_not_specialize=["rows"])
def _engram_qk_squares(
    key0, key1, key2, key3, query, squares, rows, WIDTH: tl.constexpr, CHANNELS: tl.constexpr,
    TILE_ROWS: tl.constexpr, TILE_COLS: tl.constexpr,
):
    channel = tl.program_id(2)
    key = tl.where(channel < 2, tl.where(channel == 0, key0, key1), tl.where(channel == 2, key2, key3))
    row = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    col = tl.program_id(1) * TILE_COLS + tl.arange(0, TILE_COLS)
    index = row[:, None] * WIDTH + col[None, :]
    mask = row[:, None] < rows
    k = tl.load(key + index, mask, other=0).to(tl.float32)
    q = tl.load(query + (row[:, None] * CHANNELS + channel) * WIDTH + col[None, :],
                mask, other=0).to(tl.float32)
    offset = squares + channel * (2 * rows * WIDTH) + index
    tl.store(offset, k * k, mask)
    tl.store(offset + rows * WIDTH, q * q, mask)


@triton.jit(do_not_specialize=["rows"])
def _engram_qk_products(
    key0, key1, key2, key3, query, key_weight, query_weight, inv0, inv1, inv2, inv3, output, rows,
    WIDTH: tl.constexpr, CHANNELS: tl.constexpr,
    TILE_ROWS: tl.constexpr, TILE_COLS: tl.constexpr,
):
    channel = tl.program_id(2)
    key = tl.where(channel < 2, tl.where(channel == 0, key0, key1), tl.where(channel == 2, key2, key3))
    row = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    col = tl.program_id(1) * TILE_COLS + tl.arange(0, TILE_COLS)
    index = row[:, None] * WIDTH + col[None, :]
    mask = row[:, None] < rows
    k = tl.load(key + index, mask, other=0).to(tl.float32)
    q = tl.load(query + (row[:, None] * CHANNELS + channel) * WIDTH + col[None, :],
                mask, other=0).to(tl.float32)
    inverse = tl.where(channel < 2, tl.where(channel == 0, inv0, inv1), tl.where(channel == 2, inv2, inv3))
    inv_offset = inverse + row
    ik = tl.load(inv_offset, row < rows, other=0)
    iq = tl.load(inv_offset + rows, row < rows, other=0)
    kw = tl.load(key_weight + channel * WIDTH + col).to(tl.float32)
    qw = tl.load(query_weight + channel * WIDTH + col).to(tl.float32)
    nk = ((k * ik[:, None]) * kw[None, :]).to(tl.bfloat16).to(tl.float32)
    nq = ((q * iq[:, None]) * qw[None, :]).to(tl.bfloat16).to(tl.float32)
    tl.store(output + channel * rows * WIDTH + index, (nk * nq).to(tl.bfloat16), mask)


def engram_batched_dots(keys, query, key_weight, query_weight):
    rows, channels, width = query.shape
    tile_rows, tile_cols = 8, 1024
    grid = (triton.cdiv(rows, tile_rows), width // tile_cols, channels)
    squares = torch.empty((channels, 2, rows, width), device=query.device, dtype=torch.float32)
    _engram_qk_squares[grid](*keys, query, squares, rows, width, channels, tile_rows, tile_cols)
    # Keep the original per-channel reduction geometry and BF16 rounding.
    eps = torch.finfo(torch.float32).eps
    inverses = tuple((channel.view(-1, width).mean(-1) + eps).rsqrt() for channel in squares.unbind(0))
    del squares
    products = query.new_empty((channels, rows, width))
    _engram_qk_products[grid](
        *keys, query, key_weight, query_weight, *inverses, products,
        rows, width, channels, tile_rows, tile_cols, enable_fp_fusion=False,
    )
    return tuple(channel.sum(-1) for channel in products.unbind(0))


@triton.jit(do_not_specialize=["tokens"])
def _apply_lookup_norm_with_inverse(
    rows, inverse, gamma, output, tokens,
    HEADS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr, CORES: tl.constexpr,
):
    tiles = tl.cdiv(tokens, BLOCK)
    tiles_per_core = tl.cdiv(tiles, CORES)
    start = tl.program_id(0) * tiles_per_core
    stop = tl.minimum(start + tiles_per_core, tiles)
    col = tl.arange(0, triton.next_power_of_2(HEADS * WIDTH))
    gamma_live = col < HEADS * WIDTH
    weight = tl.load(gamma + col, gamma_live, other=0)
    for tile in range(start, stop):
        token = tile * BLOCK + tl.arange(0, BLOCK)
        index = token[:, None] * (HEADS * WIDTH) + col[None, :]
        valid = (token[:, None] < tokens) & gamma_live[None, :]
        value = tl.load(rows + index, valid, other=0)
        rstd = tl.load(inverse + token[:, None] * HEADS + col[None, :] // WIDTH, valid, other=0)
        tl.store(output + index, (value * rstd) * weight[None, :], valid)


@triton.jit(do_not_specialize=["groups"])
def _apply_lookup_norm(
    rows, gamma, output, groups, EPS: tl.constexpr,
    HEADS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr, CORES: tl.constexpr,
):
    tiles = tl.cdiv(groups, BLOCK)
    tiles_per_core = tl.cdiv(tiles, CORES)
    start = tl.program_id(0) * tiles_per_core
    stop = tl.minimum(start + tiles_per_core, tiles)
    col = tl.arange(0, triton.next_power_of_2(WIDTH))
    group_lane = tl.arange(0, BLOCK)
    weight = tl.load(gamma + (group_lane[:, None] % HEADS) * WIDTH + col[None, :],
                     col[None, :] < WIDTH, other=0)
    for tile in range(start, stop):
        group = tile * BLOCK + group_lane
        valid = (group[:, None] < groups) & (col[None, :] < WIDTH)
        index = group[:, None] * WIDTH + col[None, :]
        value = tl.load(rows + index, valid, other=0)
        mean = tl.sum(value * value, axis=1) * (1.0 / WIDTH)
        # CANN takes the FP32 sqrt first, then corrects the reciprocal.
        # A direct rsqrt rounds differently on Ascend 950.
        root = tl.sqrt(mean + EPS)
        inv = 1.0 / root
        inv = tl.fma(inv, tl.fma(-root, inv, 1.0), inv)
        tl.store(output + index, (value * inv[:, None]) * weight, valid)


def apply_lookup_norm(rows, gamma, dtype, eps, *, inverse=None):
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

    tokens, heads, width = rows.shape
    out = torch.empty_like(rows, dtype=dtype)
    init_device_properties_triton()
    if inverse is None:
        groups = tokens * heads
        block = 64
        cores = min(triton.cdiv(groups, block), get_vectorcore_num())
        _apply_lookup_norm[(cores,)](rows, gamma, out, groups, eps, heads, width, block, cores,
                                     enable_fp_fusion=False)
    else:
        # The kernel spans all local heads per token. Bound its live tile as TP shrinks.
        block = min(16, max(1, 4096 // triton.next_power_of_2(heads * width)))
        cores = min(triton.cdiv(tokens, block), get_vectorcore_num())
        _apply_lookup_norm_with_inverse[(cores,)](rows, inverse, gamma, out, tokens, heads, width, block, cores,
                                                  enable_fp_fusion=False)
    return out
