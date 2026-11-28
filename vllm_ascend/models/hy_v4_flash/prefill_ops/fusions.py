# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 Flash prefill ops fusions."""

import torch
import torch_npu
from vllm.triton_utils import tl, triton

from vllm_ascend.quantization.utils import DynamicMXQuantScaleAlg

from ..attention_ops import build_mla_logits as _build_window_logits
from ..attention_ops import logical_positions, reuse_prefill_metadata


@triton.jit(do_not_specialize=["rows"])
def _engram_gate_epilogue(
    dot, gates, rows, SCALE: tl.constexpr, CHANNEL: tl.constexpr, CHANNELS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    raw = tl.load(dot + row, row < rows, other=0)
    dtype: tl.constexpr = dot.dtype.element_ty
    scaled = (raw.to(tl.float32) * SCALE).to(dtype).to(tl.float32)
    magnitude = tl.maximum(tl.abs(scaled), 1.0e-6).to(dtype).to(tl.float32)
    root = tl.sqrt(magnitude).to(dtype).to(tl.float32)
    signed = (root * tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))).to(dtype).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-signed))
    tl.store(gates + row * CHANNELS + CHANNEL, gate.to(dtype), row < rows)


def engram_gate_epilogue(dot, gates, channel, scale):
    rows, channels = gates.shape
    block = 256
    _engram_gate_epilogue[(triton.cdiv(rows, block),)](
        dot,
        gates,
        rows,
        scale,
        channel,
        channels,
        block,
        enable_fp_fusion=False,
    )


@triton.jit(do_not_specialize=["rows"])
def _engram_gates(
    dot0,
    dot1,
    dot2,
    dot3,
    gates,
    rows,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = tl.program_id(1)
    dot = tl.where(channel < 2, tl.where(channel == 0, dot0, dot1), tl.where(channel == 2, dot2, dot3))
    raw = tl.load(dot + row, row < rows, other=0)
    dtype: tl.constexpr = dot.dtype.element_ty
    scaled = (raw.to(tl.float32) * SCALE).to(dtype).to(tl.float32)
    magnitude = tl.maximum(tl.abs(scaled), 1.0e-6).to(dtype).to(tl.float32)
    root = tl.sqrt(magnitude).to(dtype).to(tl.float32)
    signed = (root * tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))).to(dtype).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-signed))
    tl.store(gates + row * 4 + channel, gate.to(dtype), row < rows)


def engram_gates(dots, scale):
    rows = dots[0].numel()
    gates = dots[0].new_empty((rows, 4))
    block = 256
    _engram_gates[(triton.cdiv(rows, block), 4)](*dots, gates, rows, scale, block, enable_fp_fusion=False)
    return gates


@triton.jit(do_not_specialize=["tokens"])
def _engram_residual(
    hidden,
    hashes,
    gates,
    scales,
    signal,
    gamma,
    inverse,
    output,
    tokens,
    WIDTH: tl.constexpr,
    CHANNELS: tl.constexpr,
    HEADS: tl.constexpr,
    HASH_ROW: tl.constexpr,
    HASH_COL: tl.constexpr,
    BLOCK: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    token = tl.program_id(0)
    channel = tl.program_id(1)
    col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    head = tl.arange(0, triton.next_power_of_2(HEADS))
    hashes_in_row = tl.load(hashes + token * HASH_ROW + head * HASH_COL, head < HEADS, other=-1)
    active = tl.sum((hashes_in_row >= 0).to(tl.int32), 0) > 0
    gate = tl.load(gates + token * CHANNELS + channel).to(tl.float32)
    scale = tl.load(scales + channel * WIDTH + col, col < WIDTH, other=0)
    value = tl.load(signal + token * WIDTH + col, col < WIDTH, other=0).to(tl.float32)
    if NORMALIZE:
        weight = tl.load(gamma + col, col < WIDTH, other=0)
        row_inverse = tl.load(inverse + token)
        value = (weight * value) * row_inverse
    offset = (token * CHANNELS + channel) * WIDTH + col
    orig = tl.load(hidden + offset, col < WIDTH, other=0).to(tl.float32)
    dtype: tl.constexpr = hidden.dtype.element_ty
    residual = ((gate + scale) * value).to(dtype).to(tl.float32)
    tl.store(output + offset, (orig + tl.where(active, residual, 0.0)).to(dtype), col < WIDTH)


def engram_residual(hidden, hashes, gates, scales, signal, *, gamma=None, inverse=None):
    tokens, channels, width = hidden.shape
    output = torch.empty_like(hidden)
    block = triton.next_power_of_2(width)
    _engram_residual[(tokens, channels, triton.cdiv(width, block))](
        hidden,
        hashes,
        gates,
        scales,
        signal,
        signal if gamma is None else gamma,
        signal if inverse is None else inverse,
        output,
        tokens,
        width,
        channels,
        hashes.shape[1],
        *hashes.stride(),
        block,
        gamma is not None,
        enable_fp_fusion=False,
    )
    return output


@triton.jit(do_not_specialize=["tokens"])
def _engram_gate_residual(
    hidden,
    hashes,
    dot0,
    dot1,
    dot2,
    dot3,
    scales,
    signal,
    gamma,
    inverse,
    output,
    tokens,
    WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    HASH_ROW: tl.constexpr,
    HASH_COL: tl.constexpr,
    GATE_SCALE: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    channel = tl.program_id(1)
    col = tl.program_id(2) * COLS + tl.arange(0, COLS)
    head = tl.arange(0, triton.next_power_of_2(HEADS))
    valid_row = row < tokens
    h = tl.load(
        hashes + row[:, None] * HASH_ROW + head[None, :] * HASH_COL,
        valid_row[:, None] & (head[None, :] < HEADS),
        other=-1,
    )
    active = tl.sum((h >= 0).to(tl.int32), 1) > 0

    dot = tl.where(channel < 2, tl.where(channel == 0, dot0, dot1), tl.where(channel == 2, dot2, dot3))
    raw = tl.load(dot + row, valid_row, other=0)
    dtype: tl.constexpr = hidden.dtype.element_ty
    scaled = (raw.to(tl.float32) * GATE_SCALE).to(dtype).to(tl.float32)
    magnitude = tl.maximum(tl.abs(scaled), 1.0e-6).to(dtype).to(tl.float32)
    root = tl.sqrt(magnitude).to(dtype).to(tl.float32)
    signed = (root * tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))).to(dtype).to(tl.float32)
    gate = (1.0 / (1.0 + tl.exp(-signed))).to(dtype).to(tl.float32)

    valid = valid_row[:, None] & (col[None, :] < WIDTH)
    value = tl.load(signal + row[:, None] * WIDTH + col[None, :], valid, other=0).to(tl.float32)
    weight = tl.load(gamma + col, col < WIDTH, other=0)
    rstd = tl.load(inverse + row, valid_row, other=0)
    value = (weight[None, :] * value) * rstd[:, None]
    scale = tl.load(scales + channel * WIDTH + col, col < WIDTH, other=0)
    offset = (row[:, None] * 4 + channel) * WIDTH + col[None, :]
    orig = tl.load(hidden + offset, valid, other=0).to(tl.float32)
    residual = ((gate[:, None] + scale[None, :]) * value).to(dtype).to(tl.float32)
    tl.store(output + offset, (orig + tl.where(active[:, None], residual, 0.0)).to(dtype), valid)


def engram_gate_residual(hidden, hashes, dots, scales, value, gamma, inverse, scale):
    tokens, channels, width = hidden.shape
    result = torch.empty_like(hidden)
    rows, cols = 8, 1024
    _engram_gate_residual[(triton.cdiv(tokens, rows), channels, triton.cdiv(width, cols))](
        hidden,
        hashes,
        *dots,
        scales,
        value,
        gamma,
        inverse,
        result,
        tokens,
        width,
        hashes.shape[1],
        *hashes.stride(),
        scale,
        rows,
        cols,
        enable_fp_fusion=False,
    )
    return result


@triton.jit(do_not_specialize=["elements"])
def _mask_local_experts(expert_map, ids, weights, output, elements, EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < elements
    expert = tl.load(ids + index, valid, other=0).to(tl.int32)
    expert = tl.where(expert < 0, expert + EXPERTS, expert)
    local = tl.load(expert_map + expert, valid, other=-1)
    weight = tl.load(weights + index, valid, other=0)
    tl.store(output + index, weight * (local != -1).to(weight.dtype), valid)


def mask_local_experts(expert_map, ids, weights, *, dtype=None):
    output = torch.empty_like(weights, dtype=dtype)
    block = 1024
    _mask_local_experts[(triton.cdiv(ids.numel(), block),)](
        expert_map,
        ids,
        weights,
        output,
        ids.numel(),
        expert_map.numel(),
        block,
    )
    return output


@triton.jit(do_not_specialize=["rows"])
def _gather_mrope_factors(
    cache,
    positions,
    cosine,
    sine,
    rows,
    POS_SIZE: tl.constexpr,
    POS_AXIS: tl.constexpr,
    POS_ROW: tl.constexpr,
    HEIGHT_BITS: tl.constexpr,
    WIDTH_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, 64)
    channel = lane // 2
    hbits = tl.full((), HEIGHT_BITS, tl.uint32)
    wbits = tl.full((), WIDTH_BITS, tl.uint32)
    axis = ((hbits >> channel) & 1) + 2 * ((wbits >> channel) & 1)
    position = tl.load(positions + axis[None, :] * POS_AXIS + row[:, None] * POS_ROW, row[:, None] < rows, other=0)
    position = tl.where(position < 0, position + POS_SIZE, position).to(tl.int32)
    index = position * 64 + channel[None, :]
    cos = tl.load(cache + index, row[:, None] < rows, other=0)
    sin = tl.load(cache + index + 32, row[:, None] < rows, other=0)
    offset = row[:, None] * 64 + lane[None, :]
    tl.store(cosine + offset, cos, row[:, None] < rows)
    tl.store(sine + offset, sin, row[:, None] < rows)


def mrope_channel_bits(section):
    return tuple(sum(1 << col for col in range(32) if col % 3 == axis and col < section[axis] * 3) for axis in (1, 2))


def gather_mrope_factors(cache, positions, channel_bits):
    rows = positions.shape[1]
    cosine = torch.empty((1, rows, 1, 64), dtype=cache.dtype, device=cache.device)
    sine = torch.empty_like(cosine)
    block = 16
    _gather_mrope_factors[(triton.cdiv(rows, block),)](
        cache,
        positions,
        cosine,
        sine,
        rows,
        cache.shape[0],
        *positions.stride(),
        *channel_bits,
        block,
    )
    return cosine, sine


@triton.jit
def _split_rope_projection(projection, main, rotary, rows, MAIN: tl.constexpr, ROPE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.arange(0, triton.next_power_of_2(MAIN))
    b = tl.arange(0, ROPE)
    base = row[:, None] * (MAIN + ROPE)
    first = tl.load(projection + base + a[None, :], (row[:, None] < rows) & (a[None, :] < MAIN), other=0)
    second = tl.load(projection + base + MAIN + b[None, :], row[:, None] < rows, other=0)
    tl.store(main + row[:, None] * MAIN + a[None, :], first, (row[:, None] < rows) & (a[None, :] < MAIN))
    tl.store(rotary + row[:, None] * ROPE + b[None, :], second, row[:, None] < rows)


@triton.jit(do_not_specialize=["rows"])
def _pack_attention_projections(
    q,
    kv,
    q_nope,
    q_pe,
    kv_latent,
    k_pe,
    rows,
    HEADS: tl.constexpr,
    NOPE: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    Q_TILE: tl.constexpr,
    KV_TILE: tl.constexpr,
):
    if tl.program_id(1) == 0:
        _split_rope_projection(q, q_nope, q_pe, rows * HEADS, NOPE, ROPE, Q_TILE)
    else:
        _split_rope_projection(kv, kv_latent, k_pe, rows, LATENT, ROPE, KV_TILE)


def pack_attention_projections(query, kv, rope_dim):
    rows, heads, q_width = query.shape
    nope_dim = q_width - rope_dim
    latent_dim = kv.shape[-1] - rope_dim
    q_nope = query.new_empty((rows, heads, nope_dim))
    q_pe = query.new_empty((rows, heads * rope_dim))
    latent = kv.new_empty((rows, latent_dim))
    k_pe = kv.new_empty((rows, rope_dim))
    q_tile = heads * 8
    kv_tile = 8
    _pack_attention_projections[(triton.cdiv(rows, kv_tile), 2)](
        query,
        kv,
        q_nope,
        q_pe,
        latent,
        k_pe,
        rows,
        heads,
        nope_dim,
        latent_dim,
        rope_dim,
        q_tile,
        kv_tile,
    )
    return q_nope, q_pe, latent, k_pe


@triton.jit
def _add_halves(source, width: tl.constexpr):
    lane = tl.arange(0, width // 2)
    return tl.gather(source, lane, 0) + tl.gather(source, lane + width // 2, 0)


@triton.jit
def _fold_1024(source):
    source = _add_halves(source, 1024)
    source = _add_halves(source, 512)
    source = _add_halves(source, 256)
    return _add_halves(source, 128)


@triton.jit
def _reduce_3072(a, b, c):
    # Combine the first and last 1024 elements before folding to 64 lanes.
    return tl.sum(_fold_1024(a + c) + _fold_1024(b), 0)


@triton.jit
def _rstd(a, b, c, eps: tl.constexpr):
    mean = _reduce_3072(a * a, b * b, c * c) * (1.0 / 3072)
    return tl.div_rn(1.0, tl.sqrt(mean + eps))


@triton.jit
def _prod(key, query, k_inv, q_inv, key_weight, query_weight, index):
    kw = tl.load(key_weight + index).to(tl.float32)
    qw = tl.load(query_weight + index).to(tl.float32)
    # Round both RMSNorm outputs and their product to BF16.
    kn = ((key * k_inv) * kw).to(tl.bfloat16).to(tl.float32)
    qn = ((query * q_inv) * qw).to(tl.bfloat16).to(tl.float32)
    return (kn * qn).to(tl.bfloat16).to(tl.float32)


@triton.jit(do_not_specialize=["key_stride"])
def _engram_gate_dot_ordered(
    key,
    hidden,
    key_weight,
    query_weight,
    dot,
    key_stride,
    CHANNEL: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, 1024)
    k = key + row * key_stride + col
    q = hidden + (row * 4 + CHANNEL) * 3072 + col
    k0 = tl.load(k).to(tl.float32)
    k1 = tl.load(k + 1024).to(tl.float32)
    k2 = tl.load(k + 2048).to(tl.float32)
    q0 = tl.load(q).to(tl.float32)
    q1 = tl.load(q + 1024).to(tl.float32)
    q2 = tl.load(q + 2048).to(tl.float32)
    kinv = _rstd(k0, k1, k2, EPS)
    qinv = _rstd(q0, q1, q2, EPS)
    p0 = _prod(k0, q0, kinv, qinv, key_weight, query_weight, col)
    p1 = _prod(k1, q1, kinv, qinv, key_weight, query_weight, col + 1024)
    p2 = _prod(k2, q2, kinv, qinv, key_weight, query_weight, col + 2048)
    dot_value = _reduce_3072(p0, p1, p2)
    tl.store(dot + row, dot_value.to(tl.bfloat16))


def gate_dot(key, hidden, key_weight, query_weight, channel, epsilon):
    tokens = hidden.shape[0]
    dot = hidden.new_empty(tokens)
    _engram_gate_dot_ordered[(tokens,)](
        key,
        hidden,
        key_weight,
        query_weight,
        dot,
        key.stride(0),
        channel,
        epsilon,
        enable_fp_fusion=False,
    )
    return dot


@triton.jit
def _load_square(value, offset):
    x = tl.load(value + offset).to(tl.float32)
    return x * x


@triton.jit(do_not_specialize=["value_stride"])
def _engram_gate_value_inverse(value, inverse, value_stride, EPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 1024)
    offset = row * value_stride + col

    # CANN's 3072-wide row reduction adds the tail to the first half,
    # folds each half into a 64-lane vector, then reduces their sum.
    left = _load_square(value, offset) + _load_square(value, offset + 2048)
    right = _load_square(value, offset + 1024)
    mean = tl.sum(_fold_1024(left) + _fold_1024(right), 0) * (1.0 / 3072)
    tl.store(inverse + row, tl.div_rn(1.0, tl.sqrt(mean + EPS)))


def gate_value_inverse(value, epsilon):
    tokens = value.shape[0]
    inverse = value.new_empty(tokens, dtype=torch.float32)
    _engram_gate_value_inverse[(tokens,)](
        value,
        inverse,
        value.stride(0),
        epsilon,
        enable_fp_fusion=False,
    )
    return inverse


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
    key0,
    key1,
    key2,
    key3,
    query,
    squares,
    rows,
    WIDTH: tl.constexpr,
    CHANNELS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    channel = tl.program_id(2)
    key = tl.where(channel < 2, tl.where(channel == 0, key0, key1), tl.where(channel == 2, key2, key3))
    row = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    col = tl.program_id(1) * TILE_COLS + tl.arange(0, TILE_COLS)
    index = row[:, None] * WIDTH + col[None, :]
    mask = row[:, None] < rows
    k = tl.load(key + index, mask, other=0).to(tl.float32)
    q = tl.load(query + (row[:, None] * CHANNELS + channel) * WIDTH + col[None, :], mask, other=0).to(tl.float32)
    offset = squares + channel * (2 * rows * WIDTH) + index
    tl.store(offset, k * k, mask)
    tl.store(offset + rows * WIDTH, q * q, mask)


@triton.jit(do_not_specialize=["rows"])
def _engram_qk_products(
    key0,
    key1,
    key2,
    key3,
    query,
    key_weight,
    query_weight,
    inv0,
    inv1,
    inv2,
    inv3,
    output,
    rows,
    WIDTH: tl.constexpr,
    CHANNELS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    channel = tl.program_id(2)
    key = tl.where(channel < 2, tl.where(channel == 0, key0, key1), tl.where(channel == 2, key2, key3))
    row = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    col = tl.program_id(1) * TILE_COLS + tl.arange(0, TILE_COLS)
    index = row[:, None] * WIDTH + col[None, :]
    mask = row[:, None] < rows
    k = tl.load(key + index, mask, other=0).to(tl.float32)
    q = tl.load(query + (row[:, None] * CHANNELS + channel) * WIDTH + col[None, :], mask, other=0).to(tl.float32)
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
        *keys,
        query,
        key_weight,
        query_weight,
        *inverses,
        products,
        rows,
        width,
        channels,
        tile_rows,
        tile_cols,
        enable_fp_fusion=False,
    )
    return tuple(channel.sum(-1) for channel in products.unbind(0))


@triton.jit(do_not_specialize=["tokens"])
def _apply_lookup_norm_with_inverse(
    rows,
    inverse,
    gamma,
    output,
    tokens,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    CORES: tl.constexpr,
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
    rows,
    gamma,
    output,
    groups,
    EPS: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    CORES: tl.constexpr,
):
    tiles = tl.cdiv(groups, BLOCK)
    tiles_per_core = tl.cdiv(tiles, CORES)
    start = tl.program_id(0) * tiles_per_core
    stop = tl.minimum(start + tiles_per_core, tiles)
    col = tl.arange(0, triton.next_power_of_2(WIDTH))
    group_lane = tl.arange(0, BLOCK)
    weight = tl.load(gamma + (group_lane[:, None] % HEADS) * WIDTH + col[None, :], col[None, :] < WIDTH, other=0)
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
        _apply_lookup_norm[(cores,)](rows, gamma, out, groups, eps, heads, width, block, cores, enable_fp_fusion=False)
    else:
        # The kernel spans all local heads per token. Bound its live tile as TP shrinks.
        block = min(16, max(1, 4096 // triton.next_power_of_2(heads * width)))
        cores = min(triton.cdiv(tokens, block), get_vectorcore_num())
        _apply_lookup_norm_with_inverse[(cores,)](
            rows, inverse, gamma, out, tokens, heads, width, block, cores, enable_fp_fusion=False
        )
    return out


def build_mla_logits(scores, candidates, sink, scale):
    """Build FP32 logits with the learned sink in the last column."""
    logits = (scores.float() * scale).masked_fill(candidates[:, None, :] < 0, -torch.inf)
    sink_logits = sink.view(1, -1, 1).expand(scores.shape[0], -1, -1)
    return torch.cat((logits, sink_logits), dim=-1)


def gate_dynamic_mx_quant(value: torch.Tensor, gate_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply BF16 sigmoid and multiplication before CANN's MXFP8 quantizer."""
    product = value * gate_logits.sigmoid()
    quantized, scale = torch_npu.npu_dynamic_mx_quant(
        product, dst_type=torch.float8_e4m3fn, scale_alg=DynamicMXQuantScaleAlg.MAX_EXPONENT
    )
    return quantized, scale.view(torch.uint8)


# Bound the live QK/PV intermediates to four existing attention chunks.
_SOFTMAX_BATCH_CHUNKS = 4
_MAX_GATHER_BYTES = 72 * 1024 * 1024


def _window_geometry(device, first, rows, history_start, width, group_rows):
    def build():
        positions = logical_positions(first, first + rows, device)
        offsets = torch.arange(1 - width, 1, device=device)
        selected = (positions[:, None] + offsets).clamp_min(-1)
        indices = selected - history_start if history_start else selected
        indices = indices.clamp_min(0).flatten()
        return tuple(zip(selected.split(group_rows), indices.split(group_rows * width)))

    return reuse_prefill_metadata(device, ("early_swa", first, rows, history_start, width, group_rows), build)


def attend_window_chunks(query, keys, first, history_start, width, sink, scale, latent_dim, chunk, dtype):
    """Keep each QK/PV matrix size and batch the independent softmax rows."""
    rows, heads, dim = query.shape
    keys = keys.float()
    chunk_bytes = chunk * width * dim * keys.element_size()
    group_chunks = min(_SOFTMAX_BATCH_CHUNKS, max(1, _MAX_GATHER_BYTES // chunk_bytes))
    group_rows = group_chunks * chunk
    geometry = _window_geometry(query.device, first, rows, history_start, width, group_rows)
    for group, (selected, indices) in enumerate(geometry):
        first_row = group * group_rows
        count = selected.size(0)
        gathered = keys.index_select(0, indices).view(count, width, dim)
        scores = torch.empty((count, heads, width), dtype=torch.float32, device=query.device)
        key_chunks = gathered.split(chunk)
        for j, (score_rows, selected_kv) in enumerate(zip(scores.split(chunk), key_chunks)):
            offset = j * chunk
            q = query[first_row + offset : first_row + offset + score_rows.size(0)]
            torch.bmm(q, selected_kv.transpose(1, 2), out=score_rows)
        logits = _build_window_logits(scores, selected, sink, scale)
        probabilities = logits.softmax(dim=-1)
        for j, selected_kv in enumerate(key_chunks):
            offset = j * chunk
            probs = probabilities[offset : offset + selected_kv.size(0), :, :-1]
            context = torch.bmm(probs, selected_kv[..., :latent_dim]).to(dtype)
            yield first_row + offset, context
