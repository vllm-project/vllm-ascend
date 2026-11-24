# SPDX-License-Identifier: Apache-2.0
"""Exact-layout helpers for dedicated HY4 Prefill."""

import torch
from vllm.triton_utils import tl, triton


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


@triton.jit(do_not_specialize=["tokens"])
def _engram_residual(
    hidden,
    hashes,
    gates,
    scales,
    signal,
    output,
    tokens,
    WIDTH: tl.constexpr,
    CHANNELS: tl.constexpr,
    HEADS: tl.constexpr,
    HASH_ROW: tl.constexpr,
    HASH_COL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    channel = tl.program_id(1)
    col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    head = tl.arange(0, triton.next_power_of_2(HEADS))
    hashes_in_row = tl.load(hashes + token * HASH_ROW + head * HASH_COL, head < HEADS, other=-1)
    active = tl.sum((hashes_in_row >= 0).to(tl.int32), 0) > 0
    gate = tl.load(gates + token * CHANNELS + channel).to(tl.float32)
    scale = tl.load(scales + channel * WIDTH + col, col < WIDTH, other=0)
    value = tl.load(signal + token * WIDTH + col, col < WIDTH, other=0)
    offset = (token * CHANNELS + channel) * WIDTH + col
    orig = tl.load(hidden + offset, col < WIDTH, other=0).to(tl.float32)
    dtype: tl.constexpr = hidden.dtype.element_ty
    residual = ((gate + scale) * value).to(dtype).to(tl.float32)
    tl.store(output + offset, (orig + tl.where(active, residual, 0.0)).to(dtype), col < WIDTH)


def engram_residual(hidden, hashes, gates, scales, signal):
    tokens, channels, width = hidden.shape
    output = torch.empty_like(hidden)
    block = triton.next_power_of_2(width)
    _engram_residual[(tokens, channels, triton.cdiv(width, block))](
        hidden,
        hashes,
        gates,
        scales,
        signal,
        output,
        tokens,
        width,
        channels,
        hashes.shape[1],
        *hashes.stride(),
        block,
        enable_fp_fusion=False,
    )
    return output


@triton.jit(do_not_specialize=["elements"])
def _mask_local_experts(expert_map, ids, weights, output, elements, EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < elements
    expert = tl.load(ids + index, valid, other=0).to(tl.int32)
    expert = tl.where(expert < 0, expert + EXPERTS, expert)
    local = tl.load(expert_map + expert, valid, other=-1)
    weight = tl.load(weights + index, valid, other=0)
    tl.store(output + index, weight * (local != -1).to(weight.dtype), valid)


def mask_local_experts(expert_map, ids, weights):
    output = torch.empty_like(weights)
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
