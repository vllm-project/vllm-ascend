# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mistral Large 4 MLA query preparation for BF16 decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _round_bf16(value):
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.to(tl.float32, bitcast=True)


@triton.jit
def _ssmax_split_rope(
    query,
    log_positions,
    weight,
    bias,
    cos,
    sin,
    nope,
    pe,
    HEADS: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    SIN_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    log_pos = _round_bf16(tl.load(log_positions + token).to(tl.float32))
    slope = _round_bf16(tl.load(weight + head).to(tl.float32))
    intercept = _round_bf16(tl.load(bias + head).to(tl.float32))
    # Keep the original BF16 rounding after each SSMax operation.
    scale = _round_bf16(_round_bf16(log_pos * slope) + intercept)

    source = query + row * 128
    lane = tl.arange(0, 64)
    content = _round_bf16(tl.load(source + lane).to(tl.float32) * scale)
    tl.store(nope + row * 64 + lane, content)

    pair = tl.arange(0, 32)
    even = _round_bf16(tl.load(source + 64 + 2 * pair).to(tl.float32) * scale)
    odd = _round_bf16(tl.load(source + 65 + 2 * pair).to(tl.float32) * scale)
    c = tl.load(cos + token * COS_STRIDE + pair).to(tl.float32)
    s = tl.load(sin + token * SIN_STRIDE + pair).to(tl.float32)
    # npu_interleave_rope consumes adjacent pairs and writes split halves.
    tl.store(pe + row * 64 + pair, _round_bf16(even * c - odd * s))
    tl.store(pe + row * 64 + 32 + pair, _round_bf16(odd * c + even * s))


def try_ssmax_split_rope(query, log_positions, weight, bias, cos, sin):
    """Return (q_nope, rotated q_pe), or None for the existing fallback."""
    if query.ndim != 3:
        return None
    tokens, heads, dim = query.shape
    if (
        query.device.type != "npu"
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
        or dim != 128
        or not 0 < tokens <= 8
        or not 0 < heads <= 64
        or log_positions.shape != (tokens,)
        or weight.shape != (heads,)
        or bias.shape != (heads,)
        or not all(x.device == query.device and x.is_contiguous() for x in (log_positions, weight, bias))
        or not all(
            x is not None
            and x.shape == (tokens, 1, 1, 64)
            and x.device == query.device
            and x.dtype == query.dtype
            and x.stride(-1) == 1
            for x in (cos, sin)
        )
    ):
        return None

    nope = query.new_empty((tokens, heads, 64))
    pe = torch.empty_like(nope)
    _ssmax_split_rope[(tokens * heads,)](
        query,
        log_positions,
        weight,
        bias,
        cos,
        sin,
        nope,
        pe,
        heads,
        cos.stride(0),
        sin.stride(0),
        enable_fp_fusion=False,
    )
    return nope, pe
