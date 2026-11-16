# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""BF16 query preparation for Mistral Large 4's FP8 Indexer."""

import torch
import triton
import triton.language as tl

from vllm_ascend.ops.triton.mistral_mla_q import _round_bf16
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

_HEADS = 32
_HEAD_DIM = 128
_ROPE_DIM = 64
_DECODE_HEAD_TILE = 4


@triton.jit(do_not_specialize=["tokens"])
def _ssmax_rope_and_pack(
    query,
    query_out,
    weights,
    weights_out,
    log_positions,
    scale_weight,
    scale_bias,
    cos,
    sin,
    tokens,
    COS_STRIDE: tl.constexpr,
    SIN_STRIDE: tl.constexpr,
    WEIGHT_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    PACK_WEIGHTS: tl.constexpr,
):
    head_tiles: tl.constexpr = HEADS // HEAD_TILE
    pair = tl.arange(0, ROPE_DIM // 2)
    tail = ROPE_DIM + tl.arange(0, HEAD_DIM - ROPE_DIM)
    for tile in tl.range(tl.program_id(0), tokens * head_tiles, tl.num_programs(0)):
        token = tile // head_tiles
        head_tile = tile % head_tiles
        heads = head_tile * HEAD_TILE + tl.arange(0, HEAD_TILE)
        row = (token * HEADS + heads[:, None]) * HEAD_DIM
        log_pos = _round_bf16(tl.load(log_positions + token).to(tl.float32))
        slope = _round_bf16(tl.load(scale_weight + heads).to(tl.float32))
        bias = _round_bf16(tl.load(scale_bias + heads).to(tl.float32))
        # Preserve the BF16 rounding after each original SSMax operation.
        scale = _round_bf16(_round_bf16(log_pos * slope) + bias)[:, None]

        left = row + 2 * pair[None, :]
        right = left + 1
        x = _round_bf16(tl.load(query + left).to(tl.float32) * scale)
        y = _round_bf16(tl.load(query + right).to(tl.float32) * scale)
        c = tl.load(cos + token * COS_STRIDE + pair).to(tl.float32)[None, :]
        s = tl.load(sin + token * SIN_STRIDE + pair).to(tl.float32)[None, :]
        tl.store(query_out + left, _round_bf16(x * c - y * s))
        tl.store(query_out + right, _round_bf16(y * c + x * s))
        offset = row + tail[None, :]
        tl.store(query_out + offset, _round_bf16(tl.load(query + offset).to(tl.float32) * scale))

        if PACK_WEIGHTS:
            # One tile owns the entire aligned 64-byte output row.
            if head_tile == 0:
                weight_columns = tl.arange(0, HEADS)
                values = tl.load(weights + token * WEIGHT_STRIDE + weight_columns)
                tl.store(weights_out + token * HEADS + weight_columns, values)


def try_fused_indexer_query(
    query: torch.Tensor,
    weights: torch.Tensor,
    log_positions: torch.Tensor,
    slope: torch.Tensor,
    bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Prepare the Indexer query and contiguous score weights on supported layouts."""
    if query.ndim != 3:
        return None
    tokens, heads, head_dim = query.shape
    pack_weights = not weights.is_contiguous()
    if (
        tokens == 0
        or (heads, head_dim) != (_HEADS, _HEAD_DIM)
        or query.device.type != "npu"
        or query.dtype != torch.bfloat16
        or not query.is_contiguous()
        or weights.shape != (tokens, _HEADS)
        or weights.dtype != torch.bfloat16
        or weights.device != query.device
        or (pack_weights and weights.stride() != (_HEAD_DIM + _HEADS, 1))
        or log_positions.shape != (tokens,)
        or slope.shape != (_HEADS,)
        or bias.shape != (_HEADS,)
        or not all(
            t.device == query.device and t.is_contiguous() and t.dtype in (torch.float32, torch.bfloat16)
            for t in (log_positions, slope, bias)
        )
        or not all(
            t is not None
            and t.ndim >= 2
            and t.shape[0] == tokens
            and t.shape[1:-1].numel() == 1
            and t.shape[-1] in (_ROPE_DIM // 2, _ROPE_DIM)
            and t.stride(-1) == 1
            and t.device == query.device
            and t.dtype == torch.bfloat16
            for t in (cos, sin)
        )
    ):
        return None

    query_out = torch.empty_like(query)
    weights_out = weights.new_empty((tokens, _HEADS)) if pack_weights else weights
    vector_cores = get_vectorcore_num()
    head_tile = _HEADS if tokens >= vector_cores else _DECODE_HEAD_TILE
    grid = (min(tokens * _HEADS // head_tile, vector_cores),)
    _ssmax_rope_and_pack[grid](
        query,
        query_out,
        weights,
        weights_out,
        log_positions,
        slope,
        bias,
        cos,
        sin,
        tokens,
        cos.stride(0),
        sin.stride(0),
        weights.stride(0),
        _HEADS,
        _HEAD_DIM,
        _ROPE_DIM,
        head_tile,
        pack_weights,
        enable_fp_fusion=False,
    )
    return query_out, weights_out
