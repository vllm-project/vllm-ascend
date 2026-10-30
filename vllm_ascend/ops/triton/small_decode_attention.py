# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged decode attention with device-resident lengths for short sequences."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

_SEQUENCE_TILE = 128
_NUM_SPLITS = 4
# Each task owns a complete 32-byte store, avoiding cross-core partial writes.
_STATS_STRIDE = 8


@triton.jit
def _decode_attention_partial(
    query_ptr,
    key_ptr,
    value_ptr,
    block_table_ptr,
    seq_lens_ptr,
    partial_ptr,
    stats_ptr,
    TABLE_STRIDE: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    TILE: tl.constexpr,
    SPLITS: tl.constexpr,
    KEY_BLOCK_STRIDE: tl.constexpr,
    VALUE_BLOCK_STRIDE: tl.constexpr,
    STATS_STRIDE: tl.constexpr,
):
    columns = tl.arange(0, HEAD_DIM)
    lanes = tl.arange(0, TILE)
    for task in range(tl.program_id(0), NUM_TOKENS * NUM_HEADS * SPLITS, tl.num_programs(0)):
        head_task = task // SPLITS
        split = task % SPLITS
        token = head_task // NUM_HEADS
        head = head_task % NUM_HEADS
        kv_head = head // (NUM_HEADS // NUM_KV_HEADS)
        seq_len = tl.load(seq_lens_ptr + token)
        query = tl.load(query_ptr + head_task * HEAD_DIM + columns).to(tl.float32)
        maximum = tl.full((), float("-inf"), tl.float32)
        denominator = tl.full((), 0.0, tl.float32)
        accumulator = tl.full((HEAD_DIM,), 0.0, tl.float32)
        # Interleave sequence tiles across splits, keeping softmax in FP32.
        for start in range(split * TILE, seq_len, TILE * SPLITS):
            positions = start + lanes
            block = tl.load(
                block_table_ptr + token * TABLE_STRIDE + positions // BLOCK_SIZE,
                mask=positions < seq_len,
                other=0,
            ).to(tl.int64)
            offsets = ((positions % BLOCK_SIZE) * NUM_KV_HEADS + kv_head)[:, None] * HEAD_DIM + columns[None, :]
            keys = tl.load(
                key_ptr + block[:, None] * KEY_BLOCK_STRIDE + offsets,
                mask=(positions < seq_len)[:, None],
                other=0,
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], 1) * SCALE
            scores = tl.where(positions < seq_len, scores, float("-inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, 0))
            correction = tl.exp(maximum - next_maximum)
            probabilities = tl.exp(scores - next_maximum)
            values = tl.load(
                value_ptr + block[:, None] * VALUE_BLOCK_STRIDE + offsets,
                mask=(positions < seq_len)[:, None],
                other=0,
            ).to(tl.float32)
            accumulator = accumulator * correction + tl.sum(values * probabilities[:, None], 0)
            denominator = denominator * correction + tl.sum(probabilities, 0)
            maximum = next_maximum
        tl.store(partial_ptr + task * HEAD_DIM + columns, accumulator)
        stats_lanes = tl.arange(0, STATS_STRIDE)
        tl.store(stats_ptr + task * STATS_STRIDE + stats_lanes, tl.where(stats_lanes == 0, maximum, denominator))


@triton.jit
def _decode_attention_reduce(
    partial_ptr,
    stats_ptr,
    output_ptr,
    NUM_TOKENS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    STATS_STRIDE: tl.constexpr,
):
    columns = tl.arange(0, HEAD_DIM)
    splits = tl.arange(0, SPLITS)
    for task in range(tl.program_id(0), NUM_TOKENS * NUM_HEADS, tl.num_programs(0)):
        base = task * SPLITS + splits
        maxima = tl.load(stats_ptr + base * STATS_STRIDE)
        denominators = tl.load(stats_ptr + base * STATS_STRIDE + 1)
        maximum = tl.max(maxima, 0)
        # Empty splits, including a wholly padded request, contribute zero.
        factors = tl.where(denominators > 0, tl.exp(maxima - maximum), 0.0)
        partials = tl.load(partial_ptr + base[:, None] * HEAD_DIM + columns[None, :])
        numerator = tl.sum(partials * factors[:, None], 0)
        denominator = tl.sum(denominators * factors, 0)
        tl.store(output_ptr + task * HEAD_DIM + columns, numerator / tl.maximum(denominator, 1e-20))


def small_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    output: torch.Tensor,
) -> None:
    """Write one-query-per-request attention into a contiguous output buffer.

    The caller gates short BF16 decode shapes. Cache inner dimensions are
    [block_size, kv_heads, head_dim]; the leading block stride may include
    the other K/V component or physical-page padding. Device length and table
    buffers must retain their addresses across graph replay.
    """
    num_tokens, num_heads, head_dim = query.shape
    _, block_size, num_kv_heads, _ = key_cache.shape
    partials = torch.empty((num_tokens, num_heads, _NUM_SPLITS, head_dim), device=query.device, dtype=torch.float32)
    stats = torch.empty((num_tokens, num_heads, _NUM_SPLITS, _STATS_STRIDE), device=query.device, dtype=torch.float32)
    num_cores = get_vectorcore_num()
    _decode_attention_partial[(min(num_cores, num_tokens * num_heads * _NUM_SPLITS),)](
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partials,
        stats,
        block_table.stride(0),
        num_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        scale,
        _SEQUENCE_TILE,
        _NUM_SPLITS,
        key_cache.stride(0),
        value_cache.stride(0),
        _STATS_STRIDE,
    )
    _decode_attention_reduce[(min(num_cores, num_tokens * num_heads),)](
        partials, stats, output, num_tokens, num_heads, head_dim, _NUM_SPLITS, _STATS_STRIDE
    )
