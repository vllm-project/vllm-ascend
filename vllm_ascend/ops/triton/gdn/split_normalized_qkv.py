# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _split_normalized_qkv_kernel(
    packed,
    query,
    key,
    value,
    row_stride: tl.constexpr,
    column_stride: tl.constexpr,
    KEY_HEADS: tl.constexpr,
    VALUE_HEADS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    ROWS: tl.constexpr,
):
    tile = tl.program_id(0)
    token = tl.program_id(1)
    # A tile stays within one token so each component can be loaded in head order.
    key_tiles = tl.cdiv(KEY_HEADS, ROWS)
    packed += token * row_stride
    if tile < 2 * key_tiles:
        first_head = (tile % key_tiles) * ROWS
        heads = first_head + tl.arange(0, ROWS)
        cols = tl.arange(0, triton.next_power_of_2(KEY_DIM))
        index = heads[:, None] * KEY_DIM + cols[None, :]
        mask = (heads[:, None] < KEY_HEADS) & (cols[None, :] < KEY_DIM)
        source = (tile // key_tiles) * KEY_HEADS * KEY_DIM + index
        x = tl.load(packed + source * column_stride, mask, 0).to(tl.float32)
        y = x * tl.rsqrt(tl.sum(x * x, 1) + 1e-6)[:, None]
        target = tl.where(tile < key_tiles, query, key)
        tl.store(target + token * KEY_HEADS * KEY_DIM + index, y, mask)
    else:
        v_heads = (tile - 2 * key_tiles) * ROWS + tl.arange(0, ROWS)
        v_cols = tl.arange(0, triton.next_power_of_2(VALUE_DIM))
        v_index = v_heads[:, None] * VALUE_DIM + v_cols[None, :]
        v_mask = (v_heads[:, None] < VALUE_HEADS) & (v_cols[None, :] < VALUE_DIM)
        v = tl.load(packed + (2 * KEY_HEADS * KEY_DIM + v_index) * column_stride, v_mask, 0)
        tl.store(value + token * VALUE_HEADS * VALUE_DIM + v_index, v, v_mask)


def split_normalized_decode_qkv(
    packed: torch.Tensor,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack the convolution output for GDN verification and L2-normalize each Q/K head."""
    num_tokens = packed.shape[0]
    key_heads = key_dim // head_k_dim
    value_heads = value_dim // head_v_dim
    q = packed.new_empty((1, num_tokens, key_heads, head_k_dim))
    k = torch.empty_like(q)
    v = packed.new_empty((1, num_tokens, value_heads, head_v_dim))
    if num_tokens:
        rows = min(16, triton.next_power_of_2(max(key_heads, value_heads)))
        grid = (2 * triton.cdiv(key_heads, rows) + triton.cdiv(value_heads, rows), num_tokens)
        _split_normalized_qkv_kernel[grid](
            packed,
            q,
            k,
            v,
            packed.stride(0),
            packed.stride(1),
            key_heads,
            value_heads,
            head_k_dim,
            head_v_dim,
            ROWS=rows,
        )
    return q, k, v
