# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram hash kernels; model code owns buffers and launch scheduling."""

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens", "num_query_rows"])
def _engram_req_index_kernel(
    query_start_loc,
    req_ids,
    num_tokens,
    num_query_rows,
    query_stride,
    BLOCK_T: tl.constexpr,
):
    """Map each token to the request row whose chunk contains it.

    Split out of ``_hash_ids_kernel``: the Ascend bishengir backend aborts on
    the per-token search when it shares a kernel with the hash body
    ("LLVM ERROR: The buffer memory has been released"), while both halves
    compile on their own. Hoisting also runs the search once per token instead
    of once per (token, layer) program.
    """
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = token < num_tokens
    lo = tl.full((BLOCK_T,), 0, tl.int32)
    hi = tl.full((BLOCK_T,), num_query_rows, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(query_start_loc + (mid + 1) * query_stride, lo < hi, other=0)
        right = token >= end
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    tl.store(req_ids + token, tl.minimum(lo, num_query_rows - 1), mask=valid)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_slots",
        "num_table_rows",
        "max_blocks",
    ]
)
def _hash_ids_kernel(
    input_ids,
    token_map,
    dead_mask,
    positions,
    block_table,
    query_start_loc,
    req_ids,
    multipliers,
    primes,
    offsets,
    cache,
    lookback_token_ids,
    lookback_dead_mask,
    output,
    num_tokens,
    num_slots,
    pad_id,
    input_stride,
    mask_stride,
    position_stride,
    table_stride,
    table_col_stride,
    query_stride,
    num_table_rows,
    max_blocks,
    cache_block_size,
    MAX_NGRAM: tl.constexpr,
    num_heads,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    dead_id,
    lookback_depth,
    lookback_row_stride,
    lookback_col_stride,
    lookback_mask_row_stride,
    lookback_mask_col_stride,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    num_layers = tl.num_programs(1)
    valid = token < num_tokens
    # The request search lives in _engram_req_index_kernel (see its docstring).
    req = tl.load(req_ids + token, valid, other=0).to(tl.int64)
    chunk_idx = tl.load(query_start_loc + req * query_stride)
    chunk_idx = tl.minimum(chunk_idx, num_tokens - 1).to(tl.int64)
    chunk_start = tl.load(positions + chunk_idx * position_stride)
    position = tl.load(positions + token * position_stride, valid, other=0).to(tl.int64)
    head = tl.arange(0, BLOCK_H)
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    for shift in tl.static_range(MAX_NGRAM):
        lookback = position - shift
        in_batch = lookback >= chunk_start
        batch_idx = tl.maximum(token - shift, 0)
        batch_token = tl.load(input_ids + batch_idx * input_stride, valid & in_batch, other=0)
        batch_source = tl.load(token_map + batch_token, valid & in_batch, other=0)
        batch_dead = tl.load(dead_mask + batch_idx * mask_stride, valid & in_batch, other=False)
        batch_source = tl.where(batch_dead, dead_id, batch_source)

        col = chunk_start - 1 - lookback
        in_window = valid & ~in_batch & (col >= 0) & (col < lookback_depth)
        col = tl.minimum(tl.maximum(col, 0), lookback_depth - 1)
        window_token = tl.load(
            lookback_token_ids + req * lookback_row_stride + col * lookback_col_stride,
            in_window,
            other=-1,
        )
        known = in_window & (window_token >= 0)
        window_source = tl.load(token_map + window_token, known, other=0)
        window_dead = tl.load(
            lookback_dead_mask + req * lookback_mask_row_stride + col * lookback_mask_col_stride,
            known,
            other=False,
        )
        window_source = tl.where(window_dead, dead_id, window_source)

        if cache is not None:
            clamped = tl.minimum(tl.maximum(lookback, 0), max_blocks * cache_block_size - 1)
            block_row = tl.minimum(req, num_table_rows - 1)
            needs_cache = valid & ~in_batch & ~known
            block = tl.load(
                block_table + block_row * table_stride + (clamped // cache_block_size) * table_col_stride,
                needs_cache,
                other=0,
            ).to(tl.int64)
            slot = tl.minimum(
                tl.maximum(block * cache_block_size + clamped % cache_block_size, 0),
                num_slots - 1,
            )
            fallback = tl.load(cache + slot, needs_cache, other=0)
        else:
            fallback = tl.full((BLOCK_T,), pad_id, tl.int32)
        source = tl.where(in_batch, batch_source, tl.where(known, window_source, fallback)).to(tl.int64)
        blocked |= (lookback < 0) | (source == dead_id)
        value = tl.where(blocked, pad_id, source)
        multiplier = tl.load(multipliers + layer * MAX_NGRAM + shift)
        rolling ^= value * multiplier
        if shift > 0:
            col = (shift - 1) * num_heads + head
            param_offset = layer * (MAX_NGRAM - 1) * num_heads + col
            prime = tl.load(primes + param_offset, head < num_heads, other=1)
            offset = tl.load(offsets + param_offset, head < num_heads, other=0)
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            out_offset = (token.to(tl.int64) * num_layers + layer)[:, None] * ((MAX_NGRAM - 1) * num_heads) + col[
                None, :
            ]
            tl.store(output + out_offset, hashed, valid[:, None] & (head < num_heads))
