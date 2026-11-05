# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mapped-Host refill, invalidation and packed KV layout operations.

Residency planning is provided by the native AscendC resolver.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def gather_rows_kernel(
    host_address,
    hot,
    source,
    destination,
    counts,
    mask_ptr,
    attention,
    state,
    HOST_ROWS: tl.constexpr,
    HOT_ROWS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    HOT_BLOCK: tl.constexpr,
    HOT_STRIDE_BYTES: tl.constexpr,
    TOP_K: tl.constexpr,
    ATTENTION_BLOCK: tl.constexpr,
    COMPACT: tl.constexpr,
    HAS_ATTENTION: tl.constexpr,
    HAS_STATE: tl.constexpr,
    BLOCK: tl.constexpr,
    PARALLEL_COLUMNS: tl.constexpr = 0,
):
    row = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    padding = tl.load(state + row) < 0 if HAS_STATE else False
    if PARALLEL_COLUMNS > 0:
        tl.static_assert(COMPACT and not HAS_ATTENTION)
        count = tl.minimum(TOP_K, tl.maximum(tl.load(counts + row), 0))
        # Empty compact plans do not inspect stale source/destination entries.
        # Keep enough independent programs for small misses and retain the
        # original contiguous byte-copy shape instead of a wide 2-D payload.
        if ~padding:
            for col in range(tl.program_id(1), count, PARALLEL_COLUMNS):
                index = row * TOP_K + col
                src = tl.load(source + index)
                dst = tl.load(destination + index)
                valid = (src >= 0) & (dst >= 0) & (dst < HOT_ROWS)
                if valid:
                    host = host_address.to(tl.pointer_type(tl.uint8))
                    destination_offset = dst.to(tl.int64) // HOT_BLOCK * HOT_STRIDE_BYTES + dst % HOT_BLOCK * ROW_BYTES
                    for start in range(0, ROW_BYTES, BLOCK):
                        payload = tl.load(
                            host + src.to(tl.int64) * ROW_BYTES + start + offset,
                            mask=(src < HOST_ROWS) & (start + offset < ROW_BYTES),
                            other=0,
                        )
                        tl.store(hot + destination_offset + start + offset, payload, mask=start + offset < ROW_BYTES)
    else:
        col = tl.program_id(1)
        index = row * TOP_K + col
        dst = tl.load(destination + index)
        if HAS_ATTENTION:
            value = tl.where((dst >= 0) & ~padding, dst // HOT_BLOCK * ATTENTION_BLOCK + dst % HOT_BLOCK, -1)
            tl.store(attention + index, value)
        if COMPACT:
            copy = col < tl.minimum(TOP_K, tl.maximum(tl.load(counts + row), 0))
        else:
            copy = tl.load(mask_ptr + index) != 0
        if copy & ~padding:
            src = tl.load(source + index)
            valid = (src >= 0) & (dst >= 0) & (dst < HOT_ROWS)
            if valid:
                host = host_address.to(tl.pointer_type(tl.uint8))
                destination_offset = dst.to(tl.int64) // HOT_BLOCK * HOT_STRIDE_BYTES + dst % HOT_BLOCK * ROW_BYTES
                for start in range(0, ROW_BYTES, BLOCK):
                    payload = tl.load(
                        host + src.to(tl.int64) * ROW_BYTES + start + offset,
                        mask=(src < HOST_ROWS) & (start + offset < ROW_BYTES),
                        other=0,
                    )
                    tl.store(hot + destination_offset + start + offset, payload, mask=start + offset < ROW_BYTES)


@triton.jit
def invalidate_kernel(
    dgi,
    states,
    requests,
    written,
    REQUESTS: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    REGION: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPED: tl.constexpr = False,
):
    if GROUPED:
        dgi = tl.load(dgi + tl.program_id(2)).to(tl.pointer_type(tl.int32))
    token = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(requests + token)
    state = tl.load(states + request, mask=(request >= 0) & (request < REQUESTS), other=-1)
    value = tl.load(written + token)
    valid = (state >= 0) & (state < STATE_ROWS) & (value >= 0) & (offset < REGION)
    ptr = dgi + state.to(tl.int64) * REGION + offset
    cached = tl.load(ptr, mask=valid, other=-1)
    tl.store(ptr, -1, mask=valid & (cached == value))


@triton.jit
def pack_attention_rows_kernel(
    cache,
    indices,
    latent,
    rope,
    K: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    LATENT: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    i = tl.arange(0, BLOCK)
    physical = tl.load(indices + row * INDEX_STRIDE + col)
    valid = (physical >= 0) & (physical < CACHE_ROWS)
    base = physical.to(tl.int64) // CACHE_BLOCK * CACHE_STRIDE + physical % CACHE_BLOCK * WIDTH
    value = tl.load(cache + base + i, mask=valid & (i < WIDTH), other=0)
    tl.store(latent + (row * K + col).to(tl.int64) * LATENT + i, value, mask=i < LATENT)
    tl.store(
        rope + (row * K + col).to(tl.int64) * (WIDTH - LATENT) + i - LATENT, value, mask=(i >= LATENT) & (i < WIDTH)
    )


@triton.jit
def scatter_packed_kv_kernel(
    latent,
    rope,
    cache,
    slots,
    valid_rows,
    CACHE_ROWS: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    LATENT: tl.constexpr,
    LATENT_STRIDE: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    HAS_VALID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    slot = tl.load(slots + row)
    valid = (slot >= 0) & (slot < CACHE_ROWS)
    if HAS_VALID:
        valid = valid & (row < tl.load(valid_rows))
    a = tl.load(latent + row.to(tl.int64) * LATENT_STRIDE + i, mask=valid & (i < LATENT), other=0)
    b = tl.load(rope + row.to(tl.int64) * ROPE_STRIDE + i - LATENT, mask=valid & (i >= LATENT) & (i < WIDTH), other=0)
    base = slot.to(tl.int64) // CACHE_BLOCK * CACHE_STRIDE + slot % CACHE_BLOCK * WIDTH
    tl.store(cache + base + i, tl.where(i < LATENT, a, b), mask=valid & (i < WIDTH))
