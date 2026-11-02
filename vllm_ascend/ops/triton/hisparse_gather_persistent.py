# SPDX-License-Identifier: Apache-2.0
from vllm.triton_utils import tl, triton


@triton.jit
def gather_rows_persistent_kernel(
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
    TILE_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    padding = tl.load(state + row) < 0 if HAS_STATE else False
    limit = TOP_K
    count = TOP_K
    if COMPACT:
        count = tl.minimum(TOP_K, tl.maximum(tl.load(counts + row), 0))
        if not HAS_ATTENTION:
            limit = tl.where(padding, 0, count)
    for col in range(tl.program_id(1), limit, TILE_ROWS):
        index = row * TOP_K + col
        dst = tl.load(destination + index)
        if HAS_ATTENTION:
            value = tl.where((dst >= 0) & ~padding, dst // HOT_BLOCK * ATTENTION_BLOCK + dst % HOT_BLOCK, -1)
            tl.store(attention + index, value)
        if COMPACT:
            copy = col < count
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
