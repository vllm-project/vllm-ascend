# SPDX-License-Identifier: Apache-2.0
"""C8 compact copy with a bounded total number of work instances."""

from vllm.triton_utils import tl, triton


@triton.jit
def gather_compact_global_kernel(
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
    ROWS: tl.constexpr,
    TOP_K: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    HOT_BLOCK: tl.constexpr,
    HOT_STRIDE_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
    ATTENTION_BLOCK: tl.constexpr,
    COMPACT: tl.constexpr,
    HAS_ATTENTION: tl.constexpr,
    HAS_STATE: tl.constexpr,
):
    tl.static_assert(COMPACT)
    tl.static_assert(not HAS_ATTENTION)
    tl.static_assert(not HAS_STATE)
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    offset = tl.arange(0, BLOCK)
    host = host_address.to(tl.pointer_type(tl.uint8))
    # Iterate row boundaries rather than every masked TopK position. All
    # workers cooperate on the current row; no global scratch or barrier.
    for row in range(ROWS):
        limit = tl.minimum(TOP_K, tl.maximum(0, tl.load(counts + row)))
        for col in range(worker, limit, workers):
            index = row * TOP_K + col
            src = tl.load(source + index)
            dst = tl.load(destination + index)
            if (src >= 0) & (dst >= 0) & (dst < HOT_ROWS):
                destination_offset = dst.to(tl.int64) // HOT_BLOCK * HOT_STRIDE_BYTES + dst % HOT_BLOCK * ROW_BYTES
                for start in range(0, ROW_BYTES, BLOCK):
                    payload = tl.load(
                        host + src.to(tl.int64) * ROW_BYTES + start + offset,
                        mask=(src < HOST_ROWS) & (start + offset < ROW_BYTES),
                        other=0,
                    )
                    tl.store(hot + destination_offset + start + offset, payload, mask=start + offset < ROW_BYTES)
