"""Inactive Ascend950 experiment: stable sparse-index compaction in one launch."""

from vllm.triton_utils import tl, triton


@triton.jit
def stable_compact_kernel(
    indices,
    selected,
    counts,
    ROW_STRIDE: tl.constexpr,
    TOP_K: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(indices + row.to(tl.int64) * ROW_STRIDE + col, mask=col < TOP_K, other=-1)
    valid = (col < TOP_K) & (value >= 0) & (value.to(tl.int64) < CAPACITY)
    prefix = tl.cumsum(valid.to(tl.int32), 0)
    count = tl.sum(valid.to(tl.int32), 0)
    # The two stores have disjoint destination sets: valid scatter owns
    # [0,count), and the tail fill owns [count,TOP_K). No initialization race.
    tl.store(selected + row.to(tl.int64) * TOP_K + prefix - 1, value, mask=valid)
    tl.store(selected + row.to(tl.int64) * TOP_K + col, -1, mask=(col >= count) & (col < TOP_K))
    tl.store(counts + row, count)
