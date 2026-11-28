# SPDX-License-Identifier: Apache-2.0
"""Scatter one MLA component using its physical page stride."""

from vllm.triton_utils import tl, triton


@triton.jit
def _scatter_component(
    rows,
    slots,
    cache,
    block_size: tl.constexpr,
    block_stride: tl.constexpr,
    row_stride: tl.constexpr,
    width: tl.constexpr,
    capacity: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slots + token).to(tl.int64)
    if (slot >= 0) & (slot < capacity):
        cols = tl.arange(0, width)
        row = tl.load(rows + token * width + cols)
        offset = (slot // block_size) * block_stride + (slot % block_size) * row_stride
        tl.store(cache + offset + cols, row)


def scatter_component(rows, slots, cache):
    blocks, block_size, heads, width = cache.shape
    assert heads == 1 and cache.stride(-1) == 1 and rows.is_contiguous()
    _scatter_component[(rows.shape[0],)](
        rows, slots.contiguous(), cache, block_size, cache.stride(0), cache.stride(1), width, blocks * block_size
    )
