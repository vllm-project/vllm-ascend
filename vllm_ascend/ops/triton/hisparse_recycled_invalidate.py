# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Detached candidate: invalidate recycled Host blocks across index groups.

No installation on import. This preserves invalidation membership and the
caller's stream; it does not change LRU order, KV payloads or completion fences.
"""

import operator

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d

_INT32_MAX = 2**31 - 1
_MAX_KERNEL_TILE = 256


def normalized_blocks(block_ids, block_size):
    """None keeps the original path for unsupported block-ID domains."""
    if block_size <= 0:
        return None
    try:
        blocks = tuple(sorted({operator.index(block) for block in block_ids}))
    except TypeError:
        return None
    if blocks and (blocks[0] < 0 or blocks[-1] * block_size + block_size - 1 > _INT32_MAX):
        return None
    return blocks


@triton.jit
def _invalidate_recycled_groups(
    tables,
    blocks,
    state_indices,
    STATES: tl.constexpr,
    HOT: tl.constexpr,
    BLOCK_COUNT: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    ITERATIONS: tl.constexpr,
    TILE: tl.constexpr,
):
    table = tl.load(tables + tl.program_id(0)).to(tl.pointer_type(tl.int32))
    state = tl.load(state_indices + tl.program_id(1))
    slot = tl.program_id(2) * TILE + tl.arange(0, TILE)
    valid = (state >= 0) & (state < STATES) & (slot < HOT)
    value = tl.load(table + state * HOT + slot, mask=valid, other=-1)
    valid = valid & (value >= 0)
    page = tl.maximum(value, 0) // PAGE_ROWS
    first = tl.load(blocks)
    last = tl.load(blocks + BLOCK_COUNT - 1)
    valid = valid & (page >= first) & (page <= last)
    low = tl.full((TILE,), 0, tl.int32)
    high = tl.full((TILE,), BLOCK_COUNT, tl.int32)
    for _ in tl.static_range(ITERATIONS):
        active = valid & (low < high)
        middle = low + (high - low) // 2
        key = tl.load(blocks + middle, mask=active, other=0)
        advance = active & (key < page)
        high = tl.where(active & ~advance, middle, high)
        low = tl.where(advance, middle + 1, low)
    key = tl.load(blocks + low, mask=valid & (low < BLOCK_COUNT), other=0)
    tl.store(table + state * HOT + slot, -1, mask=valid & (low < BLOCK_COUNT) & (key == page))


class RecycledGroupInvalidator:
    """One bounded static pointer table; every call reads current state/data."""

    def __init__(self):
        self._binding = None
        self._owners = None
        self._table = None

    def try_call(self, leaders, block_ids, state_indices, block_size):
        if not block_ids:
            return True
        blocks = normalized_blocks(block_ids, block_size)
        if blocks is None or not leaders:
            return False
        if any(not getattr(leader.cache_ops, "use_fused_index_ops", False) for leader in leaders):
            return False
        tables = tuple(leader.index_group.device_global_indices for leader in leaders)
        first = tables[0]
        if not (
            first.device.type == "npu"
            and first.ndim == 2
            and first.dtype == torch.int32
            and state_indices.device == first.device
            and state_indices.ndim == 1
            and state_indices.dtype in (torch.int32, torch.int64)
            and state_indices.is_contiguous()
        ):
            return False
        if any(
            table.device != first.device
            or table.dtype != first.dtype
            or table.shape != first.shape
            or not table.is_contiguous()
            for table in tables
        ):
            return False
        ranges = sorted((table.data_ptr(), table.data_ptr() + table.numel() * table.element_size()) for table in tables)
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            return False
        if torch.npu.is_current_stream_capturing():
            return False
        if not state_indices.numel() or not first.shape[1]:
            return True
        binding = (first.device, tuple(first.shape), tuple((id(table), table.data_ptr()) for table in tables))
        if self._binding is None:
            self._owners = tables
            self._table = async_tensor_h2d([table.data_ptr() for table in tables], first.device, torch.int64)
            self._binding = binding
        elif self._binding != binding:
            # Retain old pointer owners until the worker drains; don't cache an
            # unbounded series of bindings or submit using obsolete addresses.
            return False
        current_blocks = async_tensor_h2d(blocks, first.device, torch.int32)
        grid = (len(tables), state_indices.numel(), triton.cdiv(first.shape[1], _MAX_KERNEL_TILE))
        _invalidate_recycled_groups[grid](
            self._table,
            current_blocks,
            state_indices,
            first.shape[0],
            first.shape[1],
            len(blocks),
            block_size,
            max(1, len(blocks).bit_length()),
            _MAX_KERNEL_TILE,
        )
        return True
