# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Batch recycled Host-slot invalidation within Core's worker lifecycle."""

import operator

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

logger = init_logger(__name__)
_BLOCK_BUCKET = 8
_TILE = 256
_INT32_MAX = 2**31 - 1
_JIT_POINTER_ALIGNMENT = 16


def _normalize_blocks(block_ids, block_size):
    try:
        blocks = sorted({operator.index(block) for block in block_ids})
    except TypeError:
        return None
    if not blocks or blocks[0] < 0 or blocks[-1] * block_size + block_size - 1 > _INT32_MAX:
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
    def __init__(self, tables, block_size):
        self._rank = get_tensor_model_parallel_rank()
        # Retain the bound tensors and their storage through queued invalidations.
        self._tables = tables
        self._addresses = tuple(table.data_ptr() for table in tables)
        self._device = tables[0].device
        self._states, self._hot = tables[0].shape
        self._block_size = block_size
        self._stream = torch.accelerator.current_stream()
        self._pointer_table = async_tensor_h2d(list(self._addresses), self._device, torch.int64)

    @classmethod
    def bind(cls, worker):
        kernel_config = getattr(worker.vllm_config, "kernel_config", None)
        if (
            not getattr(kernel_config, "enable_jit_warmup", True)
            or get_ascend_device_type() != AscendDeviceType.A5
            or worker.backend.is_current_stream_capturing()
            or worker.kernel_block_size <= 0
            or not worker.leader_runtimes
        ):
            return None
        tables = tuple(runtime.index_group.device_global_indices for runtime in worker.leader_runtimes)
        first = tables[0]
        if first.device.type != "npu" or first.ndim != 2 or not first.numel():
            return None
        if any(
            table.device != first.device
            or table.dtype != torch.int32
            or table.shape != first.shape
            or not table.is_contiguous()
            for table in tables
        ):
            return None
        ranges = sorted((table.data_ptr(), table.data_ptr() + table.numel() * table.element_size()) for table in tables)
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            return None
        owner = cls(tables, worker.kernel_block_size)
        # Compile the live ABI before serving. The device state sentinel masks
        # every table read/write while preserving the real dtype and geometry.
        dummy_states = async_tensor_h2d([-1], first.device, torch.int64)
        if dummy_states.data_ptr() % _JIT_POINTER_ALIGNMENT:
            return None
        owner._launch([0] * _BLOCK_BUCKET, dummy_states)
        logger.info(
            "HiSparse recycled invalidation prewarmed: rank=%d groups=%d state_dtype=int64 bucket=8",
            owner._rank,
            len(tables),
        )
        return owner

    def _launch(self, blocks, indices):
        device_blocks = async_tensor_h2d(blocks, self._device, torch.int32)
        grid = (len(self._tables), indices.numel(), triton.cdiv(self._hot, _TILE))
        _invalidate_recycled_groups[grid](
            self._pointer_table,
            device_blocks,
            indices,
            self._states,
            self._hot,
            _BLOCK_BUCKET,
            self._block_size,
            _BLOCK_BUCKET.bit_length(),
            _TILE,
        )

    def try_call(self, worker, block_ids, indices) -> bool:
        # The pinned caller passes unique, live scheduler indices, not the
        # resolver's padded state table. External callers retain Core errors.
        if (
            block_ids is not worker._pending_invalid_block_ids
            or worker.kernel_block_size != self._block_size
            or worker.backend.is_current_stream_capturing()
            or torch.accelerator.current_stream() != self._stream
            or indices.ndim != 1
            or indices.dtype != torch.int64
            or indices.device != self._device
            or not indices.is_contiguous()
            or indices.storage_offset() != 0
            or not 0 < indices.numel() <= self._states
            or indices.data_ptr() % _JIT_POINTER_ALIGNMENT
            or len(worker.leader_runtimes) != len(self._tables)
        ):
            return False
        for runtime, table, address in zip(worker.leader_runtimes, self._tables, self._addresses, strict=True):
            current = runtime.index_group.device_global_indices
            if (
                current is not table
                or current.data_ptr() != address
                or current.shape != (self._states, self._hot)
                or not current.is_contiguous()
            ):
                return False
        blocks = _normalize_blocks(block_ids, self._block_size)
        if blocks is None:
            return False
        for start in range(0, len(blocks), _BLOCK_BUCKET):
            chunk = blocks[start : start + _BLOCK_BUCKET]
            # Repeating the last ID preserves membership without a sentinel.
            chunk.extend([chunk[-1]] * (_BLOCK_BUCKET - len(chunk)))
            self._launch(chunk, indices)
        return True
