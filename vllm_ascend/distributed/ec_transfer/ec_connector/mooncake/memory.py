# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend registered-memory allocator for Mooncake encoder-cache transfer."""

from __future__ import annotations

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    ContiguousAllocator,
    ProducerMemoryPool,
)

_REGISTERED_MEMORY_ALIGNMENT = 2 * 1024 * 1024  # 2MiB


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _aligned_storage_start(tensor: torch.Tensor) -> int | None:
    storage = tensor.untyped_storage()

    storage_start = storage.data_ptr()
    storage_end = storage_start + storage.nbytes()

    source_start = tensor.data_ptr()
    source_end = source_start + tensor.nbytes

    aligned_start = _align_up(
        storage_start,
        _REGISTERED_MEMORY_ALIGNMENT,
    )

    if source_start < aligned_start or source_end > storage_end:
        return None

    return aligned_start


def _allocate_aligned_tensor(capacity: int, device: torch.device) -> torch.Tensor:
    raw_tensor = torch.empty(
        capacity + _REGISTERED_MEMORY_ALIGNMENT - 1,
        dtype=torch.uint8,
        device=device,
    )
    offset = (-raw_tensor.data_ptr()) % _REGISTERED_MEMORY_ALIGNMENT
    tensor = raw_tensor.narrow(0, offset, capacity)
    assert tensor.data_ptr() % _REGISTERED_MEMORY_ALIGNMENT == 0
    return tensor


class AscendContiguousAllocator(ContiguousAllocator):
    """Allocate a 2 MiB-aligned registered-memory slab on NPU."""

    def _allocate_tensor(self, device: torch.device) -> torch.Tensor:
        """Allocate a 2 MiB-aligned registered-memory tensor on NPU."""
        return _allocate_aligned_tensor(self._capacity, device)


class AscendConsumerMemoryPool(ConsumerMemoryPool):
    """Defer NPU buffer reuse until preceding stream work completes."""

    def _record_release_event(self) -> torch.Event | None:
        pool = self.tensor
        if pool is None or pool.device.type != "npu":
            return None
        event = torch.npu.Event()
        event.record(torch.npu.current_stream(pool.device))
        return event


class AscendProducerMemoryPool(ProducerMemoryPool):
    """Copy producer tensors into registered memory on an NPU stream."""

    def _copy_to_staging(
        self,
        pool: torch.Tensor,
        staged: list[torch.Tensor],
        tensors: list[torch.Tensor],
    ) -> None:
        if pool.device.type != "npu":
            return super()._copy_to_staging(pool, staged, tensors)

        stream = getattr(self._local, "stream", None)
        if stream is None:
            stream = torch.npu.Stream(device=pool.device)
            self._local.stream = stream

        with torch.npu.stream(stream):
            for destination, source in zip(staged, tensors):
                destination.copy_(source, non_blocking=True)

        stream.synchronize()
