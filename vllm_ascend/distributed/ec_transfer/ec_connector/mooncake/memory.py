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
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)
from vllm.utils.math_utils import round_up

ASCEND_DIRECT_MEMORY_ALIGNMENT = 2 * 1024 * 1024  # 2 MiB


class AscendContiguousAllocator(ContiguousAllocator):
    """Allocate a 2 MiB-aligned registered-memory slab on NPU."""

    def _allocate_tensor(self, device: torch.device) -> torch.Tensor:
        """Allocate a 2 MiB-aligned registered-memory tensor on NPU."""
        raw_tensor = torch.empty(
            self._capacity + ASCEND_DIRECT_MEMORY_ALIGNMENT - 1,
            dtype=torch.uint8,
            device=device,
        )
        offset = (-raw_tensor.data_ptr()) % ASCEND_DIRECT_MEMORY_ALIGNMENT
        tensor = raw_tensor.narrow(0, offset, self._capacity)
        assert tensor.data_ptr() % ASCEND_DIRECT_MEMORY_ALIGNMENT == 0
        return tensor


class AscendProducerAllocator(AscendContiguousAllocator):
    """Own one registered slab partitioned into staging and bounce."""

    def __init__(
        self,
        staging_capacity: int,
        bounce_capacity: int,
    ) -> None:
        bounce_offset = round_up(
            staging_capacity,
            ASCEND_DIRECT_MEMORY_ALIGNMENT,
        )

        self.staging_capacity = staging_capacity
        self.bounce_offset = bounce_offset
        self.bounce_capacity = bounce_capacity
        self.registered_capacity = bounce_offset + bounce_capacity

        super().__init__(self.registered_capacity)

    @property
    def padding(self) -> int:
        return self.bounce_offset - self.staging_capacity

    @property
    def raw_allocation_size(self) -> int:
        return (
            self.registered_capacity
            + ASCEND_DIRECT_MEMORY_ALIGNMENT
            - 1
        )

    @property
    def bounce_tensor(self) -> torch.Tensor | None:
        tensor = self.tensor
        if tensor is None:
            return None

        return tensor.narrow(
            0,
            self.bounce_offset,
            self.bounce_capacity,
        )

    def prepare(
        self,
        device: torch.device,
        transfer: MooncakeTransfer,
    ) -> None:
        if self.tensor is not None:
            return

        super().prepare(device, transfer)

        if self.tensor is None:
            raise RuntimeError(
                "Could not initialize the Ascend Mooncake producer buffer: "
                f"staging={self.staging_capacity} bytes, "
                f"padding={self.padding} bytes, "
                f"bounce={self.bounce_capacity} bytes, "
                f"registered={self.registered_capacity} bytes, "
                f"allocation={self.raw_allocation_size} bytes"
            )

        self._free = [(0, self.staging_capacity)]


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
