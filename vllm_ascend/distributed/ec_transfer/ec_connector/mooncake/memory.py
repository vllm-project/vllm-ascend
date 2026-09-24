# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend registered-memory allocator for Mooncake encoder-cache transfer."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    ContiguousAllocator,
    ProducerMemoryPool,
    StagedSources,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    MooncakeTransfer,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

ASCEND_DIRECT_MEMORY_ALIGNMENT = 2 * 1024 * 1024  # 2 MiB

logger = init_logger(__name__)


@dataclass(frozen=True)
class _BounceLease:
    offset: int
    nbytes: int
    allocated_nbytes: int


class _BounceLeaseManager:
    """Manage complete wave leases over one existing bounce arena."""

    def __init__(self, capacity: int, alignment: int = 256) -> None:
        self.capacity = capacity
        self.alignment = alignment
        self._regions = ContiguousAllocator(capacity, alignment)
        self._condition = threading.Condition()
        self._waiters: deque[object] = deque()
        self._active: dict[int, int] = {}

    def acquire(self, nbytes: int) -> _BounceLease | None:
        if nbytes == 0:
            return None

        allocated_nbytes = round_up(nbytes, self.alignment)
        if allocated_nbytes > self.capacity:
            raise ValueError(
                f"bounce lease requires {allocated_nbytes} bytes but arena capacity is {self.capacity} bytes"
            )

        waiter = object()

        with self._condition:
            self._waiters.append(waiter)
            queued = True

            try:
                while True:
                    if self._waiters[0] is waiter:
                        region = self._regions.allocate(nbytes)

                        if region is not None:
                            offset, size = region
                            self._waiters.popleft()
                            queued = False
                            self._active[offset] = size
                            self._condition.notify_all()

                            return _BounceLease(
                                offset=offset,
                                nbytes=nbytes,
                                allocated_nbytes=size,
                            )
                    self._condition.wait()
            finally:
                if queued:
                    self._waiters.remove(waiter)
                    self._condition.notify_all()

    def release(self, lease: _BounceLease | None) -> None:
        if lease is None:
            return

        with self._condition:
            allocated_nbytes = self._active.get(lease.offset)

            if allocated_nbytes != lease.allocated_nbytes:
                raise ValueError("bounce lease is not active")

            del self._active[lease.offset]
            self._regions.free(
                lease.offset,
                lease.allocated_nbytes,
            )
            self._condition.notify_all()


class AscendContiguousAllocator(ContiguousAllocator):
    """Allocate a 2 MiB-aligned registered-memory slab on NPU."""

    def prepare(self, device: torch.device, transfer: MooncakeTransfer) -> None:
        """Prepare an aligned NPU slab without relying on upstream hooks."""
        if self.tensor is not None or self._disabled:
            return
        try:
            raw_tensor = torch.empty(
                self._capacity + ASCEND_DIRECT_MEMORY_ALIGNMENT - 1,
                dtype=torch.uint8,
                device=device,
            )
            offset = (-raw_tensor.data_ptr()) % ASCEND_DIRECT_MEMORY_ALIGNMENT
            tensor = raw_tensor.narrow(0, offset, self._capacity)
            assert tensor.data_ptr() % ASCEND_DIRECT_MEMORY_ALIGNMENT == 0
            ret = transfer.register_memory(tensor)
            if ret != 0:
                raise RuntimeError(f"Mooncake returned {ret}")
        except (RuntimeError, torch.OutOfMemoryError):
            self._disabled = True
            return
        self.tensor = tensor
        self._free = [(0, tensor.nbytes)]


class AscendProducerAllocator(AscendContiguousAllocator):
    """Own the single registered producer slab.

    Ascend direct registration requires a 2 MiB-aligned start address. A source
    tensor may begin before the first aligned address contained in its storage,
    so registering the tensor in place could cover memory that the storage does
    not own. The fallback path therefore splits such a source into an
    unregistrable prefix and an aligned suffix. It copies the prefix into a
    pre-registered bounce arena and registers the suffix directly.

    The producer normally allocates one registered slab containing both the
    primary staging pool and the shared bounce arena::

        +----------------------+-----------+---------------------+
        | staging pool         | alignment | shared bounce arena |
        |                      | padding   |                     |
        +----------------------+-----------+---------------------+

    Requests use the staging pool first. If a batch does not fit, transfer
    waves lease space from the shared bounce arena for their prefixes and use
    direct registration for their suffixes. Each fragment retains its
    destination offset, so Mooncake reconstructs the original tensor byte order
    in the consumer pool. A wave releases its bounce lease only after all
    writes using those bytes have finished.

    If allocating or registering the full slab fails, a producer with a
    nonzero bounce capacity retries once with the same allocator configured for
    a bounce-only slab::

        +---------------------+
        | shared bounce arena |
        +---------------------+

    In this degraded mode, the effective staging capacity and free list are
    empty. ProducerMemoryPool.stage() consequently returns None, routing every
    batch through the same direct-registration and bounced-prefix fallback.
    Reusing this allocator ensures that at most one producer slab is
    successfully registered. A zero bounce capacity disables both the fallback
    path and the bounce-only retry, so a staging failure remains fatal.
    """

    def __init__(
        self,
        staging_capacity: int,
        bounce_capacity: int,
    ) -> None:
        self.configured_staging_capacity = staging_capacity
        self.bounce_capacity = bounce_capacity
        self.fallback_only = staging_capacity == 0
        self._configure_layout(staging_capacity)

        super().__init__(self.registered_capacity)

    def _configure_layout(self, staging_capacity: int) -> None:
        bounce_offset = staging_capacity
        if self.bounce_capacity:
            bounce_offset = round_up(
                staging_capacity,
                ASCEND_DIRECT_MEMORY_ALIGNMENT,
            )

        self.staging_capacity = staging_capacity
        self.bounce_offset = bounce_offset
        self.registered_capacity = bounce_offset + self.bounce_capacity
        self._capacity = self.registered_capacity

    @property
    def padding(self) -> int:
        return self.bounce_offset - self.staging_capacity

    @property
    def raw_allocation_size(self) -> int:
        return self.registered_capacity + ASCEND_DIRECT_MEMORY_ALIGNMENT - 1

    @property
    def bounce_tensor(self) -> torch.Tensor | None:
        tensor = self.tensor
        if tensor is None or self.bounce_capacity == 0:
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

        if self.tensor is None and self.bounce_capacity and not self.fallback_only:
            logger.warning(
                "Could not initialize the full Ascend Mooncake producer "
                "buffer; retrying with a bounce-only buffer "
                "(staging=%d bytes, bounce=%d bytes)",
                self.configured_staging_capacity,
                self.bounce_capacity,
            )
            self.fallback_only = True
            self._configure_layout(0)
            self._disabled = False
            super().prepare(device, transfer)

        if self.tensor is None:
            raise RuntimeError(
                "Could not initialize the Ascend Mooncake producer buffer: "
                f"configured_staging={self.configured_staging_capacity} bytes, "
                f"effective_staging={self.staging_capacity} bytes, "
                f"padding={self.padding} bytes, "
                f"bounce={self.bounce_capacity} bytes, "
                f"registered={self.registered_capacity} bytes, "
                f"allocation={self.raw_allocation_size} bytes"
            )

        self._free = [(0, self.staging_capacity)] if self.staging_capacity else []


class AscendConsumerMemoryPool(ConsumerMemoryPool):
    """Defer NPU buffer reuse until preceding stream work completes."""

    def __init__(
        self,
        capacity: int,
        transfer: MooncakeTransfer,
        allocator: AscendContiguousAllocator,
    ) -> None:
        # The exact #56242 baseline does not accept an injected allocator.
        # Replace the unprepared default allocator immediately after upstream
        # initializes the pool's lifecycle bookkeeping.
        super().__init__(capacity, transfer)
        self._allocator = allocator

    def _record_release_event(self) -> torch.Event | None:
        pool = self.tensor
        if pool is None or pool.device.type != "npu":
            return None
        event = torch.npu.Event()
        event.record(torch.npu.current_stream(pool.device))
        return event


class AscendProducerMemoryPool(ProducerMemoryPool):
    """Copy producer tensors into registered memory on an NPU stream."""

    def __init__(
        self,
        capacity: int,
        transfer: MooncakeTransfer,
        allocator: AscendProducerAllocator,
    ) -> None:
        # The exact #56242 baseline does not accept an injected allocator.
        super().__init__(capacity, transfer)
        self._allocator = allocator
        self._producer_allocator = allocator
        self._bounce_lease_manager = _BounceLeaseManager(allocator.bounce_capacity)

    @property
    def bounce_tensor(self) -> torch.Tensor | None:
        return self._producer_allocator.bounce_tensor

    def acquire_bounce(self, nbytes: int) -> _BounceLease | None:
        return self._bounce_lease_manager.acquire(nbytes)

    def release_bounce(self, lease: _BounceLease | None) -> None:
        self._bounce_lease_manager.release(lease)

    def stage(self, tensors: list[torch.Tensor]) -> StagedSources | None:
        """Stage on an NPU stream without an upstream copy hook."""
        if not tensors:
            return StagedSources([], [])
        allocator = self._allocator
        staged: list[torch.Tensor] = []
        regions: list[tuple[int, int]] = []
        with self._lock:
            allocator.prepare(tensors[0].device, self._transfer)
            pool = allocator.tensor
            if pool is None:
                return None
            for tensor in tensors:
                region = allocator.allocate(tensor.nbytes)
                if region is None:
                    self._free_regions(regions)
                    return None
                regions.append(region)
                staged.append(allocator.view(region[0], tensor.nbytes, tuple(tensor.shape), tensor.dtype))
        stream = getattr(self._local, "stream", None)
        if stream is None:
            stream = torch.npu.Stream(device=pool.device)
            self._local.stream = stream

        # ProducerPushManager waits for the source's recorded NPU event before
        # submitting this copy to the I/O thread. Its current stream is not
        # necessarily the stream that produced the source tensors.
        with torch.npu.stream(stream):
            for destination, source in zip(staged, tensors):
                destination.copy_(source, non_blocking=True)

        stream.synchronize()
        return StagedSources(staged, regions)

    def copy_to_bounce(
        self,
        lease: _BounceLease,
        copies: list[tuple[torch.Tensor, int, int]],
    ) -> int:
        """Pack prefixes into a bounce lease and return its base address.

        Each copy is ``(source, offset within the lease, nbytes)``.
        """
        bounce = self.bounce_tensor
        if bounce is None:
            raise RuntimeError("Mooncake bounce arena is not prepared")
        assert copies

        stream = getattr(self._local, "stream", None)
        if stream is None:
            stream = torch.npu.Stream(device=bounce.device)
            self._local.stream = stream

        # The same source-readiness event also guards this fallback copy.
        with torch.npu.stream(stream):
            for source, bounce_offset, nbytes in copies:
                assert bounce_offset >= 0
                assert 0 < nbytes <= source.nbytes
                assert bounce_offset + nbytes <= lease.nbytes

                source_prefix = source.view(torch.uint8).view(-1).narrow(0, 0, nbytes)
                destination = bounce.narrow(
                    0,
                    lease.offset + bounce_offset,
                    nbytes,
                )
                destination.copy_(source_prefix, non_blocking=True)

        stream.synchronize()

        return bounce.data_ptr() + lease.offset
