# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend registered-memory allocator for Mooncake encoder-cache transfer."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field

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
from vllm.logger import logger
from vllm.utils.math_utils import round_up

ASCEND_DIRECT_MEMORY_ALIGNMENT = 2 * 1024 * 1024


class AscendContiguousAllocator(ContiguousAllocator):
    """Allocate a 2 MiB-aligned registered-memory slab on NPU."""

    # Mypy checks this module with upstream imports skipped, so restate the
    # inherited attributes used by the Ascend override.
    tensor: torch.Tensor | None
    _disabled: bool

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


@dataclass
class AscendStagedSources(StagedSources):
    """Keep payload views, optional headers, and their shared slab regions."""

    headers: list[torch.Tensor | None] = field(default_factory=list)


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
        """Use the Ascend allocator with upstream receive-buffer bookkeeping."""
        super().__init__(capacity, transfer)
        self._allocator = allocator

    def _record_release_event(self) -> torch.Event | None:
        """Fence prior NPU stream work before a receive region can be reused."""
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
        allocator: AscendContiguousAllocator,
    ) -> None:
        # The exact #56242 baseline does not accept an injected allocator.
        """Initialize fixed staging storage and allocation waiters."""
        super().__init__(capacity, transfer)
        self._allocator = allocator
        self._space_available = threading.Condition(self._lock)
        self._poisoned = False

    @property
    def capacity(self) -> int:
        """Return the total producer slab capacity in bytes."""
        return self._allocator._capacity

    def ensure_prepared(self, device: torch.device) -> None:
        """Ensure that the shared producer slab is allocated and registered."""
        with self._lock:
            self._allocator.prepare(device, self._transfer)
            if self.tensor is None:
                raise RuntimeError("Could not initialize the fixed Ascend Mooncake pool")

    def _allocate_bytes_locked(
        self,
        sizes: list[int],
        device: torch.device,
        *,
        wait: bool,
    ) -> list[tuple[int, int]] | None:
        """Allocate all regions atomically, optionally waiting for free space."""
        if self._poisoned:
            raise RuntimeError("Ascend producer staging pool is poisoned")
        self._allocator.prepare(device, self._transfer)
        if self.tensor is None:
            raise RuntimeError("Could not initialize the fixed Ascend Mooncake pool")
        required = sum(round_up(size, self._allocator.alignment) for size in sizes)
        if required > self.capacity:
            return None
        while True:
            regions: list[tuple[int, int]] = []
            for size in sizes:
                region = self._allocator.allocate(size)
                if region is None:
                    self._free_regions(regions)
                    break
                regions.append(region)
            else:
                return regions
            if not wait:
                return None
            self._space_available.wait()
            if self._poisoned:
                raise RuntimeError("Ascend producer staging pool is poisoned")

    def reserve_bytes(
        self,
        sizes: list[int],
        device: torch.device,
        *,
        wait: bool = False,
    ) -> AscendStagedSources | None:
        """Atomically lease complete objects, including their Store headers."""
        with self._space_available:
            regions = self._allocate_bytes_locked(sizes, device, wait=wait)
            if regions is None:
                return None
            try:
                views = [
                    self._allocator.view(offset, size, (size,), torch.uint8)
                    for (offset, _), size in zip(regions, sizes, strict=True)
                ]
                return AscendStagedSources(views, regions)
            except BaseException:
                self._free_regions(regions)
                self._space_available.notify_all()
                raise

    def _copy_staged(self, tensors: list[torch.Tensor], staged: list[torch.Tensor]) -> None:
        """Copy sources on a thread-local stream and wait for copy completion."""
        if not tensors:
            return
        pool = self._allocator.tensor
        assert pool is not None
        stream = getattr(self._local, "stream", None)
        if stream is None:
            stream = torch.npu.Stream(device=pool.device)
            self._local.stream = stream
        with torch.npu.stream(stream):
            for destination, source in zip(staged, tensors, strict=True):
                destination.copy_(source, non_blocking=True)
        stream.synchronize()

    def stage_with_headers(
        self,
        tensors: list[torch.Tensor],
        headers: list[torch.Tensor | None],
        *,
        wait: bool = True,
    ) -> AscendStagedSources | None:
        """Lease and populate complete header/payload objects in one allocation."""
        if not tensors:
            return AscendStagedSources([], [], [])
        if len(headers) != len(tensors):
            raise ValueError("one optional header is required per source")
        copies: list[torch.Tensor] = []
        destinations: list[torch.Tensor] = []
        with self._space_available:
            if self._poisoned:
                raise RuntimeError("Ascend producer staging pool is poisoned")
            sizes = [t.nbytes + (h.nbytes if h is not None else 0) for t, h in zip(tensors, headers, strict=True)]
            regions = self._allocate_bytes_locked(sizes, tensors[0].device, wait=wait)
            if regions is None:
                return None
            new_regions = iter(regions)
            result = AscendStagedSources([], list(regions), [])
            try:
                for tensor, header in zip(tensors, headers, strict=True):
                    offset, _ = next(new_regions)
                    header_bytes = header.nbytes if header is not None else 0
                    header_view = (
                        self._allocator.view(offset, header_bytes, (header_bytes,), torch.uint8)
                        if header is not None
                        else None
                    )
                    payload = self._allocator.view(
                        offset + header_bytes, tensor.nbytes, tuple(tensor.shape), tensor.dtype
                    )
                    result.tensors.append(payload)
                    result.headers.append(header_view)
                    if header is not None:
                        copies.append(header)
                        destinations.append(header_view)
                    copies.append(tensor)
                    destinations.append(payload)
            except BaseException:
                self._free_regions(regions)
                self._space_available.notify_all()
                raise
        try:
            self._copy_staged(copies, destinations)
        except BaseException:
            # An asynchronous copy may still access both source and destination.
            self.poison()
            self._transfer.retain_owners([self, result, *copies])
            raise
        return result

    def stage(self, tensors: list[torch.Tensor]) -> AscendStagedSources | None:
        """Stage payloads immediately or return None when space is unavailable."""
        return self.stage_with_headers(tensors, [None] * len(tensors), wait=False)

    def stage_wait(self, tensors: list[torch.Tensor]) -> AscendStagedSources | None:
        """Wait for enough staging space to copy the complete payload batch."""
        return self.stage_with_headers(tensors, [None] * len(tensors), wait=True)

    def release(self, staged: StagedSources) -> None:
        """Recycle confirmed regions unless uncertain completion poisoned the pool."""
        if not staged.regions:
            return
        with self._space_available:
            if self._poisoned:
                return
            self._free_regions(staged.regions)
            staged.regions.clear()
            self._space_available.notify_all()

    def release_after(
        self,
        staged: StagedSources,
        futures: Sequence[Future[None]],
        *,
        retain: Callable[[], None],
        safe_errors: tuple[type[BaseException], ...] = (),
    ) -> None:
        """Release after all readers settle; retain once on uncertain completion.

        GET callers must settle their copy Event first. PUT callers may list
        confirmed operation failures in safe_errors; send failures are unsafe.
        Callbacks run on completing threads without acquiring staging space.
        """
        futures = tuple(futures)
        if not futures:
            self.release(staged)
            return
        lock = threading.Lock()
        remaining, uncertain = len(futures), False

        def completed(future: Future[None]) -> None:
            """Account for one reader and release only after all readers finish safely."""
            nonlocal remaining, uncertain
            try:
                error = future.exception()
            except BaseException as exc:
                error = exc
            if isinstance(error, safe_errors):
                error = None
            with lock:
                first_error = error is not None and not uncertain
                uncertain |= error is not None
                remaining -= 1
                reusable = remaining == 0 and not uncertain
            if first_error:
                retain()
                logger.error("Transfer completion is unconfirmed; retaining staging: %s", error)
            if reusable:
                try:
                    self.release(staged)
                except BaseException:
                    retain()
                    logger.exception("Failed to reclaim completed transfer; retaining staging")

        for future in futures:
            future.add_done_callback(completed)

    def poison(self) -> None:
        """Prevent slab reuse and wake allocation waiters after uncertain I/O."""
        with self._space_available:
            self._poisoned = True
            self._space_available.notify_all()
