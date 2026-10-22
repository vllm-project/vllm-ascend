# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend registered-memory allocator for Mooncake encoder-cache transfer."""

from __future__ import annotations

import threading
import weakref
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
from vllm.utils.math_utils import round_up

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.bounce import (
    ASCEND_DIRECT_MEMORY_ALIGNMENT,
)


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


class AscendProducerAllocator(AscendContiguousAllocator):
    """One fixed registration; legacy bounce bytes are staging capacity too."""

    def __init__(self, staging_capacity: int, bounce_capacity: int) -> None:
        self.staging_capacity = staging_capacity + bounce_capacity
        self.registered_capacity = self.staging_capacity
        super().__init__(self.staging_capacity)

    def prepare(self, device: torch.device, transfer: MooncakeTransfer) -> None:
        super().prepare(device, transfer)
        if self.tensor is None:
            raise RuntimeError("Could not initialize the fixed Ascend Mooncake staging pool")


@dataclass
class AscendStagedSources(StagedSources):
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
        self._space_available = threading.Condition(self._lock)
        self._poisoned = False
        # GET results have independent cache tensors. These optional completed
        # copies can be borrowed by P2P or reclaimed immediately under pressure.
        self._loaded: dict[int, tuple[weakref.ReferenceType, AscendStagedSources]] = {}

    @property
    def capacity(self) -> int:
        return self._producer_allocator.staging_capacity

    def ensure_prepared(self, device: torch.device) -> None:
        """Ensure that the shared producer slab is allocated and registered."""
        with self._lock:
            self._producer_allocator.prepare(device, self._transfer)

    def _discard_loaded(self, key: int) -> None:
        with self._space_available:
            if self._poisoned:
                return
            cached = self._loaded.pop(key, None)
            if cached is not None:
                self._free_regions(cached[1].regions)
                self._space_available.notify_all()

    def cache_loaded(
        self,
        tensor: torch.Tensor,
        payload: torch.Tensor,
        header: torch.Tensor,
        region: tuple[int, int],
    ) -> None:
        """Take a completed GET lease; cache tensor owns independent storage."""
        with self._space_available:
            if self._poisoned:
                raise RuntimeError("Ascend producer staging pool is poisoned")
            key = id(tensor)
            self._loaded[key] = (
                weakref.ref(tensor),
                AscendStagedSources([payload], [region], [header]),
            )
            weakref.finalize(tensor, self._discard_loaded, key)

    def _allocate_bytes_locked(
        self,
        sizes: list[int],
        device: torch.device,
        *,
        wait: bool,
    ) -> list[tuple[int, int]] | None:
        if self._poisoned:
            raise RuntimeError("Ascend producer staging pool is poisoned")
        self._producer_allocator.prepare(device, self._transfer)
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
            # Completed GET staging is optional, never a reason to block.
            victim = next(iter(self._loaded), None)
            if victim is not None:
                _, staged = self._loaded.pop(victim)
                self._free_regions(staged.regions)
                continue
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
        if not tensors:
            return AscendStagedSources([], [], [])
        if len(headers) != len(tensors):
            raise ValueError("one optional header is required per source")
        copies: list[torch.Tensor] = []
        destinations: list[torch.Tensor] = []
        with self._space_available:
            if self._poisoned:
                raise RuntimeError("Ascend producer staging pool is poisoned")
            borrowed = {
                id(t): self._loaded[id(t)][1]
                for t in tensors
                if id(t) in self._loaded and self._loaded[id(t)][0]() is t
            }
            # Keep every selected cached lease out of pressure reclamation.
            missing = [i for i, t in enumerate(tensors) if id(t) not in borrowed]
            sizes = [tensors[i].nbytes + (headers[i].nbytes if headers[i] is not None else 0) for i in missing]
            borrowed_bytes = sum(size for item in borrowed.values() for _, size in item.regions)
            if borrowed_bytes + sum(round_up(n, self._allocator.alignment) for n in sizes) > self.capacity:
                # Cached GET includes header/padding; copying only the payload
                # can make an otherwise oversized combination fit.
                for key, cached in borrowed.items():
                    del self._loaded[key]
                    self._free_regions(cached.regions)
                borrowed = {}
                sizes = [t.nbytes + (h.nbytes if h is not None else 0) for t, h in zip(tensors, headers, strict=True)]
            taken = {key: self._loaded.pop(key) for key in borrowed}
            try:
                regions = self._allocate_bytes_locked(sizes, tensors[0].device, wait=wait)
            except BaseException:
                self._loaded.update(taken)
                raise
            if regions is None:
                self._loaded.update(taken)
                return None
            new_regions = iter(regions)
            result = AscendStagedSources([], list(regions), [])
            try:
                for tensor, header in zip(tensors, headers, strict=True):
                    cached = borrowed.get(id(tensor))
                    if cached is not None:
                        result.tensors.append(cached.tensors[0])
                        result.headers.append(cached.headers[0])
                        continue
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
                for key, cached in borrowed.items():
                    result.regions.extend(cached.regions)
            except BaseException:
                self._loaded.update(taken)
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
        return self.stage_with_headers(tensors, [None] * len(tensors), wait=False)

    def stage_wait(self, tensors: list[torch.Tensor]) -> AscendStagedSources | None:
        return self.stage_with_headers(tensors, [None] * len(tensors), wait=True)

    def release(self, staged: StagedSources) -> None:
        if not staged.regions:
            return
        with self._space_available:
            if self._poisoned:
                return
            self._free_regions(staged.regions)
            staged.regions.clear()
            self._space_available.notify_all()

    def poison(self) -> None:
        with self._space_available:
            self._poisoned = True
            self._space_available.notify_all()
