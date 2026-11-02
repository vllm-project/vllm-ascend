# SPDX-License-Identifier: Apache-2.0
"""Retain PTA copy tickets behind backend-owned fences on the caller's stream."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Generation:
    signature: tuple
    queue: Any
    pending: list[Any] = field(default_factory=list)
    failed: bool = False


class HostDMALifetime:
    """Own native tickets without extending the common HiSparse backend API.

    Device bindings are weak until submission, so profiling cache release does
    not retain an unused HBM slab. QueuedHostDMA retains independent byte aliases
    of the actual payload storage while a copy is pending. Polling only queries
    fences; blocking is restricted to unpin/explicit cold teardown.
    """

    def __init__(
        self,
        queue_factory: Callable | None = None,
        event_factory: Callable | None = None,
        current_stream: Callable | None = None,
    ) -> None:
        self._queue_factory = queue_factory
        self._event_factory = event_factory
        self._current_stream = current_stream
        self._hosts: dict[int, Any] = {}
        self._devices: dict[tuple, weakref.ReferenceType] = {}
        self._generations: list[_Generation] = []
        self._uncertain_owners: list[tuple] = []
        self._failed = False
        self._polling = False

    def retain_host(self, tensor) -> None:
        """Retain the exact registered extent, rather than allocator padding."""
        if id(tensor) not in self._hosts and self._hosts:
            # Registration is a cold boundary. Finish reads of a previous
            # resident slab before a replacement pool can start using it.
            self.drain()
        self._hosts[id(tensor)] = tensor

    def replace_device_owners(self, tensors) -> None:
        bindings = {}
        for tensor in tensors:
            while getattr(tensor, "_base", None) is not None:
                tensor = tensor._base
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            bindings[key] = weakref.ref(tensor, self._device_released)
        if not bindings:
            raise RuntimeError("HiSparse PTA DMA requires a real device allocation.")
        if self._devices and bindings.keys() != self._devices.keys():
            # Cache replacement may reuse the same Host addresses. Retaining
            # old storage alone would not order its writes before the new
            # generation; finish those fences at this cold allocation boundary.
            self.drain()
        else:
            self.poll()
        self._devices = bindings

    def _device_released(self, reference) -> None:
        # A profiling allocation can disappear without Host unpin. Query any
        # completed fences and discard their storage aliases before reallocation.
        try:
            self.poll()
        except Exception:
            # Never discard storage on an uncertain native/query failure. The
            # next explicit operation raises and teardown retains its owners.
            self._failed = True

    def _dependencies(self):
        if self._queue_factory is None:
            from .queued_dma import QueuedHostDMA

            self._queue_factory = QueuedHostDMA
        if self._event_factory is None or self._current_stream is None:
            import torch

            self._event_factory = self._event_factory or torch.npu.Event
            self._current_stream = self._current_stream or torch.npu.current_stream

    def _owners(self) -> tuple:
        devices = tuple(reference() for reference in self._devices.values())
        if not self._hosts or not devices or any(owner is None for owner in devices):
            raise RuntimeError("HiSparse DMA payload bindings expired or are incomplete.")
        return tuple(self._hosts.values()) + devices

    def submit(self, source, destination, sizes) -> None:
        self.poll()
        self._dependencies()
        owners = self._owners()
        signature = (tuple(self._hosts), tuple(self._devices))
        generation = next((value for value in self._generations if value.signature == signature), None)
        if generation is None:
            generation = _Generation(signature, self._queue_factory())
            self._generations.append(generation)
        event = self._event_factory()
        generation.pending.append(event)
        try:
            generation.queue.submit(source, destination, sizes, owners, event)
            event.record(self._current_stream())
        except BaseException:
            generation.failed = self._failed = True
            self._uncertain_owners.append(owners)
            raise

    def poll(self) -> None:
        if self._failed:
            raise RuntimeError("Uncertain PTA DMA; retain payload owners until process teardown.")
        if self._polling:
            return
        self._polling = True
        try:
            for generation in tuple(self._generations):
                while generation.pending and generation.pending[0].query():
                    generation.queue.release(generation.pending[0])
                    generation.pending.pop(0)
                if not generation.pending:
                    generation.queue.drain()
                    self._generations.remove(generation)
        except BaseException:
            self._failed = True
            raise
        finally:
            self._polling = False

    def drain(self) -> None:
        """Called only before actual Host unpin, after the common device drain."""
        if self._failed:
            raise RuntimeError("Cannot unpin an uncertain PTA DMA payload.")
        try:
            for generation in tuple(self._generations):
                if generation.failed:
                    raise RuntimeError("Cannot retire an unrecorded PTA completion fence.")
                for event in tuple(generation.pending):
                    if not event.query():
                        event.synchronize()
                    generation.queue.release(event)
                    generation.pending.pop(0)
                generation.queue.drain()
                self._generations.remove(generation)
        except BaseException:
            self._failed = True
            raise

    def release_host(self, tensor) -> None:
        if self._generations or self._failed:
            raise RuntimeError("Host ownership cannot retire before a successful DMA drain.")
        self._hosts.pop(id(tensor))
        if not self._hosts:
            self._devices.clear()
