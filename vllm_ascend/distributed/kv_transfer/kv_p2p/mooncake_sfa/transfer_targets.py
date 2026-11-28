# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
import weakref
from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate

import torch


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.element_size() * tensor.numel())


@dataclass(frozen=True, eq=False)
class DSAPDMooncakeTransferTarget:
    """Local-only KV destination identity captured for one registry layout."""

    generation: int
    base_ptr: int
    nbytes: int
    tensor_ref: weakref.ReferenceType[torch.Tensor]
    offloaded: bool
    tensor_offset: int = 0


class _MooncakeRangeIndex:
    """Address-sorted ranges with their original registration positions."""

    def __init__(self, ranges: list[tuple[int, int]]) -> None:
        self.order = sorted(range(len(ranges)), key=lambda position: ranges[position][0])
        self.starts = [ranges[position][0] for position in self.order]
        self.ends = [ranges[position][0] + ranges[position][1] for position in self.order]
        # A predecessor alone is insufficient when registered ranges overlap.
        self.prefix_max_ends = list(accumulate(self.ends, max))


class MooncakeKVRegistry:
    """Validate borrowed KV views for one Mooncake worker, without allocating KV.

    The cache manager owns the storage. Weak tensor references and generation
    checks reject stale targets without extending an old cache's lifetime.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cpu_buffers: list[tuple[int, int]] = []
        self._offloaded_tensors: list[tuple[int, int, weakref.ReferenceType[torch.Tensor]]] = []
        self._offloaded_byte_views: dict[int, tuple[weakref.ReferenceType[torch.Tensor], torch.Tensor]] = {}
        self._cpu_buffer_index: _MooncakeRangeIndex | None = None
        self._offloaded_tensor_index: _MooncakeRangeIndex | None = None
        self._generation = 0

    def register_cpu_buffer(self, ptr: int, size: int) -> None:
        if ptr <= 0 or size <= 0:
            return
        entry = (int(ptr), int(size))
        with self._lock:
            if entry not in self._cpu_buffers:
                self._cpu_buffers.append(entry)
                self._cpu_buffer_index = None
                self._generation += 1

    def register_offloaded_tensor(self, tensor: torch.Tensor) -> None:
        ptr = int(tensor.data_ptr())
        size = _tensor_nbytes(tensor)
        if ptr <= 0 or size <= 0:
            return

        with self._lock:
            live_entries = []
            already_registered = False
            for start, length, tensor_ref in self._offloaded_tensors:
                existing = tensor_ref()
                if existing is None:
                    continue
                live_entries.append((start, length, tensor_ref))
                already_registered = already_registered or (start == ptr and length == size)
            if not already_registered:
                live_entries.append((ptr, size, weakref.ref(tensor)))
            if not already_registered or len(live_entries) != len(self._offloaded_tensors):
                self._generation += 1
            self._offloaded_tensors = live_entries
            self._offloaded_tensor_index = None

    def create_transfer_targets(
        self, tensor_layers: list[list[torch.Tensor]]
    ) -> list[list[DSAPDMooncakeTransferTarget]] | None:
        """Classify known KV tensors once, without inferring kinds from device.

        Ambiguous aliases and partial registrations retain the address-based
        path: resolving one whole tensor must not change subrange precedence.
        """
        with self._lock:
            live_entries = [
                (start, start + length, tensor)
                for start, length, tensor_ref in self._offloaded_tensors
                if (tensor := tensor_ref()) is not None
            ]
            result = []
            for tensors in tensor_layers:
                targets = []
                for tensor in tensors:
                    start = int(tensor.data_ptr())
                    length = _tensor_nbytes(tensor)
                    if start <= 0 or length <= 0:
                        return None
                    # Direct layouts may have gaps between blocks/heads.
                    span = int(tensor.element_size()) * (
                        1 + sum((dim - 1) * stride for dim, stride in zip(tensor.shape, tensor.stride()))
                    )
                    end = start + span
                    overlaps = [(a, b, t) for a, b, t in live_entries if a < end and start < b]
                    if overlaps:
                        if len(overlaps) != 1:
                            return None
                        base, limit, registered = overlaps[0]
                        if start < base or end > limit or span != length or not registered.is_contiguous():
                            return None
                        target = DSAPDMooncakeTransferTarget(
                            self._generation, start, length, weakref.ref(registered), True, start - base
                        )
                    else:
                        if any(a < end and start < a + size for a, size in self._cpu_buffers):
                            return None
                        target = DSAPDMooncakeTransferTarget(self._generation, start, span, weakref.ref(tensor), False)
                    targets.append(target)
                result.append(targets)
            return result

    def prepare_transfer_targets(
        self, targets: list[DSAPDMooncakeTransferTarget], ptrs: list[int], lengths: list[int]
    ) -> list[tuple[torch.Tensor, int] | None]:
        """Validate local descriptors and obtain views without address lookup."""
        if not len(targets) == len(ptrs) == len(lengths):
            raise ValueError("Mooncake transfer target count must match address and length counts.")
        with self._lock:
            prepared: dict[DSAPDMooncakeTransferTarget, tuple[torch.Tensor]] = {}
            result: list[tuple[torch.Tensor, int] | None] = []
            for target, ptr, length in zip(targets, ptrs, lengths):
                if target.generation != self._generation:
                    raise RuntimeError(
                        "Stale DSA Mooncake transfer targets; recreate the reader after KV layout changes."
                    )
                offset = int(ptr) - target.base_ptr
                if length < 0 or offset < 0 or offset + length > target.nbytes:
                    raise ValueError("Mooncake READ destination crosses its declared KV tensor boundary.")
                cached = prepared.get(target)
                if cached is None:
                    tensor = target.tensor_ref()
                    if tensor is None:
                        raise RuntimeError("DSA Mooncake transfer target tensor has been released.")
                    view = self._get_offloaded_byte_view(tensor) if target.offloaded else tensor
                    cached = (view,)
                    prepared[target] = cached
                result.append((cached[0], target.tensor_offset + offset) if target.offloaded else None)
            return result

    def clear(self) -> None:
        with self._lock:
            self._cpu_buffers.clear()
            self._offloaded_tensors.clear()
            self._offloaded_byte_views.clear()
            self._cpu_buffer_index = None
            self._offloaded_tensor_index = None
            self._generation += 1

    def resolve_offloaded_byte_tensor(self, ptr: int, size: int) -> tuple[torch.Tensor, int] | None:
        with self._lock:
            resolved = self.resolve_offloaded_tensor(ptr, size)
            if resolved is None:
                return None
            tensor, offset = resolved
            return self._get_offloaded_byte_view(tensor), offset

    def _get_offloaded_byte_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reuse a detached view while the caller holds the registry lock."""
        key = id(tensor)
        cached = self._offloaded_byte_views.get(key)
        if cached is None or cached[0]() is not tensor:
            # detach keeps storage alive without owning the registered Tensor.
            byte_view = tensor.view(torch.int8).reshape(-1).detach()

            def release(tensor_ref: weakref.ReferenceType[torch.Tensor]) -> None:
                with self._lock:
                    entry = self._offloaded_byte_views.get(key)
                    if entry is not None and entry[0] is tensor_ref:
                        self._offloaded_byte_views.pop(key)

            cached = (weakref.ref(tensor, release), byte_view)
            self._offloaded_byte_views[key] = cached
        return cached[1]

    def get_register_location(self, ptr: int, size: int) -> str | None:
        ptr = int(ptr)
        size = int(size)
        end = ptr + size
        with self._lock:
            if self._cpu_buffer_index is None:
                self._cpu_buffer_index = _MooncakeRangeIndex(self._cpu_buffers)
            index = self._cpu_buffer_index
            position = bisect_right(index.starts, ptr) - 1
            if position >= 0 and end <= index.prefix_max_ends[position]:
                return "cpu"
        return None

    def resolve_offloaded_tensor(
        self,
        ptr: int,
        size: int,
    ) -> tuple[torch.Tensor, int] | None:
        ptr = int(ptr)
        size = int(size)
        end = ptr + size
        resolved = None
        with self._lock:
            if self._offloaded_tensor_index is None:
                self._offloaded_tensor_index = _MooncakeRangeIndex(
                    [(start, length) for start, length, _ in self._offloaded_tensors]
                )
            index = self._offloaded_tensor_index
            position = bisect_right(index.starts, ptr) - 1
            first_registration = len(self._offloaded_tensors)
            # Disjoint ranges need at most one candidate check. Overlapping
            # ranges retain the first-registered live Tensor's precedence.
            while position >= 0 and index.prefix_max_ends[position] >= end:
                registration = index.order[position]
                if registration < first_registration and index.ends[position] >= end:
                    start, _, tensor_ref = self._offloaded_tensors[registration]
                    tensor = tensor_ref()
                    if tensor is not None:
                        resolved = (tensor, ptr - start)
                        first_registration = registration
                        if registration == 0:
                            break
                position -= 1
            # Dead weak references are pruned by registration or clear(), not
            # by rebuilding the entire address book on every lookup. The index
            # holds only integer metadata and never owns a Tensor.
        return resolved
