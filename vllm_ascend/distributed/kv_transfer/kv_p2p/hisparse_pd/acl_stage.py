# SPDX-License-Identifier: Apache-2.0
"""PTA-owned stage and event-fenced copies with retained Host owners."""

from __future__ import annotations

import bisect
import importlib.util
import sys
import threading
from pathlib import Path

import torch

STAGE_BYTES = 16 * 1024 * 1024


class PTAStageBuffer:
    """A bounded tensor owner, retained until DMA and transport have drained.

    Single contiguous extents use Tensor.copy_. Discontiguous extents use one
    PTA OpCommand task whose small C++ compatibility boundary submits the
    installed CANN batch ABI. Host aliases come from existing retained tensors;
    this object never registers, unregisters or reconstructs a raw Host owner.
    """

    def __init__(self, device, set_device=None, capacity=STAGE_BYTES, *, host_owners=()):
        if type(device) is not int or device < 0 or type(capacity) is not int or capacity != STAGE_BYTES:
            raise ValueError("PTA stage requires a device and exactly16MiB capacity")
        self.device = device
        self.capacity = self.requested_bytes = capacity
        self.address = self.host_scratch = 0
        self.device_tensor = None
        self._reorder_tensor = self._descriptor_host = self._descriptor_device = None
        self.stream = self.completion = None
        self.allocation_range = None
        self.closed = self._started = self._ready = self._uncertain = False
        self._lock = threading.RLock()
        self._owner_cookie = id(self)
        self._pending_owners = []
        self._pending_copy = None
        self._borrowed_mappings = [
            owner
            for owner in host_owners
            if not isinstance(owner, torch.Tensor)
            and all(hasattr(owner, name) for name in ("address_of", "host_address", "nbytes", "closed", "device"))
        ]
        self._host_views = {}
        for owner in host_owners:
            if isinstance(owner, torch.Tensor) and owner.device.type == "cpu":
                if not owner.is_contiguous():
                    raise ValueError("Stage Host owners must be contiguous")
                view = owner.view(torch.uint8).reshape(-1)
                if view.numel():
                    self._host_views[(view.data_ptr(), view.numel())] = view
        self._ranges = sorted((start, start + count, view) for (start, count), view in self._host_views.items())
        self._starts = [item[0] for item in self._ranges]
        for first, second in zip(self._ranges, self._ranges[1:]):
            if first[1] > second[0]:
                raise ValueError("Ambiguous overlapping Host owners")
        self.operations = dict(
            host_to_device=0,
            device_to_host=0,
            tensor_copies=0,
            batch_submissions=0,
            pageable_tensor_fallbacks=0,
            packed_bytes=0,
            unpacked_bytes=0,
            reorder_submissions=0,
            reorder_bytes=0,
            reordered_original_extents=0,
            reordered_DMA_extents=0,
        )
        self.last_batch_status = None
        self.records = []

    @staticmethod
    def _extension():
        if not hasattr(torch.ops.hisparse_pta_host2511, "batch_copy"):
            torch.ops.load_library(str(Path(__file__).with_name("hisparse_pta_host2511.so")))
        return torch.ops.hisparse_pta_host2511

    def allocate(self):
        with self._lock:
            if self._started or self.closed:
                raise RuntimeError("PTA stage allocation cannot be repeated")
            self._started = True
            self._extension()  # Resolve the actual ABI before allocating.
            with torch.npu.device(self.device):
                self.device_tensor = torch.empty(self.capacity, dtype=torch.uint8, device=f"npu:{self.device}")
                self.address = self.device_tensor.data_ptr()
                storage = self.device_tensor.untyped_storage()
                self.allocation_range = dict(base=storage.data_ptr(), bytes=storage.nbytes(), allocator="PTA")
                if (
                    self.address <= 0
                    or self.address % 32
                    or self.address < storage.data_ptr()
                    or self.address + self.capacity > storage.data_ptr() + storage.nbytes()
                ):
                    raise RuntimeError("Stage is not an aligned view inside its PTA owner")
                self.stream = torch.npu.Stream(device=self.device)
                self.completion = torch.npu.Event()
                self.device_tensor.record_stream(self.stream)
                self._ready = True
                self.records.append(
                    dict(
                        operation="PTA tensor allocation",
                        bytes=self.capacity,
                        address=self.address,
                        host_scratch_allocated=False,
                    )
                )

    def _range(self, pointer, length):
        index = bisect.bisect_right(self._starts, pointer) - 1
        if index >= 0:
            start, end, view = self._ranges[index]
            if start <= pointer and pointer + length <= end:
                return start, end, view
        raise ValueError("Transfer descriptor escapes retained Host owners")

    def _resolve(self, pointer, length):
        start, _, view = self._range(pointer, length)
        return view[pointer - start : pointer - start + length]

    def _segments(self, segments):
        if self.closed or not self._ready or self._uncertain:
            raise RuntimeError("PTA stage is unavailable or has undrained work")
        if not isinstance(segments, (tuple, list)) or not 1 <= len(segments) <= 256:
            raise ValueError("Copy requires1..256 Host extents")
        total = 0
        ranges = []
        # Validate every original extent before coalescing or submitting work.
        for pointer, length in segments:
            if (
                type(pointer) is not int
                or type(length) is not int
                or pointer <= 0
                or length <= 0
                or pointer + length >= 2**63
            ):
                raise ValueError("Invalid Host extent")
            self._range(pointer, length)
            total += length
            ranges.append((pointer, length))
        if total > self.capacity:
            raise ValueError("Packed Host extents exceed16MiB")
        merged = []
        for pointer, length in ranges:
            if merged and merged[-1][0] + merged[-1][1] == pointer:
                start, previous = merged[-1]
                try:
                    self._range(start, previous + length)
                except ValueError:
                    pass
                else:
                    merged[-1] = (start, previous + length)
                    continue
            merged.append((pointer, length))
        return [(p, n, None) for p, n in merged], total

    def _batchable(self, views):
        for pointer, length, tensor in views:
            for mapping in self._borrowed_mappings:
                if (
                    not mapping.closed
                    and mapping.device == torch.device("npu", self.device)
                    and mapping.host_address <= pointer
                    and pointer + length <= mapping.host_address + mapping.nbytes
                ):
                    mapping.address_of(tensor)  # Check the current borrowed mapping.
                    break
            else:
                if not tensor.is_pinned():
                    return False
        return True

    def _reorder_plan(self, views):
        # Input/CPU owners were already validated. Only non-overlapping D2H
        # extents may be reordered, and only if physical adjacency reduces
        # the DMA extent count by at least four times.
        if len(views) < 8:
            return None
        offset = 0
        indexed = []
        for pointer, length, _ in views:
            indexed.append((pointer, length, offset))
            offset += length
        ordered = sorted(indexed)
        if any(p + n > q for (p, n, _), (q, _, _) in zip(ordered, ordered[1:])):
            return None
        groups = []
        for pointer, length, _ in ordered:
            if groups and groups[-1][0] + groups[-1][1] == pointer:
                start, previous = groups[-1]
                try:
                    self._range(start, previous + length)
                except ValueError:
                    pass
                else:
                    groups[-1] = (start, previous + length)
                    continue
            groups.append((pointer, length))
        if len(groups) * 4 > len(views):
            return None
        max_length = max(n for _, n, _ in indexed)
        max_tiles = (max_length + 4095) // 4096
        useful_tiles = sum((n + 4095) // 4096 for _, n, _ in indexed)
        if useful_tiles * 2 < len(views) * max_tiles:
            return None
        destination = 0
        descriptors = []
        for _, length, source in ordered:
            descriptors.append((source, destination, length))
            destination += length
        assert destination == offset <= self.capacity
        # No state mutation or GPU work has happened during this plan.
        merged = [(p, n, self._resolve(p, n)) for p, n in groups]
        return merged, descriptors, max_length

    @staticmethod
    def _reorder_extension():
        name = "hisparse_stage_reorder2581"
        module = sys.modules.get(name)
        if module is None:
            source = Path(__file__).with_name("stage_reorder2581.py")
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module

    def _submit_reorder(self, plan):
        merged, descriptors, max_length = plan
        if self._reorder_tensor is None:
            # Only this existing stream submits work. The preceding call's
            # completion was confirmed before these banks may be overwritten.
            self._reorder_tensor = torch.empty_like(self.device_tensor)
            self._descriptor_device = torch.empty((256, 3), dtype=torch.int64, device=self.device_tensor.device)
            self._descriptor_host = torch.empty((256, 3), dtype=torch.int64, device="cpu", pin_memory=True)
            assert self._descriptor_host.is_pinned()
            self._reorder_tensor.record_stream(self.stream)
            self._descriptor_device.record_stream(self.stream)
        self._pending_owners.extend((self._reorder_tensor, self._descriptor_device, self._descriptor_host))
        count = len(descriptors)
        self._descriptor_host[:count].copy_(torch.tensor(descriptors, dtype=torch.int64, device="cpu"))
        self._descriptor_device[:count].copy_(self._descriptor_host[:count], non_blocking=True)
        self._reorder_extension().reorder(
            self.device_tensor, self._reorder_tensor, self._descriptor_device, count, max_length
        )
        self.operations["reorder_submissions"] += 1
        self.operations["reorder_bytes"] += sum(n for _, n, _ in merged)
        self.operations["reordered_original_extents"] += count
        self.operations["reordered_DMA_extents"] += len(merged)
        return merged

    def _copy(self, segments, *, to_device, wait=True):
        views, total = self._segments(segments)
        if not to_device:
            # The packed legacy unpack executes serially. Keep that behavior
            # for overlapping destination views rather than batching races.
            destinations = sorted((pointer, pointer + length) for pointer, length, _ in views)
            overlap = any(end > following for (_, end), (following, _) in zip(destinations, destinations[1:]))
        else:
            overlap = False
        reorder_plan = self._reorder_plan(views) if not to_device and not overlap else None
        # One retained base per Host range covers all its validated fragments.
        # This also bounds the C++ owner's coverage search and detach work.
        bases = {}
        for pointer, length, _ in views:
            index = bisect.bisect_right(self._starts, pointer) - 1
            start, end, base = self._ranges[index]
            if pointer < start or pointer + length > end:
                raise ValueError("Transfer descriptor escapes retained Host owners")
            bases[index] = base
        self._pending_owners = [self.device_tensor, *bases.values()]
        self._uncertain = True
        ticket = None
        with torch.npu.device(self.device), torch.npu.stream(self.stream):
            transfer_tensor = self.device_tensor
            transfer_address = self.address
            # Validate the few merged destination views first. They cover all
            # the already-validated original extents, within the same retained
            # CPU owners. Keep the original per-fragment path when one merged
            # range is not covered by a live mapping or pinned allocation.
            reordered = reorder_plan is not None and self._batchable(reorder_plan[0])
            if reordered:
                views = self._submit_reorder(reorder_plan)
                transfer_tensor = self._reorder_tensor
                transfer_address = transfer_tensor.data_ptr()
                batchable = True
            else:
                views = [(p, n, self._resolve(p, n)) for p, n, _ in views]
                batchable = self._batchable(views)
            if len(views) == 1 or overlap or not batchable:
                if len(views) > 1 and not overlap and not batchable:
                    self.operations["pageable_tensor_fallbacks"] += 1
                offset = 0
                for _, length, host in views:
                    device = transfer_tensor[offset : offset + length]
                    if to_device:
                        device.copy_(host, non_blocking=True)
                    else:
                        host.copy_(device, non_blocking=True)
                    offset += length
                self.operations["tensor_copies"] += len(views)
            else:
                starts = []
                offset = 0
                for _, length, _ in views:
                    starts.append(transfer_address + offset)
                    offset += length
                hosts = [p for p, _, _ in views]
                source, destination = (hosts, starts) if to_device else (starts, hosts)
                # The C++ op snapshots descriptors and retains payloads through
                # the completion event, including when queue submission raises.
                ticket = self._extension().batch_copy(
                    torch.tensor(source, dtype=torch.int64, device="cpu"),
                    torch.tensor(destination, dtype=torch.int64, device="cpu"),
                    torch.tensor([n for _, n, _ in views], dtype=torch.int64, device="cpu"),
                    0 if to_device else 1,
                    self._pending_owners,
                    self.device,
                    self._owner_cookie,
                )
                self.operations["batch_submissions"] += 1
            self.completion.record(self.stream)
        token = object()
        self._pending_copy = (token, ticket, to_device)
        if wait:
            self._finish_copy(token)
        return token

    def _finish_copy(self, token):
        pending = self._pending_copy
        if pending is None or pending[0] is not token:
            raise ValueError("Copy completion requires its exact pending owner")
        _, ticket, to_device = pending
        # Leave all state retained if either completion check is uncertain.
        self.completion.synchronize()
        if ticket is not None:
            self.last_batch_status = self._extension().copy_status(ticket)
            if self.last_batch_status[4] != 0 or not self.last_batch_status[6]:
                raise RuntimeError("PTA batch submission completion remains unconfirmed")
            self._extension().release_copy(ticket, True, False)
        self._pending_owners.clear()
        self._uncertain = False
        self._pending_copy = None
        self.operations["host_to_device" if to_device else "device_to_host"] += 1

    def begin_copy_to_host_segments(self, segments):
        with self._lock:
            return self._copy(segments, to_device=False, wait=False)

    def finish_copy_to_host(self, token):
        with self._lock:
            self._finish_copy(token)

    def copy_from_host(self, source, length):
        self.copy_from_host_segments([(source, length)])

    def copy_to_host(self, destination, length):
        self.copy_to_host_segments([(destination, length)])

    def copy_from_host_segments(self, segments):
        with self._lock:
            self._copy(segments, to_device=True)

    def copy_to_host_segments(self, segments):
        with self._lock:
            self._copy(segments, to_device=False)

    def close(self, *, device_drained=False, transport_closed=False):
        with self._lock:
            if self.closed:
                return
            if device_drained is not True or transport_closed is not True:
                raise RuntimeError("Retain PTA owners until device drain and transport teardown")
            extension = self._extension()
            for ticket in extension.pending_copies(self._owner_cookie):
                extension.release_copy(ticket, True, True)
            self._ready = False
            self.completion = self.stream = None
            self._pending_owners.clear()
            self._pending_copy = None
            self.device_tensor = None
            self._reorder_tensor = self._descriptor_host = self._descriptor_device = None
            self._ranges.clear()
            self._starts.clear()
            self._host_views.clear()
            self._borrowed_mappings.clear()
            self.address = self.host_scratch = 0
            self._uncertain = False
            self.closed = True


ACLStageBuffer = PTAStageBuffer
