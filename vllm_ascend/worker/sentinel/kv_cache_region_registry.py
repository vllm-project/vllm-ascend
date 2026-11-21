#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Consume allocator-owned KV byte maps for UCE recovery."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from heapq import merge
from typing import Any

import torch
from vllm.logger import logger


@dataclass(frozen=True)
class BlockBytes:
    offset: int
    stride: int
    length: int
    count: int

    def interval(self, block: int) -> tuple[int, int]:
        """Return this block span as a half-open byte interval relative to the allocation."""
        start = self.offset + block * self.stride
        return start, start + self.length

    def intersect(self, start: int, end: int) -> range:
        """Return block IDs whose spans overlap the given allocation-relative byte interval."""
        if self.stride == 0:
            return range(self.count) if self.offset < end and start < self.offset + self.length else range(0)
        first = max(0, (start - self.offset - self.length) // self.stride + 1)
        last = min(self.count - 1, (end - 1 - self.offset) // self.stride)
        return range(first, last + 1)


@dataclass
class CacheAllocation:
    buffer: torch.Tensor
    num_blocks: int
    spans: list[BlockBytes] = field(default_factory=list)
    padding: list[tuple[int, int, int]] = field(default_factory=list)
    views: list[torch.Tensor] = field(default_factory=list)

    def owners(self, start: int, end: int) -> set[int]:
        """Return all block IDs owning bytes in the allocation-relative interval, including padding."""
        result = {block for span in self.spans for block in span.intersect(start, end)}
        result.update(block for lo, hi, block in self.padding if lo < end and start < hi)
        return result

    def ranges(self, blocks: set[int]) -> list[tuple[int, int]]:
        """Return merged allocation-relative byte intervals that clearing the given blocks would modify."""
        return merge_ranges(
            [span.interval(block) for span in self.spans for block in blocks]
            + [(lo, hi) for lo, hi, block in self.padding if block in blocks]
        )

    def seal(self) -> None:
        """Assign holes/alignment slack to a neighboring block, never omit bytes.

        Padding has no tensor consumer. Its owner is deliberately conservative;
        alias consumers of data bytes are represented by every span, not deduped.
        """
        if not self.spans:
            raise ValueError("KV allocation has no registered block storage")
        cursor = 0
        previous = 0

        def intervals_for(span):
            return ((*span.interval(block), block) for block in range(span.count))

        # Stream the sorted ownership intervals instead of materializing one
        # Python tuple for every block of every cache plane at startup.
        intervals = merge(*(intervals_for(span) for span in self.spans))
        self.padding.clear()
        for start, end, block in intervals:
            if start > cursor:
                self.padding.append((cursor, start, block))
            if end > cursor:
                cursor, previous = end, block
        if cursor < self.buffer.numel():
            self.padding.append((cursor, self.buffer.numel(), previous))


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent half-open byte intervals."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
    return merged


class CacheAllocationMap:
    """Explicit storage contract built by the KV allocator, before serving."""

    def __init__(self, num_blocks: int):
        if num_blocks <= 0:
            raise ValueError("KV allocation requires a positive scheduler block count")
        self.num_blocks = num_blocks
        self.allocations: dict[tuple[str, int], CacheAllocation] = {}

    def own(self, raw: torch.Tensor) -> None:
        """Declare exclusive KV ownership of an allocator-created backing storage.

        Only dedicated cache buffers may be passed, never arbitrary tensor views
        into weights/metadata. All aliases of the same backing share one record.
        """
        storage = raw.untyped_storage()
        key = (str(raw.device), storage.data_ptr())
        if key not in self.allocations:
            buffer = torch.empty(0, dtype=torch.uint8, device=raw.device)
            buffer.set_(storage, 0, (storage.nbytes(),), (1,))
            self.allocations[key] = CacheAllocation(buffer, self.num_blocks)
        self.allocations[key].views.append(raw)

    def block_bytes(
        self, raw: torch.Tensor, *, offset_bytes: int, block_stride_bytes: int, block_size_bytes: int
    ) -> None:
        """Register allocator-provided physical geometry, relative to storage.

        This is the layout-independent entry point. An allocator using an opaque
        device format must supply its byte geometry here instead of tensor strides.
        Multiple calls describe K/V/state/scale planes and overlapping views.
        """
        key = (str(raw.device), raw.untyped_storage().data_ptr())
        allocation = self.allocations.get(key)
        if allocation is None:
            raise ValueError("KV block bytes do not belong to an owned allocation")
        span = BlockBytes(offset_bytes, block_stride_bytes, block_size_bytes, self.num_blocks)
        if (
            offset_bytes < 0
            or block_stride_bytes < 0
            or block_size_bytes <= 0
            or span.interval(self.num_blocks - 1)[1] > allocation.buffer.numel()
        ):
            raise ValueError("Invalid KV block byte geometry")
        if span not in allocation.spans:
            allocation.spans.append(span)
        allocation.views.append(raw)
        raw._hbm_allocation_map = self

    def block_view(self, view: torch.Tensor, *, rows_per_block: int, block_axis: int = 0) -> None:
        """Declare an allocator's block-indexed view; record byte geometry once.

        The caller supplies the block axis and row grouping used by its kernel.
        Inner-axis order and element dtype have no recovery-time significance.
        A bounding interval may include holes; alias closure accounts for their
        other owners before any byte is zeroed.
        """
        if view.ndim < 1:
            raise ValueError("KV block view requires a block dimension")
        block_axis %= view.ndim
        storage = view.untyped_storage()
        key = (str(view.device), storage.data_ptr())
        allocation = self.allocations.get(key)
        if allocation is None:
            raise ValueError("KV view does not alias an allocator-owned storage")
        if view.device.type == "npu":
            import torch_npu

            if torch_npu.get_npu_format(view) != 2:
                raise ValueError("Opaque KV format requires allocator-provided block_bytes geometry")
        if rows_per_block <= 0 or view.shape[block_axis] != self.num_blocks * rows_per_block:
            raise ValueError("KV view does not cover the declared scheduler block space")
        if any(step < 0 for step in view.stride()):
            raise ValueError("KV allocation requires nonnegative byte strides")
        elem = view.element_size()
        stride = view.stride(block_axis) * rows_per_block * elem
        length = (
            1
            + (rows_per_block - 1) * view.stride(block_axis)
            + sum(
                (size - 1) * step
                for axis, (size, step) in enumerate(zip(view.shape, view.stride()))
                if axis != block_axis
            )
        ) * elem
        offset = view.data_ptr() - allocation.buffer.data_ptr()
        self.block_bytes(view, offset_bytes=offset, block_stride_bytes=stride, block_size_bytes=length)

    def seal(self) -> None:
        for allocation in self.allocations.values():
            allocation.seal()


def register_kv_allocations(raw_tensors, kv_caches, num_blocks: int) -> None:
    """Adapter for V1/V2's block-major reshape output contract.

    These allocators materialize exactly num_blocks logical pages per layer;
    splitting/replication produces consecutive rows within each logical page.
    They supply the views while they still own all raw buffers, including scale
    planes and shared slots. No KVCacheSpec/token-compression/model whitelist.
    Other allocators can call own()/block_view() with an explicit different axis.
    """

    def tensors(value):
        if isinstance(value, torch.Tensor):
            if value.numel():
                yield value
        elif isinstance(value, dict):
            for part in value.values():
                yield from tensors(part)
        elif isinstance(value, (tuple, list)):
            for part in value:
                yield from tensors(part)
        elif value is not None:
            raise ValueError("Allocator returned unregistered KV storage container")

    mapping = CacheAllocationMap(num_blocks)
    views = list(tensors(kv_caches))
    for raw in tensors(raw_tensors):
        mapping.own(raw)
    for view in views:
        if view.ndim < 1 or view.shape[0] % num_blocks:
            raise ValueError("Block-major allocator returned a non-integral page grouping")
        mapping.block_view(view, rows_per_block=view.shape[0] // num_blocks)
    mapping.seal()


@dataclass(frozen=True)
class KvCacheRegion:
    """One contiguous KV-cache allocation (layer tensor or K/V split)."""

    device_id: int
    rank: int
    layer_name: str
    kind: str
    data_ptr: int
    nbytes: int

    def contains(self, addr: int) -> bool:
        return self.data_ptr <= addr < self.data_ptr + self.nbytes

    def contains_range(self, start: int, size: int) -> bool:
        return self.data_ptr <= start and start + size <= self.data_ptr + self.nbytes


class KvCacheRegionRegistry:
    """Bind allocation-time ownership records; never infer layout during retry."""

    def __init__(self) -> None:
        self.regions: list[KvCacheRegion] = []
        self._tensors: list[torch.Tensor] = []
        self._allocations = []
        self.uce_ranges: list[tuple[int, int]] = []
        self.num_blocks: int | None = None

    def clear(self) -> None:
        self.regions.clear()
        self._tensors.clear()
        self._allocations.clear()
        self.uce_ranges.clear()
        self.num_blocks = None

    def register(self, kv_caches: Any, rank: int = 0, device_id: int | None = None) -> None:
        """Bind allocator-owned storage maps and reject incomplete or conflicting ownership."""
        self.clear()
        mappings = {}
        for layer, kind, tensor in _iter_kv_tensors(kv_caches):
            if not tensor.numel():
                continue
            mapping = getattr(tensor, "_hbm_allocation_map", None)
            if mapping is None:
                raise ValueError(f"KV allocator did not register {layer}/{kind}")
            mappings[id(mapping)] = mapping
        seen = set()
        for mapping in mappings.values():
            if self.num_blocks is not None and self.num_blocks != mapping.num_blocks:
                raise ValueError("KV allocations disagree on scheduler block count")
            self.num_blocks = mapping.num_blocks
            for allocation in mapping.allocations.values():
                tensor = allocation.buffer
                key = (str(tensor.device), tensor.data_ptr())
                if key in seen:
                    raise ValueError("Shared KV storage must have a single complete ownership map")
                seen.add(key)
                dev = device_id if device_id is not None else tensor.device.index
                self.regions.append(
                    KvCacheRegion(
                        device_id=dev if dev is not None else -1,
                        rank=rank,
                        layer_name=f"allocation_{len(self.regions)}",
                        kind="backing",
                        data_ptr=tensor.data_ptr(),
                        nbytes=tensor.numel(),
                    )
                )
                self._tensors.append(tensor)
                self._allocations.append(allocation)
        logger.info("Registered %d KV backing allocations for UCE recovery (rank=%s)", len(self.regions), rank)

    def lookup(self, addr: int) -> KvCacheRegion | None:
        return next((region for region in self.regions if region.contains(addr)), None)

    def _covered_allocations(self, start: int, size: int) -> list[int]:
        """Return region indices fully covering the UCE address range, rejecting any gap outside KV memory."""
        if size <= 0:
            raise ValueError("UCE range must have positive size")
        cursor = start
        matches = []
        for index in sorted(range(len(self.regions)), key=lambda i: self.regions[i].data_ptr):
            region = self.regions[index]
            lo, hi = region.data_ptr, region.data_ptr + region.nbytes
            if hi <= cursor or lo >= start + size:
                continue
            if lo > cursor:
                break
            matches.append(index)
            cursor = min(start + size, hi)
            if cursor == start + size:
                return matches
        raise ValueError("UCE range includes memory outside owned KV allocations")

    def recover_uce_addresses(self, addr_entries: list[dict]) -> KvCacheRegion | None:
        """Validate all UCE ranges, save them, mark matched tensors safe, and return one representative region."""
        # Validate every range, including multi-allocation faults, before making
        # any storage safe. A range crossing a non-KV gap is not recoverable.
        matches = set()
        ranges = []
        try:
            for entry in addr_entries:
                start, size = int(entry["ptr"]), int(entry["size"])
                matches.update(self._covered_allocations(start, size))
                ranges.append((start, size))
        except (KeyError, TypeError, ValueError):
            return None
        if not matches:
            return None
        self.uce_ranges = ranges
        for index in sorted(matches):
            allocation = self._allocations[index]
            for tensor in [allocation.buffer, *allocation.views]:
                self._update_tensor_to_safe(tensor, self.regions[index])
        return self.regions[min(matches)]

    @staticmethod
    def _update_tensor_to_safe(tensor: torch.Tensor, region: KvCacheRegion) -> None:
        """Mark a KV tensor safe through torch_npu recovery without clearing its contents."""
        import torch_npu

        torch_npu.npu._recovery.update_npu_tensor_to_safe(tensor)
        logger.debug("Marked KV storage alias safe: allocation=%s rank=%s", region.layer_name, region.rank)

    def refresh_pointers(self) -> None:
        """Reject allocations whose address or size changed since KV region registration."""
        # A reset must preserve storage identity for retained prefixes. Moving
        # allocations without rebuilding the ownership map cannot be accepted.
        for region, tensor in zip(self.regions, self._tensors):
            if tensor.data_ptr() != region.data_ptr or tensor.numel() != region.nbytes:
                raise RuntimeError("KV allocation changed during device recovery")

    def validate_block_space(self, num_blocks: int) -> None:
        """Validate the requested block count against the registered layout and check storage identity."""
        if not self._allocations or self.num_blocks != num_blocks:
            raise RuntimeError("KV allocation and scheduler block spaces differ")
        self.refresh_pointers()

    def partial_block_ids(self, num_blocks: int) -> set[int]:
        """Map saved UCE address ranges to local block IDs and include all byte-sharing aliases."""
        self.validate_block_space(num_blocks)
        blocks = set()
        for start, size in self.uce_ranges:
            for index in self._covered_allocations(start, size):
                region = self.regions[index]
                lo = max(0, start - region.data_ptr)
                hi = min(region.nbytes, start + size - region.data_ptr)
                blocks.update(self._allocations[index].owners(lo, hi))
        return self.expand_aliases(blocks)

    def expand_aliases(self, blocks: set[int]) -> set[int]:
        """Include every block whose bytes will be changed by this clear plan."""
        if any(type(block) is not int or not 0 <= block < self.num_blocks for block in blocks):
            raise ValueError("Invalid scheduler block ID in KV clear plan")
        result = set(blocks)
        frontier = set(blocks)
        while frontier:
            expanded = set()
            for allocation in self._allocations:
                for lo, hi in allocation.ranges(frontier):
                    expanded.update(allocation.owners(lo, hi))
            frontier = expanded - result
            result.update(frontier)
        return result

    def invalidate_blocks(self, block_ids: list[int]) -> None:
        """Validate the agreed block plan and zero its local byte ranges without reallocating KV storage."""
        blocks = set(block_ids)
        if not blocks:
            return
        self.refresh_pointers()
        if self.expand_aliases(blocks) != blocks:
            raise RuntimeError("Worker consensus omitted aliased KV blocks")
        # Prepare every allocation before any write. Zero raw byte intervals,
        # including scale/state/page/alignment padding, not just tensor data.
        plans = [(allocation.buffer, allocation.ranges(blocks)) for allocation in self._allocations]
        for tensor, ranges in plans:
            for start, end in ranges:
                tensor[start:end].zero_()
        logger.info("HBM retry invalidated %d DP-local blocks using allocation byte maps", len(blocks))


def _iter_kv_tensors(kv_caches: Any) -> Iterable[tuple[Any, str, torch.Tensor]]:
    if kv_caches is None:
        return
    if isinstance(kv_caches, torch.Tensor):
        yield "kv_cache", "combined", kv_caches
        return
    if isinstance(kv_caches, dict):
        for name, value in kv_caches.items():
            yield from _iter_named(name, value)
        return
    if isinstance(kv_caches, (list, tuple)):
        for idx, value in enumerate(kv_caches):
            yield from _iter_named(idx, value)


def _iter_named(name: Any, value: Any) -> Iterable[tuple[Any, str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield name, "combined", value
        return
    if isinstance(value, dict):
        for key, part in value.items():
            yield from _iter_named(f"{name}/{key}", part)
        return
    if isinstance(value, (list, tuple)):
        kinds = ("key", "value", "scale")
        for i, part in enumerate(value):
            kind = kinds[i] if i < len(kinds) else f"part{i}"
            if isinstance(part, torch.Tensor):
                yield name, kind, part
            else:
                yield from _iter_named(name, part)
