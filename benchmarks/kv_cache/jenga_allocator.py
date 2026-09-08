# SPDX-License-Identifier: Apache-2.0

"""Dependency-free prototype of a Jenga-style heterogeneous page allocator.

This module deliberately models allocation only. It does not replace vLLM's
BlockPool, prefix-cache eviction, or device tensor binding. The prototype lets
us measure whether typed small pages backed by reusable large pages are worth
integrating into those runtime paths.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PageType:
    """One cache object's physical allocation granularity."""

    name: str
    page_size_bytes: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("page type name must not be empty")
        if self.page_size_bytes <= 0:
            raise ValueError("page_size_bytes must be positive")


@dataclass(frozen=True, slots=True)
class SmallPageId:
    """A typed page ID; IDs from different page types are not interchangeable."""

    page_type: str
    large_page_id: int
    slot: int


@dataclass(frozen=True, slots=True)
class RequestDemand:
    """A request's lifetime and peak number of pages for each cache type."""

    request_id: str
    start_step: int
    duration_steps: int
    page_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.start_step < 0:
            raise ValueError("start_step must be non-negative")
        if self.duration_steps <= 0:
            raise ValueError("duration_steps must be positive")
        if any(count < 0 for count in self.page_counts.values()):
            raise ValueError("page counts must be non-negative")


@dataclass(frozen=True, slots=True)
class AllocatorSnapshot:
    total_managed_bytes: int
    pool_tail_bytes: int
    useful_bytes: int
    active_reserved_bytes: int
    active_large_pages: int
    allocated_small_pages: int
    layout_tail_bytes: int
    free_slot_bytes: int
    allocated_pages_by_type: Mapping[str, int]

    @property
    def active_fragmentation_bytes(self) -> int:
        return self.active_reserved_bytes - self.useful_bytes

    @property
    def active_utilization(self) -> float:
        if self.active_reserved_bytes == 0:
            return 1.0
        return self.useful_bytes / self.active_reserved_bytes

    @property
    def pool_utilization(self) -> float:
        if self.total_managed_bytes == 0:
            return 0.0
        return self.useful_bytes / self.total_managed_bytes

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["active_fragmentation_bytes"] = self.active_fragmentation_bytes
        result["active_utilization"] = self.active_utilization
        result["pool_utilization"] = self.pool_utilization
        return result


class OutOfPagesError(RuntimeError):
    """Raised when an atomic request allocation cannot be satisfied."""


class RequestAllocator(Protocol):
    def allocate_request(self, request_id: str, page_counts: Mapping[str, int]) -> None: ...

    def free_request(self, request_id: str) -> None: ...

    def snapshot(self) -> AllocatorSnapshot: ...


@dataclass(slots=True)
class _LargePage:
    page_type: str | None = None
    preferred_request_id: str | None = None
    allocated_slots: set[int] | None = None

    def __post_init__(self) -> None:
        if self.allocated_slots is None:
            self.allocated_slots = set()


def exact_lcm_page_size(page_types: Iterable[PageType]) -> int:
    """Return the lossless large-page size used by Jenga's LCM layout."""

    sizes = [page_type.page_size_bytes for page_type in page_types]
    if not sizes:
        raise ValueError("at least one page type is required")
    return math.lcm(*sizes)


class HeterogeneousPageAllocator:
    """Typed small pages carved from dynamically reusable large pages.

    A large page is assigned to one cache type while it contains live small
    pages. Once all its small pages are freed, it returns to the global pool and
    may be reassigned to any type. Allocation prefers a partially used page
    belonging to the same request, then any compatible partially used page,
    before consuming an empty large page. Packing existing typed slabs first is
    essential: reserving one large page per request would turn the LCM page
    itself into a new source of internal fragmentation.

    ``large_page_size_bytes`` may be the exact LCM (zero layout-tail waste) or
    a bounded superpage size (less coarse reclamation, possibly some tail waste).
    """

    def __init__(
        self,
        total_memory_bytes: int,
        page_types: Iterable[PageType],
        large_page_size_bytes: int | None = None,
    ) -> None:
        if total_memory_bytes <= 0:
            raise ValueError("total_memory_bytes must be positive")
        page_types = tuple(page_types)
        if not page_types:
            raise ValueError("at least one page type is required")
        self.page_types = {page_type.name: page_type for page_type in page_types}
        if len(self.page_types) != len(page_types):
            raise ValueError("page type names must be unique")

        if large_page_size_bytes is None:
            large_page_size_bytes = exact_lcm_page_size(page_types)
        if large_page_size_bytes < max(page.page_size_bytes for page in page_types):
            raise ValueError("large page must fit every small page type")

        self.total_memory_bytes = total_memory_bytes
        self.large_page_size_bytes = large_page_size_bytes
        self.num_large_pages = total_memory_bytes // large_page_size_bytes
        if self.num_large_pages == 0:
            raise ValueError("memory budget cannot fit one large page")
        self.total_managed_bytes = self.num_large_pages * large_page_size_bytes
        self.pool_tail_bytes = total_memory_bytes - self.total_managed_bytes
        self._capacities = {
            name: large_page_size_bytes // page.page_size_bytes for name, page in self.page_types.items()
        }
        self._large_pages = [_LargePage() for _ in range(self.num_large_pages)]
        self._request_pages: dict[str, list[SmallPageId]] = defaultdict(list)

    def allocate_request(self, request_id: str, page_counts: Mapping[str, int]) -> None:
        if request_id in self._request_pages:
            raise ValueError(f"request {request_id!r} is already allocated")
        unknown_types = set(page_counts) - self.page_types.keys()
        if unknown_types:
            raise ValueError(f"unknown page types: {sorted(unknown_types)}")
        if any(count < 0 for count in page_counts.values()):
            raise ValueError("page counts must be non-negative")

        self._request_pages[request_id] = []
        try:
            for page_type, count in page_counts.items():
                for _ in range(count):
                    self._request_pages[request_id].append(self._allocate_one(request_id, page_type))
        except OutOfPagesError:
            self.free_request(request_id)
            raise

    def _allocate_one(self, request_id: str, page_type: str) -> SmallPageId:
        capacity = self._capacities[page_type]

        preferred = self._find_partial_page(page_type, capacity, request_id)
        if preferred is not None:
            return self._take_slot(preferred, page_type, request_id)

        shared = self._find_partial_page(page_type, capacity, preferred_request_id=None)
        if shared is not None:
            self._large_pages[shared].preferred_request_id = None
            return self._take_slot(shared, page_type, request_id)

        for large_page_id, large_page in enumerate(self._large_pages):
            if large_page.page_type is None:
                large_page.page_type = page_type
                large_page.preferred_request_id = request_id
                return self._take_slot(large_page_id, page_type, request_id)

        raise OutOfPagesError(f"no {page_type!r} page is available")

    def _find_partial_page(
        self,
        page_type: str,
        capacity: int,
        preferred_request_id: str | None,
    ) -> int | None:
        for large_page_id, large_page in enumerate(self._large_pages):
            if large_page.page_type != page_type:
                continue
            if preferred_request_id is not None and large_page.preferred_request_id != preferred_request_id:
                continue
            assert large_page.allocated_slots is not None
            if len(large_page.allocated_slots) < capacity:
                return large_page_id
        return None

    def _take_slot(self, large_page_id: int, page_type: str, request_id: str) -> SmallPageId:
        large_page = self._large_pages[large_page_id]
        assert large_page.allocated_slots is not None
        capacity = self._capacities[page_type]
        slot = next(slot for slot in range(capacity) if slot not in large_page.allocated_slots)
        large_page.allocated_slots.add(slot)
        if large_page.preferred_request_id is None and len(large_page.allocated_slots) == 1:
            large_page.preferred_request_id = request_id
        return SmallPageId(page_type, large_page_id, slot)

    def free_request(self, request_id: str) -> None:
        pages = self._request_pages.pop(request_id, [])
        for page_id in pages:
            large_page = self._large_pages[page_id.large_page_id]
            assert large_page.allocated_slots is not None
            large_page.allocated_slots.remove(page_id.slot)
            if not large_page.allocated_slots:
                large_page.page_type = None
                large_page.preferred_request_id = None

    def request_pages(self, request_id: str) -> tuple[SmallPageId, ...]:
        """Return the typed physical locations currently owned by a request."""

        return tuple(self._request_pages.get(request_id, ()))

    def snapshot(self) -> AllocatorSnapshot:
        counts: Counter[str] = Counter()
        active_large_pages = 0
        layout_tail_bytes = 0
        free_slot_bytes = 0
        for large_page in self._large_pages:
            if large_page.page_type is None:
                continue
            active_large_pages += 1
            assert large_page.allocated_slots is not None
            page_type = self.page_types[large_page.page_type]
            capacity = self._capacities[large_page.page_type]
            allocated = len(large_page.allocated_slots)
            counts[large_page.page_type] += allocated
            layout_tail_bytes += self.large_page_size_bytes - capacity * page_type.page_size_bytes
            free_slot_bytes += (capacity - allocated) * page_type.page_size_bytes

        useful_bytes = sum(self.page_types[name].page_size_bytes * count for name, count in counts.items())
        return AllocatorSnapshot(
            total_managed_bytes=self.total_managed_bytes,
            pool_tail_bytes=self.pool_tail_bytes,
            useful_bytes=useful_bytes,
            active_reserved_bytes=active_large_pages * self.large_page_size_bytes,
            active_large_pages=active_large_pages,
            allocated_small_pages=sum(counts.values()),
            layout_tail_bytes=layout_tail_bytes,
            free_slot_bytes=free_slot_bytes,
            allocated_pages_by_type=dict(counts),
        )


class UniformPageAllocator:
    """Baseline with one physical page size shared by every cache group.

    ``objects_per_page`` models virtual block splitting: an enlarged Attention
    scheduler block can contain several kernel-sized cache objects, while the
    last partially filled page still reserves the entire uniform physical page.
    """

    def __init__(
        self,
        total_memory_bytes: int,
        page_types: Iterable[PageType],
        physical_page_size_bytes: int | None = None,
        objects_per_page: Mapping[str, int] | None = None,
    ) -> None:
        page_types = tuple(page_types)
        if not page_types:
            raise ValueError("at least one page type is required")
        self.page_types = {page_type.name: page_type for page_type in page_types}
        if len(self.page_types) != len(page_types):
            raise ValueError("page type names must be unique")
        self.page_size_bytes = physical_page_size_bytes or max(page.page_size_bytes for page in page_types)
        if self.page_size_bytes < max(page.page_size_bytes for page in page_types):
            raise ValueError("uniform physical page must fit every cache object")
        self.objects_per_page = {name: 1 for name in self.page_types}
        if objects_per_page:
            unknown_types = set(objects_per_page) - self.page_types.keys()
            if unknown_types:
                raise ValueError(f"unknown page types: {sorted(unknown_types)}")
            self.objects_per_page.update(objects_per_page)
        if any(count <= 0 for count in self.objects_per_page.values()):
            raise ValueError("objects_per_page values must be positive")
        self.num_pages = total_memory_bytes // self.page_size_bytes
        if self.num_pages == 0:
            raise ValueError("memory budget cannot fit one page")
        self.total_managed_bytes = self.num_pages * self.page_size_bytes
        self.pool_tail_bytes = total_memory_bytes - self.total_managed_bytes
        self._free_page_ids = list(range(self.num_pages - 1, -1, -1))
        self._request_pages: dict[str, list[tuple[int, str, int]]] = {}

    def allocate_request(self, request_id: str, page_counts: Mapping[str, int]) -> None:
        if request_id in self._request_pages:
            raise ValueError(f"request {request_id!r} is already allocated")
        unknown_types = set(page_counts) - self.page_types.keys()
        if unknown_types:
            raise ValueError(f"unknown page types: {sorted(unknown_types)}")
        required = sum(math.ceil(count / self.objects_per_page[page_type]) for page_type, count in page_counts.items())
        if required > len(self._free_page_ids):
            raise OutOfPagesError("uniform page pool is exhausted")

        allocated: list[tuple[int, str, int]] = []
        for page_type, count in page_counts.items():
            slots_per_page = self.objects_per_page[page_type]
            remaining = count
            while remaining:
                useful_objects = min(remaining, slots_per_page)
                allocated.append((self._free_page_ids.pop(), page_type, useful_objects))
                remaining -= useful_objects
        self._request_pages[request_id] = allocated

    def free_request(self, request_id: str) -> None:
        pages = self._request_pages.pop(request_id, [])
        self._free_page_ids.extend(page_id for page_id, _, _ in pages)

    def snapshot(self) -> AllocatorSnapshot:
        counts: Counter[str] = Counter()
        allocated = 0
        for pages in self._request_pages.values():
            for _, page_type, useful_objects in pages:
                counts[page_type] += useful_objects
                allocated += 1
        useful_bytes = sum(self.page_types[name].page_size_bytes * count for name, count in counts.items())
        active_reserved_bytes = allocated * self.page_size_bytes
        return AllocatorSnapshot(
            total_managed_bytes=self.total_managed_bytes,
            pool_tail_bytes=self.pool_tail_bytes,
            useful_bytes=useful_bytes,
            active_reserved_bytes=active_reserved_bytes,
            active_large_pages=allocated,
            allocated_small_pages=sum(counts.values()),
            layout_tail_bytes=active_reserved_bytes - useful_bytes,
            free_slot_bytes=0,
            allocated_pages_by_type=dict(counts),
        )


@dataclass(frozen=True, slots=True)
class ReplayResult:
    accepted_requests: int
    rejected_requests: int
    peak_useful_bytes: int
    peak_reserved_bytes: int
    peak_fragmentation_bytes: int
    minimum_active_utilization: float
    final_snapshot: AllocatorSnapshot

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["final_snapshot"] = self.final_snapshot.to_dict()
        return result


def replay_workload(allocator: RequestAllocator, requests: Iterable[RequestDemand]) -> ReplayResult:
    """Replay arrivals/completions; completions at a step happen before arrivals."""

    events: list[tuple[int, int, RequestDemand]] = []
    for request in requests:
        events.append((request.start_step, 1, request))
        events.append((request.start_step + request.duration_steps, 0, request))
    events.sort(key=lambda item: (item[0], item[1], item[2].request_id))

    accepted_ids: set[str] = set()
    accepted = 0
    rejected = 0
    peak_useful = 0
    peak_reserved = 0
    peak_fragmentation = 0
    minimum_utilization = 1.0

    for _, event_kind, request in events:
        if event_kind == 0:
            if request.request_id in accepted_ids:
                allocator.free_request(request.request_id)
                accepted_ids.remove(request.request_id)
        else:
            try:
                allocator.allocate_request(request.request_id, request.page_counts)
            except OutOfPagesError:
                rejected += 1
            else:
                accepted += 1
                accepted_ids.add(request.request_id)

        snapshot = allocator.snapshot()
        peak_useful = max(peak_useful, snapshot.useful_bytes)
        peak_reserved = max(peak_reserved, snapshot.active_reserved_bytes)
        peak_fragmentation = max(peak_fragmentation, snapshot.active_fragmentation_bytes)
        if snapshot.active_reserved_bytes:
            minimum_utilization = min(minimum_utilization, snapshot.active_utilization)

    return ReplayResult(
        accepted_requests=accepted,
        rejected_requests=rejected,
        peak_useful_bytes=peak_useful,
        peak_reserved_bytes=peak_reserved,
        peak_fragmentation_bytes=peak_fragmentation,
        minimum_active_utilization=minimum_utilization,
        final_snapshot=allocator.snapshot(),
    )
