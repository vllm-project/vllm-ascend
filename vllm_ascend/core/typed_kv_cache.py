# SPDX-License-Identifier: Apache-2.0
"""Experimental typed-page KV cache primitives.

This module implements the scheduler/worker boundary needed by the Jenga-style
runtime experiment.  Scheduler-visible block IDs stay local to a KV cache
group.  A group-specific address table translates them to byte offsets in the
shared physical arena, so heterogeneous pages can be placed and retyped
without forcing their sizes into an exact-LCM superpage.

The implementation intentionally excludes prefix caching and KV transfer.  It
is selected only when the default-off ``VLLM_ASCEND_ENABLE_TYPED_KV_CACHE``
experiment is enabled.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

TYPED_KV_CACHE_PLAN_ATTR = "_ascend_typed_kv_cache_plan"


class TypedWorkerKVCacheSpecs(dict):
    """Carry the worker-selected native block size through the cache-spec RPC.

    A worker's VllmConfig is private when using a multiprocessing executor.
    The engine therefore needs this metadata alongside the returned specs,
    before upstream groups and rewrites their padded block sizes.
    """

    def __init__(self, specs: Mapping, native_attention_block_size: int) -> None:
        if type(native_attention_block_size) is not int or native_attention_block_size <= 0:
            raise ValueError("typed worker cache specs require a positive native attention block size")
        super().__init__(specs)
        self.native_attention_block_size = native_attention_block_size


def get_typed_worker_attention_block_size(worker_specs: Iterable[Mapping]) -> int:
    """Require every nonempty worker result to agree on the native block size."""
    sizes = set()
    for specs in worker_specs:
        if not specs:
            continue
        size = getattr(specs, "native_attention_block_size", None)
        if type(size) is not int or size <= 0:
            raise ValueError("typed KV cache requires native attention block metadata from every worker")
        sizes.add(size)
    if len(sizes) != 1:
        raise ValueError("typed KV cache requires one consistent native attention block size across workers")
    return sizes.pop()


@dataclass(frozen=True, slots=True)
class TypedPageSpec:
    """Physical page and token granularity for one KV cache group."""

    group_id: int
    page_size_bytes: int
    block_size_tokens: int

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError("group_id must be non-negative")
        if self.page_size_bytes <= 0:
            raise ValueError("page_size_bytes must be positive")
        if self.block_size_tokens <= 0:
            raise ValueError("block_size_tokens must be positive")


@dataclass(frozen=True, slots=True)
class TypedBlockId:
    """A block address whose group is part of its type."""

    group_id: int
    superpage_id: int
    slot: int


@dataclass(slots=True)
class TypedKVCacheBlock:
    """Minimal no-prefix-cache block understood by single-type managers.

    Keeping this object independent of vLLM makes the address allocator usable
    in offline tests.  The no-prefix-cache manager only relies on these fields;
    hash/event features are intentionally outside the MVP.
    """

    block_id: int
    ref_cnt: int = 0
    block_hash: object | None = None
    block_hash_num_tokens: int | None = None
    is_null: bool = False

    def reset_hash(self) -> None:
        self.block_hash = None
        self.block_hash_num_tokens = None


@dataclass(frozen=True, slots=True)
class TypedKVCachePlan:
    """Storage and address plan shared by scheduler and worker."""

    specs: tuple[TypedPageSpec, ...]
    superpage_size_bytes: int
    num_superpages: int
    region_offsets_bytes: tuple[int, ...] = ()
    region_num_blocks: tuple[int, ...] = ()
    page_address_tables_bytes: tuple[tuple[int, ...], ...] = ()
    address_space_size_bytes: int = 0

    @classmethod
    def exact_lcm(
        cls,
        specs: Iterable[TypedPageSpec],
        total_memory_bytes: int,
    ) -> TypedKVCachePlan:
        specs = tuple(specs)
        if not specs:
            raise ValueError("at least one typed page spec is required")
        if len({spec.group_id for spec in specs}) != len(specs):
            raise ValueError("group IDs must be unique")
        if total_memory_bytes <= 0:
            raise ValueError("total_memory_bytes must be positive")
        superpage_size = math.lcm(*(spec.page_size_bytes for spec in specs))
        num_superpages = total_memory_bytes // superpage_size
        # Superpage zero is a shared read-safe NULL address. It is never handed
        # to a request, so at least one allocatable superpage is also required.
        if num_superpages < 2:
            raise ValueError(
                "typed KV cache needs one null superpage and at least one "
                "allocatable superpage: "
                f"superpage_size_bytes={superpage_size}, "
                f"total_memory_bytes={total_memory_bytes}, "
                f"page_sizes={[spec.page_size_bytes for spec in specs]}"
            )
        return cls(specs, superpage_size, num_superpages)

    @classmethod
    def partitioned(
        cls,
        specs: Iterable[TypedPageSpec],
        total_memory_bytes: int,
        num_blocks_by_group: Iterable[int],
    ) -> TypedKVCachePlan:
        """Build bounded, group-contiguous regions when exact LCM is too large.

        Block zero in every region is reserved as that group's NULL block. The
        remaining IDs are dense, so existing kernels keep using the conventional
        ``block_id * page_size`` formula against a group-specific tensor view.
        """

        specs = tuple(specs)
        block_counts = tuple(num_blocks_by_group)
        if not specs or len(specs) != len(block_counts):
            raise ValueError("partitioned plan must size every typed group")
        if any(count < 2 for count in block_counts):
            raise ValueError("every typed region needs a null and a data block")
        offsets = []
        offset = 0
        for spec, count in zip(specs, block_counts):
            offsets.append(offset)
            offset += spec.page_size_bytes * count
        if offset > total_memory_bytes:
            raise ValueError("partitioned typed regions exceed the memory budget")
        return cls(
            specs=specs,
            superpage_size_bytes=0,
            num_superpages=0,
            region_offsets_bytes=tuple(offsets),
            region_num_blocks=block_counts,
        )

    @classmethod
    def addressed(
        cls,
        specs: Iterable[TypedPageSpec],
        total_memory_bytes: int,
        page_address_tables_bytes: Iterable[Iterable[int]] | None = None,
    ) -> TypedKVCachePlan:
        """Build group-specific logical-page to physical-offset tables.

        Entry zero of every table maps to byte offset zero and is the shared
        read-safe NULL page.  Other entries are candidate placements in the
        common byte arena.  Candidate intervals may overlap across groups;
        ``TypedAddressPool`` prevents overlapping live allocations and makes a
        released interval available to any compatible page type.
        """

        specs = tuple(specs)
        if not specs:
            raise ValueError("at least one typed page spec is required")
        if tuple(spec.group_id for spec in specs) != tuple(range(len(specs))):
            raise ValueError("addressed plans require contiguous group IDs")
        if total_memory_bytes <= 0:
            raise ValueError("total_memory_bytes must be positive")

        null_page_size = max(spec.page_size_bytes for spec in specs)
        if total_memory_bytes < null_page_size:
            raise ValueError("typed address space cannot hold the shared NULL page")

        if page_address_tables_bytes is None:
            tables = []
            for spec in specs:
                first_data_offset = (
                    (null_page_size + spec.page_size_bytes - 1) // spec.page_size_bytes * spec.page_size_bytes
                )
                tables.append(
                    (0,)
                    + tuple(
                        range(
                            first_data_offset,
                            total_memory_bytes - spec.page_size_bytes + 1,
                            spec.page_size_bytes,
                        )
                    )
                )
        else:
            tables = [tuple(table) for table in page_address_tables_bytes]

        if len(tables) != len(specs):
            raise ValueError("address table must be provided for every group")
        for spec, table in zip(specs, tables):
            if len(table) < 2 or table[0] != 0:
                raise ValueError("every group address table needs a NULL and a data page")
            if len(set(table)) != len(table):
                raise ValueError("group address table contains duplicate offsets")
            for offset in table:
                if offset < 0 or offset % spec.page_size_bytes:
                    raise ValueError("page offset must be non-negative and page aligned")
                if offset + spec.page_size_bytes > total_memory_bytes:
                    raise ValueError("page address exceeds the typed address space")

        return cls(
            specs=specs,
            superpage_size_bytes=0,
            num_superpages=0,
            page_address_tables_bytes=tuple(tables),
            address_space_size_bytes=total_memory_bytes,
        )

    @property
    def is_partitioned(self) -> bool:
        return bool(self.region_offsets_bytes)

    @property
    def is_addressed(self) -> bool:
        return bool(self.page_address_tables_bytes)

    @property
    def total_managed_bytes(self) -> int:
        if self.is_addressed:
            return self.address_space_size_bytes
        if self.is_partitioned:
            last = len(self.specs) - 1
            return self.region_offsets_bytes[last] + self.region_num_blocks[last] * self.specs[last].page_size_bytes
        return self.superpage_size_bytes * self.num_superpages

    def num_blocks(self, group_id: int) -> int:
        if self.is_addressed:
            return len(self.page_address_tables_bytes[group_id])
        if self.is_partitioned:
            return self.region_num_blocks[group_id]
        return self.num_superpages * self.capacity(group_id)

    def region_offset_bytes(self, group_id: int) -> int:
        if self.is_addressed:
            raise ValueError("addressed plans do not have dense group regions")
        return self.region_offsets_bytes[group_id] if self.is_partitioned else 0

    def spec(self, group_id: int) -> TypedPageSpec:
        for spec in self.specs:
            if spec.group_id == group_id:
                return spec
        raise KeyError(f"unknown KV cache group {group_id}")

    def capacity(self, group_id: int) -> int:
        if self.is_partitioned or self.is_addressed:
            raise ValueError("this plan does not have superpage capacity")
        spec = self.spec(group_id)
        quotient, remainder = divmod(self.superpage_size_bytes, spec.page_size_bytes)
        if remainder:
            raise ValueError("exact-LCM plan contains a non-divisible page")
        return quotient

    def to_group_local_id(self, block_id: TypedBlockId) -> int:
        capacity = self.capacity(block_id.group_id)
        if not 0 <= block_id.superpage_id < self.num_superpages:
            raise ValueError("superpage ID is out of range")
        if not 0 <= block_id.slot < capacity:
            raise ValueError("slot is out of range")
        return block_id.superpage_id * capacity + block_id.slot

    def from_group_local_id(self, group_id: int, group_local_id: int) -> TypedBlockId:
        if group_local_id < 0:
            raise ValueError("group-local block ID must be non-negative")
        capacity = self.capacity(group_id)
        superpage_id, slot = divmod(group_local_id, capacity)
        block_id = TypedBlockId(group_id, superpage_id, slot)
        # Reuse the bounds validation in the forward mapping.
        self.to_group_local_id(block_id)
        return block_id

    def byte_offset(self, block_id: TypedBlockId) -> int:
        spec = self.spec(block_id.group_id)
        if self.is_addressed:
            if block_id.superpage_id != 0:
                raise ValueError("addressed block IDs do not use superpages")
            if not 0 <= block_id.slot < self.num_blocks(block_id.group_id):
                raise ValueError("block ID is out of range")
            return self.page_address_tables_bytes[block_id.group_id][block_id.slot]
        if self.is_partitioned:
            if block_id.superpage_id != 0:
                raise ValueError("partitioned block IDs do not use superpages")
            if not 0 <= block_id.slot < self.num_blocks(block_id.group_id):
                raise ValueError("block ID is out of range")
            return self.region_offset_bytes(block_id.group_id) + block_id.slot * spec.page_size_bytes
        self.to_group_local_id(block_id)
        return block_id.superpage_id * self.superpage_size_bytes + block_id.slot * spec.page_size_bytes

    def slot_mapping(self, group_id: int, block_id: int, offset: int) -> int:
        spec = self.spec(group_id)
        if not 0 <= offset < spec.block_size_tokens:
            raise ValueError("token offset is outside the logical block")
        if self.is_addressed:
            physical_block_id = self.physical_block_id(group_id, block_id)
            return physical_block_id * spec.block_size_tokens + offset
        if self.is_partitioned:
            if not 0 <= block_id < self.num_blocks(group_id):
                raise ValueError("group-local block ID is out of range")
        else:
            self.from_group_local_id(group_id, block_id)
        return block_id * spec.block_size_tokens + offset

    def physical_block_id(self, group_id: int, logical_block_id: int) -> int:
        """Translate a group-local logical ID to a kernel-visible page ID."""

        if not self.is_addressed:
            return logical_block_id
        if not 0 <= logical_block_id < self.num_blocks(group_id):
            raise ValueError("group-local block ID is out of range")
        page_size = self.spec(group_id).page_size_bytes
        offset = self.page_address_tables_bytes[group_id][logical_block_id]
        physical_block_id, remainder = divmod(offset, page_size)
        if remainder:
            raise AssertionError("validated page address became unaligned")
        return physical_block_id

    def kernel_page_address_table(
        self,
        group_id: int,
        blocks_per_page: int = 1,
    ) -> tuple[int, ...]:
        """Return logical-kernel-block to physical-kernel-block translation."""

        if blocks_per_page <= 0:
            raise ValueError("blocks_per_page must be positive")
        result = []
        for logical_block_id in range(self.num_blocks(group_id)):
            physical_block_id = self.physical_block_id(group_id, logical_block_id)
            base = physical_block_id * blocks_per_page
            result.extend(base + slot for slot in range(blocks_per_page))
        return tuple(result)


def typed_block_ids_to_zero(
    block_ids: Iterable[int],
    *,
    num_computed_tokens: int,
    block_size_tokens: int,
    prefix_caching_enabled: bool,
) -> tuple[int, ...]:
    """Select newly allocated pages without clearing prefix-cache hits.

    A scheduled new request carries its complete group block table. When a
    prefix was reused, the leading pages already contain valid KV/state and
    must not be cleared. Block zero is the shared read-safe NULL page.
    """

    block_ids = tuple(block_ids)
    if isinstance(num_computed_tokens, bool) or not isinstance(num_computed_tokens, int) or num_computed_tokens < 0:
        raise ValueError("num_computed_tokens must be a non-negative integer")
    if isinstance(block_size_tokens, bool) or not isinstance(block_size_tokens, int) or block_size_tokens <= 0:
        raise ValueError("block_size_tokens must be a positive integer")

    first_new_block = 0
    if prefix_caching_enabled:
        first_new_block, remainder = divmod(
            num_computed_tokens,
            block_size_tokens,
        )
        if remainder:
            raise ValueError(
                "typed prefix cache only supports common-prefix hits aligned to every group's physical page"
            )
        if first_new_block > len(block_ids):
            raise ValueError("computed prefix exceeds the typed block table")
    return tuple(block_id for block_id in block_ids[first_new_block:] if block_id != 0)


def get_typed_kv_cache_plan(kv_cache_config) -> TypedKVCachePlan | None:
    """Read the experimental plan carried by ``KVCacheConfig``.

    ``KVCacheConfig`` is an upstream dataclass without extension fields. A
    private dynamic attribute keeps this experiment source-compatible while
    still surviving deepcopy and scheduler/worker serialization.
    """

    plan = getattr(kv_cache_config, TYPED_KV_CACHE_PLAN_ATTR, None)
    if plan is not None and not isinstance(plan, TypedKVCachePlan):
        raise TypeError("invalid typed KV cache plan on KVCacheConfig")
    return plan


def set_typed_kv_cache_plan(kv_cache_config, plan: TypedKVCachePlan) -> None:
    setattr(kv_cache_config, TYPED_KV_CACHE_PLAN_ATTR, plan)


@dataclass(slots=True)
class _Superpage:
    owner_group_id: int | None = None
    allocated_slots: set[int] | None = None

    def __post_init__(self) -> None:
        if self.allocated_slots is None:
            self.allocated_slots = set()


class TypedSuperpagePool:
    """Global pool with group-specific BlockPool-compatible facades."""

    def __init__(self, plan: TypedKVCachePlan) -> None:
        self.plan = plan
        self._superpages = [_Superpage() for _ in range(plan.num_superpages)]
        # Superpage zero backs NULL_BLOCK_ID for every group and is never typed.
        self._superpages[0].owner_group_id = -1
        self._blocks: dict[tuple[int, int], TypedKVCacheBlock] = {}
        self._facades = {spec.group_id: TypedGroupBlockPool(self, spec.group_id) for spec in plan.specs}

    def for_group(self, group_id: int) -> TypedGroupBlockPool:
        try:
            return self._facades[group_id]
        except KeyError as exc:
            raise KeyError(f"unknown KV cache group {group_id}") from exc

    def get_num_free_superpages(self) -> int:
        return sum(page.owner_group_id is None for page in self._superpages[1:])

    def get_usage(self) -> float:
        allocatable = self.plan.num_superpages - 1
        if allocatable == 0:
            return 1.0
        return 1.0 - self.get_num_free_superpages() / allocatable

    def additional_superpages_required(self, block_counts: Mapping[int, int]) -> int:
        """Predict the atomic cost of a multi-group allocation."""

        required = 0
        for group_id, count in block_counts.items():
            if count < 0:
                raise ValueError("block counts must be non-negative")
            capacity = self.plan.capacity(group_id)
            free_slots = sum(
                capacity - len(page.allocated_slots or ())
                for page in self._superpages[1:]
                if page.owner_group_id == group_id
            )
            remaining = max(count - free_slots, 0)
            required += math.ceil(remaining / capacity)
        return required

    def _allocate(self, group_id: int, count: int) -> list[TypedKVCacheBlock]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if self.additional_superpages_required({group_id: count}) > (self.get_num_free_superpages()):
            raise ValueError(f"cannot allocate {count} blocks for group {group_id}")

        allocated: list[TypedKVCacheBlock] = []
        for _ in range(count):
            superpage_id = self._find_page_with_free_slot(group_id)
            page = self._superpages[superpage_id]
            if page.owner_group_id is None:
                page.owner_group_id = group_id
            assert page.allocated_slots is not None
            capacity = self.plan.capacity(group_id)
            slot = next(i for i in range(capacity) if i not in page.allocated_slots)
            page.allocated_slots.add(slot)
            local_id = self.plan.to_group_local_id(TypedBlockId(group_id, superpage_id, slot))
            block = self._blocks.get((group_id, local_id))
            if block is None:
                block = TypedKVCacheBlock(local_id)
                self._blocks[group_id, local_id] = block
            if block.ref_cnt != 0 or block.is_null:
                raise AssertionError("typed block bookkeeping is corrupt")
            block.ref_cnt = 1
            allocated.append(block)
        return allocated

    def _find_page_with_free_slot(self, group_id: int) -> int:
        capacity = self.plan.capacity(group_id)
        for superpage_id, page in enumerate(self._superpages[1:], start=1):
            if page.owner_group_id == group_id and len(page.allocated_slots or ()) < capacity:
                return superpage_id
        for superpage_id, page in enumerate(self._superpages[1:], start=1):
            if page.owner_group_id is None:
                return superpage_id
        raise ValueError("typed superpage pool is exhausted")

    def _free(self, group_id: int, blocks: Iterable[TypedKVCacheBlock]) -> None:
        for block in blocks:
            if block.is_null:
                continue
            if self._blocks.get((group_id, block.block_id)) is not block:
                raise ValueError("block belongs to a different typed group")
            typed_id = self.plan.from_group_local_id(group_id, block.block_id)
            page = self._superpages[typed_id.superpage_id]
            if page.owner_group_id != group_id:
                raise ValueError("block belongs to a different typed group")
            assert page.allocated_slots is not None
            if typed_id.slot not in page.allocated_slots or block.ref_cnt != 1:
                raise ValueError("block is not currently allocated")
            block.ref_cnt = 0
            block.reset_hash()
            page.allocated_slots.remove(typed_id.slot)
            if not page.allocated_slots:
                page.owner_group_id = None


class TypedGroupBlockPool:
    """Subset of BlockPool used by one SingleTypeKVCacheManager.

    Prefix-cache methods are deliberately absent.  The coordinator must only
    construct these facades with ``enable_caching=False``.
    """

    enable_caching = False
    enable_kv_cache_events = False

    def __init__(self, pool: TypedSuperpagePool, group_id: int) -> None:
        self._pool = pool
        self.group_id = group_id
        self.null_block = TypedKVCacheBlock(0)
        self.null_block.is_null = True
        self.num_gpu_blocks = pool.plan.num_superpages * pool.plan.capacity(group_id)

    def get_new_blocks(self, num_blocks: int) -> list[TypedKVCacheBlock]:
        return self._pool._allocate(self.group_id, num_blocks)

    def free_blocks(self, blocks: Iterable[TypedKVCacheBlock]) -> None:
        self._pool._free(self.group_id, blocks)

    def get_num_free_blocks(self) -> int:
        capacity = self._pool.plan.capacity(self.group_id)
        compatible_slots = sum(
            capacity - len(page.allocated_slots or ())
            for page in self._pool._superpages[1:]
            if page.owner_group_id == self.group_id
        )
        return compatible_slots + self._pool.get_num_free_superpages() * capacity

    def get_usage(self) -> float:
        return self._pool.get_usage()


class TypedAdmissionPool:
    """Aggregate BlockPool view used by KVCacheManager admission checks."""

    enable_kv_cache_events = False

    def __init__(self, pool: TypedSuperpagePool) -> None:
        self.typed_pool = pool
        self.num_gpu_blocks = pool.plan.num_superpages - 1

    def get_num_free_blocks(self) -> int:
        # In the typed coordinator, one "block" at this boundary means one
        # physical superpage. The coordinator predicts demand in the same unit.
        return self.typed_pool.get_num_free_superpages()

    def get_usage(self) -> float:
        return self.typed_pool.get_usage()

    def take_events(self) -> list[object]:
        # Event production is intentionally excluded from the no-prefix-cache
        # MVP, but KVCacheManager polls this interface unconditionally.
        return []


class TypedAddressPool:
    """Variable-page allocator backed by explicit group address tables."""

    def __init__(self, plan: TypedKVCachePlan) -> None:
        if not plan.is_addressed:
            raise ValueError("TypedAddressPool requires an addressed plan")
        self.plan = plan
        self._null_page_size = max(spec.page_size_bytes for spec in plan.specs)
        self._allocated_intervals: dict[tuple[int, int], tuple[int, int]] = {}
        self._blocks: dict[tuple[int, int], TypedKVCacheBlock] = {}
        self._facades = {spec.group_id: TypedAddressGroupBlockPool(self, spec.group_id) for spec in plan.specs}

    def for_group(self, group_id: int) -> TypedAddressGroupBlockPool:
        try:
            return self._facades[group_id]
        except KeyError as exc:
            raise KeyError(f"unknown KV cache group {group_id}") from exc

    @staticmethod
    def _overlaps(
        interval: tuple[int, int],
        occupied: Iterable[tuple[int, int]],
    ) -> bool:
        start, end = interval
        return any(start < other_end and other_start < end for other_start, other_end in occupied)

    def _occupied(self) -> list[tuple[int, int]]:
        return [
            (0, self._null_page_size),
            *self._allocated_intervals.values(),
        ]

    def _candidate_ids(self, group_id: int) -> list[int]:
        """Prefer large pages low and small pages high to limit fragmentation."""

        spec = self.plan.spec(group_id)
        smallest_page = min(item.page_size_bytes for item in self.plan.specs)
        ids = list(range(1, self.plan.num_blocks(group_id)))
        return list(reversed(ids)) if spec.page_size_bytes == smallest_page else ids

    def _find_allocations(
        self,
        group_id: int,
        count: int,
        occupied: list[tuple[int, int]],
    ) -> list[int] | None:
        if count < 0:
            raise ValueError("block counts must be non-negative")
        if count == 0:
            return []
        spec = self.plan.spec(group_id)
        selected = []
        for logical_id in self._candidate_ids(group_id):
            if (group_id, logical_id) in self._allocated_intervals:
                continue
            start = self.plan.page_address_tables_bytes[group_id][logical_id]
            interval = (start, start + spec.page_size_bytes)
            if self._overlaps(interval, occupied):
                continue
            selected.append(logical_id)
            occupied.append(interval)
            if len(selected) == count:
                return selected
        return selected if len(selected) == count else None

    def can_allocate(self, block_counts: Mapping[int, int]) -> bool:
        """Simulate the manager allocation order without mutating the pool."""

        occupied = self._occupied()
        for group_id in sorted(block_counts):
            if self._find_allocations(group_id, block_counts[group_id], occupied) is None:
                return False
        return True

    def _allocate(self, group_id: int, count: int) -> list[TypedKVCacheBlock]:
        occupied = self._occupied()
        logical_ids = self._find_allocations(group_id, count, occupied)
        if logical_ids is None:
            raise ValueError(f"cannot allocate {count} blocks for group {group_id}")

        spec = self.plan.spec(group_id)
        blocks = []
        for logical_id in logical_ids:
            start = self.plan.page_address_tables_bytes[group_id][logical_id]
            self._allocated_intervals[group_id, logical_id] = (
                start,
                start + spec.page_size_bytes,
            )
            block = self._blocks.get((group_id, logical_id))
            if block is None:
                block = TypedKVCacheBlock(logical_id)
                self._blocks[group_id, logical_id] = block
            if block.ref_cnt != 0 or block.is_null:
                raise AssertionError("typed address bookkeeping is corrupt")
            block.ref_cnt = 1
            blocks.append(block)
        return blocks

    def _free(self, group_id: int, blocks: Iterable[TypedKVCacheBlock]) -> None:
        for block in blocks:
            if block.is_null:
                continue
            key = (group_id, block.block_id)
            if self._blocks.get(key) is not block or block.ref_cnt != 1:
                raise ValueError("block is not allocated from this address pool")
            if key not in self._allocated_intervals:
                raise ValueError("typed address interval is not allocated")
            del self._allocated_intervals[key]
            block.ref_cnt = 0
            block.reset_hash()

    def get_num_free_blocks(self, group_id: int) -> int:
        occupied = self._occupied()
        spec = self.plan.spec(group_id)
        result = 0
        for logical_id in self._candidate_ids(group_id):
            if (group_id, logical_id) in self._allocated_intervals:
                continue
            start = self.plan.page_address_tables_bytes[group_id][logical_id]
            if not self._overlaps((start, start + spec.page_size_bytes), occupied):
                result += 1
        return result

    def get_usage(self) -> float:
        usable = self.plan.total_managed_bytes - self._null_page_size
        if usable <= 0:
            return 1.0
        used = sum(end - start for start, end in self._allocated_intervals.values())
        return used / usable


class TypedAddressGroupBlockPool:
    """BlockPool facade exposing group-local logical block IDs."""

    enable_caching = False
    enable_kv_cache_events = False

    def __init__(self, pool: TypedAddressPool, group_id: int) -> None:
        self._pool = pool
        self.group_id = group_id
        self.num_gpu_blocks = pool.plan.num_blocks(group_id)
        self.null_block = TypedKVCacheBlock(0, ref_cnt=1, is_null=True)

    def get_new_blocks(self, num_blocks: int) -> list[TypedKVCacheBlock]:
        return self._pool._allocate(self.group_id, num_blocks)

    def free_blocks(self, blocks: Iterable[TypedKVCacheBlock]) -> None:
        self._pool._free(self.group_id, blocks)

    def get_num_free_blocks(self) -> int:
        return self._pool.get_num_free_blocks(self.group_id)

    def get_usage(self) -> float:
        return self._pool.get_usage()


class TypedAddressAdmissionPool:
    """Boolean admission facade for atomic variable-page placement checks."""

    enable_kv_cache_events = False
    num_gpu_blocks = 0

    def __init__(self, pool: TypedAddressPool) -> None:
        self.typed_pool = pool

    def get_num_free_blocks(self) -> int:
        return 0

    def get_usage(self) -> float:
        return self.typed_pool.get_usage()

    def take_events(self) -> list[object]:
        return []


class TypedRegionPool:
    """Fixed-size group regions used when an exact-LCM superpage is impractical."""

    def __init__(self, plan: TypedKVCachePlan) -> None:
        if not plan.is_partitioned:
            raise ValueError("TypedRegionPool requires a partitioned plan")
        self.plan = plan
        self._facades = {spec.group_id: TypedRegionGroupBlockPool(self, spec.group_id) for spec in plan.specs}

    def for_group(self, group_id: int) -> TypedRegionGroupBlockPool:
        return self._facades[group_id]

    def can_allocate(self, block_counts: Mapping[int, int]) -> bool:
        return all(count <= self._facades[group_id].get_num_free_blocks() for group_id, count in block_counts.items())

    def get_usage(self) -> float:
        used = total = 0
        for spec in self.plan.specs:
            facade = self._facades[spec.group_id]
            usable = facade.num_gpu_blocks - 1
            total += usable * spec.page_size_bytes
            used += (usable - facade.get_num_free_blocks()) * spec.page_size_bytes
        return used / total if total else 1.0


class TypedRegionGroupBlockPool:
    """Dense BlockPool-compatible facade over one partitioned byte region."""

    enable_caching = False
    enable_kv_cache_events = False

    def __init__(self, pool: TypedRegionPool, group_id: int) -> None:
        self._pool = pool
        self.group_id = group_id
        self.num_gpu_blocks = pool.plan.num_blocks(group_id)
        self.null_block = TypedKVCacheBlock(0, ref_cnt=1, is_null=True)
        self._blocks = {block_id: TypedKVCacheBlock(block_id) for block_id in range(1, self.num_gpu_blocks)}
        self._free_ids = set(self._blocks)

    def get_new_blocks(self, num_blocks: int) -> list[TypedKVCacheBlock]:
        if num_blocks < 0 or num_blocks > len(self._free_ids):
            raise ValueError(f"cannot allocate {num_blocks} blocks for group {self.group_id}")
        block_ids = sorted(self._free_ids)[:num_blocks]
        blocks = []
        for block_id in block_ids:
            self._free_ids.remove(block_id)
            block = self._blocks[block_id]
            if block.ref_cnt != 0:
                raise AssertionError("typed region bookkeeping is corrupt")
            block.ref_cnt = 1
            blocks.append(block)
        return blocks

    def free_blocks(self, blocks: Iterable[TypedKVCacheBlock]) -> None:
        for block in blocks:
            if block.is_null:
                continue
            if self._blocks.get(block.block_id) is not block or block.ref_cnt != 1:
                raise ValueError("block is not allocated from this typed region")
            block.ref_cnt = 0
            block.reset_hash()
            self._free_ids.add(block.block_id)

    def get_num_free_blocks(self) -> int:
        return len(self._free_ids)

    def get_usage(self) -> float:
        return self._pool.get_usage()


class TypedRegionAdmissionPool:
    """Boolean admission facade for atomic multi-region capacity checks."""

    enable_kv_cache_events = False
    num_gpu_blocks = 0

    def __init__(self, pool: TypedRegionPool) -> None:
        self.typed_pool = pool

    def get_num_free_blocks(self) -> int:
        # The coordinator returns zero when every group fits and one otherwise.
        return 0

    def get_usage(self) -> float:
        return self.typed_pool.get_usage()

    def take_events(self) -> list[object]:
        return []


def make_group_byte_view(raw_tensor, plan: TypedKVCachePlan, group_id: int):
    """Return the kernel byte view indexed by physical page ID.

    ``raw_tensor`` must be a one-dimensional uint8 tensor.  Exact divisibility
    is what removes periodic tails and preserves the existing kernel indexing.
    The import is intentionally local so scheduler-only processes need not load
    torch through this experimental module.
    """

    import torch

    if raw_tensor.dtype not in (torch.int8, torch.uint8) or raw_tensor.ndim != 1:
        raise ValueError("raw_tensor must be a one-dimensional byte tensor")
    if raw_tensor.numel() < plan.total_managed_bytes:
        raise ValueError("raw tensor is smaller than the typed KV cache plan")
    page_size = plan.spec(group_id).page_size_bytes
    if plan.is_addressed:
        offset = 0
        last_physical_page = max(
            plan.physical_block_id(group_id, logical_id) for logical_id in range(plan.num_blocks(group_id))
        )
        size = (last_physical_page + 1) * page_size
    else:
        offset = plan.region_offset_bytes(group_id)
        size = plan.num_blocks(group_id) * page_size
    return raw_tensor.narrow(0, offset, size).view(-1, page_size)


def make_block_byte_view(
    raw_tensor,
    plan: TypedKVCachePlan,
    group_id: int,
    logical_block_id: int,
):
    """Return one logical page using the plan's physical byte address."""

    import torch

    if raw_tensor.dtype not in (torch.int8, torch.uint8) or raw_tensor.ndim != 1:
        raise ValueError("raw_tensor must be a one-dimensional byte tensor")
    if not 0 <= logical_block_id < plan.num_blocks(group_id):
        raise ValueError("group-local block ID is out of range")
    spec = plan.spec(group_id)
    typed_id = (
        TypedBlockId(group_id, 0, logical_block_id)
        if plan.is_addressed or plan.is_partitioned
        else plan.from_group_local_id(group_id, logical_block_id)
    )
    offset = plan.byte_offset(typed_id)
    return raw_tensor.narrow(0, offset, spec.page_size_bytes)
