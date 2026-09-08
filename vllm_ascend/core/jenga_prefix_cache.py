# SPDX-License-Identifier: Apache-2.0
"""Reference Jenga two-level prefix-cache allocation policy.

This module contains the scheduler-side policy only.  It deliberately has no
vLLM or device-runtime dependency, which makes the exact-LCM ownership,
eviction, and prefix metadata rules independently testable.

"Migration" in this policy is an ownership rebind of a fully evictable large
page.  Cached metadata is invalidated and the physical bytes are reused in
place; no KV payload is copied between memory tiers.  Consequently
``bytes_copied`` is always zero.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum, IntEnum

from .typed_kv_cache import TypedBlockId, TypedKVCachePlan

_DRY_RUN_PROTECTION = object()


class JengaOutOfPagesError(ValueError):
    """Raised when an atomic allocation cannot satisfy every group."""


class PageState(Enum):
    """Lifecycle of a small page inside a typed large page."""

    EMPTY = "empty"
    EVICTABLE = "evictable"
    USED = "used"


class AllocationTier(IntEnum):
    """The five allocation choices, in the order defined by Jenga."""

    REQUEST_AFFINE_EMPTY = 1
    EMPTY_LARGE_PAGE = 2
    EVICTABLE_LARGE_PAGE = 3
    ANY_EMPTY = 4
    EVICTABLE_SMALL_PAGE = 5


@dataclass(frozen=True, slots=True)
class JengaPageHandle:
    """A request lease for a group-local page.

    ``generation`` prevents a stale lease from addressing a page whose
    physical storage was subsequently reused.
    """

    request_id: Hashable
    group_id: int
    block_id: int
    superpage_id: int
    slot: int
    generation: int


@dataclass(frozen=True, slots=True)
class JengaCachedPage:
    """Read-only description returned by a namespaced prefix lookup."""

    group_id: int
    block_id: int
    superpage_id: int
    slot: int
    generation: int
    state: PageState
    block_hash: Hashable
    prefix_length: int
    last_access: float


@dataclass(frozen=True, slots=True)
class EvictedPage:
    group_id: int
    block_id: int
    superpage_id: int
    slot: int
    block_hash: Hashable


@dataclass(frozen=True, slots=True)
class AllocationDecision:
    """One small-page selection made by the five-tier policy."""

    handle: JengaPageHandle
    tier: AllocationTier
    previous_owner_group_id: int | None = None
    evicted_pages: tuple[EvictedPage, ...] = ()


@dataclass(frozen=True, slots=True)
class JengaAllocationPlan:
    """Result of either a dry run or a committed atomic allocation."""

    request_id: Hashable
    decisions: tuple[AllocationDecision, ...]
    committed: bool
    bytes_copied: int = 0

    def for_group(self, group_id: int) -> tuple[JengaPageHandle, ...]:
        return tuple(decision.handle for decision in self.decisions if decision.handle.group_id == group_id)


@dataclass(frozen=True, slots=True)
class JengaPrefixCacheStats:
    allocations: int
    cache_hits: int
    small_page_evictions: int
    large_page_evictions: int
    ownership_rebinds: int
    tier_counts: tuple[int, int, int, int, int]
    bytes_copied: int = 0


@dataclass(frozen=True, slots=True)
class JengaPageSnapshot:
    group_id: int
    block_id: int
    superpage_id: int
    slot: int
    state: PageState
    block_hash: Hashable | None
    prefix_length: int
    last_access: float
    generation: int
    active_requests: tuple[tuple[Hashable, int], ...]


@dataclass(frozen=True, slots=True)
class JengaLargePageSnapshot:
    superpage_id: int
    owner_group_id: int | None
    affinity_request_ids: frozenset[Hashable]
    pages: tuple[JengaPageSnapshot, ...]

    @property
    def affinity_request_id(self) -> Hashable | None:
        """Backward-compatible scalar view when there is one association."""

        if len(self.affinity_request_ids) != 1:
            return None
        return next(iter(self.affinity_request_ids))


@dataclass(frozen=True, slots=True)
class JengaPoolSnapshot:
    large_pages: tuple[JengaLargePageSnapshot, ...]
    hash_entries: frozenset[tuple[int, Hashable, int, int]]
    stats: JengaPrefixCacheStats


@dataclass(slots=True)
class _SmallPage:
    state: PageState = PageState.EMPTY
    block_hash: Hashable | None = None
    prefix_length: int = 0
    last_access: float = 0.0
    generation: int = 0
    active_requests: dict[Hashable, int] = field(default_factory=dict)

    def clone(self) -> _SmallPage:
        return _SmallPage(
            state=self.state,
            block_hash=self.block_hash,
            prefix_length=self.prefix_length,
            last_access=self.last_access,
            generation=self.generation,
            active_requests=self.active_requests.copy(),
        )


@dataclass(slots=True)
class _LargePage:
    owner_group_id: int | None = None
    # Jenga associates every small page in one large page with one preferred
    # request.  This placement hint is deliberately independent of the set of
    # requests that currently lease cached children.
    affinity_request_id: Hashable | None = None
    pages: list[_SmallPage] = field(default_factory=list)

    def clone(self) -> _LargePage:
        return _LargePage(
            owner_group_id=self.owner_group_id,
            affinity_request_id=self.affinity_request_id,
            pages=[page.clone() for page in self.pages],
        )


@dataclass(slots=True)
class _MutableStats:
    allocations: int = 0
    cache_hits: int = 0
    small_page_evictions: int = 0
    large_page_evictions: int = 0
    ownership_rebinds: int = 0
    tier_counts: list[int] = field(default_factory=lambda: [0] * 5)

    def clone(self) -> _MutableStats:
        return _MutableStats(
            allocations=self.allocations,
            cache_hits=self.cache_hits,
            small_page_evictions=self.small_page_evictions,
            large_page_evictions=self.large_page_evictions,
            ownership_rebinds=self.ownership_rebinds,
            tier_counts=self.tier_counts.copy(),
        )


@dataclass(slots=True)
class _PoolState:
    large_pages: list[_LargePage]
    hash_index: dict[tuple[int, Hashable], set[tuple[int, int]]]
    stats: _MutableStats
    next_generation: int = 1

    def clone(self) -> _PoolState:
        return _PoolState(
            large_pages=[page.clone() for page in self.large_pages],
            hash_index={key: positions.copy() for key, positions in self.hash_index.items()},
            stats=self.stats.clone(),
            next_generation=self.next_generation,
        )


class JengaPrefixCache:
    """Exact-LCM two-level allocator with layer-type prefix caching.

    Group IDs are layer-type/cache-group IDs.  All prefix hashes are keyed by
    ``(group_id, block_hash)`` so identical token hashes in heterogeneous layer
    types cannot alias each other.
    """

    def __init__(self, plan: TypedKVCachePlan) -> None:
        if plan.is_addressed or plan.is_partitioned:
            raise ValueError("JengaPrefixCache requires an exact-LCM plan")
        if plan.num_superpages < 2:
            raise ValueError("JengaPrefixCache needs an allocatable large page")
        self.plan = plan
        self._group_ids = frozenset(spec.group_id for spec in plan.specs)
        self._state = _PoolState(
            large_pages=[_LargePage(owner_group_id=-1)] + [_LargePage() for _ in range(plan.num_superpages - 1)],
            hash_index={},
            stats=_MutableStats(),
        )
        self.check_invariants()

    @staticmethod
    def small_page_lru_key(
        last_access: float,
        prefix_length: int,
        block_id: int,
    ) -> tuple[float, int, int]:
        """LRU key: oldest, then longest prefix, then lowest block ID."""

        return (last_access, -prefix_length, block_id)

    @property
    def stats(self) -> JengaPrefixCacheStats:
        value = self._state.stats
        return JengaPrefixCacheStats(
            allocations=value.allocations,
            cache_hits=value.cache_hits,
            small_page_evictions=value.small_page_evictions,
            large_page_evictions=value.large_page_evictions,
            ownership_rebinds=value.ownership_rebinds,
            tier_counts=tuple(value.tier_counts),  # type: ignore[arg-type]
            bytes_copied=0,
        )

    def dry_run(
        self,
        request_id: Hashable,
        block_counts: Mapping[int, int],
        *,
        protected: Iterable[JengaPageHandle | JengaCachedPage] = (),
    ) -> JengaAllocationPlan:
        """Plan a multi-group allocation without changing cache state.

        ``protected`` models prefix hits found before admission.  EVICTABLE
        pages in that set are temporarily treated as USED in the speculative
        state, preventing an earlier group allocation from evicting a later
        group's hit.
        """

        simulated, decisions = self._simulate(request_id, block_counts, protected)
        # Validate the speculative result as well as the live state.  This is
        # intentionally stronger than a capacity-only admission calculation.
        self._check_state_invariants(simulated)
        return JengaAllocationPlan(request_id, tuple(decisions), committed=False)

    def can_allocate(
        self,
        request_id: Hashable,
        block_counts: Mapping[int, int],
        *,
        protected: Iterable[JengaPageHandle | JengaCachedPage] = (),
    ) -> bool:
        try:
            self.dry_run(request_id, block_counts, protected=protected)
        except JengaOutOfPagesError:
            return False
        return True

    def allocate(
        self,
        request_id: Hashable,
        block_counts: Mapping[int, int],
        *,
        protected: Iterable[JengaPageHandle | JengaCachedPage] = (),
    ) -> JengaAllocationPlan:
        """Atomically allocate all requested pages or leave state unchanged."""

        simulated, decisions = self._simulate(request_id, block_counts, protected)
        self._check_state_invariants(simulated)
        self._state = simulated
        return JengaAllocationPlan(request_id, tuple(decisions), committed=True)

    def mark_cacheable(
        self,
        handle: JengaPageHandle,
        block_hash: Hashable,
        prefix_length: int,
        last_access: float,
    ) -> None:
        """Attach prefix metadata while a request still uses the page."""

        self._validate_hash(block_hash)
        self._validate_prefix_metadata(prefix_length, last_access)
        page = self._page_for_handle(handle, require_lease=True)
        if page.state is not PageState.USED:
            raise ValueError("only a used page can become prefix-cacheable")
        if page.block_hash is not None:
            self._remove_hash(self._state, handle.group_id, page.block_hash, handle.superpage_id, handle.slot)
        page.block_hash = block_hash
        page.prefix_length = prefix_length
        page.last_access = last_access
        self._add_hash(self._state, handle.group_id, block_hash, handle.superpage_id, handle.slot)
        self.check_invariants()

    def update_last_access(
        self,
        handles: Iterable[JengaPageHandle],
        last_access: float,
    ) -> None:
        self._validate_timestamp(last_access)
        for handle in handles:
            page = self._page_for_handle(handle, require_lease=True)
            page.last_access = last_access
        self.check_invariants()

    def release(
        self,
        handles: Iterable[JengaPageHandle],
        last_access: float | None = None,
    ) -> None:
        """Release request leases, retaining cacheable pages as EVICTABLE."""

        if last_access is not None:
            self._validate_timestamp(last_access)
        materialized = tuple(handles)
        validated: list[tuple[JengaPageHandle, _SmallPage]] = []
        release_counts: dict[tuple[int, int, Hashable], int] = {}
        # Validate the whole batch before changing any page.  Besides making a
        # failed release atomic, this catches a duplicate handle that would
        # release more leases than the request owns.
        for handle in materialized:
            page = self._page_for_handle(handle)
            key = (handle.superpage_id, handle.slot, handle.request_id)
            requested_count = release_counts.get(key, 0) + 1
            if requested_count > page.active_requests.get(handle.request_id, 0):
                raise ValueError("request does not hold this page")
            release_counts[key] = requested_count
            validated.append((handle, page))

        touched_superpages: set[int] = set()
        for handle, page in validated:
            lease_count = page.active_requests.get(handle.request_id, 0)
            # The complete batch was checked above, so this cannot fail after
            # an earlier page has already been mutated.
            assert lease_count > 0
            if lease_count == 1:
                del page.active_requests[handle.request_id]
            else:
                page.active_requests[handle.request_id] = lease_count - 1
            if last_access is not None:
                page.last_access = last_access
            if page.active_requests:
                page.state = PageState.USED
            elif page.block_hash is not None:
                page.state = PageState.EVICTABLE
            else:
                self._make_empty(page)
            touched_superpages.add(handle.superpage_id)

        for superpage_id in touched_superpages:
            large_page = self._state.large_pages[superpage_id]
            if large_page.pages and all(page.state is PageState.EMPTY for page in large_page.pages):
                large_page.owner_group_id = None
                large_page.affinity_request_id = None
                large_page.pages.clear()
        self.check_invariants()

    def get_cached_page(
        self,
        group_id: int,
        block_hash: Hashable,
    ) -> JengaCachedPage | None:
        """Look up a prefix page within one group-ID namespace."""

        self._validate_group(group_id)
        self._validate_hash(block_hash)
        positions = self._state.hash_index.get((group_id, block_hash), ())
        if not positions:
            return None
        superpage_id, slot = min(
            positions,
            key=lambda position: self._block_id(group_id, *position),
        )
        page = self._state.large_pages[superpage_id].pages[slot]
        return self._cached_snapshot(group_id, superpage_id, slot, page)

    def get_cached_page_by_block(
        self,
        group_id: int,
        block_id: int,
    ) -> JengaCachedPage | None:
        """Return cached metadata for one exact group-local block ID."""

        position = self._position_for_block_id(group_id, block_id)
        if position is None:
            return None
        superpage_id, slot = position
        page = self._state.large_pages[superpage_id].pages[slot]
        if page.block_hash is None or page.state is PageState.EMPTY:
            return None
        return self._cached_snapshot(group_id, superpage_id, slot, page)

    def acquire_cached(
        self,
        request_id: Hashable,
        group_id: int,
        block_hash: Hashable,
        last_access: float,
    ) -> JengaPageHandle | None:
        """Acquire a cache hit and protect it from both eviction levels."""

        self._validate_request_id(request_id)
        self._validate_timestamp(last_access)
        cached = self.get_cached_page(group_id, block_hash)
        if cached is None:
            return None
        return self.acquire_cached_block(
            request_id,
            group_id,
            cached.block_id,
            last_access,
            expected_generation=cached.generation,
            expected_hash=cached.block_hash,
        )

    def acquire_cached_block(
        self,
        request_id: Hashable,
        group_id: int,
        block_id: int,
        last_access: float,
        *,
        expected_generation: int | None = None,
        expected_hash: Hashable | None = None,
    ) -> JengaPageHandle | None:
        """Acquire one exact cached physical page by group-local block ID.

        This avoids ambiguity when several pages carry the same prefix hash.
        The returned lease embeds the current generation; callers that retained
        a lookup result can additionally pass its generation and hash for an
        atomic stale-reference check.
        """

        self._validate_request_id(request_id)
        self._validate_timestamp(last_access)
        position = self._position_for_block_id(group_id, block_id)
        if position is None:
            return None
        superpage_id, slot = position
        large_page = self._state.large_pages[superpage_id]
        page = large_page.pages[slot]
        if page.block_hash is None or page.state is PageState.EMPTY:
            return None
        if expected_generation is not None and page.generation != expected_generation:
            raise ValueError("cached block generation is stale")
        if expected_hash is not None:
            self._validate_hash(expected_hash)
            if page.block_hash != expected_hash:
                raise ValueError("cached block hash is stale")
        page.active_requests[request_id] = page.active_requests.get(request_id, 0) + 1
        page.state = PageState.USED
        page.last_access = last_access
        self._state.stats.cache_hits += 1
        handle = JengaPageHandle(
            request_id=request_id,
            group_id=group_id,
            block_id=block_id,
            superpage_id=superpage_id,
            slot=slot,
            generation=page.generation,
        )
        self.check_invariants()
        return handle

    def get_request_handle(
        self,
        request_id: Hashable,
        group_id: int,
        block_id: int,
    ) -> JengaPageHandle | None:
        """Reconstruct a generation-safe handle for a current USED lease."""

        self._validate_request_id(request_id)
        position = self._position_for_block_id(group_id, block_id)
        if position is None:
            return None
        superpage_id, slot = position
        page = self._state.large_pages[superpage_id].pages[slot]
        if page.state is not PageState.USED or page.active_requests.get(request_id, 0) <= 0:
            return None
        return JengaPageHandle(
            request_id=request_id,
            group_id=group_id,
            block_id=block_id,
            superpage_id=superpage_id,
            slot=slot,
            generation=page.generation,
        )

    def clear_request_affinity(self, request_id: Hashable) -> None:
        """Forget placement preference without invalidating cached data."""

        self._validate_request_id(request_id)
        for large_page in self._state.large_pages[1:]:
            if large_page.affinity_request_id == request_id:
                large_page.affinity_request_id = None

    def reset(self, *, reset_stats: bool = True) -> bool:
        """Clear cached metadata only when no page is in use.

        A live request makes reset fail atomically and return ``False``.  On a
        successful reset every allocatable large page becomes unowned.  Policy
        counters reset by default; callers doing lifetime telemetry can pass
        ``reset_stats=False``.  The generation counter is intentionally kept so
        pre-reset handles can never become valid again.
        """

        if any(page.state is PageState.USED for large_page in self._state.large_pages[1:] for page in large_page.pages):
            return False
        stats = _MutableStats() if reset_stats else self._state.stats.clone()
        self._state = _PoolState(
            large_pages=[_LargePage(owner_group_id=-1)] + [_LargePage() for _ in range(self.plan.num_superpages - 1)],
            hash_index={},
            stats=stats,
            next_generation=self._state.next_generation,
        )
        self.check_invariants()
        return True

    def large_page(self, superpage_id: int) -> JengaLargePageSnapshot:
        if not 0 <= superpage_id < self.plan.num_superpages:
            raise ValueError("superpage ID is out of range")
        return self._large_page_snapshot(superpage_id, self._state.large_pages[superpage_id])

    def snapshot(self) -> JengaPoolSnapshot:
        hash_entries = frozenset(
            (group_id, block_hash, superpage_id, slot)
            for (group_id, block_hash), positions in self._state.hash_index.items()
            for superpage_id, slot in positions
        )
        return JengaPoolSnapshot(
            large_pages=tuple(
                self._large_page_snapshot(superpage_id, large_page)
                for superpage_id, large_page in enumerate(self._state.large_pages)
            ),
            hash_entries=hash_entries,
            stats=self.stats,
        )

    def check_invariants(self) -> None:
        self._check_state_invariants(self._state)

    def _simulate(
        self,
        request_id: Hashable,
        block_counts: Mapping[int, int],
        protected: Iterable[JengaPageHandle | JengaCachedPage],
    ) -> tuple[_PoolState, list[AllocationDecision]]:
        self._validate_request_id(request_id)
        counts = self._normalise_counts(block_counts)
        simulated = self._state.clone()
        protected_positions = self._protect_in_simulation(simulated, protected)
        decisions: list[AllocationDecision] = []
        try:
            for group_id, count in counts:
                for _ in range(count):
                    decisions.append(self._allocate_one(simulated, request_id, group_id))
        except JengaOutOfPagesError:
            # The live object was never mutated; discard the speculative state.
            raise
        self._unprotect_simulation(simulated, protected_positions)
        return simulated, decisions

    def _protect_in_simulation(
        self,
        state: _PoolState,
        protected: Iterable[JengaPageHandle | JengaCachedPage],
    ) -> tuple[tuple[int, int], ...]:
        protected_positions: set[tuple[int, int]] = set()
        for reference in protected:
            if isinstance(reference, JengaPageHandle):
                live_page = self._page_for_handle(reference)
                group_id = reference.group_id
                superpage_id = reference.superpage_id
                slot = reference.slot
                generation = reference.generation
            elif isinstance(reference, JengaCachedPage):
                self._validate_group(reference.group_id)
                if not 0 < reference.superpage_id < self.plan.num_superpages:
                    raise ValueError("protected cached page has an invalid superpage ID")
                live_large_page = self._state.large_pages[reference.superpage_id]
                if live_large_page.owner_group_id != reference.group_id:
                    raise ValueError("protected cached page is stale after an ownership rebind")
                if not 0 <= reference.slot < len(live_large_page.pages):
                    raise ValueError("protected cached page has an invalid slot")
                live_page = live_large_page.pages[reference.slot]
                group_id = reference.group_id
                superpage_id = reference.superpage_id
                slot = reference.slot
                generation = reference.generation
                if live_page.block_hash != reference.block_hash:
                    raise ValueError("protected cached page hash is stale")
            else:
                raise TypeError("protected entries must be page handles or cached pages")

            if live_page.generation != generation or live_page.block_hash is None:
                raise ValueError("only a current cached page can be protected")
            if self._state.large_pages[superpage_id].owner_group_id != group_id:
                raise ValueError("protected page belongs to a different group")
            position = (superpage_id, slot)
            if position in protected_positions:
                continue
            simulated_page = state.large_pages[superpage_id].pages[slot]
            if simulated_page.state is PageState.EVICTABLE:
                simulated_page.state = PageState.USED
                simulated_page.active_requests[_DRY_RUN_PROTECTION] = 1
                protected_positions.add(position)
        return tuple(sorted(protected_positions))

    @staticmethod
    def _unprotect_simulation(
        state: _PoolState,
        protected_positions: Iterable[tuple[int, int]],
    ) -> None:
        for superpage_id, slot in protected_positions:
            page = state.large_pages[superpage_id].pages[slot]
            if page.active_requests.pop(_DRY_RUN_PROTECTION, 0) != 1:
                raise AssertionError("temporary admission protection was lost")
            if page.active_requests:
                page.state = PageState.USED
            elif page.block_hash is not None:
                page.state = PageState.EVICTABLE
            else:
                raise AssertionError("a protected prefix page lost its hash")

    def _allocate_one(
        self,
        state: _PoolState,
        request_id: Hashable,
        group_id: int,
    ) -> AllocationDecision:
        # 1. An empty small page already associated with this request.
        candidate = self._first_empty(state, group_id, request_id)
        if candidate is not None:
            return self._claim_empty(state, request_id, group_id, candidate, AllocationTier.REQUEST_AFFINE_EMPTY)

        # 2. A globally empty (unowned) large page.
        for superpage_id, large_page in enumerate(state.large_pages[1:], start=1):
            if large_page.owner_group_id is None:
                self._bind_large_page(state, superpage_id, group_id, request_id)
                return self._claim_empty(
                    state,
                    request_id,
                    group_id,
                    (superpage_id, 0),
                    AllocationTier.EMPTY_LARGE_PAGE,
                )

        # 3. Globally evict the LRU large page only when every child is
        # independently EVICTABLE, then rebind its ownership in place.
        large_candidates = [
            superpage_id
            for superpage_id, large_page in enumerate(state.large_pages[1:], start=1)
            if large_page.pages and all(page.state is PageState.EVICTABLE for page in large_page.pages)
        ]
        if large_candidates:
            superpage_id = min(
                large_candidates,
                key=lambda page_id: self._large_page_lru_key(state, page_id),
            )
            old_owner = state.large_pages[superpage_id].owner_group_id
            evicted = self._rebind_large_page(state, superpage_id, group_id, request_id)
            decision = self._claim_empty(
                state,
                request_id,
                group_id,
                (superpage_id, 0),
                AllocationTier.EVICTABLE_LARGE_PAGE,
                previous_owner_group_id=old_owner,
                evicted_pages=evicted,
            )
            return decision

        # 4. An arbitrary empty small page of the requested type.
        candidate = self._first_empty(state, group_id, affinity_request_id=None, match_affinity=False)
        if candidate is not None:
            return self._claim_empty(state, request_id, group_id, candidate, AllocationTier.ANY_EMPTY)

        # 5. Evict the target type's LRU small page.
        small_candidates: list[tuple[int, int]] = []
        for superpage_id, large_page in enumerate(state.large_pages[1:], start=1):
            if large_page.owner_group_id != group_id:
                continue
            small_candidates.extend(
                (superpage_id, slot) for slot, page in enumerate(large_page.pages) if page.state is PageState.EVICTABLE
            )
        if small_candidates:
            superpage_id, slot = min(
                small_candidates,
                key=lambda position: self._small_page_position_lru_key(state, group_id, *position),
            )
            page = state.large_pages[superpage_id].pages[slot]
            assert page.block_hash is not None
            evicted = (
                EvictedPage(
                    group_id=group_id,
                    block_id=self._block_id(group_id, superpage_id, slot),
                    superpage_id=superpage_id,
                    slot=slot,
                    block_hash=page.block_hash,
                ),
            )
            self._evict_small_page(state, group_id, superpage_id, slot)
            return self._claim_empty(
                state,
                request_id,
                group_id,
                (superpage_id, slot),
                AllocationTier.EVICTABLE_SMALL_PAGE,
                evicted_pages=evicted,
            )

        raise JengaOutOfPagesError(
            f"cannot allocate a page for group {group_id}; USED pages are never eviction candidates"
        )

    def _first_empty(
        self,
        state: _PoolState,
        group_id: int,
        affinity_request_id: Hashable | None,
        *,
        match_affinity: bool = True,
    ) -> tuple[int, int] | None:
        for superpage_id, large_page in enumerate(state.large_pages[1:], start=1):
            if large_page.owner_group_id != group_id:
                continue
            if match_affinity and affinity_request_id != large_page.affinity_request_id:
                continue
            for slot, page in enumerate(large_page.pages):
                if page.state is PageState.EMPTY:
                    return superpage_id, slot
        return None

    def _claim_empty(
        self,
        state: _PoolState,
        request_id: Hashable,
        group_id: int,
        position: tuple[int, int],
        tier: AllocationTier,
        *,
        previous_owner_group_id: int | None = None,
        evicted_pages: tuple[EvictedPage, ...] = (),
    ) -> AllocationDecision:
        superpage_id, slot = position
        large_page = state.large_pages[superpage_id]
        page = large_page.pages[slot]
        if large_page.owner_group_id != group_id or page.state is not PageState.EMPTY:
            raise AssertionError("the five-tier selector returned a non-empty page")
        page.state = PageState.USED
        page.active_requests[request_id] = 1
        page.generation = state.next_generation
        state.next_generation += 1
        state.stats.allocations += 1
        state.stats.tier_counts[int(tier) - 1] += 1
        handle = JengaPageHandle(
            request_id=request_id,
            group_id=group_id,
            block_id=self._block_id(group_id, superpage_id, slot),
            superpage_id=superpage_id,
            slot=slot,
            generation=page.generation,
        )
        return AllocationDecision(
            handle=handle,
            tier=tier,
            previous_owner_group_id=previous_owner_group_id,
            evicted_pages=evicted_pages,
        )

    def _bind_large_page(
        self,
        state: _PoolState,
        superpage_id: int,
        group_id: int,
        request_id: Hashable,
    ) -> None:
        large_page = state.large_pages[superpage_id]
        if large_page.owner_group_id is not None or large_page.pages:
            raise AssertionError("only an empty large page can be initially bound")
        large_page.owner_group_id = group_id
        large_page.affinity_request_id = request_id
        large_page.pages = [_SmallPage() for _ in range(self.plan.capacity(group_id))]

    def _rebind_large_page(
        self,
        state: _PoolState,
        superpage_id: int,
        group_id: int,
        request_id: Hashable,
    ) -> tuple[EvictedPage, ...]:
        large_page = state.large_pages[superpage_id]
        old_owner = large_page.owner_group_id
        if old_owner is None or not large_page.pages:
            raise AssertionError("cannot rebind an empty large page through eviction")
        if not all(page.state is PageState.EVICTABLE for page in large_page.pages):
            raise AssertionError("large-page eviction requires every child to be EVICTABLE")

        evicted: list[EvictedPage] = []
        for slot, page in enumerate(large_page.pages):
            if page.block_hash is None:
                raise AssertionError("an EVICTABLE page must have a prefix hash")
            evicted.append(
                EvictedPage(
                    group_id=old_owner,
                    block_id=self._block_id(old_owner, superpage_id, slot),
                    superpage_id=superpage_id,
                    slot=slot,
                    block_hash=page.block_hash,
                )
            )
            self._remove_hash(state, old_owner, page.block_hash, superpage_id, slot)

        state.stats.small_page_evictions += len(evicted)
        state.stats.large_page_evictions += 1
        if old_owner != group_id:
            state.stats.ownership_rebinds += 1
        large_page.owner_group_id = group_id
        large_page.affinity_request_id = request_id
        large_page.pages = [_SmallPage() for _ in range(self.plan.capacity(group_id))]
        return tuple(evicted)

    def _evict_small_page(
        self,
        state: _PoolState,
        group_id: int,
        superpage_id: int,
        slot: int,
    ) -> None:
        page = state.large_pages[superpage_id].pages[slot]
        if page.state is not PageState.EVICTABLE or page.block_hash is None:
            raise AssertionError("small-page eviction requires an EVICTABLE cached page")
        self._remove_hash(state, group_id, page.block_hash, superpage_id, slot)
        self._make_empty(page)
        state.stats.small_page_evictions += 1

    @staticmethod
    def _make_empty(page: _SmallPage) -> None:
        if page.active_requests:
            raise AssertionError("a leased page cannot become EMPTY")
        page.state = PageState.EMPTY
        page.block_hash = None
        page.prefix_length = 0
        page.last_access = 0.0

    def _large_page_lru_key(self, state: _PoolState, superpage_id: int) -> tuple[float, int]:
        large_page = state.large_pages[superpage_id]
        return max(page.last_access for page in large_page.pages), superpage_id

    def _small_page_position_lru_key(
        self,
        state: _PoolState,
        group_id: int,
        superpage_id: int,
        slot: int,
    ) -> tuple[float, int, int]:
        page = state.large_pages[superpage_id].pages[slot]
        return self.small_page_lru_key(
            page.last_access,
            page.prefix_length,
            self._block_id(group_id, superpage_id, slot),
        )

    def _block_id(self, group_id: int, superpage_id: int, slot: int) -> int:
        return self.plan.to_group_local_id(TypedBlockId(group_id, superpage_id, slot))

    def _position_for_block_id(self, group_id: int, block_id: int) -> tuple[int, int] | None:
        self._validate_group(group_id)
        if isinstance(block_id, bool) or not isinstance(block_id, int) or block_id < 0:
            raise ValueError("block_id must be a non-negative integer")
        try:
            typed_id = self.plan.from_group_local_id(group_id, block_id)
        except ValueError as exc:
            raise ValueError("group-local block ID is out of range") from exc
        if typed_id.superpage_id == 0:
            return None
        large_page = self._state.large_pages[typed_id.superpage_id]
        if large_page.owner_group_id != group_id:
            return None
        if typed_id.slot >= len(large_page.pages):
            raise AssertionError("owned large page has an invalid group capacity")
        return typed_id.superpage_id, typed_id.slot

    def _page_for_handle(
        self,
        handle: JengaPageHandle,
        *,
        require_lease: bool = False,
    ) -> _SmallPage:
        self._validate_request_id(handle.request_id)
        self._validate_group(handle.group_id)
        if not 0 < handle.superpage_id < self.plan.num_superpages:
            raise ValueError("page handle has an invalid superpage ID")
        large_page = self._state.large_pages[handle.superpage_id]
        if large_page.owner_group_id != handle.group_id:
            raise ValueError("page handle is stale after an ownership rebind")
        if not 0 <= handle.slot < len(large_page.pages):
            raise ValueError("page handle has an invalid slot")
        if handle.block_id != self._block_id(handle.group_id, handle.superpage_id, handle.slot):
            raise ValueError("page handle has an invalid group-local block ID")
        page = large_page.pages[handle.slot]
        if page.generation != handle.generation:
            raise ValueError("page handle is stale after page reuse")
        if require_lease and page.active_requests.get(handle.request_id, 0) <= 0:
            raise ValueError("request does not hold this page")
        return page

    @staticmethod
    def _add_hash(
        state: _PoolState,
        group_id: int,
        block_hash: Hashable,
        superpage_id: int,
        slot: int,
    ) -> None:
        state.hash_index.setdefault((group_id, block_hash), set()).add((superpage_id, slot))

    @staticmethod
    def _remove_hash(
        state: _PoolState,
        group_id: int,
        block_hash: Hashable,
        superpage_id: int,
        slot: int,
    ) -> None:
        key = (group_id, block_hash)
        positions = state.hash_index.get(key)
        if positions is None or (superpage_id, slot) not in positions:
            raise AssertionError("prefix hash index is inconsistent")
        positions.remove((superpage_id, slot))
        if not positions:
            del state.hash_index[key]

    def _normalise_counts(self, block_counts: Mapping[int, int]) -> tuple[tuple[int, int], ...]:
        result = []
        for group_id, count in block_counts.items():
            self._validate_group(group_id)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("block counts must be non-negative integers")
            if count:
                result.append((group_id, count))
        return tuple(sorted(result))

    def _validate_group(self, group_id: int) -> None:
        if group_id not in self._group_ids:
            raise KeyError(f"unknown KV cache group {group_id}")

    @staticmethod
    def _validate_request_id(request_id: Hashable) -> None:
        if request_id is None:
            raise ValueError("request_id must not be None")
        try:
            hash(request_id)
        except TypeError as exc:
            raise TypeError("request_id must be hashable") from exc

    @staticmethod
    def _validate_hash(block_hash: Hashable) -> None:
        if block_hash is None:
            raise ValueError("block_hash must not be None")
        try:
            hash(block_hash)
        except TypeError as exc:
            raise TypeError("block_hash must be hashable") from exc

    @classmethod
    def _validate_prefix_metadata(cls, prefix_length: int, last_access: float) -> None:
        if isinstance(prefix_length, bool) or not isinstance(prefix_length, int) or prefix_length < 0:
            raise ValueError("prefix_length must be a non-negative integer")
        cls._validate_timestamp(last_access)

    @staticmethod
    def _validate_timestamp(last_access: float) -> None:
        if not isinstance(last_access, (int, float)) or not math.isfinite(last_access):
            raise ValueError("last_access must be finite")

    def _cached_snapshot(
        self,
        group_id: int,
        superpage_id: int,
        slot: int,
        page: _SmallPage,
    ) -> JengaCachedPage:
        if page.block_hash is None:
            raise AssertionError("cached snapshot requested for a hashless page")
        return JengaCachedPage(
            group_id=group_id,
            block_id=self._block_id(group_id, superpage_id, slot),
            superpage_id=superpage_id,
            slot=slot,
            generation=page.generation,
            state=page.state,
            block_hash=page.block_hash,
            prefix_length=page.prefix_length,
            last_access=page.last_access,
        )

    def _large_page_snapshot(
        self,
        superpage_id: int,
        large_page: _LargePage,
    ) -> JengaLargePageSnapshot:
        pages: list[JengaPageSnapshot] = []
        if large_page.owner_group_id not in (None, -1):
            group_id = large_page.owner_group_id
            for slot, page in enumerate(large_page.pages):
                pages.append(
                    JengaPageSnapshot(
                        group_id=group_id,
                        block_id=self._block_id(group_id, superpage_id, slot),
                        superpage_id=superpage_id,
                        slot=slot,
                        state=page.state,
                        block_hash=page.block_hash,
                        prefix_length=page.prefix_length,
                        last_access=page.last_access,
                        generation=page.generation,
                        active_requests=tuple(sorted(page.active_requests.items(), key=lambda item: repr(item[0]))),
                    )
                )
        return JengaLargePageSnapshot(
            superpage_id=superpage_id,
            owner_group_id=large_page.owner_group_id,
            affinity_request_ids=(
                frozenset() if large_page.affinity_request_id is None else frozenset((large_page.affinity_request_id,))
            ),
            pages=tuple(pages),
        )

    def _check_state_invariants(self, state: _PoolState) -> None:
        if len(state.large_pages) != self.plan.num_superpages:
            raise AssertionError("large-page count diverged from the exact-LCM plan")
        null_page = state.large_pages[0]
        if null_page.owner_group_id != -1 or null_page.pages or null_page.affinity_request_id is not None:
            raise AssertionError("superpage zero must remain reserved and metadata-free")

        expected_hash_index: dict[tuple[int, Hashable], set[tuple[int, int]]] = {}
        for superpage_id, large_page in enumerate(state.large_pages[1:], start=1):
            owner = large_page.owner_group_id
            if owner is None:
                if large_page.pages or large_page.affinity_request_id is not None:
                    raise AssertionError("an empty large page must not retain typed metadata")
                continue
            if owner not in self._group_ids:
                raise AssertionError("large page has an unknown owner group")
            if len(large_page.pages) != self.plan.capacity(owner):
                raise AssertionError("large page has the wrong small-page capacity")
            if all(page.state is PageState.EMPTY for page in large_page.pages):
                raise AssertionError("a completely empty large page must be returned to level zero")

            for slot, page in enumerate(large_page.pages):
                if page.state is PageState.EMPTY:
                    if page.block_hash is not None or page.active_requests or page.prefix_length != 0:
                        raise AssertionError("EMPTY page retains live or cached metadata")
                elif page.state is PageState.EVICTABLE:
                    if page.block_hash is None or page.active_requests:
                        raise AssertionError("EVICTABLE page must be cached and unleased")
                elif page.state is PageState.USED:
                    if not page.active_requests:
                        raise AssertionError("USED page must have an active request lease")
                else:
                    raise AssertionError("unknown small-page state")

                if page.block_hash is not None:
                    expected_hash_index.setdefault((owner, page.block_hash), set()).add((superpage_id, slot))
                if page.generation < 0:
                    raise AssertionError("page generation must be non-negative")

        if expected_hash_index != state.hash_index:
            raise AssertionError("prefix hash index does not match typed pages")
        if state.stats.tier_counts and sum(state.stats.tier_counts) != state.stats.allocations:
            raise AssertionError("allocation-tier statistics are inconsistent")
        if len(state.stats.tier_counts) != 5:
            raise AssertionError("allocation-tier statistics must cover five levels")
