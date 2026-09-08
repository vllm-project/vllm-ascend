# SPDX-License-Identifier: Apache-2.0
"""vLLM v0.27.1 metadata adapter for the exact-LCM Jenga cache.

The policy in :mod:`jenga_prefix_cache` owns physical large-page placement,
request affinity, and the two eviction levels.  vLLM's ``BlockPool`` remains
the owner of ``KVCacheBlock`` objects and prefix-hash metadata.  This adapter
keeps the two views synchronized while presenting the interfaces used by the
vLLM single-type cache managers.

The first runtime version intentionally supports whole-page prefix hits only.
KV events/metrics, connector-directed eviction or offload, and partial-hit
copy-on-write are rejected (or, for partial-cache registration, ignored) at a
clear boundary instead of silently corrupting ownership metadata.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock

from .jenga_prefix_cache import (
    JengaAllocationPlan,
    JengaCachedPage,
    JengaPageHandle,
    JengaPrefixCache,
    PageState,
)
from .typed_kv_cache import TypedKVCachePlan

_PARTIAL_COW_ERROR = (
    "the exact-LCM Jenga runtime does not support partial-prefix copy-on-write; disable fine-grained partial hash hits"
)


@dataclass(frozen=True, slots=True)
class _Operation:
    request_id: Hashable
    timestamp: float


@dataclass(slots=True)
class _PreparedAllocation:
    request_id: Hashable
    plan: JengaAllocationPlan
    pending: dict[int, deque[JengaPageHandle]]
    consumed: int = 0
    consumed_blocks: list[tuple[int, KVCacheBlock, JengaPageHandle]] = field(default_factory=list)
    touched_blocks: list[tuple[int, KVCacheBlock, JengaPageHandle | None]] = field(default_factory=list)

    @property
    def num_pending(self) -> int:
        return sum(len(handles) for handles in self.pending.values())


@dataclass(frozen=True, slots=True)
class _DeferredFree:
    request_id: Hashable
    group_id: int
    block: KVCacheBlock
    handle: JengaPageHandle
    timestamp: float


class JengaPrefixRuntimePool:
    """Global exact-LCM allocator and aggregate ``BlockPool`` facade.

    A separate upstream ``BlockPool`` is created for every KV cache group so
    block IDs remain local to that group's physical page size.  Allocation is
    a two-step transaction:

    1. ``prepare_allocation`` commits one atomic multi-group policy decision.
    2. group managers consume those exact decisions through ``get_new_blocks``.

    The coordinator must wrap the complete operation in ``request_context``
    and call ``finish_allocation``.  This makes touches, allocations, caching,
    and frees for one request share a single eviction timestamp.
    """

    enable_caching = True
    enable_kv_cache_events = False
    num_gpu_blocks = 0  # Aggregate admission uses a 0/1 coordinator sentinel.

    def __init__(
        self,
        plan: TypedKVCachePlan,
        hash_block_size: int,
        *,
        enable_kv_cache_events: bool = False,
        metrics_collector: object | None = None,
    ) -> None:
        if plan.is_addressed or plan.is_partitioned:
            raise ValueError("Jenga prefix runtime requires an exact-LCM plan")
        if isinstance(hash_block_size, bool) or not isinstance(hash_block_size, int) or hash_block_size <= 0:
            raise ValueError("hash_block_size must be a positive integer")
        if enable_kv_cache_events:
            raise NotImplementedError("Jenga prefix runtime does not emit KV cache events")
        if metrics_collector is not None:
            raise NotImplementedError("Jenga prefix runtime metrics are not implemented")

        self.plan = plan
        self.hash_block_size = hash_block_size
        self.policy = JengaPrefixCache(plan)
        self._clock = 0
        self._poisoned_error: str | None = None
        self._operation: ContextVar[_Operation | None] = ContextVar(f"jenga_prefix_operation_{id(self)}", default=None)
        self._prepared: _PreparedAllocation | None = None
        # ``pop_blocks_for_free`` removes a request from manager bookkeeping
        # before an outer caller returns the flattened blocks to this aggregate
        # pool.  Keep an identity-only hand-off so that later free cannot be
        # confused with an equal-valued block from another group.
        self._deferred_frees: dict[int, _DeferredFree] = {}

        self._metadata_pools: dict[int, BlockPool] = {}
        self._facades: dict[int, JengaPrefixGroupBlockPool] = {}
        self._block_owners: dict[int, tuple[int, KVCacheBlock]] = {}
        for spec in plan.specs:
            group_id = spec.group_id
            metadata_pool = BlockPool(
                num_gpu_blocks=plan.num_blocks(group_id),
                enable_caching=True,
                hash_block_size=hash_block_size,
                enable_kv_cache_events=False,
                metrics_collector=None,
            )
            self._reserve_null_superpage(metadata_pool, group_id)
            self._metadata_pools[group_id] = metadata_pool
            for block in metadata_pool.blocks:
                self._block_owners[id(block)] = (group_id, block)

        self._facades = {group_id: JengaPrefixGroupBlockPool(self, group_id) for group_id in self._metadata_pools}
        first_group_id = min(self._facades)
        self.null_block = self._facades[first_group_id].null_block

    def _reserve_null_superpage(self, metadata_pool: BlockPool, group_id: int) -> None:
        """Remove every group-local view of physical superpage zero."""

        capacity = self.plan.capacity(group_id)
        if metadata_pool.null_block.block_id != 0:
            raise AssertionError("upstream BlockPool no longer reserves block zero")
        for block_id in range(1, capacity):
            block = metadata_pool.blocks[block_id]
            if block.ref_cnt != 0:
                raise AssertionError("new upstream block unexpectedly has a reference")
            metadata_pool.free_block_queue.remove(block)
            # These IDs alias the shared NULL large page.  They are deliberately
            # not marked is_null: only canonical block ID zero may enter manager
            # block tables as padding.
            block.ref_cnt = 1

    def for_group(self, group_id: int) -> JengaPrefixGroupBlockPool:
        try:
            return self._facades[group_id]
        except KeyError as exc:
            raise KeyError(f"unknown KV cache group {group_id}") from exc

    @contextmanager
    def request_context(
        self,
        request_id: Hashable,
        *,
        timestamp: float | None = None,
    ) -> Iterator[JengaPrefixRuntimePool]:
        """Bind manager callbacks to a request and one aligned timestamp."""

        self._validate_request_id(request_id)
        current = self._operation.get()
        if current is not None:
            if current.request_id != request_id:
                raise RuntimeError("cannot nest Jenga contexts for different requests")
            if timestamp is not None and float(timestamp) != current.timestamp:
                raise RuntimeError("a nested Jenga context cannot change its timestamp")
            yield self
            return

        if timestamp is None:
            self._clock += 1
            operation_timestamp = float(self._clock)
        else:
            operation_timestamp = float(timestamp)
            if not math.isfinite(operation_timestamp):
                raise ValueError("timestamp must be finite")
            self._clock = max(self._clock, math.ceil(operation_timestamp))
        token = self._operation.set(_Operation(request_id, operation_timestamp))
        try:
            yield self
        finally:
            self._operation.reset(token)

    def can_allocate(
        self,
        request_id: Hashable,
        counts_by_group: Mapping[int, int],
        *,
        protected_blocks: Mapping[int, Sequence[KVCacheBlock]] | Iterable[KVCacheBlock] = (),
    ) -> bool:
        self._raise_if_poisoned()
        protected = self._protected_references(protected_blocks)
        return self.policy.can_allocate(request_id, counts_by_group, protected=protected)

    def prepare_allocation(
        self,
        request_id: Hashable,
        counts_by_group: Mapping[int, int],
        *,
        protected_blocks: Mapping[int, Sequence[KVCacheBlock]] | Iterable[KVCacheBlock] = (),
    ) -> JengaAllocationPlan:
        """Commit and stage one atomic allocation for all group managers."""

        self._raise_if_poisoned()
        self._require_operation(request_id)
        if self._prepared is not None:
            raise RuntimeError(f"request {self._prepared.request_id!r} still has a prepared allocation")
        protected = self._protected_references(protected_blocks)
        preview = self.policy.dry_run(request_id, counts_by_group, protected=protected)
        self._validate_allocation_plan(preview)
        committed = self.policy.allocate(request_id, counts_by_group, protected=protected)
        pending = {group_id: deque() for group_id in self._facades}
        for decision in committed.decisions:
            pending[decision.handle.group_id].append(decision.handle)
        self._prepared = _PreparedAllocation(request_id, committed, pending)
        try:
            if self._decision_signature(preview) != self._decision_signature(committed):
                raise AssertionError("Jenga allocation changed between dry run and commit")
            self._synchronize_evictions(committed)
        except Exception:
            # Keep policy leases and upstream metadata internally consistent
            # even when an invariant guard or a future BlockPool version fails
            # while publishing the committed decision.
            self.abort_allocation(request_id)
            raise
        return committed

    def finish_allocation(self, request_id: Hashable) -> JengaAllocationPlan:
        """Close a transaction after every staged handle was consumed."""

        self._require_operation(request_id)
        prepared = self._require_prepared(request_id)
        if prepared.num_pending:
            remaining = {group_id: len(handles) for group_id, handles in prepared.pending.items() if handles}
            raise RuntimeError(f"prepared Jenga blocks were not consumed: {remaining}")
        self._prepared = None
        return prepared.plan

    def cancel_allocation(self, request_id: Hashable) -> None:
        """Cancel an unconsumed transaction without exposing policy leases."""

        operation = self._require_operation(request_id)
        prepared = self._require_prepared(request_id)
        if prepared.consumed:
            raise RuntimeError("cannot cancel after a staged Jenga block was consumed")
        handles = tuple(handle for pending in prepared.pending.values() for handle in pending)
        if handles:
            self.policy.release(handles, last_access=operation.timestamp)
        self.policy.clear_request_affinity(request_id)
        self._prepared = None

    def abort_allocation(self, request_id: Hashable) -> None:
        """Abort a prepared transaction, including already consumed pages.

        This is the failure-only counterpart of :meth:`cancel_allocation`.
        Acquisition logs let it reverse both cache-hit touches and staged-page
        consumption before the coordinator restores manager bookkeeping.
        Prefix entries evicted by the original allocation stay evicted, but
        both allocators are left usable for a retry or a subsequent request.
        """

        operation = self._require_operation(request_id)
        prepared = self._require_prepared(request_id)
        handles: list[JengaPageHandle] = []
        for decision in prepared.plan.decisions:
            handle = decision.handle
            current = self.policy.get_request_handle(
                request_id,
                handle.group_id,
                handle.block_id,
            )
            if current is None:
                raise RuntimeError("aborted Jenga allocation lease disappeared")
            if current != handle:
                raise RuntimeError("aborted Jenga page generation changed")
            handles.append(current)
        for _, _, handle in prepared.touched_blocks:
            if handle is None:
                continue
            current = self.policy.get_request_handle(
                request_id,
                handle.group_id,
                handle.block_id,
            )
            if current != handle:
                raise RuntimeError("aborted Jenga cache-hit lease changed")
            handles.append(current)

        metadata_releases: dict[int, list[KVCacheBlock]] = {group_id: [] for group_id in self._metadata_pools}
        for group_id, block, _ in prepared.consumed_blocks:
            metadata_releases[group_id].append(block)
        for group_id, block, _ in prepared.touched_blocks:
            metadata_releases[group_id].append(block)
        required_references: dict[int, tuple[KVCacheBlock, int]] = {}
        for blocks in metadata_releases.values():
            for block in blocks:
                identity = id(block)
                canonical, count = required_references.get(identity, (block, 0))
                if canonical is not block:
                    raise RuntimeError("aborted Jenga block identity changed")
                required_references[identity] = (block, count + 1)
        if any(block.ref_cnt < count for block, count in required_references.values()):
            raise RuntimeError("aborted Jenga block lost an upstream reference")

        if handles:
            self.policy.release(handles, last_access=operation.timestamp)
        for group_id, blocks in metadata_releases.items():
            if blocks:
                self._metadata_pools[group_id].free_blocks(reversed(blocks))
        self._prepared = None

    def has_prepared_allocation(self, request_id: Hashable) -> bool:
        prepared = self._prepared
        return prepared is not None and prepared.request_id == request_id

    def clear_request_affinity(self, request_id: Hashable) -> None:
        """End the AE request-to-large-page association after request free."""

        self.policy.clear_request_affinity(request_id)

    def _consume(self, group_id: int, num_blocks: int) -> list[KVCacheBlock]:
        if isinstance(num_blocks, bool) or not isinstance(num_blocks, int) or num_blocks < 0:
            raise ValueError("num_blocks must be a non-negative integer")
        if num_blocks == 0:
            return []
        operation = self._require_operation()
        prepared = self._require_prepared(operation.request_id)
        pending = prepared.pending[group_id]
        if num_blocks > len(pending):
            raise ValueError(f"group {group_id} requested {num_blocks} blocks but only {len(pending)} were prepared")
        handles = [pending[index] for index in range(num_blocks)]
        metadata_pool = self._metadata_pools[group_id]
        blocks = [metadata_pool.blocks[handle.block_id] for handle in handles]
        for block in blocks:
            if block.ref_cnt != 0 or block.is_null:
                raise AssertionError("staged Jenga block is not free upstream")
            if block.block_hash is not None:
                raise AssertionError("staged Jenga block retains a stale prefix hash")

        removed: list[KVCacheBlock] = []
        try:
            for block in blocks:
                metadata_pool.free_block_queue.remove(block)
                removed.append(block)
                # Required even after the explicit eviction synchronization:
                # this is the final stale-hash guard at the allocation boundary.
                if metadata_pool._maybe_evict_cached_block(block):
                    raise AssertionError("an EMPTY Jenga page unexpectedly had a hash")
                block.ref_cnt += 1
        except Exception:
            # Restore every block already removed from the upstream free queue.
            # Pending handles are deliberately left staged so the coordinator's
            # abort path can release the policy transaction as one unit.
            for block in removed:
                if block.ref_cnt == 0:
                    block.ref_cnt = 1
            if removed:
                metadata_pool.free_blocks(reversed(removed))
            raise
        for _ in range(num_blocks):
            pending.popleft()
        prepared.consumed += num_blocks
        prepared.consumed_blocks.extend((group_id, block, handle) for block, handle in zip(blocks, handles))
        return blocks

    def get_cached_block(self, block_hash: Any, kv_cache_group_ids: list[int]) -> list[KVCacheBlock] | None:
        cached: list[KVCacheBlock] = []
        for group_id in kv_cache_group_ids:
            facade = self.for_group(group_id)
            group_hit = facade.get_cached_block(block_hash, [group_id])
            if not group_hit:
                return None
            cached.append(group_hit[0])
        return cached

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Route a flattened cache-hit list by object identity."""

        grouped: dict[int, list[KVCacheBlock]] = {}
        for block in blocks:
            if block.is_null:
                continue
            group_id, canonical = self._owner_of(block)
            grouped.setdefault(group_id, []).append(canonical)
        for group_id, group_blocks in grouped.items():
            self._facades[group_id].touch(group_blocks)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Route flattened blocks safely; equal-valued foreign objects fail."""

        materialized = [block for block in ordered_blocks if not block.is_null]
        if not materialized:
            return
        deferred = [id(block) in self._deferred_frees for block in materialized]
        if any(deferred):
            if not all(deferred):
                raise ValueError("cannot mix deferred and direct Jenga frees")
            self._finish_deferred_free(materialized)
            return

        grouped: dict[int, list[KVCacheBlock]] = {}
        for block in materialized:
            group_id, canonical = self._owner_of(block)
            grouped.setdefault(group_id, []).append(canonical)
        for group_id, group_blocks in grouped.items():
            self._facades[group_id].free_blocks(group_blocks)

    def release_for_deferred_free(
        self,
        request_id: Hashable,
        ordered_blocks: Iterable[KVCacheBlock],
        *,
        timestamp: float | None = None,
    ) -> None:
        """Register policy leases for a later outer aggregate free.

        This is the adapter for ``coordinator.pop_blocks_for_free``.  The
        request lease remains USED until the later ``free_blocks`` call, so
        unrelated requests may keep allocating without evicting a page whose
        upstream reference has not yet been returned.
        """

        self._validate_request_id(request_id)
        if self._prepared is not None:
            raise RuntimeError("cannot defer a free during a prepared allocation")
        operation = self._operation.get()
        if operation is not None:
            if operation.request_id != request_id:
                raise RuntimeError("deferred free request does not match request_context")
            release_timestamp = operation.timestamp
            if timestamp is not None and float(timestamp) != release_timestamp:
                raise RuntimeError("deferred free cannot change the active timestamp")
        elif timestamp is None:
            self._clock += 1
            release_timestamp = float(self._clock)
        else:
            release_timestamp = float(timestamp)
            if not math.isfinite(release_timestamp):
                raise ValueError("timestamp must be finite")
            self._clock = max(self._clock, math.ceil(release_timestamp))

        materialized = [block for block in ordered_blocks if not block.is_null]
        pending: list[_DeferredFree] = []
        seen: set[int] = set()
        for block in materialized:
            identity = id(block)
            if identity in seen or identity in self._deferred_frees:
                raise ValueError("a Jenga block cannot be deferred twice")
            seen.add(identity)
            group_id, canonical = self._owner_of(block)
            if canonical.ref_cnt <= 0:
                raise ValueError("cannot defer an unreferenced Jenga block")
            handle = self.policy.get_request_handle(request_id, group_id, canonical.block_id)
            if handle is None:
                raise ValueError("request does not own this deferred Jenga block")
            pending.append(_DeferredFree(request_id, group_id, canonical, handle, release_timestamp))

        for deferred in pending:
            self._deferred_frees[id(deferred.block)] = deferred

    def _finish_deferred_free(self, blocks: Sequence[KVCacheBlock]) -> None:
        grouped: dict[int, list[KVCacheBlock]] = {}
        releases: dict[float, list[JengaPageHandle]] = {}
        seen: set[int] = set()
        for block in blocks:
            identity = id(block)
            if identity in seen:
                raise ValueError("a deferred Jenga block was returned twice")
            seen.add(identity)
            pending = self._deferred_frees.get(identity)
            if pending is None or pending.block is not block:
                raise ValueError("block does not match a pending deferred Jenga free")
            current_handle = self.policy.get_request_handle(
                pending.request_id, pending.group_id, pending.block.block_id
            )
            if current_handle != pending.handle:
                raise ValueError("deferred Jenga lease changed before aggregate free")
            grouped.setdefault(pending.group_id, []).append(block)
            releases.setdefault(pending.timestamp, []).append(pending.handle)
        for timestamp, handles in releases.items():
            self.policy.release(handles, last_access=timestamp)
        for group_id, group_blocks in grouped.items():
            self._metadata_pools[group_id].free_blocks(group_blocks)
        for block in blocks:
            del self._deferred_frees[id(block)]

    def cache_full_blocks(self, *args: Any, **kwargs: Any) -> None:
        group_id = kwargs.get("kv_cache_group_id")
        if group_id is None and len(args) >= 6:
            group_id = args[5]
        self.for_group(group_id).cache_full_blocks(*args, **kwargs)

    def cache_partial_block(self, *args: Any, **kwargs: Any) -> None:
        # Whole-page/common-LCM mode deliberately does not publish a partial
        # hash.  Returning None makes the upstream manager skip partial COW.
        return None

    def move_block_hashes(self, src_block: KVCacheBlock, dst_block: KVCacheBlock) -> None:
        raise NotImplementedError(_PARTIAL_COW_ERROR)

    def evict_blocks(self, block_ids: set[int]) -> None:
        raise NotImplementedError("connector-directed block eviction is unsupported by the Jenga runtime")

    def emit_cached_block_events(self, *args: Any, **kwargs: Any) -> None:
        return None

    def reset_prefix_cache(self) -> bool:
        """Clear all hashes only when no request or transaction is live."""

        if self._prepared is not None or self._deferred_frees:
            return False
        for group_id, metadata_pool in self._metadata_pools.items():
            reserved = self.plan.capacity(group_id)
            if any(block.ref_cnt != 0 for block in metadata_pool.blocks[reserved:]):
                return False
        if not self.policy.reset():
            return False
        for group_id, metadata_pool in self._metadata_pools.items():
            reserved = self.plan.capacity(group_id)
            for block in metadata_pool.blocks[reserved:]:
                metadata_pool._maybe_evict_cached_block(block)
        return True

    def get_num_free_blocks(self) -> int:
        """Aggregate admission sentinel; the typed coordinator returns 0/1."""

        return 0

    def get_usage(self) -> float:
        allocatable = self.plan.num_superpages - 1
        if allocatable <= 0:
            return 1.0
        owned = sum(large_page.owner_group_id is not None for large_page in self.policy.snapshot().large_pages[1:])
        return owned / allocatable

    def take_events(self) -> list[object]:
        return []

    def _touch_group(self, group_id: int, blocks: Sequence[KVCacheBlock]) -> None:
        operation = self._require_operation()
        prepared = self._prepared
        if prepared is not None and prepared.request_id != operation.request_id:
            raise RuntimeError(f"request {prepared.request_id!r} still has a prepared allocation")
        metadata_pool = self._metadata_pools[group_id]
        materialized = [block for block in blocks if not block.is_null]
        handles: list[JengaPageHandle] = []
        acquired: list[tuple[int, KVCacheBlock, JengaPageHandle | None]] = []
        reference_counts: dict[int, tuple[KVCacheBlock, int]] = {}
        for block in materialized:
            identity = id(block)
            if identity not in reference_counts:
                reference_counts[identity] = (block, block.ref_cnt)
        try:
            for block in materialized:
                self._validate_group_block(group_id, block)
                if block.block_hash is None:
                    raise ValueError("only a cached Jenga block can be touched")
                reference = self._cached_reference(group_id, block)
                if reference is None:
                    raise AssertionError("upstream hash has no matching Jenga page")
                handle = self.policy.acquire_cached_block(
                    operation.request_id,
                    group_id,
                    block.block_id,
                    operation.timestamp,
                    expected_generation=reference.generation,
                    expected_hash=reference.block_hash,
                )
                if handle is None:
                    raise AssertionError("Jenga page disappeared during cache-hit touch")
                handles.append(handle)
                acquired.append((group_id, block, handle))
            metadata_pool.touch(materialized)
        except Exception:
            # BlockPool.touch mutates one block at a time.  If a future metrics
            # hook throws midway, undo precisely the increments that landed.
            touched: list[KVCacheBlock] = []
            for block, old_ref_cnt in reference_counts.values():
                delta = block.ref_cnt - old_ref_cnt
                if delta < 0:
                    raise RuntimeError("failed Jenga touch decreased a reference count")
                touched.extend([block] * delta)
            if touched:
                metadata_pool.free_blocks(reversed(touched))
            if handles:
                self.policy.release(handles)
            raise
        if prepared is not None:
            prepared.touched_blocks.extend(acquired)

    def _free_group(self, group_id: int, blocks: Iterable[KVCacheBlock]) -> None:
        operation = self._require_operation()
        materialized = [block for block in blocks if not block.is_null]
        handles: list[JengaPageHandle] = []
        for block in materialized:
            self._validate_group_block(group_id, block)
            if id(block) in self._deferred_frees:
                raise ValueError("deferred Jenga block must be freed through the aggregate pool")
            if block.ref_cnt <= 0:
                raise ValueError("cannot free an unreferenced Jenga block")
            handle = self.policy.get_request_handle(operation.request_id, group_id, block.block_id)
            if handle is None:
                raise ValueError("request does not own this Jenga block")
            handles.append(handle)
        if handles:
            self.policy.release(handles, last_access=operation.timestamp)
        self._metadata_pools[group_id].free_blocks(materialized)

    def _cache_full_group(
        self,
        group_id: int,
        request: Any,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        block_mask: list[bool] | None,
    ) -> None:
        operation = self._require_operation(request.request_id)
        new_blocks = blocks[num_cached_blocks:num_full_blocks]
        if block_mask is not None and len(block_mask) != len(new_blocks):
            raise ValueError("block_mask does not match the newly full blocks")
        cacheable: list[tuple[int, KVCacheBlock, JengaPageHandle]] = []
        for offset, block in enumerate(new_blocks):
            if block.is_null or (block_mask is not None and not block_mask[offset]):
                continue
            self._validate_group_block(group_id, block)
            handle = self.policy.get_request_handle(request.request_id, group_id, block.block_id)
            if handle is None:
                raise ValueError("request cannot cache a Jenga block it does not own")
            cacheable.append((offset, block, handle))

        metadata_pool = self._metadata_pools[group_id]
        metadata_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=block_size,
            kv_cache_group_id=group_id,
            block_mask=block_mask,
        )
        for _, block, handle in cacheable:
            if block.block_hash is None or block.block_hash_num_tokens is None:
                raise AssertionError("upstream BlockPool did not publish a full hash")
            self.policy.mark_cacheable(
                handle,
                block.block_hash,
                prefix_length=block.block_hash_num_tokens,
                last_access=operation.timestamp,
            )

    def _protected_references(
        self,
        protected_blocks: Mapping[int, Sequence[KVCacheBlock]] | Iterable[KVCacheBlock],
    ) -> tuple[JengaCachedPage, ...]:
        pairs: list[tuple[int, KVCacheBlock]] = []
        if isinstance(protected_blocks, Mapping):
            for group_id, blocks in protected_blocks.items():
                self.for_group(group_id)
                pairs.extend((group_id, block) for block in blocks if not block.is_null)
        else:
            for block in protected_blocks:
                if block.is_null:
                    continue
                group_id, canonical = self._owner_of(block)
                pairs.append((group_id, canonical))

        references: list[JengaCachedPage] = []
        seen: set[tuple[int, int]] = set()
        for group_id, block in pairs:
            self._validate_group_block(group_id, block)
            key = (group_id, block.block_id)
            if key in seen:
                continue
            seen.add(key)
            reference = self._cached_reference(group_id, block)
            if reference is None:
                raise ValueError("only current cached blocks can be protected")
            references.append(reference)
        return tuple(references)

    def _cached_reference(self, group_id: int, block: KVCacheBlock) -> JengaCachedPage | None:
        reference = self.policy.get_cached_page_by_block(group_id, block.block_id)
        if reference is None:
            return None
        if block.block_hash != reference.block_hash:
            raise AssertionError("upstream and Jenga prefix hashes diverged")
        return reference

    def _validate_allocation_plan(self, allocation: JengaAllocationPlan) -> None:
        evicted_keys = {
            (evicted.group_id, evicted.block_id)
            for decision in allocation.decisions
            for evicted in decision.evicted_pages
        }
        for decision in allocation.decisions:
            handle = decision.handle
            block = self._metadata_pools[handle.group_id].blocks[handle.block_id]
            if block.ref_cnt != 0 or block.is_null:
                raise AssertionError("Jenga selected a non-free upstream block")
            if block.block_hash is not None and (handle.group_id, handle.block_id) not in evicted_keys:
                raise AssertionError("Jenga selected a block with an unplanned stale hash")
            for evicted in decision.evicted_pages:
                old_block = self._metadata_pools[evicted.group_id].blocks[evicted.block_id]
                if old_block.ref_cnt != 0:
                    raise AssertionError("Jenga tried to evict an in-use upstream block")
                if old_block.block_hash != evicted.block_hash:
                    raise AssertionError("Jenga and upstream eviction hashes diverged")

    def _synchronize_evictions(self, allocation: JengaAllocationPlan) -> None:
        expected: dict[tuple[int, int], JengaCachedPage] = {}
        for decision in allocation.decisions:
            for evicted in decision.evicted_pages:
                key = (evicted.group_id, evicted.block_id)
                previous = expected.setdefault(key, evicted)
                if previous.block_hash != evicted.block_hash:
                    raise AssertionError("one Jenga page has conflicting eviction hashes")
        # Validate the complete batch before removing the first hash.  The
        # cleanup below is then deterministic even if a future BlockPool hook
        # unexpectedly raises partway through the loop.
        for (group_id, block_id), evicted in expected.items():
            block = self._metadata_pools[group_id].blocks[block_id]
            if block.ref_cnt != 0:
                raise AssertionError("Jenga tried to synchronize an in-use eviction")
            if block.block_hash != evicted.block_hash:
                raise AssertionError("planned Jenga eviction hash changed before synchronization")
        try:
            for (group_id, block_id), _ in expected.items():
                block = self._metadata_pools[group_id].blocks[block_id]
                if not self._metadata_pools[group_id]._maybe_evict_cached_block(block):
                    raise AssertionError("planned Jenga eviction had no upstream hash")
        except Exception:
            cleanup_errors: list[str] = []
            for (group_id, block_id), evicted in expected.items():
                block = self._metadata_pools[evicted.group_id].blocks[evicted.block_id]
                if block.block_hash is None:
                    continue
                if block.block_hash != evicted.block_hash:
                    cleanup_errors.append(f"group={group_id} block={block_id}: hash changed")
                    continue
                try:
                    self._metadata_pools[group_id]._maybe_evict_cached_block(block)
                except Exception as error:  # pragma: no cover - fail-closed guard
                    cleanup_errors.append(f"group={group_id} block={block_id}: {error}")
            if cleanup_errors:
                self._poisoned_error = "failed to finish Jenga eviction cleanup: " + "; ".join(cleanup_errors)
                raise RuntimeError(self._poisoned_error)
            raise

    @staticmethod
    def _decision_signature(
        allocation: JengaAllocationPlan,
    ) -> tuple[tuple[int, int, int, tuple[tuple[int, int], ...]], ...]:
        return tuple(
            (
                int(decision.tier),
                decision.handle.group_id,
                decision.handle.block_id,
                tuple((page.group_id, page.block_id) for page in decision.evicted_pages),
            )
            for decision in allocation.decisions
        )

    def _group_num_free_blocks(self, group_id: int) -> int:
        capacity = self.plan.capacity(group_id)
        free = 0
        for large_page in self.policy.snapshot().large_pages[1:]:
            if large_page.owner_group_id is None:
                free += capacity
            elif large_page.owner_group_id == group_id:
                free += sum(page.state is not PageState.USED for page in large_page.pages)
            elif large_page.pages and all(page.state is PageState.EVICTABLE for page in large_page.pages):
                free += capacity
        return free

    def _owner_of(self, block: KVCacheBlock) -> tuple[int, KVCacheBlock]:
        owner = self._block_owners.get(id(block))
        if owner is None or owner[1] is not block:
            raise ValueError("block was not created by this Jenga runtime")
        return owner

    def _validate_group_block(self, group_id: int, block: KVCacheBlock) -> None:
        owner_group, canonical = self._owner_of(block)
        if owner_group != group_id or canonical is not block:
            raise ValueError("block belongs to a different Jenga KV cache group")

    def _require_operation(self, request_id: Hashable | None = None) -> _Operation:
        operation = self._operation.get()
        if operation is None:
            raise RuntimeError("Jenga manager operation requires request_context()")
        if request_id is not None and operation.request_id != request_id:
            raise RuntimeError(f"active Jenga request is {operation.request_id!r}, not {request_id!r}")
        return operation

    def _require_prepared(self, request_id: Hashable) -> _PreparedAllocation:
        prepared = self._prepared
        if prepared is None or prepared.request_id != request_id:
            raise RuntimeError(f"request {request_id!r} has no prepared Jenga allocation")
        return prepared

    def _raise_if_poisoned(self) -> None:
        if self._poisoned_error is not None:
            raise RuntimeError(
                f"Jenga prefix runtime is unavailable after incomplete eviction cleanup: {self._poisoned_error}"
            )

    @staticmethod
    def _validate_request_id(request_id: Hashable) -> None:
        if request_id is None:
            raise ValueError("request_id must not be None")
        try:
            hash(request_id)
        except TypeError as exc:
            raise TypeError("request_id must be hashable") from exc


class JengaPrefixGroupBlockPool:
    """Group-local ``BlockPool`` facade consumed by one vLLM manager."""

    enable_caching = True
    enable_kv_cache_events = False

    def __init__(self, runtime: JengaPrefixRuntimePool, group_id: int) -> None:
        self._runtime = runtime
        self.group_id = group_id
        metadata_pool = runtime._metadata_pools[group_id]
        self.num_gpu_blocks = metadata_pool.num_gpu_blocks
        self.hash_block_size = runtime.hash_block_size
        self.null_block = metadata_pool.null_block

    @property
    def blocks(self) -> list[KVCacheBlock]:
        return self._runtime._metadata_pools[self.group_id].blocks

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        return self._runtime._consume(self.group_id, num_blocks)

    def get_cached_block(self, block_hash: Any, kv_cache_group_ids: list[int]) -> list[KVCacheBlock] | None:
        if any(group_id != self.group_id for group_id in kv_cache_group_ids):
            raise ValueError("a group facade cannot look up another KV cache group")
        hit = self._runtime._metadata_pools[self.group_id].get_cached_block(block_hash, kv_cache_group_ids)
        if hit:
            for block in hit:
                if self._runtime._cached_reference(self.group_id, block) is None:
                    raise AssertionError("upstream returned a stale Jenga prefix hit")
        return hit

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        self._runtime._touch_group(self.group_id, blocks)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        self._runtime._free_group(self.group_id, ordered_blocks)

    def cache_full_blocks(
        self,
        request: Any,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        if kv_cache_group_id != self.group_id:
            raise ValueError("cache metadata belongs to a different KV cache group")
        self._runtime._cache_full_group(
            self.group_id,
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            block_mask,
        )

    def cache_partial_block(self, *args: Any, **kwargs: Any) -> None:
        return None

    def move_block_hashes(self, src_block: KVCacheBlock, dst_block: KVCacheBlock) -> None:
        raise NotImplementedError(_PARTIAL_COW_ERROR)

    def evict_blocks(self, block_ids: set[int]) -> None:
        raise NotImplementedError("connector-directed block eviction is unsupported by the Jenga runtime")

    def emit_cached_block_events(self, *args: Any, **kwargs: Any) -> None:
        return None

    def reset_prefix_cache(self) -> bool:
        return self._runtime.reset_prefix_cache()

    def get_num_free_blocks(self) -> int:
        return self._runtime._group_num_free_blocks(self.group_id)

    def get_usage(self) -> float:
        return self._runtime.get_usage()

    def take_events(self) -> list[object]:
        return []
