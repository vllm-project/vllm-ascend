# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-fixed cache pools and joint admission with append-only history.

The scheduler owns these IDs. Device state, stream ordering and transfer leases
belong to the worker and connector. Prefix reuse is not supported here.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    FullAttentionManager,
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import get_request_state_plan

Allocation = TypeVar("Allocation")


class FixedStateManager(CircularBufferManager):
    """One slot per request, admitted against this manager's own pool."""

    def get_num_blocks_to_allocate(self, request_id, *args, **kwargs):
        # The upstream scalar gate counts history pages only. The joint
        # admission gate below checks request slots before invoking it.
        return 0


def create_request_state_managers(
    config: KVCacheConfig,
    history_pool: BlockPool,
    *,
    scheduler_block_size: int,
    dcp_world_size: int,
    pcp_world_size: int,
    max_model_len: int,
    max_in_flight_tokens: int,
) -> tuple[SingleTypeKVCacheManager, ...]:
    """Create each manager once, directly on its declared pool.

    The initial contract is one append-only history group plus independent
    request-slot groups. Each BlockPool retains its own reserved null block.
    """
    plan = get_request_state_plan(config)
    if plan is None:
        raise ValueError("Request-state managers require an explicit pool plan")
    if history_pool.enable_caching:
        raise ValueError("Request-state pools do not support prefix caching")
    groups = {group_id for pool in (plan.history, *plan.request_states) for group_id in pool.spec.group_ids}
    if groups != set(range(len(config.kv_cache_groups))):
        raise ValueError("Pool plan must cover every cache group")
    if plan.history.capacity != config.num_blocks:
        raise ValueError("Request-state allocation requires one history pool matching num_blocks")
    managers = []
    for group_id, group in enumerate(config.kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = next(iter(spec.kv_cache_specs.values()))
        pool_plan = plan.pool_for_group(group_id)
        if pool_plan.spec.allocation_unit == "token_block":
            manager = get_manager_for_kv_cache_spec(
                kv_cache_spec=spec,
                block_pool=history_pool,
                enable_caching=False,
                kv_cache_group_id=group_id,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
                scheduler_block_size=scheduler_block_size,
                needs_kv_cache_zeroing=config.needs_kv_cache_zeroing,
            )
            if not isinstance(manager, FullAttentionManager):
                raise ValueError("Request-state rollback requires append-only full-attention history")
        else:
            manager = FixedStateManager(
                spec,
                block_pool=BlockPool(pool_plan.capacity, False, scheduler_block_size),
                enable_caching=False,
                kv_cache_group_id=group_id,
                scheduler_block_size=scheduler_block_size,
                dcp_world_size=1,
                pcp_world_size=1,
                needs_kv_cache_zeroing=False,
            )
        managers.append(manager)
    return tuple(managers)


@dataclass
class _AllocationSnapshot:
    manager: SingleTypeKVCacheManager
    had_blocks: bool
    blocks: tuple[Any, ...]
    had_cached: bool
    cached: int | None
    new_block_ids: tuple[int, ...]

    @classmethod
    def capture(cls, manager: SingleTypeKVCacheManager, request_id: str) -> "_AllocationSnapshot":
        cached = getattr(manager, "num_cached_block", {})
        return cls(
            manager,
            request_id in manager.req_to_blocks,
            tuple(manager.req_to_blocks.get(request_id, ())),
            request_id in cached,
            cached.get(request_id),
            tuple(getattr(manager, "new_block_ids", ())),
        )


def _rollback_appends(snapshots: Sequence[_AllocationSnapshot], request_id: str) -> None:
    # Validate every prefix before returning any blocks. A non-append-only
    # mutation cannot be undone safely by this transaction.
    for snapshot in snapshots:
        current = snapshot.manager.req_to_blocks.get(request_id, ())
        if tuple(current[: len(snapshot.blocks)]) != snapshot.blocks:
            raise RuntimeError("State allocation rollback requires append-only request resources")
    for snapshot in reversed(snapshots):
        manager = snapshot.manager
        current = manager.req_to_blocks.get(request_id, ())
        manager.block_pool.free_blocks(reversed(current[len(snapshot.blocks) :]))
        if snapshot.had_blocks:
            manager.req_to_blocks[request_id] = list(snapshot.blocks)
        else:
            manager.req_to_blocks.pop(request_id, None)
        if hasattr(manager, "num_cached_block"):
            if snapshot.had_cached:
                manager.num_cached_block[request_id] = snapshot.cached
            else:
                manager.num_cached_block.pop(request_id, None)
        if hasattr(manager, "new_block_ids"):
            manager.new_block_ids[:] = snapshot.new_block_ids


def reserve_append_only(
    managers: Sequence[SingleTypeKVCacheManager], request_id: str, allocate: Callable[[], Allocation | None]
) -> Allocation | None:
    """Rollback this attempt's appends on rejection or exception.

    Valid only for the single-threaded, prefix-cache-disabled scheduler with
    append-only history and fixed request slots. This is not DMA cancellation.
    """
    snapshots = [_AllocationSnapshot.capture(manager, request_id) for manager in managers]
    try:
        result = allocate()
    except Exception:
        _rollback_appends(snapshots, request_id)
        raise
    if result is None:
        _rollback_appends(snapshots, request_id)
    return result


def allocate_request_slots(
    managers: Sequence[SingleTypeKVCacheManager], request_id: str, allocate: Callable[[], Allocation | None]
) -> Allocation | None:
    """Gate missing request slots, then use the existing history allocator."""
    missing = [
        manager
        for manager in managers
        if isinstance(manager, FixedStateManager) and not manager.req_to_blocks.get(request_id)
    ]
    if not missing:
        # Resident requests retain the existing decode path, without copying
        # their potentially long history tables on every step.
        return allocate()
    if any(manager.block_pool.get_num_free_blocks() == 0 for manager in missing):
        return None
    return reserve_append_only(managers, request_id, allocate)
