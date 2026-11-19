# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure planning primitives for token-history and per-request state pools.

These descriptors do not allocate memory or change the active KV cache builder.
Byte counts describe device storage on one worker, including layout padding.
"""

from dataclasses import dataclass
from typing import Literal


def _check_count(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


@dataclass(frozen=True)
class CachePoolSpec:
    """Groups sharing an allocation unit and a local ID space.

    ``bytes_per_unit`` is the total physical storage for one ID across all
    backings in this pool, not the size of a single layer or tensor. Different
    tensors may share a pool without sharing a backing allocation.
    """

    pool_id: str
    group_ids: tuple[int, ...]
    allocation_unit: Literal["token_block", "request_slot"]
    bytes_per_unit: int

    def __post_init__(self) -> None:
        if not isinstance(self.pool_id, str) or not self.pool_id.strip():
            raise ValueError("pool_id must be a non-empty string")
        if not isinstance(self.group_ids, tuple) or not self.group_ids:
            raise ValueError("group_ids must be a non-empty tuple")
        for group_id in self.group_ids:
            _check_count("group_id", group_id)
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError("group_ids must be unique within a pool")
        if self.allocation_unit not in ("token_block", "request_slot"):
            raise ValueError(f"Unsupported allocation unit: {self.allocation_unit}")
        _check_count("bytes_per_unit", self.bytes_per_unit, 1)


@dataclass(frozen=True)
class CachePoolPlan:
    """Physical capacity, including any reserved/null IDs.

    For example, a request-state pool serving C requests with one null slot
    needs capacity C + 1. This class never adds another implicit null slot.
    """

    spec: CachePoolSpec
    capacity: int

    def __post_init__(self) -> None:
        _check_count("capacity", self.capacity, 1)

    @property
    def size_bytes(self) -> int:
        return self.capacity * self.spec.bytes_per_unit


@dataclass(frozen=True)
class RequestStateMemoryBudget:
    """Affine device-memory budget: fixed bytes + history blocks * page bytes.

    Workspace is explicit because temporary Decode staging may not appear in
    KV cache tensors. Callers must include it here only if it has not already
    been deducted from the available budget by profiling.
    """

    history_page_bytes: int
    request_state_bytes: int
    workspace_bytes: int = 0

    def __post_init__(self) -> None:
        _check_count("history_page_bytes", self.history_page_bytes, 1)
        _check_count("request_state_bytes", self.request_state_bytes)
        _check_count("workspace_bytes", self.workspace_bytes)

    @property
    def fixed_bytes(self) -> int:
        return self.request_state_bytes + self.workspace_bytes

    def bytes_for_history_blocks(self, num_blocks: int) -> int:
        _check_count("num_blocks", num_blocks)
        return self.fixed_bytes + num_blocks * self.history_page_bytes

    def history_capacity(self, available_memory: int) -> int:
        """Return physical block capacity; zero does not mean fixed pools fit."""
        _check_count("available_memory", available_memory)
        return max(0, (available_memory - self.fixed_bytes) // self.history_page_bytes)

    def resolve_history_capacity(
        self,
        available_memory: int,
        *,
        min_history_blocks: int,
        num_history_blocks: int | None = None,
    ) -> int:
        """Resolve or validate capacity against the full worker budget.

        Both counts include reserved blocks. The caller supplies the minimum
        required by admission (including the null block); this helper cannot
        infer it from a model's maximum sequence length.
        """
        _check_count("min_history_blocks", min_history_blocks, 1)
        capacity = self.history_capacity(available_memory)
        if num_history_blocks is not None:
            _check_count("num_history_blocks", num_history_blocks, 1)
            capacity = num_history_blocks
        if capacity < min_history_blocks:
            raise ValueError(f"History capacity {capacity} is below required minimum {min_history_blocks}")
        required = self.bytes_for_history_blocks(capacity)
        if required > available_memory:
            raise ValueError(f"Pool plan requires {required} bytes, but only {available_memory} bytes are available")
        return capacity


@dataclass(frozen=True)
class RequestStatePoolPlan:
    """One growing history pool and independently sized fixed state pools.

    This is a planning result, not a KVCacheConfig or a runtime allocator.
    No physical memory borrowing or ID translation is implied.
    """

    history: CachePoolPlan
    request_states: tuple[CachePoolPlan, ...]
    workspace_bytes: int = 0

    def __post_init__(self) -> None:
        if self.history.spec.allocation_unit != "token_block":
            raise ValueError("History must use token_block allocation")
        if not isinstance(self.request_states, tuple) or not self.request_states:
            raise ValueError("request_states must be a non-empty tuple")
        _check_count("workspace_bytes", self.workspace_bytes)
        pool_ids: set[str] = set()
        group_ids: set[int] = set()
        for pool in (self.history, *self.request_states):
            if pool is not self.history and pool.spec.allocation_unit != "request_slot":
                raise ValueError("Request states must use request_slot allocation")
            if pool.spec.pool_id in pool_ids:
                raise ValueError(f"Duplicate pool ID: {pool.spec.pool_id}")
            if group_ids.intersection(pool.spec.group_ids):
                raise ValueError("A cache group must belong to exactly one pool")
            pool_ids.add(pool.spec.pool_id)
            group_ids.update(pool.spec.group_ids)

    @property
    def budget(self) -> RequestStateMemoryBudget:
        return RequestStateMemoryBudget(
            history_page_bytes=self.history.spec.bytes_per_unit,
            request_state_bytes=sum(pool.size_bytes for pool in self.request_states),
            workspace_bytes=self.workspace_bytes,
        )

    @property
    def size_bytes(self) -> int:
        return self.budget.bytes_for_history_blocks(self.history.capacity)

    def pool_for_group(self, group_id: int) -> CachePoolPlan:
        _check_count("group_id", group_id)
        for pool in (self.history, *self.request_states):
            if group_id in pool.spec.group_ids:
                return pool
        raise ValueError(f"No pool owns cache group {group_id}")

    def validate_group_coverage(self, num_groups: int) -> None:
        """Require every group in the builder's output to have exactly one owner."""
        _check_count("num_groups", num_groups)
        actual = {group_id for pool in (self.history, *self.request_states) for group_id in pool.spec.group_ids}
        expected = set(range(num_groups))
        if actual != expected:
            raise ValueError(
                f"Cache group coverage mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )


def plan_request_state_pools(
    history: CachePoolSpec,
    request_states: tuple[CachePoolPlan, ...],
    available_memory: int,
    *,
    min_history_blocks: int,
    workspace_bytes: int = 0,
    num_history_blocks: int | None = None,
) -> RequestStatePoolPlan:
    """Reserve fixed pools/workspace, then resolve physical history capacity.

    ``num_history_blocks`` may be an explicit override or the capacity agreed
    across workers. In either case the complete budget is checked again.
    Inputs are immutable and a failed plan leaves no partial reservation.
    """
    # Validate ownership and units before deriving the worker-local budget.
    template = RequestStatePoolPlan(CachePoolPlan(history, 1), request_states, workspace_bytes)
    capacity = template.budget.resolve_history_capacity(
        available_memory,
        min_history_blocks=min_history_blocks,
        num_history_blocks=num_history_blocks,
    )
    return RequestStatePoolPlan(CachePoolPlan(history, capacity), request_states, workspace_bytes)
