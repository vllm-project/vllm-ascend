# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure policy, slot assignment and exact integer capacity planning."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from vllm_ascend.core.layerwise_cache_plan import (
    CacheLayoutAdapter,
    CacheOwnerDescription,
    LayerwisePolicy,
    PhysicalCachePlan,
    validate_physical_plan,
)

DEFAULT_MAX_PREFETCH_LAYERS = 8


def parse_int_config(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be an integer, got {value!r}") from err


def normalize_layerwise_policy(
    physical_layers: Iterable[int], extra_config: dict[str, Any] | None = None, excluded_owners: Iterable[str] = ()
) -> LayerwisePolicy:
    layers = sorted(set(physical_layers))
    if not layers:
        raise ValueError("num_layers must be at least 1")
    config = extra_config or {}
    shared = config.get("layerwise_num_shared_buffers")
    shared = len(layers) if shared is None else parse_int_config(shared, "layerwise_num_shared_buffers")
    prefetch = config.get("layerwise_prefetch_layers")
    prefetch = (
        min(shared, DEFAULT_MAX_PREFETCH_LAYERS)
        if prefetch is None
        else parse_int_config(prefetch, "layerwise_prefetch_layers")
    )
    if shared < 1 or prefetch < 1:
        raise ValueError("layerwise shared buffers and prefetch layers must be at least 1")
    independent = config.get("layerwise_independent_layers")
    if independent is None:
        indices = [0]
    elif isinstance(independent, str) and independent.strip().lower() == "all":
        indices = list(range(len(layers)))
    elif isinstance(independent, list):
        indices = [parse_int_config(index, "layerwise_independent_layers") for index in independent]
    else:
        raise TypeError("layerwise_independent_layers must be a list of integers or 'all'")
    normalized = set()
    for index in indices:
        index = index + len(layers) if index < 0 else index
        if not 0 <= index < len(layers):
            raise ValueError(f"layerwise_independent_layers contains out-of-range layer index {index}")
        normalized.add(layers[index])
    return LayerwisePolicy(shared, prefetch, tuple(sorted(normalized)), tuple(sorted(set(excluded_owners))))


def assign_rotating_slots(owners: Iterable[CacheOwnerDescription], policy: LayerwisePolicy) -> dict[str, int]:
    """Assign linked components together, independently per compatibility class.

    Group boundaries deliberately do not restart a class's execution order.
    Models still have to explicitly authorize any resulting physical aliases.
    """
    buckets: dict[str, dict[str, list[CacheOwnerDescription]]] = {}
    result: dict[str, int] = {}
    for owner in owners:
        if owner.resident or owner.physical_layer in policy.independent_layers or owner.name in policy.excluded_owners:
            continue
        buckets.setdefault(owner.compatibility_class, {}).setdefault(owner.linked_set, []).append(owner)
    for units in buckets.values():
        ordered = sorted(units.values(), key=lambda members: (members[0].physical_layer, members[0].linked_set))
        for ordinal, members in enumerate(ordered):
            if len({owner.physical_layer for owner in members}) != 1:
                raise ValueError("Linked cache components must execute at the same physical layer")
            for owner in members:
                result[owner.name] = ordinal % policy.num_shared_buffers
    return result


def solve_capacity(required_bytes: Callable[[int], int], memory_budget: int, override: int | None = None) -> int:
    """Maximize N for a positive, monotone, unbounded physical cost function."""
    if memory_budget < 0:
        raise ValueError("Cache memory budget must be nonnegative")
    if override is not None:
        if isinstance(override, bool) or not isinstance(override, int) or override < 1:
            raise ValueError("Cache block override must be a positive integer")
        required = required_bytes(override)
        if required > memory_budget:
            raise ValueError(f"Layerwise layout needs {required} bytes, available {memory_budget} bytes")
        return override
    if required_bytes(1) > memory_budget:
        raise ValueError("Cache budget cannot hold one layerwise block")
    low, high = 1, 2
    while required_bytes(high) <= memory_budget:
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if required_bytes(middle) <= memory_budget:
            low = middle
        else:
            high = middle
    return low


def plan_layerwise_cache(
    adapter: CacheLayoutAdapter,
    groups: list[Any],
    policy: LayerwisePolicy,
    memory_budget: int,
    override: int | None = None,
) -> PhysicalCachePlan:
    num_blocks = solve_capacity(
        lambda n: adapter.build_layout(groups, policy, n).required_bytes, memory_budget, override
    )
    return materialize_layerwise_plan(adapter, groups, policy, num_blocks, memory_budget)


def materialize_layerwise_plan(
    adapter: CacheLayoutAdapter,
    groups: list[Any],
    policy: LayerwisePolicy,
    num_blocks: int,
    memory_budget: int,
) -> PhysicalCachePlan:
    """Rebuild at the agreed capacity; never solve for more worker blocks."""
    plan = adapter.build_layout(groups, policy, num_blocks).reserve(memory_budget)
    validate_physical_plan(plan)
    return plan
