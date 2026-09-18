# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Global shared-slot planning for the periodic mainline replica router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_SLOT_GAIN_EPSILON = 1e-12
# Policy 4 owns its stability and migration thresholds. Keeping these values
# internal avoids a second set of user-facing tuning knobs and, importantly,
# prevents low-signal decode windows from forcing repeated weight migration.
_MIN_EFFECTIVE_GAIN = 0.01
_REQUIRED_STABLE_WINDOWS = 2
_POST_APPLY_COOLDOWN_WINDOWS = 1
_MIGRATION_AMORTIZATION_WINDOWS = 4.0
_NORMALIZED_MIGRATION_COST = 0.02
_HEAT_EMA_ALPHA = 0.25


@dataclass(frozen=True)
class GlobalExpertPoolDecision:
    should_apply: bool
    reason: str
    placement: tuple[tuple[tuple[int, ...], ...], ...]
    gain: float
    changed_slots: int


def _logical_heat(table: np.ndarray, workload: np.ndarray, num_experts: int) -> np.ndarray:
    heat = np.zeros((table.shape[0], num_experts), dtype=np.float64)
    for layer_id in range(table.shape[0]):
        valid = table[layer_id] >= 0
        np.add.at(heat[layer_id], table[layer_id][valid], workload[layer_id][valid])
    return heat


def _routing_shares(holders: np.ndarray, routing_table_rows: int, logical_ids: Any = None) -> np.ndarray:
    """Estimate shares under uniform source ranks and routing-table rows.

    Mainline selects (row + source_rank + logical_id) % copies. Aggregate
    heat does not describe expert occurrences by row or source, so this is
    an explicit uniform-position model, not an exact replay of real traffic.
    """
    num_ranks = holders.shape[-1]
    copies = holders.sum(axis=-1, keepdims=True)
    if np.any(copies == 0):
        raise ValueError("each logical expert must have at least one holder")
    if logical_ids is None:
        logical_ids = np.arange(holders.shape[-2])
    logical_ids = np.asarray(logical_ids)[..., None]
    ordinal = np.cumsum(holders, axis=-1) - 1
    row_tail = routing_table_rows % copies
    rank_tail = num_ranks % copies
    offset = (ordinal - logical_ids) % copies
    # Full periods contribute equally. Intersect the two remaining cyclic
    # intervals instead of enumerating every source rank for every candidate.
    tail = np.maximum(0, np.minimum(rank_tail, offset + 1) - np.maximum(0, offset - row_tail + 1))
    tail += np.maximum(0, np.minimum(rank_tail, offset + copies + 1) - np.maximum(0, offset + copies - row_tail + 1))
    counts = num_ranks * (routing_table_rows // copies) + (num_ranks // copies) * row_tail + tail
    return np.where(holders, counts / (num_ranks * routing_table_rows), 0.0)


def _allocate_critical_path_replicas(
    base: np.ndarray,
    heat: np.ndarray,
    slots_per_rank: int,
    num_ranks: int,
    routing_table_rows: int,
) -> list[list[tuple[int, int]]]:
    """Greedily minimize the summed per-layer critical path under routing.

    Evaluate one destination rank at a time to avoid materializing a
    [layers, experts, destination ranks, holder ranks] tensor.
    """
    num_layers, _, _ = base.shape
    num_experts = heat.shape[1]
    holders = np.zeros((num_layers, num_experts, num_ranks), dtype=bool)
    for layer_id in range(num_layers):
        for rank_id in range(num_ranks):
            holders[layer_id, base[layer_id, rank_id], rank_id] = True

    contributions = heat[:, :, None] * _routing_shares(holders, routing_table_rows)
    rank_load = contributions.sum(axis=1)
    remaining = np.full(num_ranks, slots_per_rank, dtype=np.int64)
    targets_by_rank: list[list[tuple[int, int]]] = [[] for _ in range(num_ranks)]
    critical_gain = np.full(holders.shape, -np.inf)
    second_moment_gain = np.full(holders.shape, -np.inf)
    dirty_layers = slice(None)

    for _ in range(slots_per_rank * num_ranks):
        # Adding a replica changes only its layer's loads and contributions.
        # Keep other layers' exact scores, invalidating exhausted ranks below.
        layer_holders = holders[dirty_layers]
        layer_heat = heat[dirty_layers]
        layer_load = rank_load[dirty_layers]
        peak = layer_load.max(axis=-1)[:, None]
        second_moment = np.square(layer_load).sum(axis=-1)[:, None]
        for rank_id in range(num_ranks):
            if remaining[rank_id] == 0:
                continue
            proposed = layer_holders.copy()
            proposed[:, :, rank_id] = True
            candidate_loads = (
                layer_load[:, None, :]
                - contributions[dirty_layers]
                + layer_heat[:, :, None] * _routing_shares(proposed, routing_table_rows)
            )
            valid = (~layer_holders[:, :, rank_id]) & (layer_heat > 0)
            critical_gain[dirty_layers, :, rank_id] = np.where(valid, peak - candidate_loads.max(axis=-1), -np.inf)
            second_moment_gain[dirty_layers, :, rank_id] = np.where(
                valid, second_moment - np.square(candidate_loads).sum(axis=-1), -np.inf
            )

        best_primary = float(np.max(critical_gain))
        if not np.isfinite(best_primary):
            break
        if best_primary > _SLOT_GAIN_EPSILON:
            eligible = critical_gain >= best_primary - _SLOT_GAIN_EPSILON
        else:
            # A second-moment improvement must not increase the critical path.
            eligible = critical_gain >= -_SLOT_GAIN_EPSILON
        score = np.where(eligible, second_moment_gain, -np.inf)
        if best_primary <= _SLOT_GAIN_EPSILON and float(np.max(score)) <= _SLOT_GAIN_EPSILON:
            break

        layer_id, expert_id, rank_id = (int(value) for value in np.unravel_index(int(np.argmax(score)), score.shape))
        holders[layer_id, expert_id, rank_id] = True
        updated = heat[layer_id, expert_id] * _routing_shares(
            holders[layer_id, expert_id], routing_table_rows, expert_id
        )
        rank_load[layer_id] += updated - contributions[layer_id, expert_id]
        contributions[layer_id, expert_id] = updated
        remaining[rank_id] -= 1
        targets_by_rank[rank_id].append((layer_id, expert_id))
        if remaining[rank_id] == 0:
            critical_gain[:, :, rank_id] = -np.inf
            second_moment_gain[:, :, rank_id] = -np.inf
        dirty_layers = slice(layer_id, layer_id + 1)

    return targets_by_rank


def _align_shared_slots(
    current: np.ndarray,
    base_slots: int,
    targets_by_rank: list[list[tuple[int, int]]],
) -> np.ndarray:
    candidate = current.copy()
    candidate[:, :, base_slots:] = -1
    num_layers, num_ranks, local_slots = current.shape
    shared_slots = local_slots - base_slots
    for rank_id in range(num_ranks):
        remaining = list(targets_by_rank[rank_id])
        assigned: list[tuple[int, int] | None] = [None] * shared_slots
        old_owner: list[tuple[int, int] | None] = []
        for slot_id in range(shared_slots):
            owners = [
                (layer_id, int(current[layer_id, rank_id, base_slots + slot_id]))
                for layer_id in range(num_layers)
                if current[layer_id, rank_id, base_slots + slot_id] >= 0
            ]
            if len(owners) > 1:
                raise ValueError("one shared physical slot is owned by multiple layers")
            old_owner.append(owners[0] if owners else None)
        for slot_id, owner in enumerate(old_owner):
            if owner is not None and owner in remaining:
                assigned[slot_id] = owner
                remaining.remove(owner)
        for slot_id, owner in enumerate(old_owner):
            if assigned[slot_id] is not None or owner is None:
                continue
            same_layer = next((item for item in remaining if item[0] == owner[0]), None)
            if same_layer is not None:
                assigned[slot_id] = same_layer
                remaining.remove(same_layer)
        remaining.sort()
        for slot_id in range(shared_slots):
            if assigned[slot_id] is None and remaining:
                assigned[slot_id] = remaining.pop(0)
        if remaining:
            raise ValueError("target placement exceeds the shared slot budget")
        for slot_id, owner in enumerate(assigned):
            if owner is not None:
                layer_id, expert_id = owner
                candidate[layer_id, rank_id, base_slots + slot_id] = expert_id
    return candidate


def _rank_loads_by_layer(table: np.ndarray, heat: np.ndarray, num_ranks: int, routing_table_rows: int) -> np.ndarray:
    holders = np.zeros((table.shape[0], heat.shape[1], num_ranks), dtype=bool)
    for layer_id in range(table.shape[0]):
        for rank_id in range(num_ranks):
            experts = table[layer_id, rank_id]
            holders[layer_id, experts[experts >= 0], rank_id] = True
    return (heat[:, :, None] * _routing_shares(holders, routing_table_rows)).sum(axis=1)


def _immutable(table: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(tuple(tuple(int(value) for value in rank) for rank in layer) for layer in table)


def _migration_penalty(changed_slots: int, capacity: int) -> float:
    return _NORMALIZED_MIGRATION_COST * changed_slots / capacity / _MIGRATION_AMORTIZATION_WINDOWS


class GlobalExpertPoolPlanner:
    """Plan shared slots with conservative, implementation-owned gating."""

    def __init__(self, num_redundant_experts: int, *, routing_table_rows: int) -> None:
        if isinstance(num_redundant_experts, bool) or not isinstance(num_redundant_experts, int):
            raise TypeError("num_redundant_experts must be an integer")
        if num_redundant_experts <= 0:
            raise ValueError("policy 4 requires num_redundant_experts > 0")
        if isinstance(routing_table_rows, bool) or not isinstance(routing_table_rows, int):
            raise TypeError("routing_table_rows must be an integer")
        if routing_table_rows <= 0:
            raise ValueError("routing_table_rows must be positive")
        self.routing_table_rows = routing_table_rows
        self.slots_per_rank = int(num_redundant_experts)
        self._heat_distribution_ema: np.ndarray | None = None
        self._last_candidate: np.ndarray | None = None
        self._pending_base: np.ndarray | None = None
        self._stable_windows = 0
        self._cooldown_remaining = 0

    def _smooth_heat_distribution(self, heat: np.ndarray) -> np.ndarray:
        totals = heat.sum(axis=-1, keepdims=True)
        distribution = np.divide(
            heat,
            totals,
            out=np.zeros_like(heat, dtype=np.float64),
            where=totals > 0,
        )
        if self._heat_distribution_ema is None or self._heat_distribution_ema.shape != distribution.shape:
            smoothed = distribution
        else:
            smoothed = self._heat_distribution_ema.copy()
            observed = totals[:, 0] > 0
            smoothed[observed] = (1.0 - _HEAT_EMA_ALPHA) * smoothed[observed] + _HEAT_EMA_ALPHA * distribution[observed]
            smoothed_totals = smoothed.sum(axis=-1, keepdims=True)
            smoothed = np.divide(
                smoothed,
                smoothed_totals,
                out=np.zeros_like(smoothed),
                where=smoothed_totals > 0,
            )
        self._heat_distribution_ema = smoothed.copy()
        return smoothed * totals

    def plan(self, current_expert_table: Any, expert_workload: Any) -> GlobalExpertPoolDecision:
        table = np.asarray(current_expert_table, dtype=np.int64)
        workload = np.asarray(expert_workload, dtype=np.float64)
        if table.ndim != 3 or workload.shape != table.shape or 0 in table.shape:
            raise ValueError("policy 4 expects matching non-empty [layers, ranks, local_slots] table and workload")
        if not np.all(np.isfinite(workload)) or np.any(workload < 0):
            raise ValueError("policy 4 workload must be non-negative and finite")
        num_layers, num_ranks, local_slots = table.shape
        base_slots = local_slots - self.slots_per_rank
        if base_slots <= 0:
            raise ValueError("global slots must be smaller than local physical capacity")
        num_experts = base_slots * num_ranks
        pool_capacity = self.slots_per_rank * num_ranks
        for layer_id in range(num_layers):
            base = table[layer_id, :, :base_slots].reshape(-1)
            if np.any(base < 0) or not np.array_equal(np.sort(base), np.arange(num_experts)):
                raise ValueError(f"layer {layer_id} base slots must contain one immutable copy per expert")

        heat = self._smooth_heat_distribution(_logical_heat(table, workload, num_experts))
        targets_by_rank = _allocate_critical_path_replicas(
            table[:, :, :base_slots], heat, self.slots_per_rank, num_ranks, self.routing_table_rows
        )

        candidate = _align_shared_slots(table, base_slots, targets_by_rank)
        current_rank_loads = _rank_loads_by_layer(table, heat, num_ranks, self.routing_table_rows)
        # Stability tracks sustained benefit, not identical greedy tie breaks.
        # Revalidate the pending placement against fresh heat and the same base.
        if (
            self._last_candidate is not None
            and self._pending_base is not None
            and np.array_equal(table, self._pending_base)
        ):
            pending_loads = _rank_loads_by_layer(self._last_candidate, heat, num_ranks, self.routing_table_rows)
            current_peak = current_rank_loads.max(axis=-1)
            pending_gain = float((current_peak - pending_loads.max(axis=-1)).sum()) / max(
                float(current_peak.sum()), _SLOT_GAIN_EPSILON
            )
            pending_changes = int(
                np.count_nonzero(np.any(self._last_candidate[:, :, base_slots:] != table[:, :, base_slots:], axis=0))
            )
            pending_cost = _migration_penalty(pending_changes, pool_capacity)
            if pending_changes > 0 and pending_gain - pending_cost >= _MIN_EFFECTIVE_GAIN:
                candidate = self._last_candidate.copy()
        else:
            self._last_candidate = None
            self._stable_windows = 0
        predicted_rank_loads = _rank_loads_by_layer(candidate, heat, num_ranks, self.routing_table_rows)
        current_critical_path = current_rank_loads.max(axis=-1)
        predicted_critical_path = predicted_rank_loads.max(axis=-1)
        raw_gain = float((current_critical_path - predicted_critical_path).sum()) / max(
            float(current_critical_path.sum()), _SLOT_GAIN_EPSILON
        )
        changed_slots = int(np.count_nonzero(np.any(candidate[:, :, base_slots:] != table[:, :, base_slots:], axis=0)))
        effective_gain = raw_gain - _migration_penalty(changed_slots, pool_capacity)
        if self._last_candidate is not None and np.array_equal(candidate, self._last_candidate):
            self._stable_windows += 1
        else:
            self._last_candidate = candidate.copy()
            self._pending_base = table.copy()
            self._stable_windows = 1

        reason, should_apply = "apply", True
        if changed_slots == 0:
            reason, should_apply = "unchanged", False
        elif effective_gain < _MIN_EFFECTIVE_GAIN:
            reason, should_apply = "insufficient_gain", False
        elif self._stable_windows < _REQUIRED_STABLE_WINDOWS:
            reason, should_apply = "unstable", False
        elif self._cooldown_remaining > 0:
            reason, should_apply = "cooldown", False
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        if should_apply:
            self._cooldown_remaining = _POST_APPLY_COOLDOWN_WINDOWS
        if should_apply or changed_slots == 0 or effective_gain < _MIN_EFFECTIVE_GAIN:
            self._last_candidate = None
            self._pending_base = None
            self._stable_windows = 0

        return GlobalExpertPoolDecision(
            should_apply=should_apply,
            reason=reason,
            placement=_immutable(candidate if should_apply else table),
            gain=effective_gain,
            changed_slots=changed_slots,
        )
