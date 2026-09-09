# SPDX-License-Identifier: Apache-2.0
"""P-node global expert pool policy using cross-layer shared slots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .policy_abstract import EplbPolicy

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
    priority: tuple[float, ...]
    gain: float
    changed_slots: int


def _logical_heat(table: np.ndarray, workload: np.ndarray, num_experts: int) -> np.ndarray:
    heat = np.zeros((table.shape[0], num_experts), dtype=np.float64)
    for layer_id in range(table.shape[0]):
        valid = table[layer_id] >= 0
        np.add.at(heat[layer_id], table[layer_id][valid], workload[layer_id][valid])
    return heat


def _allocate_critical_path_replicas(
    base: np.ndarray,
    heat: np.ndarray,
    slots_per_rank: int,
    num_ranks: int,
) -> list[list[tuple[int, int]]]:
    """Greedily minimize the summed per-layer critical-path load."""
    num_layers, _, _ = base.shape
    num_experts = heat.shape[1]
    holders = np.zeros((num_layers, num_experts, num_ranks), dtype=bool)
    rank_load = np.zeros((num_layers, num_ranks), dtype=np.float64)
    for layer_id in range(num_layers):
        for rank_id in range(num_ranks):
            experts = base[layer_id, rank_id]
            holders[layer_id, experts, rank_id] = True
            rank_load[layer_id, rank_id] = float(heat[layer_id, experts].sum())

    copies = holders.sum(axis=-1).astype(np.int64)
    remaining = np.full(num_ranks, slots_per_rank, dtype=np.int64)
    targets_by_rank: list[list[tuple[int, int]]] = [[] for _ in range(num_ranks)]
    rank_indices = np.arange(num_ranks)

    for _ in range(slots_per_rank * num_ranks):
        next_share = np.divide(
            heat,
            copies + 1,
            out=np.zeros_like(heat, dtype=np.float64),
            where=copies >= 0,
        )
        current_share = np.divide(
            heat,
            copies,
            out=np.zeros_like(heat, dtype=np.float64),
            where=copies > 0,
        )
        reduction = current_share - next_share
        reduced_loads = rank_load[:, None, None, :] - holders[:, :, None, :] * reduction[:, :, None, None]
        candidate_loads = np.broadcast_to(
            reduced_loads,
            (num_layers, num_experts, num_ranks, num_ranks),
        ).copy()
        candidate_loads[:, :, rank_indices, rank_indices] += next_share[:, :, None]

        critical_gain = rank_load.max(axis=-1)[:, None, None] - candidate_loads.max(axis=-1)
        second_moment_gain = np.square(rank_load).sum(axis=-1)[:, None, None] - np.square(candidate_loads).sum(axis=-1)
        valid = (~holders) & (copies[:, :, None] < num_ranks) & (heat[:, :, None] > 0) & (remaining[None, None, :] > 0)
        critical_gain[~valid] = -np.inf
        second_moment_gain[~valid] = -np.inf

        best_primary = float(np.max(critical_gain))
        if not np.isfinite(best_primary):
            break
        if best_primary > _SLOT_GAIN_EPSILON:
            eligible = critical_gain >= best_primary - _SLOT_GAIN_EPSILON
            score = np.where(eligible, second_moment_gain, -np.inf)
        else:
            score = second_moment_gain
            if float(np.max(score)) <= _SLOT_GAIN_EPSILON:
                break

        layer_id, expert_id, rank_id = (int(value) for value in np.unravel_index(int(np.argmax(score)), score.shape))
        old_share = current_share[layer_id, expert_id]
        new_share = next_share[layer_id, expert_id]
        rank_load[layer_id, holders[layer_id, expert_id]] += new_share - old_share
        rank_load[layer_id, rank_id] += new_share
        holders[layer_id, expert_id, rank_id] = True
        copies[layer_id, expert_id] += 1
        remaining[rank_id] -= 1
        targets_by_rank[rank_id].append((layer_id, expert_id))

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


def _rank_loads_by_layer(table: np.ndarray, heat: np.ndarray, num_ranks: int) -> np.ndarray:
    result = np.zeros((table.shape[0], num_ranks), dtype=np.float64)
    for layer_id in range(table.shape[0]):
        placement = table[layer_id]
        valid_values = placement[placement >= 0]
        counts = np.bincount(valid_values, minlength=heat.shape[1]).clip(min=1)
        for rank_id in range(num_ranks):
            experts = placement[rank_id]
            experts = experts[experts >= 0]
            result[layer_id, rank_id] = float(np.sum(heat[layer_id, experts] / counts[experts]))
    return result


def _immutable(table: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(tuple(tuple(int(value) for value in rank) for rank in layer) for layer in table)


class GlobalExpertPoolPlanner:
    """Plan shared slots with conservative, implementation-owned gating."""

    def __init__(self, num_redundant_experts: int) -> None:
        if isinstance(num_redundant_experts, bool) or not isinstance(num_redundant_experts, int):
            raise TypeError("num_redundant_experts must be an integer")
        if num_redundant_experts <= 0:
            raise ValueError("policy 4 requires num_redundant_experts > 0")
        self.slots_per_rank = int(num_redundant_experts)
        self._heat_distribution_ema: np.ndarray | None = None
        self._last_candidate: np.ndarray | None = None
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
        for layer_id in range(num_layers):
            base = table[layer_id, :, :base_slots].reshape(-1)
            if np.any(base < 0) or not np.array_equal(np.sort(base), np.arange(num_experts)):
                raise ValueError(f"layer {layer_id} base slots must contain one immutable copy per expert")

        heat = self._smooth_heat_distribution(_logical_heat(table, workload, num_experts))
        targets_by_rank = _allocate_critical_path_replicas(
            table[:, :, :base_slots], heat, self.slots_per_rank, num_ranks
        )

        candidate = _align_shared_slots(table, base_slots, targets_by_rank)
        current_rank_loads = _rank_loads_by_layer(table, heat, num_ranks)
        predicted_rank_loads = _rank_loads_by_layer(candidate, heat, num_ranks)
        current_critical_path = current_rank_loads.max(axis=-1)
        predicted_critical_path = predicted_rank_loads.max(axis=-1)
        priority = current_critical_path - predicted_critical_path
        raw_gain = float(priority.sum()) / max(float(current_critical_path.sum()), _SLOT_GAIN_EPSILON)
        changed_slots = int(np.count_nonzero(np.any(candidate[:, :, base_slots:] != table[:, :, base_slots:], axis=0)))
        migration_penalty = (
            _NORMALIZED_MIGRATION_COST
            * changed_slots
            / max(1, self.slots_per_rank * num_ranks)
            / _MIGRATION_AMORTIZATION_WINDOWS
        )
        effective_gain = raw_gain - migration_penalty
        if self._last_candidate is not None and np.array_equal(candidate, self._last_candidate):
            self._stable_windows += 1
        else:
            self._last_candidate = candidate.copy()
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

        return GlobalExpertPoolDecision(
            should_apply=should_apply,
            reason=reason,
            placement=_immutable(candidate if should_apply else table),
            priority=tuple(float(value) for value in priority),
            gain=effective_gain,
            changed_slots=changed_slots,
        )


class GlobalExpertPoolEplb(EplbPolicy):
    def __init__(self, num_redundant_experts: int) -> None:
        self.planner = GlobalExpertPoolPlanner(num_redundant_experts)
        self.last_decision: GlobalExpertPoolDecision | None = None

    def rebalance_experts(self, current_expert_table, expert_workload):
        self.last_decision = self.planner.plan(current_expert_table, expert_workload)
        decision = self.last_decision
        return int(decision.should_apply), np.asarray(decision.priority), decision.placement


__all__ = ["GlobalExpertPoolEplb"]
