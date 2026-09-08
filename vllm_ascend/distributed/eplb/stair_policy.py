# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Pure NumPy implementation of the STAIR placement policy."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BalanceScore:
    mean: float
    p95: float


@dataclass(frozen=True)
class LayerPlan:
    placement: np.ndarray
    source_rank: np.ndarray
    source_slot: np.ndarray
    score: BalanceScore


@dataclass(frozen=True)
class StairPlan:
    placement: np.ndarray
    source_rank: np.ndarray
    source_slot: np.ndarray
    accepted_scores: np.ndarray


def compress_samples(samples: np.ndarray, sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Compress chronological samples into weighted, non-empty bins."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or sample_size < 1:
        raise ValueError("STAIR samples must be [steps, layers, experts]")
    bins = min(values.shape[0], sample_size)
    boundaries = np.arange(bins + 1) * values.shape[0] // bins
    weights = np.diff(boundaries).astype(np.int64)
    compressed = np.stack(
        [values[start:end].mean(axis=0, dtype=np.float64) for start, end in zip(boundaries[:-1], boundaries[1:])]
    )
    return compressed, weights


def weighted_moments(
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    covariance: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted mean and sample variance or covariance."""
    values = np.asarray(samples, dtype=np.float64)
    counts = np.asarray(weights, dtype=np.int64)
    if values.ndim != 2 or counts.shape != (values.shape[0],) or np.any(counts <= 0):
        raise ValueError("STAIR moments require [samples, experts] and positive weights")
    total = int(counts.sum())
    mean = np.sum(values * counts[:, None], axis=0, dtype=np.float64) / total
    centered = values - mean
    if total == 1:
        shape = (values.shape[1], values.shape[1]) if covariance else (values.shape[1],)
        return mean, np.zeros(shape, dtype=np.float64)
    if covariance:
        moments = (centered * counts[:, None]).T @ centered / (total - 1)
        return mean, (moments + moments.T) * 0.5
    return mean, np.sum(centered**2 * counts[:, None], axis=0, dtype=np.float64) / (total - 1)


def replica_counts(placement: np.ndarray, num_experts: int) -> np.ndarray:
    layout = np.asarray(placement, dtype=np.int64)
    if layout.ndim != 2 or np.any(layout < 0) or np.any(layout >= num_experts):
        raise ValueError("STAIR placement contains an invalid expert")
    counts = np.bincount(layout.ravel(), minlength=num_experts)
    if np.any(counts == 0) or any(len(set(row)) != len(row) for row in layout.tolist()):
        raise ValueError("STAIR placement must cover every expert without rank-local duplicates")
    return counts


def placement_score(samples: np.ndarray, weights: np.ndarray, placement: np.ndarray) -> BalanceScore:
    values = np.asarray(samples, dtype=np.float64)
    layout = np.asarray(placement, dtype=np.int64)
    counts = replica_counts(layout, values.shape[1])
    loads = np.stack([np.sum(values[:, row] / counts[row], axis=1) for row in layout], axis=1)
    totals = loads.sum(axis=1)
    imbalance = np.ones(values.shape[0], dtype=np.float64)
    active = totals > 0
    imbalance[active] = loads[active].max(axis=1) / (totals[active] / layout.shape[0])
    sample_weights = np.asarray(weights, dtype=np.int64)
    if sample_weights.shape != imbalance.shape or np.any(sample_weights <= 0):
        raise ValueError("STAIR score weights must match the samples")
    order = np.argsort(imbalance, kind="stable")
    cumulative = np.cumsum(sample_weights[order])
    nearest_rank = max(1, int(np.ceil(0.95 * int(cumulative[-1]))))
    p95 = imbalance[order[np.searchsorted(cumulative, nearest_rank, side="left")]]
    mean = np.sum(imbalance * sample_weights, dtype=np.float64) / sample_weights.sum()
    return BalanceScore(float(mean), float(p95))


def capped_min_max(
    risk: np.ndarray,
    replicas: np.ndarray,
    slots: int,
    num_ranks: int,
    experts: Iterable[int] | None = None,
) -> np.ndarray | None:
    """Allocate slots to the largest risk-per-replica expert."""
    weights = np.asarray(risk, dtype=np.float64)
    result = np.asarray(replicas, dtype=np.int64).copy()
    allowed = tuple(range(weights.size)) if experts is None else tuple(experts)
    if (
        weights.shape != result.shape
        or not np.all(np.isfinite(weights))
        or np.any(result < 1)
        or np.any(result > num_ranks)
        or slots < 0
    ):
        raise ValueError("Invalid STAIR replica allocation input")
    for _ in range(slots):
        candidates = [expert for expert in allowed if result[expert] < num_ranks]
        if not candidates:
            return None
        expert = min(candidates, key=lambda item: (-weights[item] / result[item], item))
        result[expert] += 1
    return result


def _nearby_budgets(center: int, lower: int, upper: int, width: int) -> list[int]:
    values = []
    for distance in range(width + 1):
        for value in ((center,) if distance == 0 else (center + distance, center - distance)):
            if lower <= value <= upper and value not in values:
                values.append(value)
    return values


def replica_candidates(
    risk: np.ndarray,
    total_slots: int,
    num_ranks: int,
    *,
    depth: int,
    width: int,
    limit: int,
    score: Callable[[np.ndarray], float],
) -> list[np.ndarray]:
    """Return a bounded FlashTree-style replica-vector beam."""
    weights = np.asarray(risk, dtype=np.float64)
    num_experts = weights.size
    if num_experts == 0 or total_slots < num_experts or total_slots > num_experts * num_ranks:
        raise ValueError("STAIR requires E <= physical slots <= E * ranks")
    order = sorted(range(num_experts), key=lambda expert: (-weights[expert], expert))
    group_size = (num_experts + min(depth, num_experts) - 1) // min(depth, num_experts)
    groups = [tuple(order[start : start + group_size]) for start in range(0, num_experts, group_size)]
    beam = [(np.ones(num_experts, dtype=np.int64), total_slots - num_experts)]

    for group_index, group in enumerate(groups[:-1]):
        later = tuple(expert for remaining in groups[group_index + 1 :] for expert in remaining)
        expanded = []
        for replicas, remaining in beam:
            baseline = capped_min_max(weights, replicas, remaining, num_ranks, (*group, *later))
            if baseline is None:
                continue
            center = int(np.sum(baseline[list(group)] - replicas[list(group)]))
            current_capacity = sum(num_ranks - replicas[expert] for expert in group)
            later_capacity = sum(num_ranks - replicas[expert] for expert in later)
            lower, upper = max(0, remaining - later_capacity), min(remaining, current_capacity)
            for budget in _nearby_budgets(center, lower, upper, width):
                partial = capped_min_max(weights, replicas, budget, num_ranks, group)
                if partial is None:
                    continue
                full = capped_min_max(weights, partial, remaining - budget, num_ranks, later)
                if full is not None:
                    expanded.append((partial, remaining - budget, full))
        unique = {partial.astype("<i4").tobytes(): (partial, remaining, full) for partial, remaining, full in expanded}
        ranked = sorted(unique.values(), key=lambda item: (score(item[2]), tuple(item[2]), tuple(item[0])))
        beam = [(partial, remaining) for partial, remaining, _ in ranked[:limit]]

    complete = {}
    for replicas, remaining in beam:
        candidate = capped_min_max(weights, replicas, remaining, num_ranks, groups[-1])
        if candidate is not None:
            complete[candidate.astype("<i4").tobytes()] = candidate
    return sorted(complete.values(), key=lambda item: (score(item), tuple(item)))[:limit]
