# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Pure NumPy implementation of the STAIR placement policy."""

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
