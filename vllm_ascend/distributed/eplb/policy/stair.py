# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Pure NumPy building blocks for the STAIR EPLB policy."""

from dataclasses import dataclass

import numpy as np
from vllm.distributed.eplb.policy import AbstractEplbPolicy


@dataclass(frozen=True)
class BalanceScore:
    mean: float
    p95: float


class StairEplbPolicy(AbstractEplbPolicy):
    """STAIR load statistics and placement planning."""

    @staticmethod
    def compress_samples(samples: np.ndarray, max_bins: int) -> tuple[np.ndarray, np.ndarray]:
        """Compress a chronological load window into weighted bins."""
        values = np.asarray(samples, dtype=np.float64)
        if (
            values.ndim != 3
            or values.shape[0] == 0
            or max_bins < 1
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
        ):
            raise ValueError("STAIR samples must be finite non-negative [steps, layers, experts]")
        bins = min(values.shape[0], max_bins)
        boundaries = np.arange(bins + 1) * values.shape[0] // bins
        weights = np.diff(boundaries).astype(np.int64)
        compressed = np.stack(
            [values[start:end].mean(axis=0, dtype=np.float64) for start, end in zip(boundaries[:-1], boundaries[1:])]
        )
        return compressed, weights

    @staticmethod
    def weighted_moments(samples: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the weighted mean, variance, and sample covariance."""
        values = np.asarray(samples, dtype=np.float64)
        counts = np.asarray(weights)
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or counts.shape != (values.shape[0],)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts <= 0)
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
        ):
            raise ValueError("STAIR moments require finite samples and positive integer weights")
        counts = counts.astype(np.int64, copy=False)
        total = int(counts.sum())
        mean = np.sum(values * counts[:, None], axis=0, dtype=np.float64) / total
        centered = values - mean
        if total == 1:
            variance = np.zeros(values.shape[1], dtype=np.float64)
            return mean, variance, np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
        covariance = (centered * counts[:, None]).T @ centered / (total - 1)
        covariance = (covariance + covariance.T) * 0.5
        return mean, np.diag(covariance).copy(), covariance

    @staticmethod
    def replica_counts(placement: np.ndarray, num_experts: int) -> np.ndarray:
        """Validate a layer placement and return each expert's replica count."""
        layout = np.asarray(placement)
        if (
            layout.ndim != 2
            or not np.issubdtype(layout.dtype, np.integer)
            or num_experts < 1
            or np.any(layout < 0)
            or np.any(layout >= num_experts)
        ):
            raise ValueError("STAIR placement contains an invalid expert")
        layout = layout.astype(np.int64, copy=False)
        counts = np.bincount(layout.ravel(), minlength=num_experts)
        if np.any(counts == 0) or any(len(set(row)) != len(row) for row in layout.tolist()):
            raise ValueError("STAIR placement must cover every expert without rank-local duplicates")
        return counts

    @classmethod
    def placement_score(cls, samples: np.ndarray, weights: np.ndarray, placement: np.ndarray) -> BalanceScore:
        """Return weighted mean and nearest-rank p95 imbalance."""
        values = np.asarray(samples, dtype=np.float64)
        sample_weights = np.asarray(weights)
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or sample_weights.shape != (values.shape[0],)
            or not np.issubdtype(sample_weights.dtype, np.integer)
            or np.any(sample_weights <= 0)
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
        ):
            raise ValueError("STAIR score requires finite samples and positive integer weights")
        sample_weights = sample_weights.astype(np.int64, copy=False)
        layout = np.asarray(placement)
        counts = cls.replica_counts(layout, values.shape[1])
        loads = np.stack([np.sum(values[:, row] / counts[row], axis=1) for row in layout], axis=1)
        totals = loads.sum(axis=1)
        imbalance = np.ones(values.shape[0], dtype=np.float64)
        active = totals > 0
        imbalance[active] = loads[active].max(axis=1) / (totals[active] / layout.shape[0])
        order = np.argsort(imbalance, kind="stable")
        cumulative = np.cumsum(sample_weights[order])
        p95_rank = max(1, int(np.ceil(0.95 * int(cumulative[-1]))))
        p95 = imbalance[order[np.searchsorted(cumulative, p95_rank, side="left")]]
        mean = np.sum(imbalance * sample_weights, dtype=np.float64) / sample_weights.sum()
        return BalanceScore(float(mean), float(p95))
