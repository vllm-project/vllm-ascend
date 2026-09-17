# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Pure NumPy building blocks for the STAIR EPLB policy."""

from collections.abc import Callable
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

    @staticmethod
    def expert_risk(mean: np.ndarray, variance: np.ndarray, z_score: float) -> np.ndarray:
        """Return FlashLB's mean-plus-deviation risk per expert."""
        averages = np.asarray(mean, dtype=np.float64)
        variances = np.asarray(variance, dtype=np.float64)
        if (
            averages.ndim != 1
            or variances.shape != averages.shape
            or not np.all(np.isfinite(averages))
            or not np.all(np.isfinite(variances))
            or np.any(averages < 0)
            or np.any(variances < 0)
            or not np.isfinite(z_score)
            or z_score < 0
        ):
            raise ValueError("STAIR expert moments and z-score must be finite and non-negative")
        return averages + z_score * np.sqrt(variances)

    @staticmethod
    def _allocate_extra_replicas(
        risk: np.ndarray,
        replicas: np.ndarray,
        extra_slots: int,
        num_ranks: int,
        experts: tuple[int, ...],
    ) -> np.ndarray | None:
        result = replicas.copy()
        for _ in range(extra_slots):
            candidates = [expert for expert in experts if result[expert] < num_ranks]
            if not candidates:
                return None
            expert = min(candidates, key=lambda item: (-risk[item] / result[item], item))
            result[expert] += 1
        return result

    @staticmethod
    def _nearby_budgets(center: int, lower: int, upper: int, radius: int) -> list[int]:
        budgets = []
        for distance in range(radius + 1):
            values = (center,) if distance == 0 else (center - distance, center + distance)
            budgets.extend(value for value in values if lower <= value <= upper)
        return budgets

    @classmethod
    def replica_candidates(
        cls,
        risk: np.ndarray,
        total_slots: int,
        num_ranks: int,
        *,
        num_stages: int,
        radius: int,
        beam_size: int,
        score: Callable[[np.ndarray], float],
    ) -> list[np.ndarray]:
        """Return a bounded FlashTree-style beam of replica-count vectors."""
        values = np.asarray(risk, dtype=np.float64)
        num_experts = values.size
        integer_inputs = (total_slots, num_ranks, num_stages, radius, beam_size)
        if (
            values.ndim != 1
            or num_experts == 0
            or not np.all(np.isfinite(values))
            or np.any(values < 0)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in integer_inputs)
            or num_ranks < 1
            or not num_experts <= total_slots <= num_experts * num_ranks
            or total_slots % num_ranks != 0
            or num_stages < 1
            or radius < 0
            or beam_size < 1
            or not callable(score)
        ):
            raise ValueError("Invalid STAIR replica search input")

        order = sorted(range(num_experts), key=lambda expert: (-values[expert], expert))
        stage_count = min(num_stages, num_experts)
        groups = [tuple(group) for group in np.array_split(order, stage_count)]
        beam = [(np.ones(num_experts, dtype=np.int64), total_slots - num_experts)]

        for group_index, group in enumerate(groups[:-1]):
            later = tuple(expert for remaining in groups[group_index + 1 :] for expert in remaining)
            expanded = []
            for replicas, remaining in beam:
                greedy = cls._allocate_extra_replicas(values, replicas, remaining, num_ranks, group + later)
                if greedy is None:
                    continue
                center = int(np.sum(greedy[list(group)] - replicas[list(group)]))
                group_capacity = sum(num_ranks - replicas[expert] for expert in group)
                later_capacity = sum(num_ranks - replicas[expert] for expert in later)
                lower = max(0, remaining - later_capacity)
                upper = min(remaining, group_capacity)
                for budget in cls._nearby_budgets(center, lower, upper, radius):
                    partial = cls._allocate_extra_replicas(values, replicas, budget, num_ranks, group)
                    if partial is None:
                        continue
                    complete = cls._allocate_extra_replicas(values, partial, remaining - budget, num_ranks, later)
                    if complete is not None:
                        expanded.append((partial, remaining - budget, complete))
            unique = {tuple(partial): (partial, remaining, complete) for partial, remaining, complete in expanded}
            ranked = sorted(
                unique.values(),
                key=lambda item: (float(np.max(values / item[2])), tuple(item[2]), tuple(item[0])),
            )
            beam = [(partial, remaining) for partial, remaining, _ in ranked[:beam_size]]

        complete = {}
        for replicas, remaining in beam:
            candidate = cls._allocate_extra_replicas(values, replicas, remaining, num_ranks, groups[-1])
            if candidate is not None:
                complete[tuple(candidate)] = candidate
        return sorted(complete.values(), key=lambda item: (score(item), tuple(item)))[:beam_size]
