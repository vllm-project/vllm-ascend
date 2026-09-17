# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Pure NumPy building blocks for the STAIR EPLB policy."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from vllm.distributed.eplb.policy import AbstractEplbPolicy


@dataclass(frozen=True)
class PlacementImbalance:
    """Max-to-average rank-load ratios; 1.0 means perfectly balanced."""

    mean_ratio: float
    p95_ratio: float


_ReplicaSearchState = tuple[np.ndarray, int]
# (replica counts, unallocated extra slots)


class StairEplbPolicy(AbstractEplbPolicy):
    """STAIR load statistics and placement planning."""

    @staticmethod
    def compress_load_window(load_samples: np.ndarray, max_bins: int) -> tuple[np.ndarray, np.ndarray]:
        """Compress [steps, layers, experts] into bin means and sample counts."""
        if max_bins < 1:
            raise ValueError("max_bins must be positive")
        values = np.asarray(load_samples, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("load_samples must be finite non-negative [steps, layers, experts]")
        num_bins = min(values.shape[0], max_bins)
        boundaries = np.arange(num_bins + 1) * values.shape[0] // num_bins
        sample_counts = np.diff(boundaries).astype(np.int64)
        compressed = np.stack(
            [values[start:end].mean(axis=0, dtype=np.float64) for start, end in zip(boundaries[:-1], boundaries[1:])]
        )
        return compressed, sample_counts

    @staticmethod
    def weighted_moments(
        load_samples: np.ndarray, sample_counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return frequency-weighted mean, sample variance, and covariance."""
        values = np.asarray(load_samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("load_samples must be finite non-negative [bins, experts]")
        counts = np.asarray(sample_counts)
        if counts.shape != (values.shape[0],) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
            raise ValueError("sample_counts must contain one positive integer per bin")
        counts = counts.astype(np.int64, copy=False)
        total_sample_count = int(counts.sum())
        mean = np.sum(values * counts[:, None], axis=0, dtype=np.float64) / total_sample_count
        centered = values - mean
        if total_sample_count == 1:
            variance = np.zeros(values.shape[1], dtype=np.float64)
            return mean, variance, np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
        covariance = (centered * counts[:, None]).T @ centered / (total_sample_count - 1)
        covariance = (covariance + covariance.T) * 0.5
        return mean, np.diag(covariance).copy(), covariance

    @staticmethod
    def placement_replica_counts(rank_expert_ids: np.ndarray, num_experts: int) -> np.ndarray:
        """Validate [ranks, slots] expert IDs and count each expert's replicas."""
        placement = np.asarray(rank_expert_ids)
        if placement.ndim != 2 or not np.issubdtype(placement.dtype, np.integer):
            raise ValueError("rank_expert_ids must be an integer [ranks, slots] array")
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if np.any(placement < 0) or np.any(placement >= num_experts):
            raise ValueError("rank_expert_ids contains an out-of-range expert")
        placement = placement.astype(np.int64, copy=False)
        replica_counts = np.bincount(placement.ravel(), minlength=num_experts)
        if np.any(replica_counts == 0) or any(len(set(rank)) != len(rank) for rank in placement.tolist()):
            raise ValueError("rank_expert_ids must cover every expert without rank-local duplicates")
        return replica_counts

    @classmethod
    def placement_imbalance(
        cls, load_samples: np.ndarray, sample_counts: np.ndarray, rank_expert_ids: np.ndarray
    ) -> PlacementImbalance:
        """Return weighted mean and nearest-rank p95 max-to-average ratios."""
        values = np.asarray(load_samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("load_samples must be finite non-negative [bins, experts]")
        counts = np.asarray(sample_counts)
        if counts.shape != (values.shape[0],) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
            raise ValueError("sample_counts must contain one positive integer per bin")
        counts = counts.astype(np.int64, copy=False)
        placement = np.asarray(rank_expert_ids)
        replica_counts = cls.placement_replica_counts(placement, values.shape[1])
        rank_loads = np.stack(
            [np.sum(values[:, rank_experts] / replica_counts[rank_experts], axis=1) for rank_experts in placement],
            axis=1,
        )
        sample_total_loads = rank_loads.sum(axis=1)
        imbalance_ratios = np.ones(values.shape[0], dtype=np.float64)
        nonzero_load_samples = sample_total_loads > 0
        imbalance_ratios[nonzero_load_samples] = rank_loads[nonzero_load_samples].max(axis=1) / (
            sample_total_loads[nonzero_load_samples] / placement.shape[0]
        )
        imbalance_order = np.argsort(imbalance_ratios, kind="stable")
        cumulative_sample_counts = np.cumsum(counts[imbalance_order])
        p95_rank = max(1, int(np.ceil(0.95 * int(cumulative_sample_counts[-1]))))
        p95_index = np.searchsorted(cumulative_sample_counts, p95_rank, side="left")
        p95_ratio = imbalance_ratios[imbalance_order[p95_index]]
        mean_ratio = np.sum(imbalance_ratios * counts, dtype=np.float64) / counts.sum()
        return PlacementImbalance(float(mean_ratio), float(p95_ratio))

    @staticmethod
    def expert_risk(expert_means: np.ndarray, expert_variances: np.ndarray, z_score: float) -> np.ndarray:
        """Return ``mean + z_score * sqrt(variance)`` for each expert."""
        averages = np.asarray(expert_means, dtype=np.float64)
        variances = np.asarray(expert_variances, dtype=np.float64)
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
