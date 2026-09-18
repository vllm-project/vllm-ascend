# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Pure NumPy building blocks for the STAIR EPLB policy."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from vllm.distributed.eplb.policy import AbstractEplbPolicy

from vllm_ascend.ascend_config import StairConfig


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

    @classmethod
    def gated_layer_imbalance(
        cls,
        load_samples: np.ndarray,
        sample_counts: np.ndarray,
        current_rank_expert_ids: np.ndarray,
        last_committed_mean_ratio: float | None,
        config: StairConfig,
    ) -> PlacementImbalance | None:
        """Return current imbalance when a layer passes load and hysteresis gates.

        Loads are ``[bins, experts]``, counts are ``[bins]``, and placement is
        ``[ranks, slots]``. ``last_committed_mean_ratio`` is the prediction saved
        only after a real placement commit; ``None`` or NaN means no anchor yet.
        Return the current imbalance for a nonzero window without an anchor or
        when either threshold fires; return ``None`` for an all-zero window or
        when neither threshold fires.
        """
        values = np.asarray(load_samples, dtype=np.float64)
        current_imbalance = cls.placement_imbalance(values, sample_counts, current_rank_expert_ids)
        if not np.any(values):
            return None
        if last_committed_mean_ratio is None or np.isnan(last_committed_mean_ratio):
            return current_imbalance
        if not np.isfinite(last_committed_mean_ratio) or last_committed_mean_ratio < 1:
            raise ValueError("last_committed_mean_ratio must be NaN or a finite ratio no smaller than one")

        current_balance = 1.0 / current_imbalance.mean_ratio
        committed_balance = 1.0 / last_committed_mean_ratio
        if (
            current_balance / committed_balance <= config.relative_balance_threshold
            or current_balance <= config.absolute_balance_threshold
        ):
            return current_imbalance
        return None

    # FlashTree-style search over per-expert replica counts.
    @staticmethod
    def _allocate_extra_replicas(
        expert_risks: np.ndarray,
        replica_counts: np.ndarray,
        extra_slots: int,
        num_ranks: int,
        eligible_experts: tuple[int, ...],
    ) -> np.ndarray | None:
        """Greedily allocate slots by descending risk per existing replica."""
        allocated_counts = replica_counts.copy()
        for _ in range(extra_slots):
            allocatable_experts = [expert for expert in eligible_experts if allocated_counts[expert] < num_ranks]
            if not allocatable_experts:
                return None
            expert = max(
                allocatable_experts,
                key=lambda expert_id: (expert_risks[expert_id] / allocated_counts[expert_id], -expert_id),
            )
            allocated_counts[expert] += 1
        return allocated_counts

    @staticmethod
    def _candidate_group_budgets(center_budget: int, min_budget: int, max_budget: int, budget_radius: int) -> list[int]:
        """Enumerate valid budgets from nearest to farthest from the center."""
        budgets = []
        for distance in range(budget_radius + 1):
            values = (center_budget,) if distance == 0 else (center_budget - distance, center_budget + distance)
            budgets.extend(value for value in values if min_budget <= value <= max_budget)
        return budgets

    @classmethod
    def _expand_replica_search_stage(
        cls,
        expert_risks: np.ndarray,
        search_beam: list[_ReplicaSearchState],
        current_experts: tuple[int, ...],
        later_experts: tuple[int, ...],
        *,
        num_ranks: int,
        budget_radius: int,
        beam_size: int,
    ) -> list[_ReplicaSearchState]:
        """Expand and prune one stage using a cheap replica-risk score."""
        stage_expansions = []
        eligible_experts = current_experts + later_experts
        for replica_counts, unallocated_slots in search_beam:
            baseline_completion = cls._allocate_extra_replicas(
                expert_risks, replica_counts, unallocated_slots, num_ranks, eligible_experts
            )
            if baseline_completion is None:
                continue
            center_budget = int(
                np.sum(baseline_completion[list(current_experts)] - replica_counts[list(current_experts)])
            )
            current_capacity = sum(num_ranks - replica_counts[expert] for expert in current_experts)
            later_capacity = sum(num_ranks - replica_counts[expert] for expert in later_experts)
            min_budget = max(0, unallocated_slots - later_capacity)
            max_budget = min(unallocated_slots, current_capacity)
            for budget in cls._candidate_group_budgets(center_budget, min_budget, max_budget, budget_radius):
                stage_replica_counts = cls._allocate_extra_replicas(
                    expert_risks, replica_counts, budget, num_ranks, current_experts
                )
                if stage_replica_counts is None:
                    continue
                candidate_completion = cls._allocate_extra_replicas(
                    expert_risks,
                    stage_replica_counts,
                    unallocated_slots - budget,
                    num_ranks,
                    later_experts,
                )
                if candidate_completion is not None:
                    stage_expansions.append((stage_replica_counts, unallocated_slots - budget, candidate_completion))

        unique_expansions = {
            tuple(stage_counts): (stage_counts, unallocated_slots, completion)
            for stage_counts, unallocated_slots, completion in stage_expansions
        }

        def screening_key(expansion):
            stage_counts, _, candidate_completion = expansion
            max_per_replica_risk = float(np.max(expert_risks / candidate_completion))
            # Lexicographic vectors make equal-risk screening deterministic.
            return max_per_replica_risk, tuple(candidate_completion), tuple(stage_counts)

        ranked_expansions = sorted(unique_expansions.values(), key=screening_key)
        return [
            (stage_counts, unallocated_slots) for stage_counts, unallocated_slots, _ in ranked_expansions[:beam_size]
        ]

    @classmethod
    def replica_candidates(
        cls,
        expert_risks: np.ndarray,
        total_slots: int,
        num_ranks: int,
        *,
        num_stages: int,
        budget_radius: int,
        beam_size: int,
        candidate_score: Callable[[np.ndarray], float],
    ) -> list[np.ndarray]:
        """Return at most ``beam_size`` unique candidates, best score first.

        ``candidate_score`` is lower-is-better and runs only on final candidates.
        """
        risks = np.asarray(expert_risks, dtype=np.float64)
        num_experts = risks.size
        if risks.ndim != 1 or num_experts == 0 or not np.all(np.isfinite(risks)) or np.any(risks < 0):
            raise ValueError("expert_risks must be a finite non-negative vector")
        integer_controls = (
            ("total_slots", total_slots),
            ("num_ranks", num_ranks),
            ("num_stages", num_stages),
            ("budget_radius", budget_radius),
            ("beam_size", beam_size),
        )
        for name, value in integer_controls:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if num_ranks < 1 or num_stages < 1 or budget_radius < 0 or beam_size < 1:
            raise ValueError("num_ranks, num_stages, and beam_size must be positive; budget_radius cannot be negative")
        if not num_experts <= total_slots <= num_experts * num_ranks or total_slots % num_ranks != 0:
            raise ValueError("total_slots must form an equal-capacity rank placement")
        if not callable(candidate_score):
            raise ValueError("candidate_score must be callable")

        experts_by_descending_risk = sorted(range(num_experts), key=lambda expert: (-risks[expert], expert))
        stage_count = min(num_stages, num_experts)
        expert_groups = [tuple(group) for group in np.array_split(experts_by_descending_risk, stage_count)]
        search_beam: list[_ReplicaSearchState] = [(np.ones(num_experts, dtype=np.int64), total_slots - num_experts)]

        for group_index, current_experts in enumerate(expert_groups[:-1]):
            later_experts = tuple(expert for later_group in expert_groups[group_index + 1 :] for expert in later_group)
            search_beam = cls._expand_replica_search_stage(
                risks,
                search_beam,
                current_experts,
                later_experts,
                num_ranks=num_ranks,
                budget_radius=budget_radius,
                beam_size=beam_size,
            )

        final_candidates_by_counts = {}
        for replica_counts, unallocated_slots in search_beam:
            candidate = cls._allocate_extra_replicas(
                risks, replica_counts, unallocated_slots, num_ranks, expert_groups[-1]
            )
            if candidate is not None:
                final_candidates_by_counts[tuple(candidate)] = candidate
        return sorted(
            final_candidates_by_counts.values(),
            key=lambda candidate: (candidate_score(candidate), tuple(candidate)),
        )[:beam_size]
