# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""CPU building blocks for the STAIR EPLB policy."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from vllm.distributed import get_eplb_group
from vllm.distributed.eplb.policy import AbstractEplbPolicy

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.layer_sharding import all_gather_layer_shards, assigned_layer_ids
from vllm_ascend.distributed.eplb.policy import PreparedLoadStats

_MEAN_RATIO_TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PlacementImbalance:
    """Mean and p95 max-to-average ratios; 1.0 means perfectly balanced."""

    mean_ratio: float
    p95_ratio: float


@dataclass(frozen=True)
class PlacementPlan:
    """Target experts and their pre-migration source coordinates.

    Attributes:
        rank_expert_ids: Target expert IDs ``[ranks, slots]``, indexed by
            destination rank and slot.
        source_rank_ids: Source rank for each target slot, with shape
            ``[ranks, slots]``.
        source_slot_ids: Source slot for each target slot, with shape
            ``[ranks, slots]``.

    Source coordinates index the current placement, so
    ``current[source_rank_ids, source_slot_ids] == rank_expert_ids``.
    """

    rank_expert_ids: np.ndarray
    source_rank_ids: np.ndarray
    source_slot_ids: np.ndarray


@dataclass(frozen=True)
class LayerPlan:
    """An accepted placement and its predicted imbalance."""

    placement: PlacementPlan
    predicted_imbalance: PlacementImbalance


@dataclass(frozen=True)
class StairPlan:
    """Fixed-shape placement plan for every model layer.

    The placement and source arrays are ``[layers, ranks, slots]``.
    ``predicted_mean_ratios`` is ``[layers]`` and contains NaN for layers
    without an accepted candidate; those layers keep their current placement
    and same-rank, same-slot sources. All source coordinates index the current
    placement passed to the planner. Callers may persist a predicted ratio only
    after that layer is committed successfully.
    """

    rank_expert_ids: np.ndarray
    source_rank_ids: np.ndarray
    source_slot_ids: np.ndarray
    predicted_mean_ratios: np.ndarray


_RankChoice = tuple[float, int, float, float, float]  # risk, rank, mean, variance, variance scale
_PlacementUndoState = tuple[int, int, tuple[float, float, float]]  # rank, slot, previous rank statistics


@dataclass
class _PlacementDecision:
    choices: list[_RankChoice]
    next_choice: int = 0
    tried_feasible_choice: bool = False
    undo_state: _PlacementUndoState | None = None


_ReplicaSearchState = tuple[np.ndarray, int]
# (replica counts, unallocated extra slots)
_VARIANCE_ROUNDOFF_SAFETY_FACTOR = 8


class StairEplbPolicy(AbstractEplbPolicy):
    """STAIR load statistics and placement planning."""

    def __init__(self, config: StairConfig) -> None:
        self.config = config

    @staticmethod
    def _load_bin_boundaries(num_samples: int, max_bins: int) -> tuple[np.ndarray, np.ndarray]:
        if num_samples < 1 or max_bins < 1:
            raise ValueError("load binning requires samples and a positive bin limit")
        num_bins = min(num_samples, max_bins)
        boundaries = np.arange(num_bins + 1) * num_samples // num_bins
        return boundaries, np.diff(boundaries).astype(np.int64)

    def prepare_local_load_stats(self, load_samples: torch.Tensor) -> PreparedLoadStats:
        """Compress local temporal samples into STAIR's weighted bins."""
        boundaries, samples_per_bin = self._load_bin_boundaries(load_samples.shape[0], self.config.load_window_bins)
        load_sums_per_bin = torch.stack(
            [load_samples[start:end].sum(dim=0) for start, end in zip(boundaries[:-1], boundaries[1:])]
        )
        return PreparedLoadStats(load_sums_per_bin, samples_per_bin)

    def rebalance_experts(
        self,
        weight: torch.Tensor | PreparedLoadStats,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
        *,
        last_committed_mean_ratios: np.ndarray | None = None,
        rank_node_ids: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Plan through the upstream policy contract with STAIR context.

        A tensor is either ``[layers, experts]`` or an uncompressed
        ``[samples, layers, experts]`` window. A :class:`PreparedLoadStats`
        carries pre-binned sums and sample counts; its values must already
        reside on CPU, where planning runs. Optional anchors and node IDs
        enable the full STAIR path while preserving the upstream positional
        contract. ``num_groups`` is validated for that contract but does not
        constrain STAIR placement. The CPU result is ``[layers, num_replicas]``;
        its source-coordinate attributes are ``[layers, ranks, slots]`` and
        predicted ratios are ``[layers]``.
        """
        controls = num_replicas, num_groups, num_nodes, num_ranks
        invalid_type = any(isinstance(value, bool) or not isinstance(value, int) for value in controls)
        if invalid_type or min(controls) < 1:
            raise ValueError("STAIR topology values must be positive integers")
        if num_replicas % num_ranks:
            raise ValueError("STAIR requires equal rank capacity")
        if old_global_expert_indices is None:
            raise ValueError("STAIR requires the current expert placement")

        current_map = old_global_expert_indices.cpu()
        if current_map.ndim != 2 or current_map.shape[1] != num_replicas:
            raise ValueError("current expert placement must be [layers, num_replicas]")
        if isinstance(weight, PreparedLoadStats):
            if weight.values.device.type != "cpu":
                raise ValueError("prepared STAIR load statistics must be on CPU")
            logical_load_values = weight.values.to(dtype=torch.float64).numpy()
            sample_counts = weight.sample_counts
            if sample_counts is None:
                raise ValueError("prepared STAIR load statistics require sample counts")
        else:
            logical_load_values = weight.to(device="cpu", dtype=torch.float64).numpy()
            logical_load_values = (
                logical_load_values[None, ...] if logical_load_values.ndim == 2 else logical_load_values
            )
            sample_counts = None
        if logical_load_values.ndim != 3 or logical_load_values.shape[1] != current_map.shape[0]:
            raise ValueError("weight must be [samples, layers, logical_experts] and match the placement")

        current_placement = current_map.numpy().reshape(current_map.shape[0], num_ranks, num_replicas // num_ranks)
        if last_committed_mean_ratios is None:
            last_committed_mean_ratios = np.full(current_placement.shape[0], np.nan)
        if rank_node_ids is None:
            if num_ranks % num_nodes:
                raise ValueError("STAIR cannot infer topology with unequal ranks per node")
            node_ids = np.arange(num_ranks, dtype=np.int64) // (num_ranks // num_nodes)
        else:
            node_ids = np.asarray(rank_node_ids)
            if node_ids.shape != (num_ranks,) or not np.issubdtype(node_ids.dtype, np.integer) or np.any(node_ids < 0):
                raise ValueError("rank_node_ids must contain one non-negative integer per rank")
            if len(np.unique(node_ids)) != num_nodes:
                raise ValueError("num_nodes must match the distinct rank_node_ids")
        cpu_group = get_eplb_group().cpu_group
        if cpu_group.size() != num_ranks:
            raise RuntimeError("STAIR topology does not match the stage-local EPLB group")
        if num_ranks == 1:
            plan = self.plan_rebalance(
                logical_load_values=logical_load_values,
                current_rank_expert_ids=current_placement,
                last_committed_mean_ratios=last_committed_mean_ratios,
                rank_node_ids=node_ids,
                config=self.config,
                sample_counts=sample_counts,
            )
        else:
            plan = self.plan_sharded_rebalance(
                logical_load_values=logical_load_values,
                current_rank_expert_ids=current_placement,
                last_committed_mean_ratios=last_committed_mean_ratios,
                rank_node_ids=node_ids,
                config=self.config,
                cpu_group=cpu_group,
                sample_counts=sample_counts,
            )
        self.validate_plan(
            current_placement,
            plan,
            logical_load_values.shape[2],
            rank_node_ids=node_ids,
            rank_transfer_limit=self.config.rank_transfer_limit,
            cross_node_transfer_limit=self.config.cross_node_transfer_limit,
        )
        planned_map = plan.rank_expert_ids.reshape(current_map.shape)
        target = torch.from_numpy(planned_map).to(dtype=current_map.dtype)
        # The async migration patch consumes STAIR's exact source plan.
        target.source_rank_ids = plan.source_rank_ids
        target.source_slot_ids = plan.source_slot_ids
        target.predicted_mean_ratios = plan.predicted_mean_ratios
        if sample_counts is None:
            load_bins, bin_counts = self.compress_load_window(logical_load_values, self.config.load_window_bins)
        else:
            bin_counts = np.asarray(sample_counts)
            load_bins = logical_load_values / bin_counts[:, None, None]
        # Report model-wide averages of the per-layer temporal mean and p95 ratios.
        before, after = [], []
        for layer_id in range(current_placement.shape[0]):
            before.append(self.placement_imbalance(load_bins[:, layer_id], bin_counts, current_placement[layer_id]))
            after.append(self.placement_imbalance(load_bins[:, layer_id], bin_counts, plan.rank_expert_ids[layer_id]))
        target.predicted_imbalance_summary = (
            float(np.mean([score.mean_ratio for score in before])),
            float(np.mean([score.p95_ratio for score in before])),
            float(np.mean([score.mean_ratio for score in after])),
            float(np.mean([score.p95_ratio for score in after])),
        )
        return target

    # Load modeling and placement scoring.

    @staticmethod
    def compress_load_window(load_samples: np.ndarray, max_bins: int) -> tuple[np.ndarray, np.ndarray]:
        """Compress ``[steps, layers, experts]`` into weighted bins.

        Return bin means ``[bins, layers, experts]`` and counts ``[bins]``.
        """
        values = np.asarray(load_samples, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("load_samples must be finite non-negative [steps, layers, experts]")
        # Integer boundaries spread the remainder across bins without dropping steps.
        boundaries, sample_counts = StairEplbPolicy._load_bin_boundaries(values.shape[0], max_bins)
        compressed = np.stack(
            [values[start:end].mean(axis=0, dtype=np.float64) for start, end in zip(boundaries[:-1], boundaries[1:])]
        )
        return compressed, sample_counts

    @staticmethod
    def weighted_moments(
        load_samples: np.ndarray, sample_counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return frequency-weighted moments for ``[bins, experts]`` samples.

        ``sample_counts`` is ``[bins]``. Return mean and variance ``[experts]``
        plus covariance ``[experts, experts]``.
        """
        values = np.asarray(load_samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("load_samples must be finite non-negative [bins, experts]")
        counts = np.asarray(sample_counts)
        if counts.shape != (values.shape[0],) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
            raise ValueError("sample_counts must contain one positive integer per bin")
        counts = counts.astype(np.int64, copy=False)
        # Counts restore how many original steps each compressed bin represents.
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
        """Validate ``[ranks, slots]`` IDs and return counts ``[experts]``."""
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
        """Score a ``[ranks, slots]`` placement on ``[bins, experts]`` loads.

        ``sample_counts`` is ``[bins]``. Return weighted mean and nearest-rank
        p95 max-to-average ratios.
        """
        values = np.asarray(load_samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("load_samples must be finite non-negative [bins, experts]")
        counts = np.asarray(sample_counts)
        if counts.shape != (values.shape[0],) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
            raise ValueError("sample_counts must contain one positive integer per bin")
        counts = counts.astype(np.int64, copy=False)
        placement = np.asarray(rank_expert_ids)
        replica_counts = cls.placement_replica_counts(placement, values.shape[1])
        # Planning assumes an expert's logical load is shared evenly by its replicas.
        rank_loads = np.stack(
            [np.sum(values[:, rank_experts] / replica_counts[rank_experts], axis=1) for rank_experts in placement],
            axis=1,
        )
        sample_total_loads = rank_loads.sum(axis=1)
        # A sample with no routed load is balanced by definition.
        imbalance_ratios = np.ones(values.shape[0], dtype=np.float64)
        nonzero_load_samples = sample_total_loads > 0
        imbalance_ratios[nonzero_load_samples] = rank_loads[nonzero_load_samples].max(axis=1) / (
            sample_total_loads[nonzero_load_samples] / placement.shape[0]
        )
        # Compute a weighted nearest-rank quantile without expanding the bins.
        imbalance_order = np.argsort(imbalance_ratios, kind="stable")
        cumulative_sample_counts = np.cumsum(counts[imbalance_order])
        p95_rank = max(1, int(np.ceil(0.95 * int(cumulative_sample_counts[-1]))))
        p95_index = np.searchsorted(cumulative_sample_counts, p95_rank, side="left")
        p95_ratio = imbalance_ratios[imbalance_order[p95_index]]
        mean_ratio = np.sum(imbalance_ratios * counts, dtype=np.float64) / counts.sum()
        return PlacementImbalance(float(mean_ratio), float(p95_ratio))

    @staticmethod
    def expert_risk(expert_means: np.ndarray, expert_variances: np.ndarray, z_score: float) -> np.ndarray:
        """Return ``mean + z_score * sqrt(variance)`` for ``[experts]`` inputs."""
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
        """Allocate ``[experts]`` counts from ``[experts]`` per-copy risks."""
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
        budgets: list[int] = []
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
            # Greedy completion defines the center budget explored for this group.
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
            # Reserve enough capacity for later groups while respecting this group's cap.
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

        ``expert_risks`` and every returned replica vector are ``[experts]``.
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
        # Every expert starts covered; the search distributes only redundant slots.
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

    @classmethod
    def incremental_replica_candidates(
        cls,
        risks: np.ndarray,
        current_placement: np.ndarray,
        num_ranks: int,
        max_replica_changes: int,
        *,
        num_stages: int,
        budget_radius: int,
        beam_size: int,
    ) -> list[np.ndarray]:
        """Move current replica counts toward FlashTree candidates one step."""
        current = cls.placement_replica_counts(current_placement, len(risks))
        score = lambda replicas: float(np.max(risks / replicas))
        targets = cls.replica_candidates(
            risks,
            current_placement.size,
            num_ranks,
            num_stages=num_stages,
            budget_radius=budget_radius,
            beam_size=beam_size,
            candidate_score=score,
        )
        candidates = {tuple(current): current.copy()}
        for target in targets:
            candidate = current.copy()
            for _ in range(max_replica_changes):
                receivers = np.flatnonzero(candidate < target)
                donors = np.flatnonzero(candidate > target)
                if not receivers.size or not donors.size:
                    break
                receiver = min(receivers, key=lambda expert: (-risks[expert] / candidate[expert], expert))
                donor = min(donors, key=lambda expert: (risks[expert] / (candidate[expert] - 1), expert))
                candidate = candidate.copy()
                candidate[receiver] += 1
                candidate[donor] -= 1
                candidates[tuple(candidate)] = candidate
        ordered = sorted(candidates.values(), key=lambda replicas: (score(replicas), tuple(replicas)))
        selected = ordered[:beam_size]
        if not any(np.array_equal(candidate, current) for candidate in selected):
            selected[-1] = current
        return selected

    # Covariance-aware placement and migration-source selection.

    @staticmethod
    def _updated_rank_variance(
        expert: int,
        rank_experts: np.ndarray,
        current_variance: float,
        current_scale: float,
        expert_variances: np.ndarray,
        expert_covariance: np.ndarray,
        replica_counts: np.ndarray,
    ) -> tuple[float, float]:
        """Add one replica's scaled variance and covariance to a rank.

        ``rank_experts`` contains expert IDs already placed on that rank. The
        variance and replica-count vectors are ``[experts]``; covariance is
        ``[experts, experts]``. The result uses total replica counts for load
        splitting and clips only floating-point roundoff below zero.
        """
        expert_replica_count = replica_counts[expert]
        variance_increment = expert_variances[expert] / expert_replica_count**2
        updated_scale = current_scale + abs(variance_increment)
        for existing_expert in rank_experts:
            # Each off-diagonal covariance contributes in both matrix directions.
            covariance_increment = (
                2
                * expert_covariance[expert, existing_expert]
                / (expert_replica_count * replica_counts[existing_expert])
            )
            variance_increment += covariance_increment
            updated_scale += abs(covariance_increment)
        updated_variance = current_variance + variance_increment
        # Bound cancellation error by the accumulated term magnitudes.
        num_experts = len(rank_experts) + 1
        num_terms = num_experts * (num_experts + 1) // 2
        scale = max(updated_scale, np.finfo(np.float64).tiny)
        roundoff_tolerance = _VARIANCE_ROUNDOFF_SAFETY_FACTOR * num_terms * np.finfo(np.float64).eps * scale
        if updated_variance < -roundoff_tolerance:
            raise ValueError("expert covariance produces a negative rank variance")
        return max(float(updated_variance), 0.0), updated_scale

    @staticmethod
    def _has_feasible_migration_sources(
        demands: list[tuple[int, int]],
        rank_transfer_limit: int,
        cross_node_transfer_limit: int,
        source_candidates: list[list[tuple[int, ...]]],
        compact_node_ids: np.ndarray,
        num_nodes: int,
    ) -> bool:
        """Check source feasibility for ``(destination rank, expert)`` demands."""
        if rank_transfer_limit == -1:
            rank_transfer_limit = len(demands)
        if cross_node_transfer_limit == -1:
            cross_node_transfer_limit = len(demands)
        candidates = [source_candidates[dst_rank][expert] for dst_rank, expert in demands]
        source_usage = [0] * len(compact_node_ids)
        cross_out = [0] * num_nodes
        cross_in = [0] * num_nodes
        max_cross_transfers = min(len(demands), num_nodes * cross_node_transfer_limit)
        failed_states: set[tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = set()

        def assign(demand_index: int, remaining_cross_transfers: int) -> bool:
            if demand_index == len(demands):
                return True
            state = (
                demand_index,
                remaining_cross_transfers,
                tuple(source_usage),
                tuple(cross_out),
                tuple(cross_in),
            )
            if state in failed_states:
                return False
            dst_rank, _ = demands[demand_index]
            dst_node = compact_node_ids[dst_rank]
            for src_rank in candidates[demand_index]:
                src_node = compact_node_ids[src_rank]
                crosses_node = src_node != dst_node
                if source_usage[src_rank] >= rank_transfer_limit or crosses_node > remaining_cross_transfers:
                    continue
                if crosses_node and (
                    cross_out[src_node] >= cross_node_transfer_limit or cross_in[dst_node] >= cross_node_transfer_limit
                ):
                    continue
                source_usage[src_rank] += 1
                cross_out[src_node] += crosses_node
                cross_in[dst_node] += crosses_node
                if assign(demand_index + 1, remaining_cross_transfers - crosses_node):
                    return True
                cross_in[dst_node] -= crosses_node
                cross_out[src_node] -= crosses_node
                source_usage[src_rank] -= 1
            failed_states.add(state)
            return False

        return assign(0, max_cross_transfers)

    @staticmethod
    def _migration_sources(
        current_placement: np.ndarray,
        target_placement: np.ndarray,
        rank_transfer_limit: int,
        cross_node_transfer_limit: int,
        expert_sources: list[list[int]],
        rank_node_ids: np.ndarray,
    ) -> np.ndarray | None:
        """Globally assign sources within per-rank and cross-node budgets."""
        source_rank_ids = np.full_like(target_placement, -1)
        node_ids = np.asarray(rank_node_ids)
        unique_nodes, compact_node_ids = np.unique(node_ids, return_inverse=True)
        assigned = target_placement >= 0
        retained = assigned & np.any(target_placement[:, :, None] == current_placement[:, None, :], axis=2)
        destination_ranks = np.broadcast_to(np.arange(current_placement.shape[0])[:, None], target_placement.shape)
        source_rank_ids[retained] = destination_ranks[retained]
        incoming_mask = assigned & ~retained
        incoming = incoming_mask.sum(axis=1)
        demands = [
            (int(dst_rank), int(slot), int(target_placement[dst_rank, slot]))
            for dst_rank, slot in np.argwhere(incoming_mask)
        ]
        if rank_transfer_limit == -1:
            rank_transfer_limit = len(demands)
        if cross_node_transfer_limit == -1:
            cross_node_transfer_limit = len(demands)
        if np.any(incoming > rank_transfer_limit):
            return None

        source_usage = np.zeros(current_placement.shape[0], dtype=np.int64)
        cross_out = np.zeros(len(unique_nodes), dtype=np.int64)
        cross_in = np.zeros(len(unique_nodes), dtype=np.int64)
        max_cross_transfers = min(len(demands), len(unique_nodes) * cross_node_transfer_limit)
        candidates_by_demand = [
            sorted(
                expert_sources[expert],
                key=lambda src_rank, dst_node=compact_node_ids[dst_rank]: (
                    compact_node_ids[src_rank] != dst_node,
                    src_rank,
                ),
            )
            for dst_rank, _, expert in demands
        ]
        failed_states: set[tuple[int, int, bytes, bytes, bytes]] = set()

        def assign(demand_index: int, remaining_cross_transfers: int) -> bool:
            if demand_index == len(demands):
                return True
            state = (
                demand_index,
                remaining_cross_transfers,
                source_usage.tobytes(),
                cross_out.tobytes(),
                cross_in.tobytes(),
            )
            if state in failed_states:
                return False
            dst_rank, slot, _ = demands[demand_index]
            dst_node = compact_node_ids[dst_rank]
            for src_rank in candidates_by_demand[demand_index]:
                src_node = compact_node_ids[src_rank]
                crosses_node = src_node != dst_node
                if source_usage[src_rank] >= rank_transfer_limit or crosses_node > remaining_cross_transfers:
                    continue
                if crosses_node and (
                    cross_out[src_node] >= cross_node_transfer_limit or cross_in[dst_node] >= cross_node_transfer_limit
                ):
                    continue
                source_usage[src_rank] += 1
                cross_out[src_node] += crosses_node
                cross_in[dst_node] += crosses_node
                source_rank_ids[dst_rank, slot] = src_rank
                if assign(demand_index + 1, remaining_cross_transfers - crosses_node):
                    return True
                source_rank_ids[dst_rank, slot] = -1
                cross_in[dst_node] -= crosses_node
                cross_out[src_node] -= crosses_node
                source_usage[src_rank] -= 1
            failed_states.add(state)
            return False

        # The first feasible budget minimizes cross-node transfers. Sorted
        # candidates then stabilize the selected source ranks.
        for cross_budget in range(max_cross_transfers + 1):
            if assign(0, cross_budget):
                return source_rank_ids
        return None

    @staticmethod
    def _align_target_slots(current_placement: np.ndarray, target_placement: np.ndarray) -> np.ndarray:
        """Align ``[ranks, slots]`` layouts while keeping retained slots."""
        aligned = np.full_like(target_placement, -1)
        for rank_id, target_experts in enumerate(target_placement):
            target_set = set(map(int, target_experts))
            retained_experts = set()
            for slot, expert in enumerate(current_placement[rank_id]):
                if int(expert) in target_set:
                    aligned[rank_id, slot] = expert
                    retained_experts.add(int(expert))
            empty_slots = np.flatnonzero(aligned[rank_id] < 0)
            for slot, expert in zip(empty_slots, sorted(target_set - retained_experts)):
                aligned[rank_id, slot] = expert
        return aligned

    @staticmethod
    def _source_slots(
        current_placement: np.ndarray,
        target_placement: np.ndarray,
        source_rank_ids: np.ndarray,
    ) -> np.ndarray:
        """Map three ``[ranks, slots]`` inputs to source slots of the same shape."""
        source_slot_ids = np.empty_like(target_placement)
        for dst_rank, target_experts in enumerate(target_placement):
            for dst_slot, expert in enumerate(target_experts):
                src_rank = source_rank_ids[dst_rank, dst_slot]
                source_slots = np.flatnonzero(current_placement[src_rank] == expert)
                assert source_slots.size == 1
                source_slot_ids[dst_rank, dst_slot] = source_slots[0]
        return source_slot_ids

    @classmethod
    def lpt_placement(
        cls,
        expert_means: np.ndarray,
        expert_variances: np.ndarray,
        expert_covariance: np.ndarray,
        replica_counts: np.ndarray,
        num_ranks: int,
        z_score: float,
        *,
        current_rank_expert_ids: np.ndarray,
        rank_node_ids: np.ndarray,
        rank_transfer_limit: int,
        cross_node_transfer_limit: int,
        backtrack_limit: int,
        migration_feasibility_cache: dict[tuple[tuple[int, int], ...], bool] | None = None,
    ) -> PlacementPlan | None:
        """Place replicas with deterministic covariance-aware greedy LPT.

        Mean, variance, and replica counts are ``[experts]``; covariance is
        ``[experts, experts]``. Current placement is ``[ranks, slots]`` and node
        IDs are ``[ranks]``; all returned plan arrays are ``[ranks, slots]``.
        Experts are processed by descending per-replica risk. Each replica
        chooses the legal rank with the lowest updated risk,
        breaking ties by rank ID. Each partial placement must have a source
        assignment within the per-rank and cross-node limits. ``None`` means bounded
        backtracking found no legal placement. The first source-feasible choice
        is free; each accepted alternative choice consumes one backtrack.
        ``rank_node_ids`` contains one non-negative node ID per rank; equal IDs
        mean that two ranks share a node. Final source assignment first minimizes
        cross-node transfers, then source rank IDs in target-slot order.
        Retained experts keep their current slots; incoming experts fill the
        remaining slots by expert ID. Returned source coordinates align with
        these final target slots.
        """
        means = np.asarray(expert_means, dtype=np.float64)
        variances = np.asarray(expert_variances, dtype=np.float64)
        covariance = np.asarray(expert_covariance, dtype=np.float64)
        replicas = np.asarray(replica_counts)
        num_experts = means.size
        if means.ndim != 1 or num_experts == 0:
            raise ValueError("expert_means must be a non-empty vector")
        if (
            variances.shape != means.shape
            or covariance.shape != (num_experts, num_experts)
            or replicas.shape != means.shape
        ):
            raise ValueError("STAIR LPT variance, covariance, and replica-count shapes must match expert_means")
        if not np.issubdtype(replicas.dtype, np.integer):
            raise ValueError("replica_counts must contain integers")
        if (
            not np.all(np.isfinite(means))
            or not np.all(np.isfinite(variances))
            or not np.all(np.isfinite(covariance))
            or not np.isfinite(z_score)
        ):
            raise ValueError("STAIR LPT moments and z_score must be finite")
        if np.any(means < 0) or np.any(variances < 0) or z_score < 0:
            raise ValueError("expert means, variances, and z_score must be non-negative")
        if not np.allclose(covariance, covariance.T):
            raise ValueError("expert_covariance must be symmetric")
        if not np.allclose(np.diag(covariance), variances):
            raise ValueError("expert_covariance diagonal must match expert_variances")
        covariance = (covariance + covariance.T) * 0.5
        if not isinstance(num_ranks, int) or isinstance(num_ranks, bool) or num_ranks < 1:
            raise ValueError("num_ranks must be a positive integer")
        controls = rank_transfer_limit, cross_node_transfer_limit, backtrack_limit
        invalid_type = any(isinstance(value, bool) or not isinstance(value, int) for value in controls)
        invalid_limits = rank_transfer_limit < -1 or rank_transfer_limit == 0 or cross_node_transfer_limit < -1
        if invalid_type or invalid_limits or backtrack_limit < 0:
            raise ValueError("STAIR transfer limits and backtrack_limit must be valid integers")
        replicas = replicas.astype(np.int64, copy=False)
        total_slots = int(replicas.sum())
        if np.any(replicas < 1) or np.any(replicas > num_ranks) or total_slots % num_ranks != 0:
            raise ValueError("replica_counts must fit an equal-capacity rank placement")
        current_placement = np.asarray(current_rank_expert_ids)
        if current_placement.shape != (num_ranks, total_slots // num_ranks):
            raise ValueError("current_rank_expert_ids must match the target rank capacity")
        cls.placement_replica_counts(current_placement, num_experts)
        current_placement = current_placement.astype(np.int64, copy=False)
        node_ids = np.asarray(rank_node_ids)
        if node_ids.shape != (num_ranks,) or not np.issubdtype(node_ids.dtype, np.integer) or np.any(node_ids < 0):
            raise ValueError("rank_node_ids must contain one non-negative integer per rank")
        expert_sources = [np.where(current_placement == expert)[0].tolist() for expert in range(num_experts)]
        migration_sources = cls._migration_sources
        _, compact_node_ids = np.unique(node_ids, return_inverse=True)
        num_nodes = int(compact_node_ids.max()) + 1
        source_candidates = [
            [
                tuple(
                    sorted(
                        expert_sources[expert],
                        key=lambda src_rank: (compact_node_ids[src_rank] != compact_node_ids[dst_rank], src_rank),
                    )
                )
                for expert in range(num_experts)
            ]
            for dst_rank in range(num_ranks)
        ]
        current_has_expert = np.zeros((num_ranks, num_experts), dtype=bool)
        current_has_expert[np.arange(num_ranks)[:, None], current_placement] = True
        placed_has_expert = np.zeros((num_ranks, num_experts), dtype=bool)
        incoming_demands: list[tuple[int, int]] = []
        incoming_counts = np.zeros(num_ranks, dtype=np.int64)
        if migration_feasibility_cache is None:
            migration_feasibility_cache = {(): True}

        slots_per_rank = total_slots // num_ranks
        placement = np.full((num_ranks, slots_per_rank), -1, dtype=np.int64)
        rank_sizes = np.zeros(num_ranks, dtype=np.int64)
        rank_means = np.zeros(num_ranks, dtype=np.float64)
        rank_variances = np.zeros(num_ranks, dtype=np.float64)
        rank_variance_scales = np.zeros(num_ranks, dtype=np.float64)
        scaled_variances = variances / replicas**2
        scaled_covariance = 2 * covariance / (replicas[:, None] * replicas[None, :])
        slot_ids = np.arange(slots_per_rank)
        per_replica_risks = cls.expert_risk(means, variances, z_score) / replicas
        experts_by_descending_replica_risk = sorted(
            range(num_experts), key=lambda expert: (-per_replica_risks[expert], expert)
        )

        replica_order = [expert for expert in experts_by_descending_replica_risk for _ in range(replicas[expert])]
        decisions: list[_PlacementDecision] = []
        replica_index = 0
        backtracks_used = 0

        def undo_placement(rank_id: int, slot: int, previous_state: tuple[float, float, float]) -> None:
            expert = placement[rank_id, slot]
            if not current_has_expert[rank_id, expert]:
                assert incoming_demands.pop() == (rank_id, expert)
                incoming_counts[rank_id] -= 1
            placed_has_expert[rank_id, expert] = False
            placement[rank_id, slot] = -1
            rank_sizes[rank_id] -= 1
            rank_means[rank_id] = previous_state[0]
            rank_variances[rank_id] = previous_state[1]
            rank_variance_scales[rank_id] = previous_state[2]

        # Decision i retains the remaining rank choices and undo state for copy i.
        while replica_index < len(replica_order):
            expert = replica_order[replica_index]
            if len(decisions) == replica_index:
                valid_ranks = (rank_sizes < slots_per_rank) & ~placed_has_expert[:, expert]
                if rank_transfer_limit != -1:
                    valid_ranks &= current_has_expert[:, expert] | (incoming_counts < rank_transfer_limit)
                rank_ids = np.flatnonzero(valid_ranks)
                existing_mask = slot_ids[None, :] < rank_sizes[rank_ids, None]
                existing_experts = np.where(existing_mask, placement[rank_ids], 0)
                covariance_increments = scaled_covariance[expert, existing_experts] * existing_mask
                variance_increments = scaled_variances[expert] + covariance_increments.sum(axis=1)
                updated_variances = rank_variances[rank_ids] + variance_increments
                updated_scales = (
                    rank_variance_scales[rank_ids]
                    + abs(scaled_variances[expert])
                    + np.abs(covariance_increments).sum(axis=1)
                )
                num_rank_experts = rank_sizes[rank_ids] + 1
                num_terms = num_rank_experts * (num_rank_experts + 1) // 2
                roundoff_tolerances = (
                    _VARIANCE_ROUNDOFF_SAFETY_FACTOR
                    * num_terms
                    * np.finfo(np.float64).eps
                    * np.maximum(updated_scales, np.finfo(np.float64).tiny)
                )
                if np.any(updated_variances < -roundoff_tolerances):
                    raise ValueError("expert covariance produces a negative rank variance")
                updated_variances = np.maximum(updated_variances, 0.0)
                updated_means = rank_means[rank_ids] + means[expert] / replicas[expert]
                updated_risks = updated_means + z_score * np.sqrt(updated_variances)
                rank_choices = sorted(
                    (
                        float(risk),
                        int(rank_id),
                        float(updated_mean),
                        float(updated_variance),
                        float(updated_scale),
                    )
                    for risk, rank_id, updated_mean, updated_variance, updated_scale in zip(
                        updated_risks, rank_ids, updated_means, updated_variances, updated_scales
                    )
                )
                decisions.append(_PlacementDecision(rank_choices))

            decision = decisions[replica_index]
            advanced = False
            while decision.next_choice < len(decision.choices):
                _, rank_id, updated_mean, updated_variance, updated_scale = decision.choices[decision.next_choice]
                decision.next_choice += 1
                slot = rank_sizes[rank_id]
                previous_state = rank_means[rank_id], rank_variances[rank_id], rank_variance_scales[rank_id]
                placement[rank_id, slot] = expert
                placed_has_expert[rank_id, expert] = True
                rank_sizes[rank_id] += 1
                rank_means[rank_id] = updated_mean
                rank_variances[rank_id] = updated_variance
                rank_variance_scales[rank_id] = updated_scale
                expert_is_incoming = not current_has_expert[rank_id, expert]
                if expert_is_incoming:
                    incoming_demands.append((rank_id, expert))
                    incoming_counts[rank_id] += 1
                # Search needs only source feasibility; topology cost is deferred.
                sources_are_feasible = True
                if expert_is_incoming:
                    migration_key = tuple(sorted(incoming_demands))
                    sources_are_feasible = migration_feasibility_cache.get(migration_key)
                    if sources_are_feasible is None:
                        sources_are_feasible = cls._has_feasible_migration_sources(
                            incoming_demands,
                            rank_transfer_limit,
                            cross_node_transfer_limit,
                            source_candidates,
                            compact_node_ids,
                            num_nodes,
                        )
                        migration_feasibility_cache[migration_key] = sources_are_feasible
                budget_exhausted = (
                    sources_are_feasible and decision.tried_feasible_choice and backtracks_used == backtrack_limit
                )
                if not sources_are_feasible or budget_exhausted:
                    undo_placement(rank_id, slot, previous_state)
                    if budget_exhausted:
                        return None
                    continue
                # The greedy feasible choice is free; feasible alternatives consume budget.
                if decision.tried_feasible_choice:
                    backtracks_used += 1
                else:
                    decision.tried_feasible_choice = True
                decision.undo_state = rank_id, slot, previous_state
                replica_index += 1
                advanced = True
                break
            if advanced:
                continue
            decisions.pop()
            if replica_index == 0:
                return None
            replica_index -= 1
            undo_state = decisions[replica_index].undo_state
            assert undo_state is not None
            rank_id, slot, previous_state = undo_state
            undo_placement(rank_id, slot, previous_state)

        # Slot order does not affect risk, so stabilize it before exact source selection.
        placement = cls._align_target_slots(current_placement, placement)
        sources = migration_sources(
            current_placement,
            placement,
            rank_transfer_limit,
            cross_node_transfer_limit,
            expert_sources,
            node_ids,
        )
        assert sources is not None
        source_slots = cls._source_slots(current_placement, placement, sources)
        return PlacementPlan(placement, sources, source_slots)

    # Single-layer orchestration and acceptance.

    @classmethod
    def plan_layer(
        cls,
        load_samples: np.ndarray,
        sample_counts: np.ndarray,
        current_rank_expert_ids: np.ndarray,
        rank_node_ids: np.ndarray,
        config: StairConfig,
    ) -> LayerPlan | None:
        """Return the best mean- and p95-non-regressing placement for one layer.

        Loads are ``[bins, experts]``, counts are ``[bins]``, current placement
        is ``[ranks, slots]``, and node IDs are ``[ranks]``.
        Candidates may not regress mean or p95 imbalance. The lowest predicted
        mean ratio wins; ratios within the internal absolute tolerance are tied.
        Ties minimize cross-node migrations, same-node remote migrations,
        target expert IDs, source rank IDs, then source slot IDs. Return ``None``
        when no candidate is accepted or the winner keeps the current placement.
        """
        current_placement = np.asarray(current_rank_expert_ids)
        current_imbalance = cls.placement_imbalance(load_samples, sample_counts, current_placement)
        means, variances, covariance = cls.weighted_moments(load_samples, sample_counts)
        risks = cls.expert_risk(means, variances, config.z_score)
        node_ids = np.asarray(rank_node_ids)
        num_ranks = current_placement.shape[0]
        scored_candidates = []
        migration_feasibility_cache: dict[tuple[tuple[int, int], ...], bool] = {(): True}

        # Replica risk screens the bounded beam; actual imbalance decides acceptance.
        replica_candidates = cls.incremental_replica_candidates(
            risks,
            current_placement,
            num_ranks,
            current_placement.size if config.rank_transfer_limit == -1 else num_ranks * config.rank_transfer_limit,
            num_stages=config.replica_search_num_stages,
            budget_radius=config.replica_search_radius,
            beam_size=config.replica_search_beam_size,
        )
        for replicas in replica_candidates:
            placement = cls.lpt_placement(
                means,
                variances,
                covariance,
                replicas,
                num_ranks,
                config.z_score,
                current_rank_expert_ids=current_placement,
                rank_node_ids=node_ids,
                rank_transfer_limit=config.rank_transfer_limit,
                cross_node_transfer_limit=config.cross_node_transfer_limit,
                backtrack_limit=config.placement_search_backtrack_limit,
                migration_feasibility_cache=migration_feasibility_cache,
            )
            if placement is None:
                continue
            predicted_imbalance = cls.placement_imbalance(load_samples, sample_counts, placement.rank_expert_ids)
            if (
                predicted_imbalance.mean_ratio > current_imbalance.mean_ratio
                or predicted_imbalance.p95_ratio > current_imbalance.p95_ratio
            ):
                continue

            dst_rank_ids = np.arange(num_ranks)[:, None]
            remote = placement.source_rank_ids != dst_rank_ids
            cross_node = remote & (node_ids[placement.source_rank_ids] != node_ids[:, None])
            cross_node_migrations = int(cross_node.sum())
            same_node_remote_migrations = int(remote.sum() - cross_node_migrations)
            # The remaining fields make equal-cost plans deterministic.
            tie_key = (
                cross_node_migrations,
                same_node_remote_migrations,
                tuple(placement.rank_expert_ids.ravel()),
                tuple(placement.source_rank_ids.ravel()),
                tuple(placement.source_slot_ids.ravel()),
            )
            candidate_plan = LayerPlan(placement, predicted_imbalance)
            scored_candidates.append((predicted_imbalance.mean_ratio, tie_key, candidate_plan))

        if not scored_candidates:
            return None
        minimum_mean_ratio = min(mean_ratio for mean_ratio, *_ in scored_candidates)
        tied_candidates = [
            candidate
            for candidate in scored_candidates
            if candidate[0] <= minimum_mean_ratio + _MEAN_RATIO_TIE_TOLERANCE
        ]
        _, _, selected_plan = min(tied_candidates, key=lambda candidate: candidate[1])
        if np.array_equal(selected_plan.placement.rank_expert_ids, current_placement):
            return None
        return selected_plan

    @classmethod
    def plan_rebalance(
        cls,
        logical_load_values: np.ndarray,
        current_rank_expert_ids: np.ndarray,
        last_committed_mean_ratios: np.ndarray,
        rank_node_ids: np.ndarray,
        config: StairConfig,
        layer_ids: Sequence[int] | None = None,
        sample_counts: np.ndarray | None = None,
    ) -> StairPlan:
        """Plan every eligible layer from temporal loads.

        ``logical_load_values`` contains raw ``[steps, layers, experts]``
        samples when ``sample_counts`` is absent, otherwise ``[bins, layers,
        experts]`` pre-binned sums with one positive count per bin. Current
        placement is ``[layers, ranks, slots]``,
        committed ratios are ``[layers]``, and node IDs are ``[ranks]``. A NaN
        committed ratio means that the layer has no commit anchor; its relative
        deterioration is 0 for sorting. Eligible layers are planned by
        descending current mean ratio, relative deterioration, then layer ID.
        ``layer_ids`` contains stage-local indices on the input layer axis. The
        returned plan keeps its full shape, but only those indices are
        authoritative; omitted layers are identity/NaN placeholders that must
        not be committed before the shards are gathered.
        """
        if sample_counts is None:
            load_bins, bin_sample_counts = cls.compress_load_window(logical_load_values, config.load_window_bins)
        else:
            load_sums = np.asarray(logical_load_values, dtype=np.float64)
            bin_sample_counts = np.asarray(sample_counts)
            if (
                load_sums.ndim != 3
                or load_sums.shape[0] == 0
                or not np.all(np.isfinite(load_sums))
                or np.any(load_sums < 0)
            ):
                raise ValueError("prepared loads must be finite non-negative [bins, layers, experts]")
            if (
                bin_sample_counts.shape != (load_sums.shape[0],)
                or not np.issubdtype(bin_sample_counts.dtype, np.integer)
                or np.any(bin_sample_counts <= 0)
            ):
                raise ValueError("sample_counts must contain one positive integer per bin")
            load_bins = load_sums / bin_sample_counts[:, None, None]
        current = np.asarray(current_rank_expert_ids)
        if current.ndim != 3 or 0 in current.shape or not np.issubdtype(current.dtype, np.integer):
            raise ValueError("current_rank_expert_ids must be a non-empty integer [layers, ranks, slots] array")
        if load_bins.shape[1] != current.shape[0]:
            raise ValueError("logical load and current placement layer counts must match")
        current = current.astype(np.int64, copy=False)

        anchors = np.asarray(last_committed_mean_ratios, dtype=np.float64)
        if anchors.shape != (current.shape[0],):
            raise ValueError("last_committed_mean_ratios must contain one value per layer")
        if np.any(~np.isnan(anchors) & (~np.isfinite(anchors) | (anchors < 1))):
            raise ValueError("committed mean ratios must be NaN or finite values no smaller than one")
        node_ids = np.asarray(rank_node_ids)
        if (
            node_ids.shape != (current.shape[1],)
            or not np.issubdtype(node_ids.dtype, np.integer)
            or np.any(node_ids < 0)
        ):
            raise ValueError("rank_node_ids must contain one non-negative integer per rank")
        selected_layers = range(current.shape[0]) if layer_ids is None else tuple(layer_ids)
        if (
            any(type(layer_id) is not int for layer_id in selected_layers)
            or len(set(selected_layers)) != len(selected_layers)
            or any(not 0 <= layer_id < current.shape[0] for layer_id in selected_layers)
        ):
            raise ValueError("layer_ids must contain unique valid stage-local layer IDs")

        rank_expert_ids = current.copy()
        source_rank_ids = np.broadcast_to(np.arange(current.shape[1])[None, :, None], current.shape).copy()
        source_slot_ids = np.broadcast_to(np.arange(current.shape[2])[None, None, :], current.shape).copy()
        predicted_mean_ratios = np.full(current.shape[0], np.nan, dtype=np.float64)
        layer_priority_keys = []
        for layer_id in selected_layers:
            current_imbalance = cls.gated_layer_imbalance(
                load_bins[:, layer_id], bin_sample_counts, current[layer_id], anchors[layer_id], config
            )
            if current_imbalance is None:
                continue
            relative_deterioration = (
                0.0 if np.isnan(anchors[layer_id]) else current_imbalance.mean_ratio / anchors[layer_id] - 1.0
            )
            layer_priority_keys.append((-current_imbalance.mean_ratio, -relative_deterioration, layer_id))

        for _, _, layer_id in sorted(layer_priority_keys):
            layer_plan = cls.plan_layer(load_bins[:, layer_id], bin_sample_counts, current[layer_id], node_ids, config)
            if layer_plan is None:
                continue
            rank_expert_ids[layer_id] = layer_plan.placement.rank_expert_ids
            source_rank_ids[layer_id] = layer_plan.placement.source_rank_ids
            source_slot_ids[layer_id] = layer_plan.placement.source_slot_ids
            predicted_mean_ratios[layer_id] = layer_plan.predicted_imbalance.mean_ratio

        return StairPlan(
            rank_expert_ids=rank_expert_ids,
            source_rank_ids=source_rank_ids,
            source_slot_ids=source_slot_ids,
            predicted_mean_ratios=predicted_mean_ratios,
        )

    @classmethod
    def plan_sharded_rebalance(
        cls,
        logical_load_values: np.ndarray,
        current_rank_expert_ids: np.ndarray,
        last_committed_mean_ratios: np.ndarray,
        rank_node_ids: np.ndarray,
        config: StairConfig,
        cpu_group: torch.distributed.ProcessGroup,
        sample_counts: np.ndarray | None = None,
    ) -> StairPlan:
        """Plan round-robin layer shards and gather one stage-local plan.

        Every rank in ``cpu_group`` must call this method with identical model
        inputs and configuration. The group must contain only the current PP
        stage's EPLB ranks. All ranks receive and validate the same complete
        plan; no coordinator rank assembles the result. Input and result shapes
        follow :meth:`plan_rebalance` and :class:`StairPlan`.
        """
        group_size = cpu_group.size()
        local_error = None
        local_plan_fields = None
        current = None
        try:
            current = np.asarray(current_rank_expert_ids)
            if current.ndim != 3:
                raise ValueError("current_rank_expert_ids must be a [layers, ranks, slots] array")
            num_layers = current.shape[0]
            owned_layer_ids = assigned_layer_ids(num_layers, cpu_group.rank(), group_size)
            local_plan = cls.plan_rebalance(
                logical_load_values,
                current,
                last_committed_mean_ratios,
                rank_node_ids,
                config,
                layer_ids=owned_layer_ids,
                sample_counts=sample_counts,
            )
            owned_indices = np.fromiter(owned_layer_ids, dtype=np.int64, count=len(owned_layer_ids))
            local_plan_fields = tuple(
                torch.from_numpy(plan_field[owned_indices])
                for plan_field in (
                    local_plan.rank_expert_ids,
                    local_plan.source_rank_ids,
                    local_plan.source_slot_ids,
                    local_plan.predicted_mean_ratios,
                )
            )
            num_experts = np.asarray(logical_load_values).shape[2]
        except Exception as error:
            local_error = error

        planning_succeeded = torch.tensor(local_error is None, dtype=torch.int32)
        if group_size > 1:
            torch.distributed.all_reduce(
                planning_succeeded,
                op=torch.distributed.ReduceOp.MIN,
                group=cpu_group,
            )
        if not planning_succeeded.item():
            if local_error is not None:
                raise local_error
            raise RuntimeError("STAIR layer shard planning failed on another EPLB rank")
        if current is None or local_plan_fields is None:
            raise RuntimeError("STAIR layer shard planning produced no local plan")

        def gather_owned_field(local_values: torch.Tensor) -> np.ndarray:
            return all_gather_layer_shards(local_values, num_layers, cpu_group).numpy()

        plan = StairPlan(
            rank_expert_ids=gather_owned_field(local_plan_fields[0]),
            source_rank_ids=gather_owned_field(local_plan_fields[1]),
            source_slot_ids=gather_owned_field(local_plan_fields[2]),
            predicted_mean_ratios=gather_owned_field(local_plan_fields[3]),
        )
        cls.validate_plan(
            current,
            plan,
            num_experts,
            rank_node_ids,
            config.rank_transfer_limit,
            config.cross_node_transfer_limit,
        )
        return plan

    @classmethod
    def validate_plan(
        cls,
        current_rank_expert_ids: np.ndarray,
        plan: StairPlan,
        num_experts: int,
        rank_node_ids: np.ndarray,
        rank_transfer_limit: int,
        cross_node_transfer_limit: int,
    ) -> None:
        """Validate a fixed-shape plan against its current placement.

        Current and planned arrays are ``[layers, ranks, slots]``. Every source
        coordinate must own its target expert in the current placement;
        retained experts must keep their rank and slot. Rank and cross-node
        transfer usage is counted independently for each layer. Predicted mean
        ratios are ``[layers]``: changed layers require a finite value and
        unchanged layers require NaN.
        """
        current = np.asarray(current_rank_expert_ids)
        target = np.asarray(plan.rank_expert_ids)
        source_ranks = np.asarray(plan.source_rank_ids)
        source_slots = np.asarray(plan.source_slot_ids)
        if current.ndim != 3 or 0 in current.shape or not np.issubdtype(current.dtype, np.integer):
            raise ValueError("current_rank_expert_ids must be a non-empty integer [layers, ranks, slots] array")
        if target.shape != current.shape or source_ranks.shape != current.shape or source_slots.shape != current.shape:
            raise ValueError("STAIR plan placement and source arrays must match the current placement shape")
        if not all(np.issubdtype(values.dtype, np.integer) for values in (target, source_ranks, source_slots)):
            raise ValueError("STAIR plan placement and source arrays must contain integers")
        node_ids = np.asarray(rank_node_ids)
        if node_ids.shape != (current.shape[1],) or not np.issubdtype(node_ids.dtype, np.integer):
            raise ValueError("rank_node_ids must contain one integer per rank")
        controls = num_experts, rank_transfer_limit, cross_node_transfer_limit
        invalid_type = any(isinstance(value, bool) or not isinstance(value, int) for value in controls)
        invalid_limits = rank_transfer_limit < -1 or rank_transfer_limit == 0 or cross_node_transfer_limit < -1
        if invalid_type or num_experts < 1 or invalid_limits:
            raise ValueError("STAIR expert count and transfer limits are invalid")

        ratios = np.asarray(plan.predicted_mean_ratios)
        if ratios.shape != (current.shape[0],) or not np.issubdtype(ratios.dtype, np.floating):
            raise ValueError("predicted_mean_ratios must be a floating-point value per layer")
        ratios = ratios.astype(np.float64, copy=False)
        if np.any(~np.isnan(ratios) & (~np.isfinite(ratios) | (ratios < 1))):
            raise ValueError("predicted mean ratios must be NaN or finite values no smaller than one")
        if (
            np.any(source_ranks < 0)
            or np.any(source_ranks >= current.shape[1])
            or np.any(source_slots < 0)
            or np.any(source_slots >= current.shape[2])
        ):
            raise ValueError("STAIR plan contains an out-of-range source coordinate")

        for layer_id, target_layer in enumerate(target):
            current_layer = current[layer_id]
            cls.placement_replica_counts(current_layer, num_experts)
            cls.placement_replica_counts(target_layer, num_experts)
            changed = not np.array_equal(target_layer, current_layer)
            has_candidate = not np.isnan(ratios[layer_id])
            if changed != has_candidate:
                raise ValueError("predicted_mean_ratios must be finite for changed layers and NaN for unchanged layers")

            outgoing = np.zeros(current.shape[1], dtype=np.int64)
            incoming = np.zeros(current.shape[1], dtype=np.int64)
            cross_out: dict[int, int] = {}
            cross_in: dict[int, int] = {}
            for dst_rank, target_experts in enumerate(target_layer):
                current_slots = {int(expert): slot for slot, expert in enumerate(current_layer[dst_rank])}
                for dst_slot, expert in enumerate(target_experts):
                    src_rank = int(source_ranks[layer_id, dst_rank, dst_slot])
                    src_slot = int(source_slots[layer_id, dst_rank, dst_slot])
                    if current_layer[src_rank, src_slot] != expert:
                        raise ValueError("STAIR source does not own the target expert")
                    retained_slot = current_slots.get(int(expert))
                    if retained_slot is not None:
                        if (src_rank, src_slot, dst_slot) != (dst_rank, retained_slot, retained_slot):
                            raise ValueError("retained experts must keep their current rank and slot")
                        continue
                    outgoing[src_rank] += 1
                    incoming[dst_rank] += 1
                    if rank_transfer_limit != -1 and (
                        outgoing[src_rank] > rank_transfer_limit or incoming[dst_rank] > rank_transfer_limit
                    ):
                        raise ValueError("STAIR plan exceeds a per-rank transfer limit")
                    src_node, dst_node = int(node_ids[src_rank]), int(node_ids[dst_rank])
                    if src_node != dst_node:
                        cross_out[src_node] = cross_out.get(src_node, 0) + 1
                        cross_in[dst_node] = cross_in.get(dst_node, 0) + 1
                        if cross_node_transfer_limit != -1 and (
                            cross_out[src_node] > cross_node_transfer_limit
                            or cross_in[dst_node] > cross_node_transfer_limit
                        ):
                            raise ValueError("STAIR plan exceeds a per-node cross-node transfer limit")
