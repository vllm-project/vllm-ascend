# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Statistical Temporal-Aware Incremental Rebalancing (STAIR) policy."""

import time
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import torch
from vllm.logger import logger

BALANCE_EPSILON = 1e-6
IMBALANCE_THRESHOLD = 1.01
SWAP_IMPROVEMENT_RATIO = 0.01
MAX_COMMUNICATIONS_PER_RANK_PAIR = 1
MAX_SWAP_ATTEMPTS = 100

TEMPORAL_UPDATE_THRESHOLD_RATIO = 0.9
TEMPORAL_UPDATE_THRESHOLD_VALUE = 0.85
SMALL_WORLD_SIZE = 32
SMALL_WORLD_UPDATE_THRESHOLD_RATIO = 0.95
SMALL_WORLD_UPDATE_THRESHOLD_VALUE = 0.9


def _replica_counts(placement: np.ndarray, num_experts: int) -> np.ndarray:
    return np.bincount(placement.reshape(-1), minlength=num_experts).astype(np.int64, copy=False)


def _rank_loads(expert_load: np.ndarray, placement: np.ndarray, num_experts: int) -> np.ndarray:
    replica_counts = _replica_counts(placement, num_experts)
    if np.any(replica_counts == 0):
        raise ValueError("Every logical expert must have at least one physical replica.")

    rank_loads = np.zeros((expert_load.shape[0], placement.shape[0]), dtype=np.float64)
    for rank_id, rank in enumerate(placement):
        rank_loads[:, rank_id] = np.sum(expert_load[:, rank] / replica_counts[rank], axis=1)
    return rank_loads


def _score_rank_loads(rank_loads: np.ndarray) -> float:
    total_load = np.sum(rank_loads, axis=1)
    scores = np.ones(rank_loads.shape[0], dtype=np.float64)
    nonzero = total_load > 0
    if np.any(nonzero):
        average_load = total_load[nonzero] / rank_loads.shape[1]
        scores[nonzero] = np.max(rank_loads[nonzero], axis=1) / average_load
    return float(np.mean(scores))


def compute_balance_score(expert_load: np.ndarray, placement: np.ndarray) -> float:
    """Return the mean peak-to-average rank load for one MoE layer."""
    expert_load = np.asarray(expert_load, dtype=np.float64)
    placement = np.asarray(placement, dtype=np.int64)
    if expert_load.ndim != 2:
        raise ValueError(f"expert_load must have shape [window, experts], got {expert_load.shape}.")
    if placement.ndim != 2:
        raise ValueError(f"placement must have shape [ranks, slots], got {placement.shape}.")
    if expert_load.shape[0] == 0 or expert_load.shape[1] == 0:
        raise ValueError("expert_load window and expert dimensions must be nonzero.")
    if np.any(placement < 0) or np.any(placement >= expert_load.shape[1]):
        raise ValueError("placement contains an invalid logical expert index.")
    return _score_rank_loads(_rank_loads(expert_load, placement, expert_load.shape[1]))


def _validate_layer_placement(placement: np.ndarray, num_experts: int, num_ranks: int) -> None:
    if placement.ndim != 2 or placement.shape[0] != num_ranks:
        raise ValueError("placement must have shape [num_ranks, slots_per_rank].")
    if np.any(placement < 0) or np.any(placement >= num_experts):
        raise ValueError("placement contains an invalid logical expert index.")
    replica_counts = _replica_counts(placement, num_experts)
    if np.any(replica_counts == 0):
        raise ValueError("Every logical expert must have at least one physical replica.")
    if np.any(replica_counts > num_ranks):
        raise ValueError("A logical expert cannot have more replicas than ranks.")
    for rank in placement:
        if np.unique(rank).size != rank.size:
            raise ValueError("A logical expert cannot have two replicas on the same rank.")


def _allocate_replica_counts(expert_load: np.ndarray, num_replicas: int, num_ranks: int) -> np.ndarray:
    """Allocate redundant slots greedily to the hottest per-replica load."""
    num_experts = expert_load.size
    replica_counts: npt.NDArray[np.int64] = np.ones(num_experts, dtype=np.int64)
    per_replica_load: npt.NDArray[np.float64] = expert_load.astype(np.float64, copy=True)
    for _ in range(num_replicas - num_experts):
        allocated = False
        for expert_id in np.argsort(per_replica_load, kind="stable")[::-1]:
            if replica_counts[expert_id] >= num_ranks:
                continue
            replica_counts[expert_id] += 1
            per_replica_load[expert_id] = expert_load[expert_id] / replica_counts[expert_id]
            allocated = True
            break
        if not allocated:
            raise ValueError("The requested replica count cannot be placed without duplicate experts on a rank.")
    return replica_counts


def _find_primary_experts(
    placement: np.ndarray,
    num_experts: int,
) -> tuple[list[list[int]], np.ndarray]:
    """Find the stable primary copy and the replaceable redundant slots."""
    num_ranks, slots_per_rank = placement.shape
    redundant_slots: list[list[int]] = [[] for _ in range(num_ranks)]
    source_rank: npt.NDArray[np.int64] = np.full(num_experts, -1, dtype=np.int64)
    seen: set[int] = set()
    # Column-major traversal preserves the legacy placement convention while
    # keeping this implementation independent of the Model Runner V1 policy.
    for slot_idx in range(slots_per_rank):
        for rank_idx in range(num_ranks):
            expert_id = int(placement[rank_idx, slot_idx])
            if expert_id in seen:
                redundant_slots[rank_idx].append(slot_idx)
            else:
                seen.add(expert_id)
                source_rank[expert_id] = rank_idx
    if len(seen) != num_experts:
        raise ValueError("Every logical expert must have at least one physical replica.")
    return redundant_slots, source_rank


def _place_replicas(
    expert_load: np.ndarray,
    current: np.ndarray,
    target_replica_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, defaultdict[int, set[int]]]:
    """Place target replicas with bounded point-to-point communication."""
    num_ranks, slots_per_rank = current.shape
    num_experts = expert_load.size
    redundant_slots, source_rank = _find_primary_experts(current, num_experts)
    assignments = np.full_like(current, -1)
    target_load = expert_load / target_replica_counts

    for rank_idx in range(num_ranks):
        redundant = set(redundant_slots[rank_idx])
        for slot_idx, expert_id in enumerate(current[rank_idx]):
            if slot_idx not in redundant:
                assignments[rank_idx, slot_idx] = expert_id

    rank_loads = np.zeros(num_ranks, dtype=np.float64)
    for rank_idx, rank in enumerate(assignments):
        valid = rank[rank >= 0]
        rank_loads[rank_idx] = target_load[valid].sum()

    communication = np.zeros((num_ranks, num_ranks), dtype=np.int64)
    received_by_rank: defaultdict[int, set[int]] = defaultdict(set)
    replicas_to_place: list[tuple[int, float]] = []
    for expert_id, replica_count in enumerate(target_replica_counts):
        replicas_to_place.extend((expert_id, float(target_load[expert_id])) for _ in range(replica_count - 1))
    replicas_to_place.sort(key=lambda item: item[1], reverse=True)

    for expert_id, load in replicas_to_place:
        source = int(source_rank[expert_id])
        candidate = -1
        for rank_idx in range(num_ranks):
            if not redundant_slots[rank_idx] or expert_id in assignments[rank_idx]:
                continue
            if communication[source, rank_idx] >= MAX_COMMUNICATIONS_PER_RANK_PAIR:
                continue
            if candidate == -1 or rank_loads[rank_idx] < rank_loads[candidate]:
                candidate = rank_idx
        if candidate == -1:
            continue
        slot_idx = redundant_slots[candidate].pop()
        assignments[candidate, slot_idx] = expert_id
        rank_loads[candidate] += load
        communication[source, candidate] += 1
        received_by_rank[candidate].add(expert_id)

    # Communication limits can leave slots unassigned. Fill those slots with
    # the hottest expert absent from that rank and recompute per-replica load.
    replica_counts = np.bincount(assignments[assignments >= 0], minlength=num_experts).astype(np.int64)
    per_replica_load: npt.NDArray[np.float64] = np.full(num_experts, -1.0, dtype=np.float64)
    present = replica_counts > 0
    per_replica_load[present] = expert_load[present] / replica_counts[present]
    for rank_idx in range(num_ranks):
        for slot_idx in redundant_slots[rank_idx]:
            for expert_id in np.argsort(per_replica_load, kind="stable")[::-1]:
                if expert_id in assignments[rank_idx]:
                    continue
                assignments[rank_idx, slot_idx] = expert_id
                source = int(source_rank[expert_id])
                communication[source, rank_idx] += 1
                received_by_rank[rank_idx].add(int(expert_id))
                old_count = replica_counts[expert_id]
                replica_counts[expert_id] += 1
                per_replica_load[expert_id] *= old_count / replica_counts[expert_id]
                break

    if np.any(assignments < 0):
        raise RuntimeError("STAIR failed to assign every physical expert slot.")
    rank_loads = np.sum(per_replica_load[assignments], axis=1)
    return assignments, rank_loads, communication, received_by_rank


def _swap_experts(
    assignments: np.ndarray,
    rank_loads: np.ndarray,
    expert_load: np.ndarray,
    communication: np.ndarray,
    received_by_rank: defaultdict[int, set[int]],
) -> np.ndarray:
    """Incrementally exchange resident experts to reduce the peak rank load."""
    num_ranks = assignments.shape[0]
    replica_counts = _replica_counts(assignments, expert_load.size)
    per_replica_load = expert_load / replica_counts
    minimum_gain = expert_load.sum() / num_ranks * SWAP_IMPROVEMENT_RATIO
    rank_experts = [set(int(expert) for expert in rank) for rank in assignments]

    exchanged = True
    attempts = MAX_SWAP_ATTEMPTS
    while exchanged and attempts > 0:
        attempts -= 1
        exchanged = False
        sorted_ranks = np.argsort(rank_loads, kind="stable")
        hot_rank = int(sorted_ranks[-1])
        hot_load = float(rank_loads[hot_rank])
        for cold_rank_value in sorted_ranks[:-1]:
            cold_rank = int(cold_rank_value)
            if (
                communication[cold_rank, hot_rank] >= MAX_COMMUNICATIONS_PER_RANK_PAIR
                or communication[hot_rank, cold_rank] >= MAX_COMMUNICATIONS_PER_RANK_PAIR
            ):
                continue

            best_pair: tuple[int, int] | None = None
            best_peak = hot_load
            for hot_expert in sorted(rank_experts[hot_rank]):
                if hot_expert in rank_experts[cold_rank] or hot_expert in received_by_rank[hot_rank]:
                    continue
                for cold_expert in sorted(rank_experts[cold_rank]):
                    if cold_expert in rank_experts[hot_rank] or cold_expert in received_by_rank[cold_rank]:
                        continue
                    hot_after = hot_load - per_replica_load[hot_expert] + per_replica_load[cold_expert]
                    cold_after = rank_loads[cold_rank] - per_replica_load[cold_expert] + per_replica_load[hot_expert]
                    peak_after = max(hot_after, cold_after)
                    if peak_after < best_peak:
                        best_peak = peak_after
                        best_pair = hot_expert, cold_expert

            if best_pair is None or hot_load - best_peak < minimum_gain:
                continue
            hot_expert, cold_expert = best_pair
            rank_experts[hot_rank].remove(hot_expert)
            rank_experts[cold_rank].remove(cold_expert)
            rank_experts[hot_rank].add(cold_expert)
            rank_experts[cold_rank].add(hot_expert)
            rank_loads[hot_rank] += per_replica_load[cold_expert] - per_replica_load[hot_expert]
            rank_loads[cold_rank] += per_replica_load[hot_expert] - per_replica_load[cold_expert]
            received_by_rank[hot_rank].add(cold_expert)
            received_by_rank[cold_rank].add(hot_expert)
            communication[cold_rank, hot_rank] += 1
            communication[hot_rank, cold_rank] += 1
            exchanged = True
            break

    desired = np.asarray([sorted(experts) for experts in rank_experts], dtype=np.int64)
    return desired


def _align_local_slots(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Keep resident experts in their old slots and fill only changed slots."""
    aligned = np.full_like(current, -1)
    for rank_idx in range(current.shape[0]):
        desired_experts = set(int(expert) for expert in desired[rank_idx])
        for slot_idx, expert_id in enumerate(current[rank_idx]):
            if int(expert_id) in desired_experts:
                aligned[rank_idx, slot_idx] = expert_id
                desired_experts.remove(int(expert_id))
        replacements = iter(sorted(desired_experts))
        for slot_idx in np.flatnonzero(aligned[rank_idx] < 0):
            aligned[rank_idx, slot_idx] = next(replacements)
    return aligned


def _build_incremental_candidate(
    logical_load: np.ndarray,
    current_placement: np.ndarray,
    num_ranks: int,
) -> np.ndarray:
    """Build STAIR's complete balance candidate without Model Runner V1 code."""
    num_layers, num_experts = logical_load.shape
    slots_per_rank = current_placement.shape[1] // num_ranks
    current_by_rank = current_placement.reshape(num_layers, num_ranks, slots_per_rank)
    candidate = current_by_rank.copy()

    for layer_idx in range(num_layers):
        current_layer = current_by_rank[layer_idx]
        current_score = compute_balance_score(logical_load[layer_idx : layer_idx + 1], current_layer)
        if current_score < IMBALANCE_THRESHOLD:
            continue
        target_counts = _allocate_replica_counts(logical_load[layer_idx], current_layer.size, num_ranks)
        assignments, rank_loads, communication, received_by_rank = _place_replicas(
            logical_load[layer_idx],
            current_layer,
            target_counts,
        )
        desired = _swap_experts(
            assignments,
            rank_loads,
            logical_load[layer_idx],
            communication,
            received_by_rank,
        )
        proposed = _align_local_slots(current_layer, desired)
        _validate_layer_placement(proposed, num_experts, num_ranks)
        if compute_balance_score(logical_load[layer_idx : layer_idx + 1], proposed) < current_score:
            candidate[layer_idx] = proposed

    return candidate.reshape(current_placement.shape)


class StairEplbPolicy:
    """Generate and temporally filter incremental expert placements."""

    uses_expert_load_time_series = True

    def __init__(self) -> None:
        self.average_to_peak_history: dict[int, float] = {}
        self._topology: tuple[int, int, int, int] | None = None
        self._expected_layer_placements: dict[int, np.ndarray] = {}

    def _prepare_history(
        self,
        expert_load: np.ndarray,
        current_placement: np.ndarray,
        num_ranks: int,
    ) -> None:
        topology = (
            expert_load.shape[1],
            expert_load.shape[2],
            current_placement.shape[1],
            num_ranks,
        )
        if self._topology != topology:
            self.average_to_peak_history.clear()
            self._expected_layer_placements.clear()
            self._topology = topology
            return

        for layer_id, expected in list(self._expected_layer_placements.items()):
            if layer_id >= current_placement.shape[0] or not np.array_equal(current_placement[layer_id], expected):
                self.average_to_peak_history.pop(layer_id, None)
                self._expected_layer_placements.pop(layer_id, None)

    def _needs_temporal_update(self, layer_id: int, current_score: float, num_ranks: int) -> bool:
        past_ratio = self.average_to_peak_history.get(layer_id)
        if past_ratio is None:
            return True
        if num_ranks < SMALL_WORLD_SIZE:
            threshold_ratio = SMALL_WORLD_UPDATE_THRESHOLD_RATIO
            threshold_value = SMALL_WORLD_UPDATE_THRESHOLD_VALUE
        else:
            threshold_ratio = TEMPORAL_UPDATE_THRESHOLD_RATIO
            threshold_value = TEMPORAL_UPDATE_THRESHOLD_VALUE
        current_ratio = 1.0 / current_score
        return current_ratio < past_ratio * threshold_ratio or current_ratio < threshold_value

    def rebalance_experts(
        self,
        weight: torch.Tensor,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a CPU physical-to-logical map through the vLLM policy contract."""
        if old_global_expert_indices is None:
            raise ValueError("STAIR EPLB requires the current physical-to-logical map.")
        if num_replicas <= 0 or num_groups <= 0 or num_nodes <= 0 or num_ranks <= 0:
            raise ValueError("num_replicas, num_groups, num_nodes, and num_ranks must be positive.")
        if num_replicas % num_ranks != 0:
            raise ValueError(f"num_replicas ({num_replicas}) must be divisible by num_ranks ({num_ranks}).")
        if num_ranks % num_nodes != 0:
            raise ValueError(f"num_ranks ({num_ranks}) must be divisible by num_nodes ({num_nodes}).")
        if weight.ndim != 3:
            raise ValueError(f"STAIR EPLB requires [window, layers, experts] load, got {tuple(weight.shape)}.")
        if weight.device.type != "cpu" or old_global_expert_indices.device.type != "cpu":
            raise ValueError("STAIR EPLB policy inputs must be CPU tensors.")
        if old_global_expert_indices.shape != (weight.shape[1], num_replicas):
            raise ValueError(
                "Current placement shape must be [layers, num_replicas], got "
                f"{tuple(old_global_expert_indices.shape)} for weight shape {tuple(weight.shape)} "
                f"and num_replicas={num_replicas}."
            )

        expert_load = weight.detach().to(dtype=torch.float64).contiguous().numpy()
        current = old_global_expert_indices.detach().to(dtype=torch.long).contiguous().numpy().copy()
        if not np.all(np.isfinite(expert_load)) or np.any(expert_load < 0):
            raise ValueError("expert_load must contain finite, non-negative values.")
        if expert_load.shape[0] == 0 or expert_load.shape[2] == 0:
            raise ValueError("expert_load window and expert dimensions must be nonzero.")

        slots_per_rank = num_replicas // num_ranks
        current_by_rank = current.reshape(current.shape[0], num_ranks, slots_per_rank)
        for layer_placement in current_by_rank:
            _validate_layer_placement(layer_placement, expert_load.shape[2], num_ranks)
        if not np.any(expert_load):
            return torch.from_numpy(current).to(dtype=torch.long).contiguous()

        start_time = time.perf_counter()
        candidate = _build_incremental_candidate(expert_load.sum(axis=0), current, num_ranks)
        candidate_by_rank = candidate.reshape(candidate.shape[0], num_ranks, slots_per_rank)
        self._prepare_history(expert_load, current, num_ranks)

        result = current_by_rank.copy()
        candidate_layers = np.flatnonzero(np.any(candidate != current, axis=1))
        for layer_id in candidate_layers:
            layer_load = expert_load[:, layer_id, :]
            current_score = compute_balance_score(layer_load, current_by_rank[layer_id])
            if not self._needs_temporal_update(int(layer_id), current_score, num_ranks):
                continue
            candidate_score = compute_balance_score(layer_load, candidate_by_rank[layer_id])
            if current_score - candidate_score > BALANCE_EPSILON:
                result[layer_id] = candidate_by_rank[layer_id]

        result = result.reshape(current.shape).copy()
        changed_layers = int(np.count_nonzero(np.any(result != current, axis=1)))
        logger.info(
            "STAIR EPLB policy completed in %.3f ms; proposed %d layers and accepted %d layers.",
            (time.perf_counter() - start_time) * 1000,
            candidate_layers.size,
            changed_layers,
        )
        return torch.from_numpy(result).to(dtype=torch.long).contiguous()

    def commit_layer(
        self,
        expert_load: torch.Tensor,
        layer_idx: int,
        committed_placement: torch.Tensor,
        num_ranks: int,
    ) -> None:
        """Record hysteresis only after one layer is actually committed."""
        load_array = expert_load.detach().to(dtype=torch.float64).contiguous().numpy()
        placement_array = committed_placement.detach().to(dtype=torch.long).contiguous().numpy()
        if load_array.ndim != 3 or not 0 <= layer_idx < load_array.shape[1]:
            raise ValueError("expert_load must contain the committed layer.")
        if placement_array.ndim != 1 or placement_array.size % num_ranks != 0:
            raise ValueError("committed_placement must be one-dimensional and divisible by num_ranks.")
        placement = placement_array.reshape(num_ranks, -1)
        _validate_layer_placement(placement, load_array.shape[2], num_ranks)
        score = compute_balance_score(load_array[:, layer_idx, :], placement)
        self.average_to_peak_history[layer_idx] = 1.0 / score
        self._expected_layer_placements[layer_idx] = placement_array.copy()
