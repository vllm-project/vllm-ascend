# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Temporal acceptance stage for the STAIR expert placement policy."""

import numpy as np

BALANCE_EPSILON = 1e-6
FLASH_UPDATE_THRESHOLD_RATIO = 0.9
FLASH_UPDATE_THRESHOLD_VALUE = 0.85
FLASH_SMALL_WORLD_SIZE = 32
FLASH_SMALL_WORLD_UPDATE_THRESHOLD_RATIO = 0.95
FLASH_SMALL_WORLD_UPDATE_THRESHOLD_VALUE = 0.9


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


class StairEplbPolicy:
    """Accept Swift candidates only when they improve the load time series."""

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

    def _needs_flash_update(self, layer_id: int, current_score: float, num_ranks: int) -> bool:
        past_ratio = self.average_to_peak_history.get(layer_id)
        if past_ratio is None:
            return True
        if num_ranks < FLASH_SMALL_WORLD_SIZE:
            threshold_ratio = FLASH_SMALL_WORLD_UPDATE_THRESHOLD_RATIO
            threshold_value = FLASH_SMALL_WORLD_UPDATE_THRESHOLD_VALUE
        else:
            threshold_ratio = FLASH_UPDATE_THRESHOLD_RATIO
            threshold_value = FLASH_UPDATE_THRESHOLD_VALUE
        current_ratio = 1.0 / current_score
        return current_ratio < past_ratio * threshold_ratio or current_ratio < threshold_value

    def rebalance_experts(
        self,
        expert_load: np.ndarray,
        current_placement: np.ndarray,
        candidate_placement: np.ndarray,
        num_ranks: int,
    ) -> np.ndarray:
        """Filter a complete Swift proposal with FlashLB temporal scoring."""
        expert_load = np.asarray(expert_load, dtype=np.float64)
        current_placement = np.asarray(current_placement, dtype=np.int64)
        candidate_placement = np.asarray(candidate_placement, dtype=np.int64)
        if expert_load.ndim != 3:
            raise ValueError(f"expert_load must have shape [window, layers, experts], got {expert_load.shape}.")
        if current_placement.ndim != 2:
            raise ValueError(f"current_placement must have shape [layers, replicas], got {current_placement.shape}.")
        if candidate_placement.shape != current_placement.shape:
            raise ValueError("candidate_placement and current_placement must have the same shape.")
        if expert_load.shape[0] == 0 or expert_load.shape[2] == 0:
            raise ValueError("expert_load window and expert dimensions must be nonzero.")
        if current_placement.shape[0] != expert_load.shape[1]:
            raise ValueError("expert_load and current_placement must have the same number of layers.")
        if num_ranks <= 0 or current_placement.shape[1] % num_ranks != 0:
            raise ValueError("The number of physical replicas must be divisible by num_ranks.")
        if not np.all(np.isfinite(expert_load)) or np.any(expert_load < 0):
            raise ValueError("expert_load must contain finite, non-negative values.")

        num_experts = expert_load.shape[2]
        slots_per_rank = current_placement.shape[1] // num_ranks
        current_by_rank = current_placement.reshape(current_placement.shape[0], num_ranks, slots_per_rank)
        candidate_by_rank = candidate_placement.reshape(candidate_placement.shape[0], num_ranks, slots_per_rank)
        for layer_placement in current_by_rank:
            _validate_layer_placement(layer_placement, num_experts, num_ranks)
        for layer_placement in candidate_by_rank:
            _validate_layer_placement(layer_placement, num_experts, num_ranks)
        self._prepare_history(expert_load, current_placement, num_ranks)

        result = current_by_rank.copy()
        if not np.any(expert_load):
            return result.reshape(current_placement.shape).copy()

        changed_layers = np.flatnonzero(np.any(candidate_placement != current_placement, axis=1))
        for layer_id in changed_layers:
            layer_load = expert_load[:, layer_id, :]
            current_score = compute_balance_score(layer_load, current_by_rank[layer_id])
            if not self._needs_flash_update(int(layer_id), current_score, num_ranks):
                continue
            candidate_score = compute_balance_score(layer_load, candidate_by_rank[layer_id])
            if current_score - candidate_score > BALANCE_EPSILON:
                result[layer_id] = candidate_by_rank[layer_id]

        return result.reshape(current_placement.shape).copy()

    def commit_layer(
        self,
        expert_load: np.ndarray,
        layer_id: int,
        committed_placement: np.ndarray,
        num_ranks: int,
    ) -> None:
        """Record hysteresis after one layer is actually committed."""
        expert_load = np.asarray(expert_load, dtype=np.float64)
        committed_placement = np.asarray(committed_placement, dtype=np.int64)
        if expert_load.ndim != 3 or not 0 <= layer_id < expert_load.shape[1]:
            raise ValueError("expert_load must contain the committed layer.")
        if committed_placement.ndim != 1 or committed_placement.size % num_ranks != 0:
            raise ValueError("committed_placement must be one-dimensional and divisible by num_ranks.")
        placement = committed_placement.reshape(num_ranks, -1)
        _validate_layer_placement(placement, expert_load.shape[2], num_ranks)
        score = compute_balance_score(expert_load[:, layer_id, :], placement)
        self.average_to_peak_history[layer_id] = 1.0 / score
        self._expected_layer_placements[layer_id] = committed_placement.copy()
