# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Pure NumPy implementation of the STAIR placement policy."""

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from vllm_ascend.ascend_config import StairConfig


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


def _minimum_source_cost(
    demands: list[tuple[int, int]],
    owners: dict[int, tuple[int, ...]],
    capacity: dict[tuple[int, int], int],
    node_by_rank: tuple[int, ...],
) -> int | None:
    """Solve the directed pair-capacity matching problem."""
    if not demands:
        return 0
    pairs = sorted({(src, dst) for dst, expert in demands for src in owners[expert] if capacity[(src, dst)]})
    pair_nodes = {pair: len(demands) + index + 1 for index, pair in enumerate(pairs)}
    sink = len(demands) + len(pairs) + 1
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, cap: int, cost: int) -> None:
        graph[start].append([end, len(graph[end]), cap, cost])
        graph[end].append([start, len(graph[start]) - 1, 0, -cost])

    scale = len(demands) + 1
    for index, (dst, expert) in enumerate(demands, 1):
        add_edge(0, index, 1, 0)
        for src in owners[expert]:
            if (src, dst) in pair_nodes:
                cost = 1 if node_by_rank[src] == node_by_rank[dst] else scale
                add_edge(index, pair_nodes[(src, dst)], 1, cost)
    for pair, node in pair_nodes.items():
        add_edge(node, sink, capacity[pair], 0)

    total = 0
    for _ in demands:
        distance = [10**18] * len(graph)
        parent: list[tuple[int, int] | None] = [None] * len(graph)
        distance[0] = 0
        queue, queued = deque([0]), {0}
        while queue:
            node = queue.popleft()
            queued.discard(node)
            for edge_index, edge in enumerate(graph[node]):
                target, _, cap, cost = edge
                if cap and distance[node] + cost < distance[target]:
                    distance[target] = distance[node] + cost
                    parent[target] = (node, edge_index)
                    if target not in queued:
                        queue.append(target)
                        queued.add(target)
        if parent[sink] is None:
            return None
        total += distance[sink]
        node = sink
        while node:
            previous, edge_index = parent[node]  # type: ignore[misc]
            edge = graph[previous][edge_index]
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = previous
    return total


def assign_sources(
    old_placement: np.ndarray,
    destination_experts: list[set[int]],
    node_by_rank: tuple[int, ...],
    pair_cap: int,
) -> dict[tuple[int, int], tuple[int, int]] | None:
    """Choose real sources while minimizing cross-node transfers."""
    old = np.asarray(old_placement, dtype=np.int64)
    locations: dict[int, list[tuple[int, int]]] = {}
    for rank, row in enumerate(old):
        for slot, expert in enumerate(row):
            locations.setdefault(int(expert), []).append((rank, slot))
    demands = sorted(
        (dst, expert) for dst, experts in enumerate(destination_experts) for expert in experts if expert not in old[dst]
    )
    owners = {expert: tuple(rank for rank, _ in values) for expert, values in locations.items()}
    slots = {(expert, rank): slot for expert, values in locations.items() for rank, slot in values}
    capacity = {(src, dst): pair_cap for src in range(old.shape[0]) for dst in range(old.shape[0]) if src != dst}
    target = _minimum_source_cost(demands, owners, capacity, node_by_rank)
    if target is None:
        return None

    assignment = {}
    for index, demand in enumerate(demands):
        dst, expert = demand
        for src in owners[expert]:
            pair = (src, dst)
            if not capacity.get(pair, 0):
                continue
            cost = 1 if node_by_rank[src] == node_by_rank[dst] else len(demands) + 1
            capacity[pair] -= 1
            future = _minimum_source_cost(demands[index + 1 :], owners, capacity, node_by_rank)
            if future is not None and cost + future == target:
                assignment[demand] = (src, slots[(expert, src)])
                target -= cost
                break
            capacity[pair] += 1
        else:
            return None
    return assignment


def align_slots(
    old_placement: np.ndarray,
    rank_experts: list[set[int]],
    sources: dict[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep retained experts in place and fill empty slots by expert id."""
    old = np.asarray(old_placement, dtype=np.int64)
    new = np.full_like(old, -1)
    source_rank = np.full_like(old, -1)
    source_slot = np.full_like(old, -1)
    for rank, desired in enumerate(rank_experts):
        for slot, expert in enumerate(old[rank]):
            if int(expert) in desired:
                new[rank, slot] = expert
                source_rank[rank, slot], source_slot[rank, slot] = rank, slot
        empty = iter(np.flatnonzero(new[rank] < 0))
        for expert in sorted(desired - set(old[rank])):
            slot = int(next(empty))
            new[rank, slot] = expert
            source_rank[rank, slot], source_slot[rank, slot] = sources[(rank, expert)]
    return new, source_rank, source_slot


def _post_insert_risk(
    experts: set[int],
    candidate: int,
    mean: np.ndarray,
    moments: np.ndarray,
    replicas: np.ndarray,
    z_score: float,
) -> float:
    selected = sorted((*experts, candidate))
    counts = replicas[selected].astype(np.float64)
    rank_mean = float(np.sum(mean[selected] / counts, dtype=np.float64))
    if moments.ndim == 1:
        variance = float(np.sum(moments[selected] / counts**2, dtype=np.float64))
    else:
        variance = float(np.sum(moments[np.ix_(selected, selected)] / np.outer(counts, counts), dtype=np.float64))
    result = rank_mean + z_score * np.sqrt(max(variance, 0.0))
    if not np.isfinite(result):
        raise ValueError("STAIR rank risk must be finite")
    return result


def _ordered_copies(mean: np.ndarray, moments: np.ndarray, replicas: np.ndarray, z_score: float) -> list[int]:
    diagonal = np.diag(moments) if moments.ndim == 2 else moments
    risk = mean + z_score * np.sqrt(np.maximum(diagonal, 0.0))
    copies = [
        (float(risk[expert] / replicas[expert]), expert, ordinal)
        for expert in range(len(mean))
        for ordinal in range(replicas[expert])
    ]
    copies.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [expert for _, expert, _ in copies]


def unconstrained_lpt(
    mean: np.ndarray,
    moments: np.ndarray,
    replicas: np.ndarray,
    num_ranks: int,
    z_score: float,
) -> np.ndarray:
    """Build the source-agnostic LPT placement used for screening."""
    total_slots = int(np.sum(replicas))
    if total_slots % num_ranks:
        raise ValueError("STAIR physical slots must divide evenly across ranks")
    slots_per_rank = total_slots // num_ranks
    ranks = [set() for _ in range(num_ranks)]
    for expert in _ordered_copies(mean, moments, replicas, z_score):
        candidates = [rank for rank in range(num_ranks) if len(ranks[rank]) < slots_per_rank and expert not in ranks[rank]]
        if not candidates:
            raise ValueError("STAIR replica vector has no duplicate-free placement")
        rank = min(
            candidates,
            key=lambda item: (_post_insert_risk(ranks[item], expert, mean, moments, replicas, z_score), item),
        )
        ranks[rank].add(expert)
    return np.asarray([sorted(row) for row in ranks], dtype=np.int64)


def constrained_lpt(
    mean: np.ndarray,
    moments: np.ndarray,
    replicas: np.ndarray,
    old_placement: np.ndarray,
    node_by_rank: tuple[int, ...],
    *,
    z_score: float,
    pair_cap: int,
    max_backtracks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    """Place replicas using LPT while preserving feasible source pairs."""
    old = np.asarray(old_placement, dtype=np.int64)
    if int(np.sum(replicas)) != old.size or len(node_by_rank) != old.shape[0]:
        raise ValueError("STAIR replica, placement, and topology sizes disagree")
    ranks = [set() for _ in range(old.shape[0])]
    copies = _ordered_copies(mean, moments, replicas, z_score)
    backtracks = 0

    def search(index: int) -> dict[tuple[int, int], tuple[int, int]] | None:
        nonlocal backtracks
        if index == len(copies):
            return assign_sources(old, ranks, node_by_rank, pair_cap)
        expert = copies[index]
        candidates = [rank for rank in range(old.shape[0]) if len(ranks[rank]) < old.shape[1] and expert not in ranks[rank]]
        candidates.sort(key=lambda rank: (_post_insert_risk(ranks[rank], expert, mean, moments, replicas, z_score), rank))
        for rank in candidates:
            ranks[rank].add(expert)
            feasible = assign_sources(old, ranks, node_by_rank, pair_cap) is not None
            result = search(index + 1) if feasible else None
            if result is not None:
                return result
            ranks[rank].remove(expert)
            if feasible:
                backtracks += 1
                if backtracks > max_backtracks:
                    return None
        return None

    sources = search(0)
    if sources is None:
        return None
    placement, source_rank, source_slot = align_slots(old, ranks, sources)
    cross_node = same_node = 0
    for dst, row in enumerate(source_rank):
        for src in row:
            if src != dst:
                if node_by_rank[int(src)] == node_by_rank[dst]:
                    same_node += 1
                else:
                    cross_node += 1
    return placement, source_rank, source_slot, cross_node, same_node


def passes_hysteresis(current_score: float, accepted_score: float, config: StairConfig) -> bool:
    if not config.hysteresis_enabled or np.isnan(accepted_score):
        return True
    current_balance = 1.0 / current_score
    accepted_balance = 1.0 / accepted_score
    return (
        current_balance / accepted_balance <= config.hysteresis_relative
        or current_balance <= config.hysteresis_absolute
    )


def _plan_layer(
    samples: np.ndarray,
    weights: np.ndarray,
    old: np.ndarray,
    node_by_rank: tuple[int, ...],
    config: StairConfig,
) -> LayerPlan | None:
    current = placement_score(samples, weights, old)
    mean, moments = weighted_moments(samples, weights, covariance=config.use_covariance)
    diagonal = np.diag(moments) if moments.ndim == 2 else moments
    risk = mean + config.z_score * np.sqrt(np.maximum(diagonal, 0.0))

    def screening(replicas: np.ndarray) -> float:
        try:
            placement = unconstrained_lpt(mean, moments, replicas, old.shape[0], config.z_score)
        except ValueError:
            return float("inf")
        return placement_score(samples, weights, placement).mean

    candidates = []
    for replicas in replica_candidates(
        risk,
        old.size,
        old.shape[0],
        depth=config.flash_tree_depth,
        width=config.flash_tree_width,
        limit=config.max_candidates_per_layer,
        score=screening,
    ):
        result = constrained_lpt(
            mean,
            moments,
            replicas,
            old,
            node_by_rank,
            z_score=config.z_score,
            pair_cap=config.max_expert_transfers_per_rank_pair,
            max_backtracks=config.lpt_max_backtracks,
        )
        if result is None:
            continue
        placement, source_rank, source_slot, cross_node, same_node = result
        score = placement_score(samples, weights, placement)
        relative_gain = (current.mean - score.mean) / current.mean
        if (
            relative_gain >= config.min_relative_score_improvement
            and current.mean - score.mean >= config.min_absolute_score_improvement
            and score.p95 <= current.p95 * (1 + config.p95_regression_tolerance)
        ):
            key = (cross_node, same_node, tuple(placement.ravel()), tuple(source_rank.ravel()))
            candidates.append((score.mean, key, LayerPlan(placement, source_rank, source_slot, score)))
    if not candidates:
        return None
    minimum = min(score for score, _, _ in candidates)
    tied = [item for item in candidates if item[0] <= minimum + config.score_tie_tolerance]
    return min(tied, key=lambda item: item[1])[2]


def plan_rebalance(
    logical_load: np.ndarray,
    old_placement: np.ndarray,
    accepted_scores: np.ndarray,
    node_by_rank: tuple[int, ...],
    config: StairConfig,
) -> StairPlan:
    """Run STAIR's six stages for every eligible layer."""
    samples, weights = compress_samples(logical_load, config.sample_size)
    old = np.asarray(old_placement, dtype=np.int64)
    if old.ndim != 3 or samples.shape[1] != old.shape[0] or len(node_by_rank) != old.shape[1]:
        raise ValueError("STAIR load, placement, and topology shapes disagree")
    anchors = np.asarray(accepted_scores, dtype=np.float64)
    if anchors.shape != (old.shape[0],):
        raise ValueError("STAIR accepted scores must match the layer count")

    placement = old.copy()
    source_rank = np.broadcast_to(np.arange(old.shape[1])[None, :, None], old.shape).copy()
    source_slot = np.broadcast_to(np.arange(old.shape[2])[None, None, :], old.shape).copy()
    new_scores = np.full(old.shape[0], np.nan, dtype=np.float64)
    eligible = []
    for layer in range(old.shape[0]):
        replica_counts(old[layer], samples.shape[2])
        if np.sum(samples[:, layer], dtype=np.float64) == 0:
            continue
        current = placement_score(samples[:, layer], weights, old[layer])
        if current.mean > config.imbalance_threshold and passes_hysteresis(current.mean, anchors[layer], config):
            deterioration = 0.0 if np.isnan(anchors[layer]) else current.mean / anchors[layer] - 1.0
            eligible.append((-current.mean, -deterioration, layer))

    for _, _, layer in sorted(eligible):
        result = _plan_layer(samples[:, layer], weights, old[layer], node_by_rank, config)
        if result is not None:
            placement[layer] = result.placement
            source_rank[layer] = result.source_rank
            source_slot[layer] = result.source_slot
            new_scores[layer] = result.score.mean
    return StairPlan(placement, source_rank, source_slot, new_scores)
