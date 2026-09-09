# SPDX-License-Identifier: Apache-2.0
#
# Capacity-aware per-layer expert placement utilities.
#
# This module intentionally has no dependency on torch or the EPLB runtime.  It
# can therefore be used by an offline map builder and by a background dynamic
# policy worker without importing device libraries.

from __future__ import annotations

from typing import Sequence

import numpy as np


Placement = tuple[tuple[np.ndarray, ...], ...]


def _strict_integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}")
    return result


def _strict_integer_array(value: object, name: str, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype == np.bool_ or not np.issubdtype(result.dtype, np.integer):
        raise TypeError(f"{name} must contain only integers")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    return result.astype(np.int64, copy=False)


def _normalise_load(expert_load: np.ndarray | Sequence[object]) -> np.ndarray:
    raw_load = np.asarray(expert_load)
    if raw_load.dtype == np.bool_:
        raise TypeError("expert_load must be numeric, not boolean")
    load = np.asarray(expert_load, dtype=np.float64)
    if load.ndim == 2:
        load = load[np.newaxis, ...]
    if load.ndim != 3:
        raise ValueError(
            "expert_load must have shape [layers, experts] or "
            f"[batches, layers, experts], got {load.shape}"
        )
    if 0 in load.shape:
        raise ValueError(f"expert_load dimensions must be non-zero, got {load.shape}")
    if not np.all(np.isfinite(load)):
        raise ValueError("expert_load must contain only finite values")
    if np.any(load < 0):
        raise ValueError("expert_load must be non-negative")
    return load


def capacity_aware_interleaving(
    physical_experts_by_layer: Sequence[int] | np.ndarray,
    num_ranks: int,
) -> np.ndarray:
    """Distribute variable layer capacities while balancing total rank memory.

    This is Algorithm 3 from layer-aware, applied to total physical experts.  Applying
    it to total experts is equivalent to applying it to replicas when the
    logical expert count is divisible by the EP rank count, and also handles
    the non-divisible case.
    """

    totals = _strict_integer_array(
        physical_experts_by_layer,
        "physical_experts_by_layer",
        ndim=1,
    )
    if totals.ndim != 1 or totals.size == 0:
        raise ValueError("physical_experts_by_layer must be a non-empty vector")
    if np.any(totals < 0):
        raise ValueError("physical expert counts must be non-negative")
    num_ranks = _strict_integer(num_ranks, "num_ranks", minimum=1)

    base = totals // num_ranks
    remainder = totals % num_ranks
    capacity = np.repeat(base[:, np.newaxis], num_ranks, axis=1)
    rank_totals = capacity.sum(axis=0)

    for layer, count in enumerate(remainder.tolist()):
        if count == 0:
            continue
        ordered = np.argsort(rank_totals, kind="stable")
        cutoff = rank_totals[ordered[count - 1]]
        must_choose = np.flatnonzero(rank_totals < cutoff)
        tied = np.flatnonzero(rank_totals == cutoff)
        tie_count = count - must_choose.size
        if tie_count:
            positions = np.linspace(0, tied.size - 1, tie_count).astype(np.int64)
            selected = np.concatenate((must_choose, tied[positions]))
        else:
            selected = must_choose
        capacity[layer, selected] += 1
        rank_totals[selected] += 1

    return capacity


def _replicate_layer(
    logical_load: np.ndarray,
    num_replicas: int,
    num_ranks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select replicas by repeatedly splitting the hottest effective expert."""

    num_replicas = _strict_integer(num_replicas, "num_replicas", minimum=0)
    num_ranks = _strict_integer(num_ranks, "num_ranks", minimum=1)
    num_experts = logical_load.size
    max_replicas = num_experts * (num_ranks - 1)
    if num_replicas < 0 or num_replicas > max_replicas:
        raise ValueError(
            "infeasible replica budget: each logical expert may have at most "
            f"{num_ranks} total copies, so at most {max_replicas} additional "
            f"replicas are possible; got {num_replicas}"
        )

    copy_count = np.ones(num_experts, dtype=np.int64)
    replicas: list[int] = []
    for _ in range(num_replicas):
        eligible = np.flatnonzero(copy_count < num_ranks)
        if eligible.size == 0:
            raise ValueError(
                "infeasible replica budget: all logical experts already have "
                f"the maximum {num_ranks} total copies"
            )
        source_multiplicity = np.ceil(num_ranks / copy_count[eligible])
        effective_load = logical_load[eligible] * source_multiplicity / num_ranks
        expert = int(eligible[int(np.argmax(effective_load))])
        replicas.append(expert)
        copy_count[expert] += 1
        if copy_count[expert] > num_ranks:
            raise RuntimeError(
                "layer-aware replica planner violated the per-expert copy limit"
            )

    physical_to_logical = np.concatenate(
        (
            np.arange(num_experts, dtype=np.int64),
            np.asarray(replicas, dtype=np.int64),
        )
    )
    return physical_to_logical, copy_count


def _source_ranks_per_copy(copy_count: int, num_ranks: int) -> np.ndarray:
    """Return runtime owners for ``source_rank % copy_count`` routing."""

    copy_count = _strict_integer(copy_count, "copy_count", minimum=1)
    num_ranks = _strict_integer(num_ranks, "num_ranks", minimum=1)
    if copy_count > num_ranks:
        raise ValueError("copy_count cannot exceed num_ranks")
    return np.bincount(
        np.arange(num_ranks, dtype=np.int64) % copy_count,
        minlength=copy_count,
    )


def _physical_copy_loads(
    physical_to_logical: np.ndarray,
    logical_load: np.ndarray,
    copy_count: np.ndarray,
    num_ranks: int,
) -> np.ndarray:
    occurrence = np.zeros(logical_load.size, dtype=np.int64)
    result = np.empty(physical_to_logical.size, dtype=np.float64)
    weights = [
        _source_ranks_per_copy(int(count), num_ranks) / num_ranks
        for count in copy_count.tolist()
    ]
    for physical, logical_value in enumerate(physical_to_logical.tolist()):
        logical = int(logical_value)
        copy_index = int(occurrence[logical])
        result[physical] = logical_load[logical] * weights[logical][copy_index]
        occurrence[logical] += 1
    return result


def _complete_degree_sequence_is_feasible(
    logical_remaining: np.ndarray,
    rank_remaining: np.ndarray,
) -> bool:
    """Check an unrestricted bipartite degree sequence with Gale-Ryser."""

    logical = np.asarray(logical_remaining, dtype=np.int64)
    ranks = np.asarray(rank_remaining, dtype=np.int64)
    if np.any(logical < 0) or np.any(ranks < 0):
        return False
    if int(logical.sum()) != int(ranks.sum()):
        return False

    logical = np.sort(logical[logical > 0])[::-1]
    ranks = np.sort(ranks[ranks > 0])[::-1]
    if logical.size == 0:
        return ranks.size == 0
    if ranks.size == 0:
        return False
    if int(logical[0]) > ranks.size or int(ranks[0]) > logical.size:
        return False

    counts = np.arange(1, logical.size + 1, dtype=np.int64)
    logical_prefix = np.cumsum(logical)
    rank_prefix_bound = np.minimum(
        ranks[:, None], counts[None, :]
    ).sum(axis=0)
    return bool(np.all(logical_prefix <= rank_prefix_bound))


def _residual_placement_is_feasible(
    logical_remaining: np.ndarray,
    rank_remaining: np.ndarray,
    selected_ranks: Sequence[set[int]],
) -> bool:
    """Check whether the residual duplicate-free placement has a completion."""

    logical = np.asarray(logical_remaining, dtype=np.int64)
    ranks = np.asarray(rank_remaining, dtype=np.int64)
    if logical.ndim != 1 or ranks.ndim != 1:
        return False
    if len(selected_ranks) != logical.size:
        return False
    if np.any(logical < 0) or np.any(ranks < 0):
        return False
    required = int(logical.sum())
    if required != int(ranks.sum()):
        return False
    if required == 0:
        return True

    active_logical = np.flatnonzero(logical > 0)
    active_ranks = np.flatnonzero(ranks > 0)
    for logical_id in active_logical.tolist():
        available = sum(
            rank_id not in selected_ranks[logical_id]
            for rank_id in active_ranks.tolist()
        )
        if int(logical[logical_id]) > available:
            return False

    has_active_forbidden_edges = any(
        selected_ranks[logical_id]
        for logical_id in active_logical.tolist()
    )
    if not has_active_forbidden_edges:
        return _complete_degree_sequence_is_feasible(logical, ranks)

    partial = [
        logical_id
        for logical_id in active_logical.tolist()
        if selected_ranks[logical_id]
    ]
    if len(partial) == 1:
        logical_id = partial[0]
        available = [
            rank_id
            for rank_id in active_ranks.tolist()
            if rank_id not in selected_ranks[logical_id]
        ]
        needed = int(logical[logical_id])
        residual_logical = logical.copy()
        residual_logical[logical_id] = 0
        preferred = sorted(
            available,
            key=lambda rank_id: (-int(ranks[rank_id]), rank_id),
        )[:needed]
        residual_ranks = ranks.copy()
        residual_ranks[preferred] -= 1
        return _complete_degree_sequence_is_feasible(
            residual_logical, residual_ranks
        )

    raise RuntimeError(
        "layer-aware placement order produced multiple partially placed experts"
    )


def _place_layer(
    physical_to_logical: np.ndarray,
    logical_load: np.ndarray,
    copy_count: np.ndarray,
    rank_capacity: np.ndarray,
    num_nodes: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Artifact-style capacity-aware, node-interleaved greedy placement."""

    num_ranks = rank_capacity.size
    if num_nodes <= 0 or num_ranks % num_nodes != 0:
        raise ValueError(
            f"num_nodes={num_nodes} must divide num_ranks={num_ranks}"
        )
    if int(rank_capacity.sum()) != physical_to_logical.size:
        raise ValueError(
            "rank capacity does not match physical expert count: "
            f"{rank_capacity.sum()} != {physical_to_logical.size}"
        )

    ranks_per_node = num_ranks // num_nodes
    physical_copy_load = _physical_copy_loads(
        physical_to_logical,
        logical_load,
        copy_count,
        num_ranks,
    )
    rank_load = np.zeros(num_ranks, dtype=np.float64)
    rank_count = np.zeros(num_ranks, dtype=np.int64)
    rank_remaining = rank_capacity.astype(np.int64, copy=True)
    rank_lists: list[list[int]] = [[] for _ in range(num_ranks)]
    physical_to_rank = np.full(physical_to_logical.size, -1, dtype=np.int64)

    remaining_copies = copy_count.astype(np.int64, copy=True)
    selected_ranks: list[set[int]] = [
        set() for _ in range(logical_load.size)
    ]
    # Process all copies of one logical expert contiguously. This preserves the
    # bipartite Havel-Hakimi invariant: at most one active expert has forbidden
    # ranks, so residual feasibility needs one vectorized degree check instead
    # of rebuilding a generic max-flow graph for every candidate.
    physical_by_logical = tuple(
        np.flatnonzero(physical_to_logical == logical)
        for logical in range(logical_load.size)
    )
    logical_order = sorted(
        range(logical_load.size),
        key=lambda logical: (
            -float(physical_copy_load[physical_by_logical[logical]].max()),
            logical,
        ),
    )
    physical_order = [
        int(physical)
        for logical in logical_order
        for physical in sorted(
            physical_by_logical[logical].tolist(),
            key=lambda physical: (-float(physical_copy_load[physical]), physical),
        )
    ]
    for physical in physical_order:
        logical = int(physical_to_logical[physical])
        copies_by_node = np.zeros(num_nodes, dtype=np.int64)
        for selected in selected_ranks[logical]:
            copies_by_node[selected // ranks_per_node] += 1

        eligible = [
            rank
            for rank in range(num_ranks)
            if rank_remaining[rank] > 0
            and rank not in selected_ranks[logical]
        ]
        if not eligible:
            raise RuntimeError(
                "rank capacities cannot realize a duplicate-free "
                f"placement for logical expert {logical}"
            )

        ordered_candidates = sorted(
            eligible,
            key=lambda rank: (
                float(rank_load[rank]),
                int(copies_by_node[rank // ranks_per_node]),
                -int(rank_remaining[rank]),
                rank,
            ),
        )
        target = None
        for candidate in ordered_candidates:
            remaining_copies[logical] -= 1
            rank_remaining[candidate] -= 1
            selected_ranks[logical].add(candidate)
            if _residual_placement_is_feasible(
                remaining_copies,
                rank_remaining,
                selected_ranks,
            ):
                target = candidate
                break
            selected_ranks[logical].remove(candidate)
            rank_remaining[candidate] += 1
            remaining_copies[logical] += 1

        if target is None:
            raise RuntimeError(
                "no load-ordered rank preserves residual placement "
                f"feasibility for logical expert {logical}"
            )

        physical_to_rank[physical] = target
        rank_lists[target].append(logical)
        rank_load[target] += physical_copy_load[physical]
        rank_count[target] += 1

    if not np.array_equal(rank_count, rank_capacity):
        raise RuntimeError(
            f"placement did not fill capacities: {rank_count} != {rank_capacity}"
        )
    if np.any(remaining_copies != 0) or np.any(rank_remaining != 0):
        raise RuntimeError("placement left a non-empty residual degree sequence")
    # Runtime enumerates a logical expert's copies in physical rank order.
    # Canonicalize the otherwise interchangeable planner copy ids to that order.
    for logical in range(logical_load.size):
        physical = np.flatnonzero(physical_to_logical == logical)
        physical_to_rank[physical] = np.sort(physical_to_rank[physical])
    placement = tuple(np.asarray(values, dtype=np.int64) for values in rank_lists)
    return placement, physical_to_rank


def build_layer_placement(
    aggregate_load: np.ndarray,
    replica_count_by_layer: Sequence[int] | np.ndarray,
    num_ranks: int,
    *,
    num_nodes: int = 1,
) -> tuple[
    np.ndarray,
    Placement,
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    np.ndarray,
]:
    """Build capacities and physical placement for a layer replica vector."""

    weights = np.asarray(aggregate_load, dtype=np.float64)
    replicas = _strict_integer_array(
        replica_count_by_layer,
        "replica_count_by_layer",
        ndim=1,
    )
    if weights.ndim != 2:
        raise ValueError(f"aggregate_load must have shape [L, E], got {weights.shape}")
    if replicas.shape != (weights.shape[0],):
        raise ValueError(
            f"replica vector must have shape {(weights.shape[0],)}, got {replicas.shape}"
        )
    if np.any(replicas < 0):
        raise ValueError("replica counts must be non-negative")

    num_layers, num_experts = weights.shape
    capacities = capacity_aware_interleaving(
        num_experts + replicas,
        num_ranks,
    )
    placements: list[tuple[np.ndarray, ...]] = []
    physical_to_logical: list[np.ndarray] = []
    physical_to_rank: list[np.ndarray] = []
    copy_counts = np.empty((num_layers, num_experts), dtype=np.int64)

    for layer in range(num_layers):
        phy2log, copies = _replicate_layer(
            weights[layer],
            int(replicas[layer]),
            num_ranks,
        )
        placement, phy2rank = _place_layer(
            phy2log,
            weights[layer],
            copies,
            capacities[layer],
            num_nodes,
        )
        placements.append(placement)
        physical_to_logical.append(phy2log)
        physical_to_rank.append(phy2rank)
        copy_counts[layer] = copies

    return (
        capacities,
        tuple(placements),
        tuple(physical_to_logical),
        tuple(physical_to_rank),
        copy_counts,
    )


def replay_balancedness(
    expert_load: np.ndarray | Sequence[object],
    physical_to_logical: Sequence[np.ndarray],
    physical_to_rank: Sequence[np.ndarray],
    logical_copy_count: np.ndarray,
    num_ranks: int,
) -> np.ndarray:
    """Replay source-rank modulo routing and return per-layer balancedness.

    The runtime sends all traffic from source rank ``s`` to copy
    ``s % copy_count``. The profile contains cluster-aggregate logical load,
    so each source rank contributes one ``1 / num_ranks`` share. This differs
    from equal ``load / copy_count`` splitting whenever the copy count does
    not divide the EP size, notably for three and five copies on eight ranks.
    """

    load = _normalise_load(expert_load)
    num_ranks = _strict_integer(num_ranks, "num_ranks", minimum=1)
    num_batches, num_layers, num_experts = load.shape
    if len(physical_to_logical) != num_layers or len(physical_to_rank) != num_layers:
        raise ValueError("placement layer count does not match expert_load")
    copy_table = _strict_integer_array(
        logical_copy_count,
        "logical_copy_count",
        ndim=2,
    )
    if copy_table.shape != (num_layers, num_experts):
        raise ValueError(
            "logical_copy_count must have shape "
            f"{(num_layers, num_experts)}, got {copy_table.shape}"
        )
    if np.any(copy_table < 1) or np.any(copy_table > num_ranks):
        raise ValueError("logical_copy_count entries must be in [1, num_ranks]")

    result = np.empty(num_layers, dtype=np.float64)
    for layer in range(num_layers):
        phy2log = _strict_integer_array(
            physical_to_logical[layer],
            f"physical_to_logical[{layer}]",
            ndim=1,
        )
        phy2rank = _strict_integer_array(
            physical_to_rank[layer],
            f"physical_to_rank[{layer}]",
            ndim=1,
        )
        if phy2log.shape != phy2rank.shape:
            raise ValueError(f"layer {layer} physical maps have different shapes")
        if np.any(phy2log < 0) or np.any(phy2log >= num_experts):
            raise ValueError(f"layer {layer} contains an invalid logical expert id")
        if np.any(phy2rank < 0) or np.any(phy2rank >= num_ranks):
            raise ValueError(f"layer {layer} contains an invalid rank id")

        actual_copy_count = np.bincount(phy2log, minlength=num_experts)
        if not np.array_equal(actual_copy_count, copy_table[layer]):
            raise ValueError(f"layer {layer} physical maps do not match logical_copy_count")
        gpu_load = np.zeros((num_batches, num_ranks), dtype=np.float64)
        for logical in range(num_experts):
            physical = np.flatnonzero(phy2log == logical)
            physical = physical[np.argsort(phy2rank[physical], kind="stable")]
            target_ranks = phy2rank[physical]
            if np.unique(target_ranks).size != target_ranks.size:
                raise ValueError(
                    f"layer {layer} logical expert {logical} has two copies on one rank"
                )
            source_counts = _source_ranks_per_copy(physical.size, num_ranks)
            for copy_index, rank in enumerate(target_ranks.tolist()):
                gpu_load[:, rank] += (
                    load[:, layer, logical]
                    * float(source_counts[copy_index])
                    / num_ranks
                )

        maximum = gpu_load.max(axis=1)
        balancedness = np.ones(num_batches, dtype=np.float64)
        nonzero = maximum > 0
        balancedness[nonzero] = (
            gpu_load[nonzero].mean(axis=1) / maximum[nonzero]
        )
        result[layer] = balancedness.mean()
    return result
