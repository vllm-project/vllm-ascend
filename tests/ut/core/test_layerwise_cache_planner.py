# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import pytest

from vllm_ascend.core.layerwise_cache_plan import BackingPlan, CacheOwnerDescription, unique_backings
from vllm_ascend.core.layerwise_cache_planner import assign_rotating_slots, normalize_layerwise_policy, solve_capacity


def test_policy_normalizes_stage_indices_once():
    policy = normalize_layerwise_policy([12, 10, 11], {"layerwise_independent_layers": [0, -1, 0]})
    assert policy.independent_layers == (10, 12)
    assert normalize_layerwise_policy([0, 1], {"layerwise_independent_layers": "all"}).independent_layers == (0, 1)


@pytest.mark.parametrize("value", [True, False, 0, -1])
def test_invalid_shared_buffers(value):
    with pytest.raises((TypeError, ValueError)):
        normalize_layerwise_policy([0, 1], {"layerwise_num_shared_buffers": value})


@pytest.mark.parametrize("indices", [[True], [2], [-3], "bad"])
def test_invalid_independent_layers(indices):
    with pytest.raises((TypeError, ValueError)):
        normalize_layerwise_policy([0, 1], {"layerwise_independent_layers": indices})


def test_linked_slots_are_stable_and_cross_group_boundaries():
    policy = normalize_layerwise_policy(
        range(6), {"layerwise_num_shared_buffers": 2, "layerwise_independent_layers": []}
    )
    owners = [
        CacheOwnerDescription(f"{layer}.{component}", layer, layer // 3, False, "attention", str(layer))
        for layer in range(6)
        for component in ("main", "indexer")
    ]
    expected = {owner.name: owner.physical_layer % 2 for owner in owners}
    assert assign_rotating_slots(owners, policy) == expected
    assert assign_rotating_slots(reversed(owners), policy) == expected


def test_exact_nonlinear_capacity_and_override():
    cost = lambda n: 37 + ((n * 19 + 63) // 64) * 64 + n * 7
    for budget in (108, 500, 1024, 4096):
        capacity = solve_capacity(cost, budget)
        assert cost(capacity) <= budget < cost(capacity + 1)
    assert solve_capacity(cost, 500, 1) == 1
    with pytest.raises(ValueError, match="needs .* available"):
        solve_capacity(cost, 500, 100)
    with pytest.raises(ValueError, match="one layerwise block"):
        solve_capacity(cost, 0)


def test_backings_are_deduplicated_by_identity_not_size():
    first = BackingPlan("first", 128, 16, 120)
    second = replace(first, backing_id="second")
    assert sum(b.size_bytes for b in unique_backings((first, second, first)).values()) == 256
    with pytest.raises(ValueError, match="Conflicting"):
        unique_backings((first, replace(first, size_bytes=256)))
