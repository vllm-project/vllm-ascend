# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.eplb.core.allgather import build_allgather_routing, count_local_routes, route_allgather


def planner_map(placement):
    placement = torch.tensor(placement)
    result = torch.full((placement.shape[0], int(placement.max()) + 1), -1, dtype=torch.int32)
    for rank, experts in enumerate(placement):
        result[rank, experts] = torch.arange(len(experts), dtype=torch.int32)
    return result


@pytest.mark.parametrize(
    "placement",
    [
        [[0, 1, 2], [3, 0, 1]],
        [[0, 1, 3], [2, 0, 1]],
        [[0, 1, 3], [2, 3, 1]],
        [[2, 1, 3], [0, 3, 2]],
        [[0, 1], [2, 3]],
        [[0, 1], [2, 0], [1, 2]],
    ],
)
@pytest.mark.parametrize("num_tokens", [0, 1, 2, 17, 1025])
def test_routes_match_weight_slots_exactly_once(placement, num_tokens):
    mapping = planner_map(placement)
    slots, counts = build_allgather_routing(mapping)
    logical = torch.arange(num_tokens * 2).reshape(num_tokens, 2).remainder(mapping.shape[1]).int()
    physical = route_allgather(logical, slots, counts)
    flattened = torch.tensor(placement).flatten()
    assert torch.equal(flattened[physical.long()], logical.long())
    multiplicity = torch.zeros_like(logical)
    local_size = len(placement[0])
    for rank in range(len(placement)):
        independently_routed = route_allgather(logical, *build_allgather_routing(mapping.clone()))
        multiplicity += (
            (independently_routed >= rank * local_size) & (independently_routed < (rank + 1) * local_size)
        ).int()
    assert torch.all(multiplicity == 1)


@pytest.mark.parametrize("hot_expert", [0, 1])
def test_replicas_really_share_tokens(hot_expert):
    slots, counts = build_allgather_routing(planner_map([[0, 1, 2], [3, 0, 1]]))
    physical = route_allgather(torch.full((32, 1), hot_expert, dtype=torch.int32), slots, counts)
    valid = torch.ones(32, dtype=torch.bool)
    loads = torch.stack([count_local_routes(physical, valid, rank, 3) for rank in range(2)])
    assert loads.sum(1).tolist() == [16, 16]
    assert int(loads.sum()) == 32


def test_padding_and_empty_batch_do_not_add_heat():
    slots, counts = build_allgather_routing(planner_map([[0, 1, 2], [3, 0, 1]]))
    logical = torch.tensor([[0, 2], [1, 3], [0, 3], [1, 2]], dtype=torch.int32)
    physical = route_allgather(logical, slots, counts)
    valid = torch.tensor([True, False, True, False])
    loads = torch.cat([count_local_routes(physical, valid, rank, 3) for rank in range(2)])
    reference = torch.bincount(physical[valid].long().flatten(), minlength=6)
    assert torch.equal(loads, reference)
    assert int(loads.sum()) == 4
    for rank in range(2):
        assert count_local_routes(physical[:0], valid[:0], rank, 3).tolist() == [0, 0, 0]


def test_multilayer_tables_update_in_place_and_can_change_replicated_expert():
    layouts = [[[0, 1, 2], [3, 0, 1]], [[2, 1, 3], [0, 3, 2]]]
    tables = [build_allgather_routing(planner_map(x)) for x in layouts]
    ptrs = [(slots.data_ptr(), counts.data_ptr()) for slots, counts in tables]
    layouts.reverse()
    for layer, layout in enumerate(layouts):
        new_tables = build_allgather_routing(planner_map(layout))
        for target, source in zip(tables[layer], new_tables):
            target.copy_(source)
        assert tuple(x.data_ptr() for x in tables[layer]) == ptrs[layer]
        logical = torch.arange(32).remainder(4).reshape(-1, 1).int()
        physical = route_allgather(logical, *tables[layer])
        assert torch.equal(torch.tensor(layout).flatten()[physical.long()], logical.long())


@pytest.mark.parametrize(
    "mapping",
    [
        [[0, -1], [0, -1]],
        [[0, 1], [-1, 0]],
    ],
)
def test_invalid_layout_is_rejected(mapping):
    with pytest.raises(ValueError):
        build_allgather_routing(torch.tensor(mapping))
