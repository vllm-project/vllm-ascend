#
# Copyright (c) 2025 Huawei Technologies Co, Ltd. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# CPU unit tests for the ALLGATHER token-row replica routing table (A2 V1
# path, EPLB_ALLGATHER_TOKEN_ROUTE). The exactly-once simulation mirrors
# tests/ut/ops/test_allgather_eplb.py but drives the V1 payload path:
# placement -> generate_logical_replica_map -> [1024, N] table ->
# map_to_physical -> per-rank flat-window ownership.
from types import SimpleNamespace

import torch

from vllm_ascend.distributed.eplb.tensor_ops import build_expert_replica_routing_table
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_utils import (
    generate_global_placement,
    generate_log2phy_map,
    generate_logical_replica_map,
)
from vllm_ascend.ops.fused_moe.eplb import map_to_physical

ROUTING_TABLE_NUM_ROWS = 1024


def _global_expert_map_from_placement(placement: torch.Tensor):
    """Mirror init_eplb_config: full-length per-rank maps + phys_to_logical."""
    phys_to_logical = placement.reshape(-1).contiguous()
    ep_size, num_local = placement.shape
    maps = []
    phys_offset = 0
    for rankid in range(ep_size):
        expert_map = torch.full((phys_to_logical.numel(),), -1, dtype=torch.int32)
        local_slots = torch.arange(num_local, dtype=torch.int32)
        expert_map[phys_offset + local_slots] = local_slots
        phys_offset += num_local
        maps.append(expert_map)
    return torch.stack(maps), phys_to_logical


def _build_table(placement: torch.Tensor) -> tuple[torch.Tensor, list, list]:
    gem, p2l = _global_expert_map_from_placement(placement)
    logical_to_physical, replica_count = generate_logical_replica_map(gem, p2l)
    table = build_expert_replica_routing_table(
        torch.tensor(logical_to_physical, dtype=torch.int64),
        torch.tensor(replica_count, dtype=torch.int64),
        ep_rank=0,
    )
    return table, logical_to_physical, replica_count


# Four ranks, three slots each; logical expert 1 has copies on rank 0 (flat 1)
# and rank 3 (flat 9), so per-rank rotations disagree — the same fixture as
# tests/ut/ops/test_allgather_eplb.py.
PLACEMENT = torch.cat([torch.arange(8), torch.arange(4)]).reshape(4, 3)


def test_replica_map_lists_every_copy_in_rank_slot_order():
    gem, p2l = _global_expert_map_from_placement(PLACEMENT)
    logical_to_physical, replica_count = generate_logical_replica_map(gem, p2l)

    # The redundant copies are logical experts 0-3: expert 0 lives at flat 0
    # (rank 0) and flat 8 (rank 2); expert 1 at flat 1 (rank 0) and flat 9
    # (rank 3); experts 4-7 are single-copy.
    assert replica_count == [2, 2, 2, 2, 1, 1, 1, 1]
    # flat id = local slot + rank * num_local(3)
    assert logical_to_physical[0][:2] == [0, 8]
    assert logical_to_physical[1][:2] == [1, 9]
    assert logical_to_physical[2][:2] == [2, 10]
    assert logical_to_physical[3][:2] == [3, 11]
    # Single-replica experts are padded with their own (only) copy.
    assert logical_to_physical[4] == [4, 4]
    assert logical_to_physical[7] == [7, 7]


def test_token_route_table_row_matches_rank_independent_log2phy():
    """Row r selects replica (r + logical) % count; row 0 == key % count."""
    gem, p2l = _global_expert_map_from_placement(PLACEMENT)
    table, logical_to_physical, replica_count = _build_table(PLACEMENT)

    log2phy_rank_independent = generate_log2phy_map(gem, 0, phys_to_logical=p2l, rank_independent_replica=True)

    for logical in range(8):
        reps = logical_to_physical[logical][: replica_count[logical]]
        for row in (0, 1, 2, 1023):
            expected = reps[(row + logical) % replica_count[logical]]
            assert table[row, logical].item() == expected, (row, logical)
        # Row 0 is exactly the current fixed choice for every logical expert.
        assert table[0, logical].item() == log2phy_rank_independent[logical].item()
        # Single-replica experts are constant across rows.
        if replica_count[logical] == 1:
            assert table[:, logical].eq(reps[0]).all()


def test_allgather_token_rows_execute_and_count_each_token_once():
    table, logical_to_physical, replica_count = _build_table(PLACEMENT)

    num_rows = 16
    logical_ids = torch.arange(8).expand(num_rows, -1)
    weights = torch.arange(1, 9, dtype=torch.float32).expand(num_rows, -1) / 36
    hidden = torch.arange(1, num_rows + 1, dtype=torch.float32)

    physical = map_to_physical(logical_ids, table)

    owners = torch.zeros_like(logical_ids)
    output = torch.zeros_like(hidden)
    for rank in range(4):
        local = (physical >= rank * 3) & (physical < (rank + 1) * 3)
        owners.add_(local)
        # Weight encodes the logical expert, so it is identical for both
        # replicas; the reference is therefore replica-choice independent and
        # equality means no route was dropped or double-counted.
        logical_weights = PLACEMENT.flatten()[physical.long()].float() + 1
        output += (hidden[:, None] * logical_weights * weights * local).sum(dim=1)

    reference = hidden * ((torch.arange(8).float() + 1) * weights[0]).sum()
    torch.testing.assert_close(owners, torch.ones_like(owners))
    torch.testing.assert_close(output, reference)

    # The point of token-row routing: both copies of the duplicated expert 1
    # receive a real share of the token stream (here an exact 50/50 split),
    # instead of one copy taking everything.
    copies = logical_to_physical[1][: replica_count[1]]
    expert1_routes = physical[:, 1]
    for copy in copies:
        assert expert1_routes.eq(copy).sum().item() > 0
    assert expert1_routes.eq(copies[0]).sum().item() == num_rows // 2
    assert expert1_routes.eq(copies[1]).sum().item() == num_rows // 2

    # Contrast: the fixed rank-independent choice sends every row to one copy.
    gem, p2l = _global_expert_map_from_placement(PLACEMENT)
    fixed = generate_log2phy_map(gem, 0, phys_to_logical=p2l, rank_independent_replica=True)
    fixed_physical = fixed[logical_ids]
    assert fixed_physical[:, 1].eq(fixed_physical[0, 1]).all()


def test_map_to_physical_matches_manual_row_gather():
    table, _, _ = _build_table(PLACEMENT)
    num_rows, topk = 2050, 8  # spans multiple 1024-row cycles plus a tail
    logical_ids = (torch.arange(num_rows * topk).view(num_rows, topk) % 8).to(torch.int32)

    physical = map_to_physical(logical_ids, table)

    rows = torch.arange(num_rows) % ROUTING_TABLE_NUM_ROWS
    manual = torch.gather(table[rows], 1, logical_ids.long())
    torch.testing.assert_close(physical.long(), manual.long())
    assert physical.dtype == logical_ids.dtype


def test_dsv4_flash_default_layout_replica_lists():
    """Anchor the real DSV4-Flash layout: 256 logical, 8 ranks, 16 redundant."""
    placement = generate_global_placement(256, 8, 16, num_shared_experts=0)
    gem, p2l = _global_expert_map_from_placement(placement)
    logical_to_physical, replica_count = generate_logical_replica_map(gem, p2l)

    assert replica_count[:16] == [2] * 16
    assert replica_count[16:] == [1] * 240
    for i in range(16):
        # flat i (primary) and flat 256+i (redundant copy on rank 7).
        assert logical_to_physical[i][:2] == [i, 256 + i]
    for i in range(16, 256):
        assert logical_to_physical[i][0] == i


def test_do_update_replica_routing_table_refreshes_in_place():
    table, _, _ = _build_table(PLACEMENT)
    storage_before = table.data_ptr()
    layer = SimpleNamespace(allgather_replica_routing_table=table)
    adaptor = SimpleNamespace(moe_layers=[layer])

    # A different valid placement: copies of experts 4-7 at flats 8-11.
    new_placement = torch.cat([torch.arange(8), torch.tensor([4, 5, 6, 7])]).reshape(4, 3)
    _, logical_to_physical, replica_count = _build_table(new_placement)

    VllmEplbAdaptor.do_update_replica_routing_table(
        adaptor,
        0,
        torch.tensor(logical_to_physical, dtype=torch.int64),
        torch.tensor(replica_count, dtype=torch.int64),
    )

    # Storage identity is preserved (captured ACL graphs keep reading the
    # same buffer) and the content matches a fresh build.
    assert layer.allgather_replica_routing_table.data_ptr() == storage_before
    expected, _, _ = _build_table(new_placement)
    torch.testing.assert_close(layer.allgather_replica_routing_table, expected)


def test_do_update_replica_routing_table_noop_when_disabled():
    layer = SimpleNamespace()  # no allgather_replica_routing_table attribute
    adaptor = SimpleNamespace(moe_layers=[layer])
    # Must not raise; the worker ships the payload unconditionally.
    VllmEplbAdaptor.do_update_replica_routing_table(
        adaptor, 0, torch.zeros((8, 1), dtype=torch.int64), torch.ones(8, dtype=torch.int64)
    )
