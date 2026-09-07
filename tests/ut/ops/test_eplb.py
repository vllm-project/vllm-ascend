# SPDX-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

from vllm_ascend.ops.fused_moe.eplb import (
    EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS,
    build_expert_replica_routing_table,
    map_to_physical,
    map_to_physical_and_record,
)


def _eplb_inputs():
    logical_to_physical_map = torch.tensor(
        [[0, 4, -1], [1, 5, -1], [2, -1, -1]],
        dtype=torch.int64,
    )
    logical_replica_count = torch.tensor([2, 2, 1], dtype=torch.int64)
    return logical_to_physical_map, logical_replica_count


def test_build_expert_replica_routing_table_applies_rank_and_expert_offsets():
    logical_map, replica_count = _eplb_inputs()

    rank0_routing_table = build_expert_replica_routing_table(
        logical_map,
        replica_count,
        ep_rank=0,
    )
    rank1_routing_table = build_expert_replica_routing_table(
        logical_map,
        replica_count,
        ep_rank=1,
    )

    assert rank0_routing_table.shape == (EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS, 3)
    assert rank0_routing_table.dtype == torch.int32
    torch.testing.assert_close(
        rank0_routing_table[0],
        torch.tensor([0, 5, 2], dtype=torch.int32),
    )
    torch.testing.assert_close(
        rank1_routing_table[0],
        torch.tensor([4, 1, 2], dtype=torch.int32),
    )
    torch.testing.assert_close(rank0_routing_table[1], rank1_routing_table[0])


def test_map_to_physical_uses_periodic_rows():
    logical_map, replica_count = _eplb_inputs()
    routing_table = build_expert_replica_routing_table(
        logical_map,
        replica_count,
        ep_rank=0,
    )
    topk_ids = torch.zeros(
        (EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS + 1, 2),
        dtype=torch.int64,
    )
    topk_ids[:, 1] = 1

    physical_ids = map_to_physical(topk_ids, routing_table)

    assert physical_ids[0, 0] == 0
    assert physical_ids[EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS - 1, 0] == 4
    assert physical_ids[EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS, 0] == physical_ids[0, 0]
    assert physical_ids[EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS, 1] == physical_ids[0, 1]


def test_map_to_physical_and_record_gates_load_collection():
    routing_table = torch.tensor(
        [[0, 3], [2, 1], [0, 3], [2, 1]],
        dtype=torch.int32,
    )
    topk_ids = torch.tensor(
        [[0, 1], [0, 1], [0, 1], [0, 1]],
        dtype=torch.int32,
    )
    expert_load = torch.zeros(4, dtype=torch.int32)

    physical_ids = map_to_physical_and_record(
        topk_ids,
        routing_table,
        expert_load,
        record_enabled=torch.tensor(True),
        num_unpadded_tokens=torch.tensor(3, dtype=torch.int32),
    )

    torch.testing.assert_close(
        physical_ids,
        torch.tensor([[0, 3], [2, 1], [0, 3], [2, 1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        expert_load,
        torch.tensor([2, 1, 1, 2], dtype=torch.int32),
    )

    map_to_physical_and_record(
        topk_ids,
        routing_table,
        expert_load,
        record_enabled=torch.tensor(False),
        num_unpadded_tokens=torch.tensor(4, dtype=torch.int32),
    )
    torch.testing.assert_close(
        expert_load,
        torch.tensor([2, 1, 1, 2], dtype=torch.int32),
    )
