# SPDX-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

from vllm_ascend.ops.fused_moe.eplb import (
    EPLB_LOOKUP_NUM_ROWS,
    build_physical_id_lookup,
    map_to_physical,
    record_local_expert_load,
)


def _eplb_inputs():
    logical_to_physical_map = torch.tensor(
        [[0, 4, -1], [1, 5, -1], [2, -1, -1]],
        dtype=torch.int64,
    )
    logical_replica_count = torch.tensor([2, 2, 1], dtype=torch.int64)
    return logical_to_physical_map, logical_replica_count


def test_build_physical_id_lookup_applies_rank_and_expert_offsets():
    logical_map, replica_count = _eplb_inputs()

    rank0_lookup = build_physical_id_lookup(logical_map, replica_count, ep_rank=0)
    rank1_lookup = build_physical_id_lookup(logical_map, replica_count, ep_rank=1)

    assert rank0_lookup.shape == (EPLB_LOOKUP_NUM_ROWS, 3)
    assert rank0_lookup.dtype == torch.int32
    torch.testing.assert_close(rank0_lookup[0], torch.tensor([0, 5, 2], dtype=torch.int32))
    torch.testing.assert_close(rank1_lookup[0], torch.tensor([4, 1, 2], dtype=torch.int32))
    torch.testing.assert_close(rank0_lookup[1], rank1_lookup[0])


def test_map_to_physical_uses_periodic_rows():
    logical_map, replica_count = _eplb_inputs()
    lookup = build_physical_id_lookup(logical_map, replica_count, ep_rank=0)
    topk_ids = torch.zeros((EPLB_LOOKUP_NUM_ROWS + 1, 2), dtype=torch.int64)
    topk_ids[:, 1] = 1

    physical_ids = map_to_physical(topk_ids, lookup)

    assert physical_ids[0, 0] == 0
    assert physical_ids[EPLB_LOOKUP_NUM_ROWS - 1, 0] == 4
    assert physical_ids[EPLB_LOOKUP_NUM_ROWS, 0] == physical_ids[0, 0]
    assert physical_ids[EPLB_LOOKUP_NUM_ROWS, 1] == physical_ids[0, 1]


def test_record_local_expert_load_updates_only_current_rank_slice():
    expert_load = torch.zeros(6, dtype=torch.int32)

    record_local_expert_load(
        expert_tokens=torch.tensor([3, 5], dtype=torch.int64),
        group_list_type=1,
        expert_load_view=expert_load,
        ep_rank=1,
        ep_size=3,
    )

    torch.testing.assert_close(expert_load, torch.tensor([0, 0, 3, 5, 0, 0], dtype=torch.int32))


def test_record_local_expert_load_converts_cumulative_group_list():
    expert_load = torch.zeros(4, dtype=torch.int32)

    record_local_expert_load(
        expert_tokens=torch.tensor([2, 7], dtype=torch.int64),
        group_list_type=0,
        expert_load_view=expert_load,
        ep_rank=0,
        ep_size=2,
    )

    torch.testing.assert_close(expert_load, torch.tensor([2, 5, 0, 0], dtype=torch.int32))
