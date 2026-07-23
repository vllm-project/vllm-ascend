# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

from vllm_ascend.ops.fused_moe.eplb import map_and_record


def _eplb_inputs():
    topk_ids = torch.tensor([[0, 1], [0, 0], [-1, 1]], dtype=torch.int32)
    logical_to_physical_map = torch.tensor([[0, 2], [1, -1]], dtype=torch.int32)
    logical_replica_count = torch.tensor([2, 1], dtype=torch.int32)
    expert_load = torch.zeros(3, dtype=torch.int64)
    return topk_ids, logical_to_physical_map, logical_replica_count, expert_load


def test_map_and_record_matches_knuth_replica_selection():
    topk_ids, logical_map, replica_count, expert_load = _eplb_inputs()

    physical_ids = map_and_record(
        topk_ids,
        logical_map,
        replica_count,
        expert_load,
        torch.tensor(True),
        torch.tensor(2, dtype=torch.int32),
    )

    torch.testing.assert_close(
        physical_ids,
        torch.tensor([[0, 1], [2, 2], [-1, 1]], dtype=torch.int32),
    )
    torch.testing.assert_close(expert_load, torch.tensor([1, 1, 2], dtype=torch.int64))


def test_map_and_record_honors_record_switch():
    topk_ids, logical_map, replica_count, expert_load = _eplb_inputs()

    map_and_record(
        topk_ids,
        logical_map,
        replica_count,
        expert_load,
        torch.tensor(False),
    )

    torch.testing.assert_close(expert_load, torch.zeros_like(expert_load))
