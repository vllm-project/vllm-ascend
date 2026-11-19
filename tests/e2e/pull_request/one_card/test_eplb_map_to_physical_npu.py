# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.distributed.eplb import eplb_state
from vllm_ascend.ops.fused_moe.eplb import map_to_physical_for_ascend, record_expert_tokens


def test_async_mapping_refresh_on_npu_does_not_compare_device_maps():
    physical = torch.tensor([[1, 0]], dtype=torch.int32, device="npu")
    snapshot = torch.tensor([[0, 1]], dtype=torch.int32, device="npu")
    model_state = SimpleNamespace(
        model=SimpleNamespace(moe_layers=[SimpleNamespace(eplb_state=None)]),
        physical_to_logical_map=physical,
        _allgather_physical_map=snapshot,
        expert_load_pass=torch.ones(1, 2, dtype=torch.int32, device="npu"),
        expert_load_window=torch.ones(2, 1, 2, dtype=torch.int32, device="npu"),
        _load_mapping_generation=0,
    )
    with patch.object(torch, "equal", side_effect=AssertionError("device map comparison synchronizes")):
        eplb_state.refresh_model_routing_tables(model_state, layer_idx=0)
    torch.npu.synchronize()

    assert model_state._allgather_physical_map is snapshot
    torch.testing.assert_close(snapshot.cpu(), physical.cpu())
    assert not model_state.expert_load_pass.cpu().any()
    assert not model_state.expert_load_window.cpu().any()
    assert model_state._load_mapping_generation == 1


def test_map_and_operator_counts_run_on_npu_without_host_gating():
    routing_table = torch.tensor(
        [[0, 3], [2, 1], [0, 3], [2, 1]],
        dtype=torch.int32,
        device="npu",
    )
    topk_ids = torch.tensor(
        [[0, 1], [0, 1], [0, 1], [0, 1]],
        dtype=torch.int32,
        device="npu",
    )
    expert_load = torch.zeros(4, dtype=torch.int32, device="npu")
    record_enabled = torch.tensor(True, device="npu")
    expert_tokens = torch.tensor([2, 1, 1, 2], dtype=torch.int32, device="npu")

    physical_ids = map_to_physical_for_ascend(topk_ids, routing_table)
    record_expert_tokens(expert_tokens, expert_load, record_enabled, 1, 0)
    torch.npu.synchronize()

    torch.testing.assert_close(
        physical_ids.cpu(),
        torch.tensor([[0, 3], [2, 1], [0, 3], [2, 1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        expert_load.cpu(),
        torch.tensor([2, 1, 1, 2], dtype=torch.int32),
    )

    record_enabled.fill_(False)
    record_expert_tokens(expert_tokens, expert_load, record_enabled, 1, 0)
    torch.npu.synchronize()
    torch.testing.assert_close(
        expert_load.cpu(),
        torch.tensor([2, 1, 1, 2], dtype=torch.int32),
    )
