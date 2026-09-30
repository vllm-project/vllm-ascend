# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.fused_moe import eplb as _eplb_ops  # noqa: F401
from vllm_ascend.ops.triton.eplb_map_record import eplb_map_and_record
from vllm_ascend.utils import enable_custom_op


def _case(tokens: int, logical_experts: int, top_k: int, physical_experts: int, table_rows: int = 7):
    assignments = torch.arange(tokens * top_k, device="npu", dtype=torch.int32).reshape(tokens, top_k)
    logical_ids = ((assignments * 13 + 3) % logical_experts).contiguous()
    rows = torch.arange(table_rows, device="npu", dtype=torch.int32)[:, None]
    columns = torch.arange(logical_experts, device="npu", dtype=torch.int32)[None, :]
    table = ((columns * 17 + rows * 5 + 1) % physical_experts).contiguous()
    return logical_ids, table


def _expected_load(
    mapped_ids: torch.Tensor,
    initial_load: torch.Tensor,
    valid_tokens: int,
    local_start: int,
    local_count: int,
) -> torch.Tensor:
    expected = initial_load.cpu().clone()
    for physical_id in mapped_ids[:valid_tokens].cpu().flatten().tolist():
        if local_start <= physical_id < local_start + local_count:
            expected[physical_id] += 1
    return expected.to(initial_load.device)


@pytest.mark.parametrize(
    "tokens,logical_experts,top_k,physical_experts,local_start,local_count,valid_tokens",
    [
        (1, 8, 6, 8, 0, 8, 0),
        (32, 16, 8, 16, 0, 16, 31),
        (64, 32, 8, 32, 0, 32, 64),
        (65, 128, 8, 128, 64, 64, 37),
        (8, 896, 16, 896, 0, 112, 1),
        (128, 896, 16, 896, 784, 112, 111),
        (128, 896, 16, 896, 0, 112, 127),
        (256, 896, 16, 896, 0, 112, 256),
        (512, 32, 6, 40, 8, 24, 511),
        (65536, 32, 8, 32, 0, 32, 65531),
    ],
)
@pytest.mark.parametrize("record_on", [False, True])
def test_mapping_and_valid_record_exact(
    tokens, logical_experts, top_k, physical_experts, local_start, local_count, valid_tokens, record_on
):
    enable_custom_op()
    logical_ids, table = _case(tokens, logical_experts, top_k, physical_experts)
    initial_load = torch.arange(physical_experts, device="npu", dtype=torch.int32)
    load = initial_load.clone()
    enabled = torch.tensor(record_on, device="npu")
    expected_ids = torch.ops.vllm.ascend_eplb_map_to_physical(logical_ids, table)

    actual_ids = eplb_map_and_record(
        logical_ids,
        table,
        load,
        enabled,
        torch.tensor(valid_tokens, device="npu", dtype=torch.int32),
        local_expert_start=local_start,
        local_expert_count=local_count,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    expected_load = (
        _expected_load(expected_ids, initial_load, valid_tokens, local_start, local_count)
        if record_on
        else initial_load
    )
    torch.testing.assert_close(load, expected_load, rtol=0, atol=0)


def test_graph_replay_uses_runtime_table_flag_and_valid_count():
    enable_custom_op()
    tokens, logical_experts, top_k, physical_experts = 65, 128, 8, 192
    logical_ids, table = _case(tokens, logical_experts, top_k, physical_experts)
    initial_load = torch.arange(physical_experts, device="npu", dtype=torch.int32)
    load = initial_load.clone()
    enabled = torch.tensor(False, device="npu")
    valid_tokens = torch.tensor(0, device="npu", dtype=torch.int32)
    eplb_map_and_record(logical_ids, table, load, enabled, valid_tokens, local_expert_start=64, local_expert_count=64)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        physical_ids = eplb_map_and_record(
            logical_ids, table, load, enabled, valid_tokens, local_expert_start=64, local_expert_count=64
        )

    expected_load = initial_load.clone()
    for shift, record_on, valid in ((1, True, 0), (7, True, 32), (2, False, 65), (11, True, 65)):
        rows = torch.arange(table.shape[0], device="npu", dtype=torch.int32)[:, None]
        columns = torch.arange(logical_experts, device="npu", dtype=torch.int32)[None, :]
        table.copy_((columns * 17 + rows * 5 + shift) % physical_experts)
        enabled.fill_(record_on)
        valid_tokens.fill_(valid)
        graph.replay()
        torch.npu.synchronize()
        expected_ids = torch.ops.vllm.ascend_eplb_map_to_physical(logical_ids, table)
        torch.testing.assert_close(physical_ids, expected_ids, rtol=0, atol=0)
        if record_on:
            expected_load = _expected_load(expected_ids, expected_load, valid, 64, 64)
        torch.testing.assert_close(load, expected_load, rtol=0, atol=0)


def test_resource_guard_is_local_physical_not_logical_expert_count():
    logical_ids, table = _case(1, 896, 16, 896)
    load = torch.zeros(9000, device="npu", dtype=torch.int32)
    enabled = torch.tensor(True, device="npu")
    with pytest.raises(ValueError, match="comparison resource budget"):
        eplb_map_and_record(logical_ids, table, load, enabled, 1, local_expert_start=0, local_expert_count=8193)


def test_int64_ids_and_invalid_padding_ids_match_mainline_mapping():
    enable_custom_op()
    logical_ids = torch.tensor([[0, 1], [-1, 2**31]], device="npu", dtype=torch.int64)
    table = torch.tensor([[1, 0]], device="npu", dtype=torch.int32)
    load = torch.zeros(2, device="npu", dtype=torch.int32)
    enabled = torch.tensor(True, device="npu")
    actual = eplb_map_and_record(logical_ids, table, load, enabled, 1, local_expert_count=2)
    expected = torch.ops.vllm.ascend_eplb_map_to_physical(logical_ids, table)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(load, torch.ones_like(load), rtol=0, atol=0)


def test_noncontiguous_logical_ids_use_the_same_mapping():
    enable_custom_op()
    full_ids, table = _case(32, 16, 8, 16)
    logical_ids = full_ids[:, ::2]
    assert not logical_ids.is_contiguous()
    load = torch.zeros(16, device="npu", dtype=torch.int32)
    enabled = torch.tensor(True, device="npu")
    actual = eplb_map_and_record(logical_ids, table, load, enabled, 31, local_expert_count=16)
    expected = torch.ops.vllm.ascend_eplb_map_to_physical(logical_ids.contiguous(), table)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(load, _expected_load(expected, torch.zeros_like(load), 31, 0, 16), rtol=0, atol=0)


def test_empty_routing_does_not_change_load():
    logical_ids = torch.empty((0, 8), device="npu", dtype=torch.int32)
    table = torch.arange(16, device="npu", dtype=torch.int32).reshape(1, 16)
    load = torch.arange(16, device="npu", dtype=torch.int32)
    initial = load.clone()
    enabled = torch.tensor(True, device="npu")
    actual = eplb_map_and_record(logical_ids, table, load, enabled, 0, local_expert_count=16)
    assert actual.shape == logical_ids.shape
    torch.testing.assert_close(load, initial, rtol=0, atol=0)
