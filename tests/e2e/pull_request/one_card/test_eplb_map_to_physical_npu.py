# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.ops.fused_moe.eplb import map_to_physical_for_ascend, record_expert_tokens


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 32, 39, 40, 41, 64, 65, 128, 256, 512, 2559, 2560, 2561, 65536])
@pytest.mark.parametrize("experts,topk", [(128, 8), (896, 16)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_map_balanced_token_ranges(tokens, experts, topk, dtype):
    rows = 7
    table = (torch.arange(rows * experts).reshape(rows, experts) * 3 + 5).to(torch.int32)
    ids = (torch.arange(tokens * topk).reshape(tokens, topk) % experts).to(dtype)
    ids[0, 0] = -1
    ids[-1, -1] = experts if dtype == torch.int32 else 2**32
    valid = (ids >= 0) & (ids < experts)
    expected = table[torch.arange(tokens)[:, None] % rows, ids.clamp(0, experts - 1).long()].to(dtype)
    expected[~valid] = -1

    mapped = map_to_physical_for_ascend(ids.npu(), table.npu())
    torch.testing.assert_close(mapped.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("group_list_type", [0, 1])
def test_map_and_record_mutable_state_graph_replay(group_list_type):
    ids = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.int64, device="npu")
    table = torch.tensor([[2, 1], [0, 3]], dtype=torch.int32, device="npu")
    counts = torch.tensor([2, 3, 5, 7], dtype=torch.int64)
    encoded = (counts.cumsum(0) if group_list_type == 0 else counts).npu()
    load = torch.arange(8, dtype=torch.int32, device="npu")
    enabled = torch.tensor(True, device="npu")
    for _ in range(3):
        map_to_physical_for_ascend(ids, table)
        record_expert_tokens(encoded, load, enabled, group_list_type, 2)
    torch.npu.synchronize()
    initial = load.cpu()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        mapped = map_to_physical_for_ascend(ids, table)
        record_expert_tokens(encoded, load, enabled, group_list_type, 2)
    load.copy_(initial)
    table.add_(11)
    graph.replay()
    expected = initial.clone()
    expected[2:6] += counts.to(expected.dtype)
    torch.testing.assert_close(load.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        mapped.cpu(), torch.tensor([[12, 13], [11, 14], [12, 13]], dtype=torch.int64), rtol=0, atol=0
    )

    enabled.fill_(False)
    graph.replay()
    torch.testing.assert_close(load.cpu(), expected, rtol=0, atol=0)


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


@pytest.mark.parametrize("count", [255, 256, 257, 896])
@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("count_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("load_dtype", [torch.int32, torch.int64])
def test_record_expert_blocks(count, group_list_type, count_dtype, load_dtype):
    counts = torch.arange(count, dtype=torch.int64) * 7 + 3 + 2**28
    encoded = (counts.cumsum(0) if group_list_type == 0 else counts).to(count_dtype)
    decoded = encoded if group_list_type == 1 else torch.diff(encoded, prepend=encoded.new_zeros(1))
    start = 3
    initial = torch.arange(count + 2 * start, dtype=load_dtype) + 2**30
    expected = initial.clone()
    expected[start : start + count] += decoded.to(load_dtype)
    load = initial.npu()
    tokens = encoded.npu()
    enabled = torch.tensor(True, device="npu")

    record_expert_tokens(tokens, load, enabled, group_list_type, start)
    torch.testing.assert_close(load.cpu(), expected, rtol=0, atol=0)
    enabled.fill_(False)
    record_expert_tokens(tokens, load, enabled, group_list_type, start)
    torch.testing.assert_close(load.cpu(), expected, rtol=0, atol=0)
