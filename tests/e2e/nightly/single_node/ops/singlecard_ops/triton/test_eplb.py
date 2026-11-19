# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.distributed.eplb.tensor_ops import EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS
from vllm_ascend.ops.fused_moe.eplb import map_to_physical_for_ascend, record_expert_tokens


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "tokens,experts,topk",
    [
        # Observed Qwen3-30B-A3B decode and prefill.
        (1, 128, 8),
        (32, 128, 8),
        (33, 128, 8),
        # Tile transitions and a routing-table wrap.
        (255, 128, 8),
        (256, 128, 8),
        (257, 128, 8),
        (1023, 128, 8),
        (1024, 128, 8),
        (1025, 128, 8),
        (65536, 896, 16),
    ],
)
def test_map_to_physical(tokens, experts, topk, dtype):
    rows = EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS
    table = torch.arange(rows * experts, dtype=torch.int32).reshape(rows, experts) * 3 + 5
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
    # Compile both kernels before capture.
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


@pytest.mark.parametrize("count", [64, 255, 256, 257, 896])
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
