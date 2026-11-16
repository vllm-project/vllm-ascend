# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch

from vllm_ascend.ops.fused_moe.eplb import record_expert_tokens


@pytest.mark.parametrize("num_experts", [0, 1, 63, 64, 112, 255, 256, 257, 511, 512, 896])
@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("count_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("load_dtype", [torch.int32, torch.int64])
def test_record_expert_blocks(num_experts, group_list_type, count_dtype, load_dtype):
    # Large synthetic counts exercise prefix-sum and destination overflow.
    counts = torch.arange(num_experts, dtype=torch.int64) * 7 + 3 + 2**28
    encoded = (counts.cumsum(0) if group_list_type == 0 else counts).to(count_dtype)
    decoded = encoded if group_list_type == 1 else torch.diff(encoded, prepend=encoded.new_zeros(1))
    start = 3
    initial = torch.arange(num_experts + 2 * start, dtype=load_dtype) + torch.iinfo(load_dtype).max - 2**27
    expected = initial.clone()
    expected[start : start + num_experts] += decoded.to(load_dtype)
    load = initial.npu()
    tokens = encoded.npu()
    enabled = torch.tensor(True, device="npu")

    record_expert_tokens(tokens, load, enabled, group_list_type, start)
    torch.testing.assert_close(load.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(tokens.cpu(), encoded, rtol=0, atol=0)

    enabled.fill_(False)
    record_expert_tokens(tokens, load, enabled, group_list_type, start)
    torch.testing.assert_close(load.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("group_list_type", [0, 1])
def test_record_mutable_state_graph_replay(group_list_type):
    # Cross the program boundary while testing the production custom-op entry.
    num_experts = 257
    start = 2
    counts = torch.arange(num_experts, dtype=torch.int64) % 7 + 1
    tokens = (counts.cumsum(0) if group_list_type == 0 else counts).npu()
    initial = torch.arange(num_experts + 2 * start, dtype=torch.int32)
    load = initial.npu()
    enabled = torch.tensor(True, device="npu")
    record_op = torch.ops.vllm.ascend_eplb_record_expert_tokens

    for _ in range(3):
        record_op(tokens, load, enabled, group_list_type, start)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        record_op(tokens, load, enabled, group_list_type, start)

    load.copy_(initial)
    updated_counts = counts * 2 + 1
    tokens.copy_(updated_counts.cumsum(0) if group_list_type == 0 else updated_counts)
    graph.replay()
    expected = initial.clone()
    expected[start : start + num_experts] += updated_counts.to(expected.dtype)
    torch.testing.assert_close(load.cpu(), expected, rtol=0, atol=0)

    enabled.fill_(False)
    graph.replay()
    torch.testing.assert_close(load.cpu(), expected, rtol=0, atol=0)

    enabled.fill_(True)
    graph.replay()
    expected[start : start + num_experts] += updated_counts.to(expected.dtype)
    torch.testing.assert_close(load.cpu(), expected, rtol=0, atol=0)
