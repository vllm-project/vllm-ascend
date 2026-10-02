# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.fused_moe.prepare_finalize import _pad_and_split_tokens


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_local_padding_graph_replay(dtype, rank):
    # Alternate capture shapes and inputs to detect stale zero tails or
    # accidental cross-graph buffer reuse; consume views inside the graph.
    captures = []
    for rows, padded in [(1, 4), (3, 4), (5, 8), (1, 8)]:
        x = torch.ones(rows, 8, dtype=dtype, device="npu")
        for _ in range(3):
            _pad_and_split_tokens(x, padded, 4, rank).clone()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = _pad_and_split_tokens(x, padded, 4, rank).clone()
        captures.append((x, padded, graph, result))
    for value in [2, 7, -3]:
        for x, padded, graph, result in reversed(captures):
            x.fill_(value)
            graph.replay()
            torch.npu.synchronize()
            reference = torch.nn.functional.pad(x.cpu(), (0, 0, 0, padded - x.shape[0])).chunk(4)[rank]
            torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)
