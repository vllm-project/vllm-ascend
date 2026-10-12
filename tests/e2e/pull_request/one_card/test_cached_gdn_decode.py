# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.v2.mamba.cached_decode import materialize_cached_gdn_decode


@pytest.mark.parametrize("width", [1, 3, 8, 17])
@pytest.mark.parametrize("graph_rows,actual_rows", [(1, 1), (8, 1), (33, 0), (33, 17), (33, 33), (64, 17)])
@pytest.mark.parametrize("stride", [1, 2])
def test_current_cached_decode_buffers(width, graph_rows, actual_rows, stride):
    torch.npu.set_device(0)
    capacity = graph_rows + 3
    state_values = (torch.arange(capacity * (width + 4), dtype=torch.int32) + 2**24 + 1).view(capacity, width + 4)
    states = state_values.to("npu")[:, :width]
    query_values = torch.arange((actual_rows + 1) * stride, dtype=torch.int32)
    query = (query_values * 5).to("npu")[::stride]
    accepted = torch.arange(max(1, actual_rows) * stride, dtype=torch.int32, device="npu")[::stride]
    tokens = torch.arange(actual_rows * 5, dtype=torch.int32, device="npu")
    outs = [
        torch.full(shape, -99, dtype=torch.int32, device="npu")
        for shape in (
            (capacity, width),
            (capacity + 1,),
            (capacity,),
            (max(1, actual_rows * 5) + 5,),
            (capacity + 1,),
        )
    ]
    out_state, out_query, out_accepted, out_token, out_lengths = outs
    out_masks = torch.ones(capacity, dtype=torch.bool, device="npu")
    stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()

    def prepare():
        materialize_cached_gdn_decode(
            states,
            query,
            accepted,
            tokens,
            out_state,
            out_query,
            out_accepted,
            out_token,
            out_masks,
            out_lengths,
            actual_rows,
            graph_rows,
        )

    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        prepare()
        torch.npu.synchronize()
        with torch.npu.graph(graph, stream=stream):
            prepare()
        snapshots = []
        for step in (1, 2, 3):
            states.add_(1)
            query.add_(1)
            accepted.add_(1)
            tokens.add_(1)
            graph.replay()
            snapshots.append([t.clone() for t in (*outs, out_masks)])
    torch.npu.synchronize()
    for step, snapshot in enumerate(snapshots, 1):
        (
            current_state,
            current_query,
            current_accepted,
            current_tokens,
            lengths,
            masks,
        ) = [t.cpu() for t in snapshot]
        expected_state = state_values[:graph_rows, :width].clone() + step
        expected_state[actual_rows:].zero_()
        torch.testing.assert_close(current_state[:graph_rows], expected_state, rtol=0, atol=0)
        offsets = torch.arange(graph_rows + 1).clamp_max(actual_rows)
        expected_query = offsets.to(torch.int32) * stride * 5 + step
        torch.testing.assert_close(current_query[: graph_rows + 1], expected_query, rtol=0, atol=0)
        expected_lengths = torch.cat([expected_query[:1], expected_query[1:] - expected_query[:-1]])
        torch.testing.assert_close(lengths[: graph_rows + 1], expected_lengths, rtol=0, atol=0)
        assert current_accepted[:graph_rows].tolist() == [i * stride + step for i in range(actual_rows)] + [1] * (
            graph_rows - actual_rows
        )
        assert masks[:graph_rows].tolist() == [True] * actual_rows + [False] * (graph_rows - actual_rows)
        assert current_tokens[: tokens.numel()].tolist() == [i + step for i in range(tokens.numel())]
        for tensor, tail in (
            (current_state, graph_rows),
            (current_query, graph_rows + 1),
            (current_accepted, graph_rows),
            (current_tokens, tokens.numel()),
            (lengths, graph_rows + 1),
        ):
            assert torch.all(tensor[tail:] == -99)
        assert torch.all(masks[graph_rows:])
