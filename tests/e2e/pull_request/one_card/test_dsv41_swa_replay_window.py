# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native window producer regression; no model checkpoint is required."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.build_window_indices import build_window_indices_triton


@pytest.mark.parametrize("window_size", [4, 128, 257])
@pytest.mark.parametrize("with_floor", [False, True])
def test_swa_window_replays_current_floor_and_padding(window_size, with_floor):
    device = torch.device("npu:0")
    positions = torch.zeros(8, dtype=torch.int64, device=device)
    visible = torch.ones(8, 1, dtype=torch.int32, device=device)
    indices = torch.empty(8, 1, window_size, dtype=torch.int32, device=device)
    lengths = torch.empty(8, 1, dtype=torch.int32, device=device)

    def forward():
        return build_window_indices_triton(
            positions,
            window_size,
            visible_lens=visible if with_floor else None,
            indices_output=indices,
            lengths_output=lengths,
        )

    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        forward()
        with torch.npu.graph(graph, stream=stream):
            forward()
    torch.npu.synchronize()

    # Keep the same graph and storage while changing live rows and replay floors.
    for pos, caps in [
        ([255, 256, 257, 258, 259, 260, 261, 262], [1, 2, 3, 4, 5, 6, 7, 8]),
        ([511, 512, 513, 514, 515, 516, 517, 518], [1, 2, 0, 0, 0, 0, 0, 0]),
        ([-1, 0, 2, 255, 256, 257, 258, 259], [0, 1, 3, 4, 0, 0, 0, 0]),
        ([255, 256, 257, 258, 259, 260, 261, 262], [1, 2, 3, 4, 5, 6, 7, 8]),
    ]:
        expected_indices = torch.full((8, 1, window_size), -1, dtype=torch.int32)
        expected_lengths = torch.zeros(8, 1, dtype=torch.int32)
        for row, position in enumerate(pos):
            count = max(0, min(position + 1, window_size))
            if with_floor:
                count = min(count, caps[row])
            expected_lengths[row, 0] = count
            expected_indices[row, 0, :count] = torch.arange(position + 1 - count, position + 1, dtype=torch.int32)
        with torch.npu.stream(stream):
            stream.wait_stream(torch.npu.default_stream())
            positions.copy_(torch.tensor(pos))
            visible.copy_(torch.tensor(caps, dtype=torch.int32).unsqueeze(1))
            indices.fill_(-99)
            lengths.fill_(-99)
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(indices.cpu(), expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(lengths.cpu(), expected_lengths, rtol=0, atol=0)
