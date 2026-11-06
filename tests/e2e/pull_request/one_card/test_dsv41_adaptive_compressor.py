# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C2 verification rollback with a fixed graph carrier and live device lengths."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.dsa_v41 import build_replay_swa_visible_lens
from vllm_ascend.ops.triton.build_window_indices import build_window_indices_triton
from vllm_ascend.ops.triton.c2_ring_metadata import build_c2_ring_metadata
from vllm_ascend.ops.triton.compressor.compressor_triton import _cube_core_num, compressor_from_projected


@pytest.mark.parametrize("max_query_len", [6, 9, 32])
@pytest.mark.parametrize("full_graph", [False, True])
def test_c2_adaptive_graph_replay_matches_sequential_reference(max_query_len, full_graph):
    torch.manual_seed(19)
    device = torch.device("npu:0")
    width, requests = 512, 4
    tokens = max(16, 2 * max_query_len)
    # Use separate request pages and retain a sentinel null page.
    state_cpu = torch.randn(8, 32, 2 * width, dtype=torch.float32) * 0.1
    state = state_cpu.to(device)
    kv = torch.zeros(tokens, width, dtype=torch.float32, device=device)
    scores = torch.zeros_like(kv)
    output = torch.empty(tokens, width, dtype=torch.bfloat16, device=device)
    starts = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
    lengths = torch.ones(requests, dtype=torch.int32, device=device)
    positions = torch.zeros(tokens, dtype=torch.int64, device=device)
    blocks = torch.tensor([[2], [5], [0], [0]], dtype=torch.int32, device=device)
    ring = torch.empty(5, requests, dtype=torch.int32, device=device)
    complete = torch.empty(tokens, dtype=torch.bool, device=device)
    source_positions = torch.empty(tokens, dtype=torch.int64, device=device)
    rope = torch.ones(1024, 1, 1, 64, dtype=torch.float32, device=device)
    cos, sin = (torch.empty(tokens, 1, 1, 64, dtype=torch.float32, device=device) for _ in range(2))
    visible = torch.empty(tokens, 1, dtype=torch.int32, device=device)
    window_indices = torch.empty(tokens, 1, 128, dtype=torch.int32, device=device)
    window_lengths = torch.empty_like(visible)
    replay_floor = torch.zeros(requests, dtype=torch.int32, device=device)
    cores = _cube_core_num()

    def forward():
        build_c2_ring_metadata(
            starts,
            lengths,
            positions,
            blocks,
            rope,
            rope,
            requests,
            tokens,
            requests,
            tokens,
            skip_update=False,
            ring_metadata_output=ring,
            complete_mask_output=complete,
            source_positions_output=source_positions,
            cos_output=cos,
            sin_output=sin,
        )
        compressor_from_projected(kv, scores, state, ring, output, max_query_len=max_query_len, num_cores=cores)
        visible.copy_(build_replay_swa_visible_lens(128, starts, lengths, replay_floor, tokens))
        build_window_indices_triton(
            positions, 128, visible_lens=visible, indices_output=window_indices, lengths_output=window_lengths
        )

    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph() if full_graph else None
    with torch.npu.stream(stream):
        stream.wait_stream(torch.npu.default_stream())
        # Warm up before capture. The capture deliberately has one-token rows,
        # whereas subsequent replays put many verification tokens in one row.
        forward()
        if graph is not None:
            with torch.npu.graph(graph, stream=stream):
                forward()
    torch.npu.synchronize()
    state.copy_(state_cpu)
    histories = [121, 255]
    snapshots = []
    expected = []
    for query_lengths in ([max_query_len, 1], [1, 1], [1, max_query_len], [2, 3], [1, 1]):
        kv_cpu = torch.randn(tokens, width, dtype=torch.float32) * 0.1
        score_cpu = torch.randn_like(kv_cpu) * 0.1
        expected_output = torch.zeros(tokens, width, dtype=torch.bfloat16)
        offsets = [0, query_lengths[0], sum(query_lengths), sum(query_lengths), sum(query_lengths)]
        input_positions = torch.full((tokens,), 299, dtype=torch.int64)
        expected_visible = torch.zeros(tokens, 1, dtype=torch.int32)
        expected_indices = torch.full((tokens, 1, 128), -1, dtype=torch.int32)
        floors = [histories[0] - 1, histories[1]]
        for req, (history, count, page) in enumerate(zip(histories, query_lengths, [2, 5])):
            for local in range(count):
                row, pos = offsets[req] + local, history + local
                input_positions[row] = pos
                keys = torch.arange(max(pos - 127, floors[req]), pos + 1, dtype=torch.int32)
                expected_visible[row] = keys.numel()
                expected_indices[row, 0, : keys.numel()] = keys
                state_cpu[page, pos % 32, :width] = kv_cpu[row]
                state_cpu[page, pos % 32, width:] = score_cpu[row]
                if pos % 2:
                    pair = state_cpu[page, [(pos - 1) % 32, pos % 32]]
                    expected_output[row] = (pair[:, :width] * pair[:, width:].softmax(dim=0)).sum(0).bfloat16()
        expected.append((expected_output, state_cpu.clone(), expected_visible, expected_indices))
        with torch.npu.stream(stream):
            stream.wait_stream(torch.npu.default_stream())
            kv.copy_(kv_cpu)
            scores.copy_(score_cpu)
            positions.copy_(input_positions)
            starts.copy_(torch.tensor(offsets, dtype=torch.int32))
            lengths.copy_(torch.tensor([h + n for h, n in zip(histories, query_lengths)] + [0, 0], dtype=torch.int32))
            replay_floor.copy_(torch.tensor([*floors, 0, 0], dtype=torch.int32))
            if graph is not None:
                graph.replay()
            else:
                forward()
            snapshots.append((output.clone(), state.clone(), visible.clone(), window_indices.clone(), window_lengths.clone()))
        # Accept only the anchor, then rewind over the rejected suffix. The
        # next C2 residual may need the first row from a much longer prior step.
        histories = [history + 1 for history in histories]
    torch.npu.synchronize()
    for (actual, actual_state, actual_visible, actual_indices, actual_lengths), (
        reference, reference_state, reference_visible, reference_indices
    ) in zip(snapshots, expected):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.cpu(), reference, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(actual_state.cpu(), reference_state, rtol=0, atol=0)
        torch.testing.assert_close(actual_visible.cpu(), reference_visible, rtol=0, atol=0)
        torch.testing.assert_close(actual_lengths.cpu(), reference_visible, rtol=0, atol=0)
        torch.testing.assert_close(actual_indices.cpu(), reference_indices, rtol=0, atol=0)
