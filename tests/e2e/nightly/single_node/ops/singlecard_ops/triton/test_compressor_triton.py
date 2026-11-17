# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.deepseek_v41.cache_config import STATE_RING_ROWS
from vllm_ascend.ops.triton.compressor.compressor_triton import (
    compressor,
    compressor_from_projected,
    compressor_ref,
    write_pcp_ring_rows,
)

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)


@pytest.mark.parametrize("ratio", [2, 3, 8, 9, 16, 128])
@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_compressor_matches_torch(ratio, packed):
    """Exercise projection, direct/blocked pooling, normalization and ring updates."""
    generator = torch.Generator().manual_seed(1024)
    hidden, width = 5120, 512
    length = max(129, 2 * ratio + 5)
    lengths = [length, ratio + 3, 5, 1] if packed else [length] * 4
    used = torch.tensor([length, ratio + 1, 0, 1], dtype=torch.int32)
    starts = torch.tensor([0, 1, min(ratio - 1, STATE_RING_ROWS - 1), 2 * ratio + 1], dtype=torch.int32)
    # Block zero is a valid page in the general compressor, unlike the PCP entry.
    blocks = torch.tensor([3, 0, 4, 1], dtype=torch.int32)
    x_shape = (sum(lengths), hidden) if packed else (4, length, hidden)
    x = torch.randn(x_shape, generator=generator).bfloat16()
    wkv = (torch.randn(width, hidden, generator=generator) / hidden**0.5).bfloat16()
    wgate = (torch.randn(width, hidden, generator=generator) / hidden**0.5).bfloat16()
    norm = torch.randn(width, generator=generator).bfloat16()
    state = torch.randn(6, STATE_RING_ROWS, 2 * width, generator=generator)
    expected_state = state.clone()
    cumulative = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int() if packed else None
    expected = compressor_ref(x, wkv, wgate, expected_state, ratio, norm, blocks, cumulative, used, starts)
    actual_state = state.npu()
    actual = compressor(
        x.npu(),
        wkv.npu(),
        wgate.npu(),
        actual_state,
        ratio,
        norm.npu(),
        state_block_table=blocks.npu(),
        cu_seqlens=cumulative.tolist() if packed else None,
        seqused=used.tolist(),
        start_pos=starts.tolist(),
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=2**-6, atol=2**-6)
    torch.testing.assert_close(actual_state.cpu(), expected_state, rtol=2**-10, atol=2**-16)
    # Projection and pooling must not change pages owned by another request.
    torch.testing.assert_close(actual_state[[2, 4, 5]].cpu(), state[[2, 4, 5]], rtol=0, atol=0)


def _projected_reference(kv, scores, state, metadata, prev_rows, prev_indices):
    """CPU pair pooling with token-aligned output, independent of Triton indexing."""
    out, updated = torch.zeros_like(kv, dtype=torch.bfloat16), state.clone()
    width = kv.shape[1]
    for request, (start, used, _, base, block) in enumerate(metadata.T.tolist()):
        if not 0 < block < state.shape[0]:
            continue
        for offset in range(used):
            position = start + offset
            if position % 2:
                if offset:
                    previous = torch.cat((kv[base + offset - 1], scores[base + offset - 1]))
                elif prev_rows is not None and prev_indices[request] >= 0:
                    previous = prev_rows[prev_indices[request]]
                else:
                    previous = state[block, (position - 1) % STATE_RING_ROWS]
                pair_k = torch.stack((previous[:width], kv[base + offset]))
                pair_s = torch.stack((previous[width:], scores[base + offset]))
                out[base + offset] = (pair_s.softmax(0) * pair_k).sum(0).bfloat16()
            updated[block, position % STATE_RING_ROWS] = torch.cat((kv[base + offset], scores[base + offset]))
    return out, updated


@pytest.mark.parametrize("width", [128, 512])
@pytest.mark.parametrize("predecessor", ["ring", "gather"])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_projected_chunks_and_replay(width, predecessor, use_graph):
    """Odd starts, long chunks, rejection-corrected starts, empty ranks and resume."""
    generator = torch.Generator().manual_seed(307)
    requests, capacity = 6, 96
    tokens = requests * capacity + 3
    state = torch.randn(7, STATE_RING_ROWS, 2 * width, generator=generator)
    ring = state.npu()
    kv = torch.randn(tokens, width, generator=generator).npu()
    scores = torch.randn(tokens, width, generator=generator).npu()
    out = torch.full((tokens, width), 17, dtype=torch.bfloat16, device="npu")
    starts = [0, 29, 1, 13, 31, 3]
    metadata = torch.tensor(
        [starts, [65, 35, 7, 0, 9, 3], [0] * requests, [r * capacity for r in range(requests)], [3, 1, 0, 4, 2, 7]],
        dtype=torch.int32,
    )
    controls = metadata.npu()
    previous = torch.randn(requests, 2 * width, generator=generator) if predecessor == "gather" else None
    previous_indices = torch.tensor([-1, 1, -1, 3, 4, -1], dtype=torch.int64) if previous is not None else None
    device_previous = previous.npu() if previous is not None else None
    device_indices = previous_indices.npu() if previous_indices is not None else None

    def run():
        return compressor_from_projected(
            kv,
            scores,
            ring,
            controls,
            out,
            max_query_len=capacity,
            num_cores=8,
            prev_rows=device_previous,
            prev_row_indices=device_indices,
        )

    graph = None
    if use_graph:
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        ring.copy_(state)

    for step, lengths in enumerate(([65, 35, 7, 0, 9, 3], [7, 9, 1, 0, 4, 1], [0] * requests, [1, 2, 0, 0, 1, 0])):
        current_k = torch.randn(tokens, width, generator=generator)
        current_s = torch.randn(tokens, width, generator=generator) * 3
        metadata[0] = torch.tensor(starts)
        metadata[1] = torch.tensor(lengths)
        expected, updated = _projected_reference(current_k, current_s, state, metadata, previous, previous_indices)
        kv.copy_(current_k)
        scores.copy_(current_s)
        controls.copy_(metadata)
        out.fill_(17)
        if graph is not None:
            graph.replay()
        else:
            assert run().data_ptr() == out.data_ptr()
        torch.testing.assert_close(out.cpu(), expected, rtol=2**-6, atol=2**-6)
        torch.testing.assert_close(ring.cpu(), state if previous is not None else updated, rtol=0, atol=0)
        if previous is not None:
            # Global tails are written only after every chunk has read its predecessor.
            indices, owners, offsets = [], [], []
            for request, length in enumerate(lengths):
                for offset in range(max(0, length - STATE_RING_ROWS), length):
                    indices.append(request * capacity + offset)
                    owners.append(request)
                    offsets.append(offset)
            write_pcp_ring_rows(
                torch.cat((kv, scores), dim=1),
                ring,
                controls,
                torch.tensor(indices, dtype=torch.int64, device="npu"),
                torch.tensor(owners, dtype=torch.int64, device="npu"),
                torch.tensor(offsets, dtype=torch.int64, device="npu"),
            )
            torch.testing.assert_close(ring.cpu(), updated, rtol=0, atol=0)
        state = updated
        # Simulate rejection after verification; the next launch reads device starts.
        starts = [start + length - (min(2, length) if step == 1 else 0) for start, length in zip(starts, lengths)]


@pytest.mark.parametrize("width", [128, 512])
@pytest.mark.parametrize("requests,tail", [(1, 1), (3, 3), (16, STATE_RING_ROWS)])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_ring_writes_match_cpu(width, requests, tail, use_graph):
    generator = torch.Generator().manual_seed(89)
    count = requests * tail + 5
    rows = torch.randn(count, 2 * width, generator=generator)
    state = torch.randn(requests + 3, STATE_RING_ROWS, 2 * width, generator=generator)
    metadata = torch.zeros(5, requests + 3, dtype=torch.int32)
    metadata[0] = torch.arange(requests + 3) * 29 + 1009
    metadata[1] = 65
    metadata[4, :requests] = torch.arange(requests, 0, -1)
    # Inactive request, null page and out-of-range page must all be ignored.
    metadata[1, requests] = 0
    metadata[4, requests:] = torch.tensor([requests + 1, 0, state.shape[0]])
    owners = torch.arange(requests).repeat_interleave(tail)
    owners = torch.cat((owners, torch.tensor([-1, metadata.shape[1], requests, requests + 1, requests + 2])))
    offsets = torch.cat((torch.arange(65 - tail, 65).repeat(requests), torch.zeros(5, dtype=torch.int64)))
    source = torch.cat(
        (torch.randperm(count, generator=generator)[: requests * tail], torch.zeros(5, dtype=torch.int64))
    )
    device_rows, ring, controls = rows.npu(), state.npu(), metadata.npu()
    source_npu, owners_npu, offsets_npu = source.npu(), owners.npu(), offsets.npu()

    def run():
        write_pcp_ring_rows(device_rows, ring, controls, source_npu, owners_npu, offsets_npu)

    graph = None
    if use_graph:
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        ring.copy_(state)
    for step in range(3):
        expected = state.clone()
        for row, request, offset in zip(source.tolist(), owners.tolist(), offsets.tolist()):
            if 0 <= request < metadata.shape[1]:
                start, used, _, _, block = metadata[:, request].tolist()
                if used > 0 and 0 < block < state.shape[0]:
                    expected[block, (start + offset) % STATE_RING_ROWS] = rows[row]
        if graph is None:
            run()
        else:
            graph.replay()
        torch.testing.assert_close(ring.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(device_rows.cpu(), rows, rtol=0, atol=0)
        state = expected
        rows = torch.randn(rows.shape, generator=generator)
        # Reuse the captured addresses with changed values and corrected starts.
        metadata[0] -= step + 1
        metadata[1, 0] = 0 if step == 0 else 65
        device_rows.copy_(rows)
        controls.copy_(metadata)


@pytest.mark.parametrize("width", [128, 512])
@torch.inference_mode()
def test_pcp_chunks_share_ring_page(width):
    """A pair spanning PCP chunks reads gathered rows before the global tail write."""
    generator = torch.Generator().manual_seed(53)
    tokens, boundary = 101, 65
    kv = torch.randn(tokens, width, generator=generator)
    scores = torch.randn(tokens, width, generator=generator)
    rows = torch.cat((kv, scores), dim=1).npu()
    state = torch.randn(3, STATE_RING_ROWS, 2 * width, generator=generator)
    ring = state.npu()
    metadata = torch.tensor(
        [[0, boundary], [boundary, tokens - boundary], [0, 0], [0, boundary], [1, 1]],
        dtype=torch.int32,
        device="npu",
    )
    out = torch.empty(tokens, width, dtype=torch.bfloat16, device="npu")
    compressor_from_projected(
        kv.npu(),
        scores.npu(),
        ring,
        metadata,
        out,
        max_query_len=boundary,
        num_cores=8,
        prev_rows=rows,
        prev_row_indices=torch.tensor([-1, boundary - 1], dtype=torch.int64, device="npu"),
    )
    expected = torch.zeros(tokens, width, dtype=torch.bfloat16)
    pairs = tokens // 2
    expected[1 : 2 * pairs : 2] = (
        (scores[: 2 * pairs].reshape(pairs, 2, width).softmax(1) * kv[: 2 * pairs].reshape(pairs, 2, width))
        .sum(1)
        .bfloat16()
    )
    torch.testing.assert_close(out.cpu(), expected, rtol=2**-6, atol=2**-6)
    torch.testing.assert_close(ring.cpu(), state, rtol=0, atol=0)
    # One canonical tail per global request; local tails can overlap ring slots.
    offsets = torch.arange(tokens - STATE_RING_ROWS, tokens, dtype=torch.int64)
    global_metadata = torch.tensor([[0], [tokens], [0], [0], [1]], dtype=torch.int32, device="npu")
    write_pcp_ring_rows(
        rows, ring, global_metadata, offsets.npu(), torch.zeros_like(offsets, device="npu"), offsets.npu()
    )
    for offset in offsets.tolist():
        state[1, offset % STATE_RING_ROWS] = torch.cat((kv[offset], scores[offset]))
    torch.testing.assert_close(ring.cpu(), state, rtol=0, atol=0)


@pytest.mark.parametrize("empty", ["tokens", "requests", "launch"])
@torch.inference_mode()
def test_empty_projected_and_ring_writes(empty):
    tokens = 0 if empty == "tokens" else 3
    kv = torch.zeros(tokens, 128, device="npu")
    ring = torch.ones(2, STATE_RING_ROWS, 256, device="npu")
    metadata = torch.zeros(5, 0 if empty == "requests" else 1, dtype=torch.int32, device="npu")
    out = torch.full_like(kv, 17, dtype=torch.bfloat16)
    actual = compressor_from_projected(
        kv,
        kv,
        ring,
        metadata,
        out,
        max_query_len=0 if empty == "launch" else tokens,
        num_cores=8,
        prev_rows=None,
        prev_row_indices=None,
    )
    indices = torch.empty(0, dtype=torch.int64, device="npu")
    write_pcp_ring_rows(torch.zeros(tokens, 256, device="npu"), ring, metadata, indices, indices, indices)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), torch.zeros(tokens, 128, dtype=torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(ring.cpu(), torch.ones(2, STATE_RING_ROWS, 256), rtol=0, atol=0)
