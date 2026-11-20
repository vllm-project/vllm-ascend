# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check C2 residual pooling and writes with interleaved physical ring pages."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.compressor.compressor_triton import (
    _cube_core_num,
    compressor_from_projected,
    write_pcp_ring_rows,
)


@pytest.mark.live_npu
def test_ring_block_stride_preserves_other_resources_and_requests():
    width, rows, blocks = 512, 32, 6
    page_elements = rows * 2 * width
    prefix, gap = 64, 256
    stride = prefix + page_elements + gap
    backing = torch.full((blocks * stride,), -123.0, device="npu", dtype=torch.float32)
    state = torch.as_strided(backing, (blocks, rows, 2 * width), (stride, 2 * width, 1), prefix)
    block_ids = [3, 1, 4]
    for block in block_ids:
        state[block].zero_()
    expected_backing = backing.cpu()
    expected_state = torch.as_strided(expected_backing, (blocks, rows, 2 * width), (stride, 2 * width, 1), prefix)
    generator = torch.Generator().manual_seed(41)
    histories = [[] for _ in block_ids]
    positions = [0] * len(block_ids)
    # Odd boundaries force residual reads; long chunks wrap and overwrite the ring.
    for lengths in ([1, 3, 2], [3, 1, 1], [29, 35, 31], [2, 1, 3], [35, 2, 1]):
        total = sum(lengths)
        kv = torch.randn(total, width, generator=generator)
        scores = torch.randn(total, width, generator=generator)
        expected_output = torch.zeros(total, width, dtype=torch.bfloat16)
        bases, cursor = [], 0
        for req, (block, length) in enumerate(zip(block_ids, lengths)):
            bases.append(cursor)
            for row in range(length):
                token = positions[req] + row
                value, score = kv[cursor + row], scores[cursor + row]
                histories[req].append((value, score))
                if token % 2:
                    pair = histories[req][-2:]
                    values = torch.stack([item[0] for item in pair])
                    weights = torch.stack([item[1] for item in pair]).softmax(dim=0)
                    expected_output[cursor + row] = (values * weights).sum(dim=0).to(torch.bfloat16)
                expected_state[block, token % rows, :width] = value
                expected_state[block, token % rows, width:] = score
            cursor += length
        # A padded request owns the null block and must perform no writes.
        metadata = torch.tensor(
            [positions + [0], list(lengths) + [0], [0] * 4, bases + [total], block_ids + [0]],
            device="npu",
            dtype=torch.int32,
        )
        output = torch.empty(total, width, device="npu", dtype=torch.bfloat16)
        compressor_from_projected(
            kv.npu(),
            scores.npu(),
            state,
            metadata,
            output,
            max_query_len=max(lengths),
            num_cores=_cube_core_num(),
            prev_rows=None,
            prev_row_indices=None,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected_output, rtol=0.01, atol=0.015625)
        # Includes gaps, other cache resources, unused pages, and null-block state.
        torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)
        positions = [start + length for start, length in zip(positions, lengths)]


@pytest.mark.live_npu
def test_pcp_tail_writes_preserve_interleaved_ring_gaps():
    width, rows, blocks = 512, 32, 4
    stride = rows * 2 * width + 256
    backing = torch.full((blocks * stride,), -123.0, device="npu", dtype=torch.float32)
    state = torch.as_strided(backing, (blocks, rows, 2 * width), (stride, 2 * width, 1), 64)
    expected = backing.cpu()
    expected_state = torch.as_strided(expected, state.shape, state.stride(), 64)
    gathered = torch.arange(4 * 2 * width, dtype=torch.float32).reshape(4, 2 * width)
    # Two real requests cross the ring boundary; a null request must not write.
    metadata = torch.tensor([[31, 63, 0], [2, 2, 1], [0, 0, 0], [0, 2, 4], [3, 1, 0]], dtype=torch.int32, device="npu")
    sources = torch.tensor([0, 1, 2, 3, 0], dtype=torch.int64, device="npu")
    requests = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int64, device="npu")
    offsets = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int64, device="npu")
    write_pcp_ring_rows(gathered.npu(), state, metadata, sources, requests, offsets)
    for block, source in [(3, 0), (1, 2)]:
        expected_state[block, 31] = gathered[source]
        expected_state[block, 0] = gathered[source + 1]
    torch.npu.synchronize()
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
