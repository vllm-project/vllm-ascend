# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool

from vllm_ascend.ops.triton.v2.spec_decode.select_context_window import (
    select_context_window_indices,
    select_uniform_context_window_indices,
)


@pytest.mark.parametrize("lengths", [[16384], [6, 16348, 1, 29], [0, 132, 133, 134], [1, 6, 29]])
@pytest.mark.parametrize("pointer_offset", [0, 1])
@pytest.mark.parametrize("speculative", [1, 5, 8])
def test_context_suffix_keeps_rejected_anchor_window(lengths, pointer_offset, speculative):
    window = 128
    keep = window + speculative
    boundaries = np.concatenate(([0], np.cumsum(lengths)))
    storage = torch.full(
        (len(boundaries) + pointer_offset,),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    query_start_loc = storage[pointer_offset:]
    query_start_loc.copy_(torch.tensor(boundaries, dtype=torch.int32))
    packed_starts = np.concatenate(([0], np.cumsum(np.minimum(lengths, keep))))
    packed_starts_pool = UvaBufferPool(len(packed_starts), torch.int64)

    indices, valid = select_context_window_indices(
        query_start_loc,
        packed_starts_pool.copy_to_uva(packed_starts),
        int(packed_starts[-1]),
        keep,
    )

    expected_indices = torch.zeros_like(indices, device="cpu")
    expected_valid = torch.zeros_like(valid, device="cpu")
    for req, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        rows = torch.arange(max(begin, end - keep), end)
        start = packed_starts[req]
        expected_indices[start : start + len(rows)] = rows
        expected_valid[start : start + len(rows)] = True
        selected = set(rows.tolist())
        for rejected in range(min(speculative, end - begin) + 1):
            accepted_end = end - rejected
            needed = set(range(max(begin, accepted_end - window), accepted_end))
            assert needed <= selected
            for graph_prefix in (0, 48):
                eager = {index for index in selected if graph_prefix <= index < accepted_end}
                graph = set(range(begin, min(graph_prefix, accepted_end)))
                assert needed <= eager | graph
                assert eager.isdisjoint(graph)
    torch.testing.assert_close(indices.cpu(), expected_indices, atol=0, rtol=0)
    torch.testing.assert_close(valid.cpu(), expected_valid, atol=0, rtol=0)


def test_context_suffix_uses_device_request_boundaries():
    query_start_loc = torch.tensor([0, 3, 4091, 4092], dtype=torch.int32, device="npu")
    packed_starts_pool = UvaBufferPool(4, torch.int64)
    packed_starts = packed_starts_pool.copy_to_uva([0, 8, 141, 147])

    indices, valid = select_context_window_indices(
        query_start_loc,
        packed_starts,
        num_context_tokens=147,
        keep=133,
    )

    expected = torch.cat((torch.arange(3), torch.arange(3958, 4091), torch.tensor([4091])))
    torch.testing.assert_close(indices[valid].cpu(), expected, atol=0, rtol=0)
    assert valid.sum().item() == 137


def test_uniform_context_suffix_uses_device_request_boundaries():
    query_start_loc = torch.tensor([0, 3, 4091, 4092], dtype=torch.int32, device="npu")

    indices, valid = select_uniform_context_window_indices(
        query_start_loc,
        num_reqs=3,
        keep=133,
    )

    expected = torch.cat((torch.arange(3), torch.arange(3958, 4091), torch.tensor([4091])))
    torch.testing.assert_close(indices[valid].cpu(), expected, atol=0, rtol=0)
    assert valid.sum().item() == 137
    assert indices.shape == (3 * 133,)
