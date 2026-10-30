# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bound FP32 temporary memory without overwriting an unfinished reader.

Five-layer retention caused the actual GLM long-mixed OOM. These real native
cases distinguish a bounded read/stream workspace from independent external
materializations, and queue readers before the next write without a Host wait.
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.worker.v2.spec_decode.draft_window_dense import (  # noqa: E402
    dense_draft_attention,
    gather_dense_draft_kv,
    prepare_dense_draft_read,
)
from vllm_ascend.worker.v2.spec_decode.draft_window_read import DirectPrefixSource  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
TOLERANCE = {"atol": 2**-7, "rtol": 2**-6}


def make_read(capacity, held):
    block = 128
    blocks = (capacity + block - 1) // block
    table = torch.arange(blocks, dtype=torch.int32, device="npu")[None]
    read = prepare_dense_draft_read(
        DirectPrefixSource(table, (0,), (blocks,), block),
        exact_device=torch.tensor([capacity], dtype=torch.int32, device="npu"),
        row_indices=torch.tensor([0], dtype=torch.int64, device="npu"),
        upper_cpu=torch.tensor([capacity], dtype=torch.int32),
        window_size=1024,
        anchor_span=128,
        query_width=8,
        causal=False,
        full_rows_cpu=torch.tensor([True]),
        retain=lambda *items: held.extend(items),
    )
    keys = torch.zeros((blocks, block, 4, 192), dtype=torch.bfloat16, device="npu")
    return read, keys, torch.zeros_like(keys), torch.ones((8, 4, 192), dtype=torch.bfloat16, device="npu")


def buffers(held, capacity):
    return {
        item.untyped_storage().data_ptr(): item.untyped_storage().nbytes()
        for item in held
        if isinstance(item, torch.Tensor) and item.dtype == torch.float32 and tuple(item.shape) == (1, capacity, 4, 192)
    }


@pytest.mark.parametrize("capacity", [1152, 64000])
def test_sequential_attention_has_bounded_private_operands(capacity):
    held: list[object] = []
    read, keys, values, query = make_read(capacity, held)
    # Compile outside the measured allocation-contract section. No timed
    # throughput claim; retaining the read is required for its whole lifetime.
    dense_draft_attention(read, query, keys, values, 1.0)
    torch.npu.synchronize()
    held.clear()
    outputs = []
    for layer in range(1, 6):
        values.fill_(layer)
        outputs.append(dense_draft_attention(read, query, keys, values, 1.0))
    torch.npu.synchronize()
    for layer, output in enumerate(outputs, 1):
        torch.testing.assert_close(output.cpu(), torch.full_like(query.cpu(), layer), **TOLERANCE)
    actual = buffers(held, capacity)
    assert len(actual) == 2, f"one read retained per-layer FP32 pairs instead of a bounded pair: {len(actual)}"
    assert sum(actual.values()) == 2 * capacity * 4 * 192 * 4


def test_standalone_gather_outputs_remain_independent():
    held: list[object] = []
    read, keys, values, _ = make_read(1152, held)
    values.fill_(1)
    first = gather_dense_draft_kv(read, keys, values, output_dtype=torch.float32)[1]
    values.fill_(2)
    second = gather_dense_draft_kv(read, keys, values, output_dtype=torch.float32)[1]
    torch.npu.synchronize()
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    torch.testing.assert_close(first, torch.ones_like(first), atol=0, rtol=0)
    torch.testing.assert_close(second, torch.full_like(second, 2), atol=0, rtol=0)


def test_queued_attention_consumes_before_same_stream_overwrite():
    held: list[object] = []
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        read, keys, values, query = make_read(1152, held)
        dense_draft_attention(read, query, keys, values, 1.0)
        left = torch.full((4096, 4096), 1 / 4096, dtype=torch.bfloat16, device="npu")
        right = torch.eye(4096, dtype=torch.bfloat16, device="npu")
        scratch = torch.empty_like(left)
    stream.synchronize()
    try:
        with torch.npu.stream(stream):
            for _ in range(128):
                torch.mm(left, right, out=scratch)
                left, scratch = scratch, left
            values.fill_(1)
            first = dense_draft_attention(read, query, keys, values, 1.0)
            reader = torch.npu.Event()
            reader.record(stream)
            if reader.query():
                pytest.skip("bounded real reader completed before pending-write boundary")
            values.fill_(2)
            second = dense_draft_attention(read, query, keys, values, 1.0)
        # Only this test's cleanup waits. Product reuse relies on the queued
        # same-stream read-before-write order, never a manufactured completion.
        stream.synchronize()
        torch.testing.assert_close(first.cpu(), torch.ones_like(query.cpu()), **TOLERANCE)
        torch.testing.assert_close(second.cpu(), torch.full_like(query.cpu(), 2), **TOLERANCE)
        assert len(buffers(held, 1152)) == 2
    finally:
        stream.synchronize()
