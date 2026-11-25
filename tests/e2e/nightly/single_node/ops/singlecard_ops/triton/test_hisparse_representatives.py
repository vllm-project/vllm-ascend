# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact int64 semantics, masked tails, contention and graph replay."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.hisparse import (
    _scatter_min_representatives_kernel,
    scatter_min_representatives,
)


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1])
def test_representative_min_int64_extremes(value):
    index = torch.zeros(257, dtype=torch.int64, device="npu")
    source = torch.full((257,), value, dtype=torch.int64, device="npu")
    out = torch.tensor([2**63 - 1, -1], dtype=torch.int64, device="npu")
    scatter_min_representatives(out, index, source)
    torch.testing.assert_close(out.cpu(), torch.tensor([value, -1], dtype=torch.int64), rtol=0, atol=0)


@pytest.mark.parametrize(
    "rows,pattern,offset",
    [
        (0, "unique", 0),
        (1, "unique", 0),
        (255, "repeated", 1),
        (256, "unique", 0),
        (257, "unique", 1),
        (257, "one", 0),
        (8193, "repeated", 0),
        (32768, "one", 1),
    ],
)
def test_representative_min_exact(rows, pattern, offset):
    generator = torch.Generator().manual_seed(42)
    capacity = max(rows, 1) if pattern == "unique" else 65
    ids = (
        torch.arange(rows, dtype=torch.int64)
        if pattern == "unique"
        else torch.randint(1 if pattern == "one" else capacity, (rows,), generator=generator, dtype=torch.int64)
    )
    # Include negative values and values outside int32; integer equality is required.
    values = torch.randint(-(2**40), 2**40, (rows,), generator=generator, dtype=torch.int64)
    initial = torch.full((capacity,), 2**50, dtype=torch.int64)
    initial[-1] = -(2**45)  # Existing output participates in amin; holes stay unchanged.
    expected = initial.clone().scatter_reduce_(0, ids, values, reduce="amin", include_self=True)

    def on_device_with_offset(cpu):
        storage = torch.empty(cpu.numel() + offset, dtype=torch.int64, device="npu")
        result = storage[offset:]
        result.copy_(cpu)
        return result

    out, index, source = map(on_device_with_offset, (initial, ids, values))
    assert scatter_min_representatives(out, index, source) is out
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_representatives_graph_replay_reads_current_inputs():
    rows, capacity = 513, 65
    index = (torch.arange(rows, dtype=torch.int64) % capacity).npu()
    source = torch.arange(rows, dtype=torch.int64, device="npu")
    out = torch.full((capacity,), 2**50, dtype=torch.int64, device="npu")
    scatter_min_representatives(out, index, source)  # Compile before capture.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        scatter_min_representatives(out, index, source)
    for shift in (17, -1000):
        source.add_(shift)
        out.fill_(2**50)
        graph.replay()
        torch.npu.synchronize()
        expected = torch.full((capacity,), 2**50, dtype=torch.int64)
        expected.scatter_reduce_(0, index.cpu(), source.cpu(), reduce="amin", include_self=True)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    graph.reset()


def test_runtime_length_and_pointer_alignment_do_not_specialize():
    params = {p.name: p for p in _scatter_min_representatives_kernel.params}
    assert not params["num_rows"].is_constexpr
    assert params["num_rows"].do_not_specialize
    assert params["BLOCK_SIZE"].is_constexpr
    for name in ("index_ptr", "source_ptr", "representatives_ptr"):
        assert params[name].do_not_specialize_on_alignment
