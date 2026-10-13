# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import pytest
import torch

from vllm_ascend.ops.triton.compute_slot_mapping import (
    _compute_slot_mapping_fused_groups_adaptive_kernel,
    _compute_slot_mapping_fused_groups_kernel,
    compute_slot_mapping_fused_groups,
)


def _make_fused_case(lengths, circular):
    block_sizes = [16, 32, 64, 128, 256, 512]
    starts = list(itertools.accumulate(lengths, initial=0))
    # Start just before a block boundary, and exercise negative circular positions
    # separately (ordinary groups require non-negative positions).
    positions = [position for length in lengths for position in range(15, 15 + length)]
    if circular == "negative":
        positions = [-1 if i % 5 == 0 else p for i, p in enumerate(positions)]
    num_tokens = len(positions)
    capacity = num_tokens + 1031
    query = torch.tensor(starts, dtype=torch.int32, device="npu")
    pos = torch.tensor(positions, dtype=torch.int64, device="npu")
    tables = []
    outputs = []
    references = []
    for group, block_size in enumerate(block_sizes):
        stride = (max(positions, default=0) // block_size) + 2 + group
        table = [[1 + group * 10000 + req * stride + block for block in range(stride)] for req in range(len(lengths))]
        tables.append(torch.tensor(table, dtype=torch.int32, device="npu"))
        # Extra storage catches writes past max_num_tokens; each group owns a
        # distinct allocation, not a row of one contiguous output tensor.
        outputs.append(torch.full((capacity + 7,), 123456, dtype=torch.int32, device="npu"))
        expected = [-1] * capacity + [123456] * 7
        for req, (start, end) in enumerate(zip(starts, starts[1:])):
            for token in range(start, end):
                position = positions[token]
                is_circular = circular == "negative" or (circular and group % 2)
                if is_circular and position < 0:
                    continue
                block = 0 if is_circular else position // block_size
                expected[token] = table[req][block] * block_size + position % block_size
        references.append(torch.tensor(expected, dtype=torch.int32))
    addresses = torch.tensor([t.data_ptr() for t in tables], dtype=torch.uint64, device="npu")
    output_addresses = torch.tensor([t.data_ptr() for t in outputs], dtype=torch.uint64, device="npu")
    strides = torch.tensor([t.stride(0) for t in tables], dtype=torch.int64, device="npu")
    sizes = torch.tensor(block_sizes, dtype=torch.int32, device="npu")
    flags = (
        torch.tensor(
            [circular == "negative" or bool(g % 2) for g in range(len(tables))], dtype=torch.int32, device="npu"
        )
        if circular
        else None
    )

    def launch():
        compute_slot_mapping_fused_groups(
            len(tables),
            len(lengths),
            num_tokens,
            capacity,
            query,
            pos,
            addresses,
            output_addresses,
            strides,
            sizes,
            min(block_sizes),
            pad_id=-1,
            is_circular_ptr=flags,
        )

    return launch, outputs, references


@pytest.mark.parametrize(
    "lengths",
    [
        [1] * 64,
        [17] * 7,
        [4097] * 3,
        [8193],
        [1, 0, 33, 2049],
        [0] * 3,
        [1, 0, 33, 1025],
        [1],
        [1537],
        [2048],
        [3072],
        [4097] * 2,
    ],
)
@pytest.mark.parametrize("circular", [False, True, "negative"])
@pytest.mark.parametrize("graph", [False, True])
def test_fused_slot_mapping(lengths, circular, graph):
    """Check both grids, independent group pointers, tails and padding exactly."""
    launch, outputs, references = _make_fused_case(lengths, circular)
    launch()
    if graph:
        torch.npu.synchronize()
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            launch()
        for output in outputs:
            output.fill_(123456)
        captured.replay()
    for actual, expected in zip(outputs, references):
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("tokens_per_request", [1, 17, 33, 65, 129, 257, 513, 4097])
@pytest.mark.parametrize("circular", [False, True])
def test_fused_request_count_reuses_compilation(tokens_per_request, circular, monkeypatch):
    """Changing requests within one launch configuration must hit the JIT cache."""
    kernel = (
        _compute_slot_mapping_fused_groups_adaptive_kernel
        if tokens_per_request <= 512
        else _compute_slot_mapping_fused_groups_kernel
    )
    compile_events = []
    original_compile = kernel._do_compile

    def record_compile(*args, **kwargs):
        compile_events.append(args[0])
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(kernel, "_do_compile", record_compile)
    test_fused_slot_mapping([tokens_per_request] * 3, circular, False)
    first_count = len(compile_events)
    for num_reqs in [4, 7, 16, 64]:
        test_fused_slot_mapping([tokens_per_request] * num_reqs, circular, False)
        assert len(compile_events) == first_count, (tokens_per_request, num_reqs, compile_events)
