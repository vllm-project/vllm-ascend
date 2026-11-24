# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact address checks for strided MLA cache reads on the NPU."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.mla_read_slots import build_mla_read_slots


def _metadata(lengths, offsets, page_size, dtype, strided):
    # Leave room for a valid negative Python index and cross-page offsets.
    columns = max(8, (max(lengths, default=0) + page_size - 1) // page_size + 4)
    step = 2 if strided else 1
    table_cpu = torch.arange(len(lengths) * columns, dtype=dtype).reshape(len(lengths), columns) % 64
    table_cpu[0, 0] = -1
    table_storage = torch.empty((len(lengths), columns * step), dtype=dtype, device="npu")
    table = table_storage[:, ::step]
    table.copy_(table_cpu)
    lengths_storage = torch.empty((len(lengths) * step,), dtype=dtype, device="npu")
    lengths_npu = lengths_storage[::step]
    lengths_npu.copy_(torch.tensor(lengths, dtype=dtype))
    offsets_npu = None
    if offsets is not None:
        offsets_storage = torch.empty((len(lengths) * step,), dtype=dtype, device="npu")
        offsets_npu = offsets_storage[::step]
        offsets_npu.copy_(torch.tensor(offsets, dtype=dtype))
    return table_cpu.tolist(), table, lengths_npu, offsets_npu


def _scalar_reference(table, lengths, offsets, page_size):
    result = []
    for request, length in enumerate(lengths):
        start = offsets[request] if offsets is not None else 0
        for token in range(length):
            column, in_page = divmod(start + token, page_size)
            result.append(table[request][column] * page_size + in_page)
    return result


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "lengths,offsets,page_size",
    [
        ([0], None, 128),
        ([1], [0], 128),
        ([129], [127], 128),
        ([0, 5], [0, -1], 128),
        ([5, 0], [3, 0], 128),
        ([127, 129], [1, 127], 128),
        ([3, 4], [3, -1], 3),
        ([9], [1], 1),
        ([3], [4], 512),
        ([3, 0, 11], [1, 0, 2], 384),
        ([0, 31, 2, 5, 0, 128, 89, 2], [0] * 8, 128),
        ([9], [1], 513),
    ],
)
def test_mla_read_slots_matches_independent_reference(lengths, offsets, page_size, dtype, strided):
    table_values, table, seq_lens, seq_offsets = _metadata(lengths, offsets, page_size, dtype, strided)
    actual = build_mla_read_slots(table, seq_lens, seq_offsets, sum(lengths), page_size)
    assert actual is not None
    assert actual.dtype == torch.int64
    assert actual.shape == (sum(lengths),)
    assert actual.cpu().tolist() == _scalar_reference(table_values, lengths, offsets, page_size)


def test_mla_read_slots_falls_back_for_large_batch_and_unsupported_dtype():
    table = torch.zeros((9, 8), dtype=torch.int32, device="npu")
    lengths = torch.ones(9, dtype=torch.int32, device="npu")
    assert build_mla_read_slots(table, lengths, None, 9, 128) is None
    assert build_mla_read_slots(table[:1].float(), lengths[:1], None, 1, 128) is None


@pytest.mark.parametrize(
    "lengths,offsets,replays",
    [
        ([257], [1], [([257], [127]), ([257], [-1])]),
        ([129, 128], [1, 127], [([0, 257], [0, -1]), ([257, 0], [127, 0]), ([1, 256], [-1, 1])]),
        (
            [0, 31, 2, 5, 0, 128, 89, 2],
            [0] * 8,
            [([31, 2, 5, 0, 128, 89, 2, 0], [1] * 8), ([2, 0, 31, 2, 5, 0, 128, 89], [-1] * 8)],
        ),
    ],
)
def test_mla_read_slots_graph_replay_uses_current_metadata(lengths, offsets, replays):
    table_values, table, seq_lens, seq_offsets = _metadata(lengths, offsets, 128, torch.int32, True)
    assert seq_offsets is not None
    for _ in range(2):
        build_mla_read_slots(table, seq_lens, seq_offsets, sum(lengths), 128)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        slots = build_mla_read_slots(table, seq_lens, seq_offsets, sum(lengths), 128)
    for new_lengths, new_offsets in replays:
        seq_lens.copy_(torch.tensor(new_lengths, dtype=torch.int32))
        seq_offsets.copy_(torch.tensor(new_offsets, dtype=torch.int32))
        table_values = [[(value + 7) % 64 for value in row] for row in table_values]
        table.copy_(torch.tensor(table_values, dtype=torch.int32))
        graph.replay()
        torch.npu.synchronize()
        assert slots.cpu().tolist() == _scalar_reference(table_values, new_lengths, new_offsets, 128)


def test_mla_read_slots_graph_replay_rejects_out_of_range_page():
    table = torch.zeros((1, 2), dtype=torch.int32, device="npu")
    lengths = torch.tensor([2], dtype=torch.int32, device="npu")
    offsets = torch.tensor([0], dtype=torch.int32, device="npu")
    build_mla_read_slots(table, lengths, offsets, 2, 128)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        slots = build_mla_read_slots(table, lengths, offsets, 2, 128)
    assert slots is not None
    # First page remains valid; only the final page is outside the table.
    offsets.fill_(255)
    with pytest.raises((IndexError, RuntimeError)):
        graph.replay()
        torch.npu.synchronize()
