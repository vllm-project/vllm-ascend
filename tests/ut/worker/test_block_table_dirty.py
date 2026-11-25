# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm_ascend import envs
from vllm_ascend.worker import block_table as bt


class HostDeviceBuffer:
    """Separate CPU tensors model the persistent device table for CPU tests."""

    def __init__(self, *size, dtype, **kwargs):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = torch.zeros_like(self.cpu)
        self.copies = []

    def copy_to_gpu(self, size=None):
        self.copies.append(size)
        self.gpu[:size].copy_(self.cpu[:size])

    def fill_(self, value):
        self.cpu.fill_(value)
        self.gpu.fill_(value)


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(bt, "CpuGpuBuffer", HostDeviceBuffer)
    monkeypatch.setattr(bt, "get_dcp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))
    monkeypatch.setattr(bt, "get_decode_context_model_parallel_world_size", lambda: 1)
    npu = MagicMock()
    npu.Event.side_effect = lambda: MagicMock()
    monkeypatch.setattr(torch, "npu", npu, raising=False)

    def scatter(packed, metadata, dst, count):
        for row, begin, length, offset in metadata[:count].tolist():
            dst[row, begin : begin + length] = packed[offset : offset + length]

    scatter_mock = MagicMock(side_effect=scatter)
    monkeypatch.setattr(bt, "scatter_block_table", scatter_mock)
    return scatter_mock


def make_table(cls=bt.OptimizedBlockTable, hybrid=False):
    return cls(128, 4, 32, 64, False, torch.device("cpu"), kernel_sizes=[32] if hybrid else [128])


def assert_valid_prefixes(actual, reference, num_reqs):
    np.testing.assert_array_equal(actual.num_blocks_per_row, reference.num_blocks_per_row)
    for row in range(num_reqs):
        size = int(reference.num_blocks_per_row[row])
        torch.testing.assert_close(actual.block_table.gpu[row, :size], reference.block_table.gpu[row, :size])


@pytest.mark.parametrize("hybrid", [False, True])
def test_random_mutations_match_full_copy(backend, hybrid):
    optimized, reference = make_table(hybrid=hybrid), make_table(bt.BlockTable, hybrid=hybrid)
    rng = np.random.default_rng(42)
    device_ptr = optimized.block_table.gpu.data_ptr()
    for step in range(300):
        # Several mutations between commits exercise dirty-range coalescing.
        for _ in range(3):
            row, other = (int(value) for value in rng.integers(0, 4, size=2))
            op = rng.choice(["add", "append", "move", "swap", "clear"])
            ids = rng.integers(1, 100, size=int(rng.integers(0, 4))).tolist()
            for table in (optimized, reference):
                if op == "add":
                    table.add_row(ids, row)
                elif op == "append" and table.num_blocks_per_row[row] < 16:
                    table.append_row(ids, row)
                elif op == "move":
                    table.move_row(row, other)
                elif op == "swap":
                    table.swap_row(row, other)
                elif op == "clear":
                    table.clear_row(row)
        num_reqs = int(rng.integers(0, 5))
        if step % 17 == 0:
            optimized.commit_block_table(4)  # Dummy/graph full-copy interleaving.
        optimized.commit_dirty_ranges(num_reqs)
        reference.commit_block_table(num_reqs)
        assert_valid_prefixes(optimized, reference, num_reqs)
    optimized.commit_dirty_ranges(4)
    reference.commit_block_table(4)
    assert_valid_prefixes(optimized, reference, 4)
    assert optimized.block_table.gpu.data_ptr() == device_ptr


def test_no_dirty_skips_transfer_and_append_only_uploads_delta(backend):
    table = make_table()
    table.add_row([10, 11], 0)
    table.commit_dirty_ranges(1)
    backend.reset_mock()
    slot = table._dirty_commit_buffer_index
    table.commit_dirty_ranges(1)
    backend.assert_not_called()
    assert table._dirty_commit_buffer_index == slot
    table.append_row([12], 0)
    table.commit_dirty_ranges(1)
    packed, metadata, _, count = backend.call_args.args
    assert count == 1
    assert packed[0].item() == 12
    assert metadata[0].tolist() == [0, 2, 1, 0]
    assert table._dirty_pack_buffers[slot].copies == [1]
    assert table._dirty_metadata_buffers[slot].copies == [1]
    assert table.block_table.copies == []


def test_events_protect_reuse_growth_and_clear(backend):
    table = make_table()
    table.add_row([1, 2], 0)
    table.commit_dirty_ranges(1)
    first = table._dirty_buffer_events[0]
    table.add_row([3, 4], 0)
    table.commit_dirty_ranges(1)
    second = table._dirty_buffer_events[1]
    first.synchronize.assert_not_called()
    table.add_row([5, 6], 0)
    table.commit_dirty_ranges(1)
    first.synchronize.assert_called_once()
    second.synchronize.assert_not_called()
    events = list(table._dirty_buffer_events)
    table.add_row(list(range(9)), 0)
    table.commit_dirty_ranges(1)
    for event in events:
        assert event.synchronize.called
    assert table._dirty_pack_capacity >= 9
    latest = [event for event in table._dirty_buffer_events if event is not None]
    table.clear()
    for event in latest:
        event.synchronize.assert_called_once()
    backend.reset_mock()
    table.commit_dirty_ranges(4)
    backend.assert_not_called()
    assert torch.count_nonzero(table.block_table.gpu) == 0


def test_inactive_rows_remain_dirty_until_activated(backend):
    table = make_table()
    table.add_row([7, 8], 3)
    table.commit_dirty_ranges(1)
    backend.assert_not_called()
    table.commit_dirty_ranges(4)
    assert table.block_table.gpu[3, :2].tolist() == [7, 8]


def test_delayed_dma_does_not_observe_overwritten_staging(backend, monkeypatch):
    pending = []
    observed = []
    expected = []

    class DelayedEvent:
        def record(self, stream):
            self.done = False
            pending.append(lambda: setattr(self, "done", True))

        def synchronize(self):
            while not self.done:
                pending.pop(0)()

    def delayed_copy(buffer, size=None):
        # Read the pinned source only when the simulated stream executes DMA.
        pending.append(lambda: buffer.gpu[:size].copy_(buffer.cpu[:size]))

    scatter = backend.side_effect

    def delayed_scatter(packed, metadata, dst, count):
        def execute():
            scatter(packed, metadata, dst, count)
            observed.append(dst[0].clone())

        pending.append(execute)

    monkeypatch.setattr(HostDeviceBuffer, "copy_to_gpu", delayed_copy)
    monkeypatch.setattr(torch.npu, "Event", DelayedEvent)
    backend.side_effect = delayed_scatter
    table = make_table()
    for step in range(20):
        size = min(step + 1, 16)  # Includes reuse and multiple reallocations.
        table.add_row([step + 1] * size, 0)
        expected.append((size, table.block_table.cpu[0, :size].clone()))
        table.commit_dirty_ranges(1)
    table._synchronize_dirty_buffers()
    assert len(observed) == len(expected)
    for result, (size, reference) in zip(observed, expected):
        torch.testing.assert_close(result[:size], reference)


@pytest.mark.parametrize("value,disabled", [(None, False), ("0", False), ("1", True)])
def test_multigroup_switch_isolates_original_class(backend, monkeypatch, value, disabled):
    name = "VLLM_ASCEND_BLOCK_TABLE_NO_COMMIT_OPTIMIZE"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    tables = bt.MultiGroupBlockTable(4, 4096, 64, False, torch.device("cpu"), [128, 256])
    assert all(type(table) is (bt.BlockTable if disabled else bt.OptimizedBlockTable) for table in tables.block_tables)
    tables.add_row(([10], [20]), 0)
    tables.commit_runtime(1)
    assert tables[0].block_table.gpu[0, 0].item() == 10
    assert tables[1].block_table.gpu[0, 0].item() == 20
    if disabled:
        backend.assert_not_called()
        assert not hasattr(tables[0], "dirty_begin")
        torch.npu.Event.assert_not_called()
    else:
        assert backend.call_count == 2
    tables.commit_block_table(4)
    assert all(table.block_table.copies[-1] == 4 for table in tables.block_tables)


@pytest.mark.parametrize("value", ["2", "true", "", "-1"])
def test_invalid_switch_rejected(monkeypatch, value):
    monkeypatch.setenv("VLLM_ASCEND_BLOCK_TABLE_NO_COMMIT_OPTIMIZE", value)
    with pytest.raises(ValueError, match="must be '0' or '1'"):
        envs.env_variables["VLLM_ASCEND_BLOCK_TABLE_NO_COMMIT_OPTIMIZE"]()
