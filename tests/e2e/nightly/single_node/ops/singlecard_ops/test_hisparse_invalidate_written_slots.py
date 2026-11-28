# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse written-slot invalidation accuracy tests on the A2/A3 arch22 kernel."""

from contextlib import contextmanager

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.hisparse.cache_ops import HiSparseNPUOps
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


@pytest.fixture(autouse=True)
def _require_arch22():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    if get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3):
        pytest.skip("requires the arch22 kernel")
    if not hasattr(torch.ops._C_ascend, "hisparse_resolve_residency"):
        import vllm_ascend.vllm_ascend_C  # type: ignore[import-not-found, import-untyped]  # noqa: F401
    assert hasattr(torch.ops._C_ascend, "hisparse_resolve_residency"), "rebuild the arch22 extension"


@contextmanager
def _graph_replay(forward):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        yield graph, stream
    finally:
        torch.npu.synchronize()
        graph.reset()


@pytest.mark.parametrize("hot_capacity", [4, 10240, 32768, 32769])
def test_hisparse_invalidate_written_slots_hot_capacity_graph(hot_capacity):
    dgi = torch.arange(2 * hot_capacity, dtype=torch.int32, device="npu").view(2, hot_capacity)
    group_table = torch.tensor([dgi.data_ptr()], dtype=torch.int64, device=dgi.device)
    initial = dgi.cpu()
    states = torch.tensor([1, 0], dtype=torch.int32, device="npu")
    requests = torch.tensor([0, 1, 1, -1], dtype=torch.int32, device="npu")
    written = torch.tensor([2 * hot_capacity - 1, 0, hot_capacity - 1, hot_capacity], dtype=torch.int64, device="npu")
    expected = initial.clone()
    expected[1, -1] = expected[0, 0] = expected[0, -1] = -1

    def forward():
        HiSparseNPUOps.hisparse_invalidate_written_slots(group_table, states, requests, written, *dgi.shape)

    forward()
    torch.npu.synchronize()
    torch.testing.assert_close(dgi.cpu(), expected)
    with _graph_replay(forward) as (graph, stream):
        with torch.npu.graph(graph, stream=stream):
            forward()
        for _ in range(10):
            dgi.copy_(initial)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(dgi.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("hot_capacity", [1, 7, 2049, 10240])
@pytest.mark.parametrize("write_repeats", [1, 8])
def test_hisparse_invalidate_written_slots_batched_dynamic_graph(dtype, hot_capacity, write_repeats):
    """Duplicate writes and changing states must not race or pollute neighbors."""
    torch.manual_seed(71)
    initial = torch.randint(-1, 128, (3, 4, hot_capacity), dtype=torch.int32)
    initial[:, :, 0] = 2**31 - 1
    if hot_capacity > 1:
        initial[:, :, 1] = 3
    backing = [torch.full((table.numel() + 9,), -73, dtype=torch.int32, device="npu") for table in initial]
    tables = [storage[1:-8].view_as(table) for storage, table in zip(backing, initial)]
    for table, original in zip(tables, initial):
        table.copy_(original)
    groups = torch.tensor([table.data_ptr() for table in tables], dtype=torch.int64, device="npu")
    states = torch.tensor([3, 1, 0, -1], dtype=torch.int32, device="npu")
    requests_cpu = torch.tensor([0, 0, 1, 0, 2, 3, -1, 9], dtype=torch.int32)
    writes_cpu = torch.tensor([3, 3, 2**31 - 1, 5, -1, 8, 9, 10], dtype=dtype)
    if dtype == torch.int64:
        writes_cpu[4] = 2**32 + 3  # Must not wrap to the valid tag 3.
        writes_cpu[-3:] = torch.tensor([2**34 + 3, -(2**34), 2**31], dtype=dtype)
    requests_cpu = requests_cpu.repeat(write_repeats)
    writes_cpu = writes_cpu.repeat(write_repeats)
    requests, writes = requests_cpu.npu(), writes_cpu.npu()

    def forward():
        HiSparseNPUOps.hisparse_invalidate_written_slots(groups, states, requests, writes, 4, hot_capacity)

    with _graph_replay(forward) as (graph, stream):
        with torch.npu.graph(graph, stream=stream):
            forward()
        for state_cpu in (torch.tensor([3, 1, 0, -1]), torch.tensor([0, 0, 2, 8])):
            states.copy_(state_cpu)
            for table, original in zip(tables, initial):
                table.copy_(original)
            graph.replay()
            torch.npu.synchronize()
            expected = initial.clone()
            for request, value in zip(requests_cpu.tolist(), writes_cpu.tolist()):
                if 0 <= request < 4 and 0 <= state_cpu[request] < 4 and 0 <= value <= 2**31 - 1:
                    row = expected[:, state_cpu[request]]
                    row[row == value] = -1
            for storage, table, wanted in zip(backing, tables, expected):
                torch.testing.assert_close(table.cpu(), wanted)
                assert storage[0].item() == -73 and torch.all(storage[-8:].cpu() == -73)


@pytest.mark.parametrize("empty", ["groups", "states", "writes", "state_count", "region"])
@pytest.mark.parametrize("empty_view", [False, True])
def test_hisparse_invalidate_written_slots_empty_inputs(empty, empty_view):
    table = torch.arange(7, dtype=torch.int32, device="npu")[None]
    groups = torch.tensor([table.data_ptr()], dtype=torch.int64, device="npu")
    states = torch.zeros(1, dtype=torch.int32, device="npu")
    requests, written = states.clone(), states.clone()

    def make_empty(tensor):
        return tensor[:0] if empty_view else torch.empty(0, dtype=tensor.dtype, device=tensor.device)

    if empty == "groups":
        groups = make_empty(groups)
    elif empty == "states":
        states = make_empty(states)
    elif empty == "writes":
        requests, written = make_empty(requests), make_empty(written)
    state_count, region = (0 if empty == "state_count" else 1), (0 if empty == "region" else 7)
    invalidate = HiSparseNPUOps.hisparse_invalidate_written_slots
    invalidate(groups, states, requests, written, state_count, region)
    torch.testing.assert_close(table.cpu(), torch.arange(7, dtype=torch.int32)[None])
    # Empty work must not hide an invalid tensor shape.
    with pytest.raises(ValueError, match="group_table"):
        invalidate(groups.view(-1, 1), states, requests, written, state_count, region)


@pytest.mark.parametrize("count", [64, 65])
def test_hisparse_invalidate_written_slots_write_count(count):
    """The shared Triton kernel has no native write-count ceiling."""
    table = torch.arange(count, dtype=torch.int32, device="npu")[None]
    groups = torch.tensor([table.data_ptr()], dtype=torch.int64, device="npu")
    states = torch.zeros(1, dtype=torch.int32, device="npu")
    requests = torch.zeros(count, dtype=torch.int32, device="npu")
    written = torch.arange(count, dtype=torch.int32, device="npu")
    HiSparseNPUOps.hisparse_invalidate_written_slots(groups, states, requests, written, 1, count)
    assert torch.all(table.cpu() == -1)


@pytest.mark.parametrize("num_tokens", [4096, 8192])
def test_hisparse_invalidate_written_slots_prefill_sorted_graph(num_tokens):
    """Large writes use the same request/slot semantics as the CPU oracle."""
    torch.manual_seed(42)
    groups, num_states, hot_capacity = 3, 5, 10240
    initial = torch.randint(-1, 20000, (groups, num_states, hot_capacity), dtype=torch.int32)
    initial[0].fill_(-1)
    initial[1, 1, 0] = 2**31 - 1
    initial[2, 2, 0] = 2**31 - 2
    tables = [table.to("npu") for table in initial]
    pointers = torch.tensor([table.data_ptr() for table in tables], dtype=torch.int64, device="npu")
    states = torch.tensor([2, 1, -1, 4], dtype=torch.int32)
    requests = torch.randint(-1, 5, (num_tokens,), dtype=torch.int32)
    written = torch.randint(-1, 20000, (num_tokens,), dtype=torch.int64)
    requests[:5] = torch.tensor([1, 0, 2, 3, 1])
    written[:5] = torch.tensor([2**31 - 1, 2**31 - 2, 1, 2**34, -(2**34)])
    device_states, device_requests, device_written = (value.to("npu") for value in (states, requests, written))

    def forward():
        HiSparseNPUOps.hisparse_invalidate_written_slots(
            pointers, device_states, device_requests, device_written, num_states, hot_capacity
        )

    def check():
        expected = initial.clone()
        for request, slot in zip(requests.tolist(), written.tolist()):
            if 0 <= request < len(states) and 0 <= states[request] < num_states and 0 <= slot <= 2**31 - 1:
                row = expected[:, states[request]]
                row[row == slot] = -1
        for group, table in enumerate(tables):
            torch.testing.assert_close(table.cpu(), expected[group], rtol=0, atol=0)

    forward()
    check()
    with _graph_replay(forward) as (graph, stream):
        with torch.npu.graph(graph, stream=stream):
            forward()
        for _ in range(3):
            for group, table in enumerate(tables):
                table.copy_(initial[group])
            requests = (requests + 1) % 5 - 1
            device_requests.copy_(requests)
            graph.replay()
            torch.npu.synchronize()
            check()
