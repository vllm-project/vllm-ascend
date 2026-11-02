# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse invalidation: upstream regression and exact state-table checks.

NPU: pytest --confcutdir=tests/e2e/pull_request/one_card <this file>
CPU software check: TRITON_INTERPRET=1 ... -k 'not npu_'
The interpreter executes the real kernel, bypassing the NPU-only Python wrapper;
it is neither Ascend code-generation nor hardware validation.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

# Load the production file itself without importing unrelated serving operators
# from ops/__init__.py. Nothing is copied, rewritten, extracted or mocked.
_KERNEL_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/ops/triton/hisparse.py"
_SPEC = importlib.util.spec_from_file_location("hisparse_invalidation_under_test", _KERNEL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
hisparse = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hisparse)


@pytest.fixture(scope="module")
def device():
    if os.environ.get("TRITON_INTERPRET") == "1":
        return torch.device("cpu")
    import torch_npu  # noqa: F401

    assert torch.npu.is_available(), "NPU tests require a released, visible Ascend device"
    return torch.device("npu", torch.npu.current_device())


def invoke(ptrs, states, requests, slots, num_state_rows, region_stride):
    if ptrs.device.type == "cpu":
        assert os.environ.get("TRITON_INTERPRET") == "1"
        if not (ptrs.numel() and slots.numel() and num_state_rows and region_stride):
            return
        hisparse.hisparse_invalidate_written_slots_kernel[
            (hisparse.triton.cdiv(region_stride, 256), num_state_rows, ptrs.numel())
        ](
            ptrs,
            states,
            requests,
            slots,
            slots.numel(),
            states.numel(),
            REGION_STRIDE=region_stride,
            BLOCK_SIZE=256,
        )
    else:
        hisparse.hisparse_invalidate_written_slots(ptrs, states, requests, slots, num_state_rows, region_stride)


def reference(tables, states, requests, slots):
    result = [table.cpu().clone() for table in tables]
    for req, slot in zip(requests.cpu().tolist(), slots.cpu().tolist()):
        if not 0 <= req < states.numel() or not 0 <= slot <= 2**31 - 1:
            continue
        state = int(states.cpu()[req])
        if not 0 <= state < result[0].shape[0]:
            continue
        for table in result:
            table[state].masked_fill_(table[state] == slot, -1)
    return result


def pointers(tables, device):
    return torch.tensor([table.data_ptr() for table in tables], dtype=torch.int64, device=device)


def assert_tables(tables, expected):
    for table, want in zip(tables, expected):
        torch.testing.assert_close(table.cpu(), want.cpu(), rtol=0, atol=0)


def test_hisparse_invalidation_covers_every_index_group(device):
    """Port of upstream test_sparse_mla_backends.py at d5301a4f5487:3344.

    Keep its tables, remapped request, padded token, unlisted group and expected
    updates. Replace only the CUDA/runtime entry with the six-argument NPU op.
    """
    tables = [
        torch.tensor(
            [[5, 7, 9, 7, 1, 2, 3, 4], [7, 1, 2, 3, 7, 7, 7, 7]],
            dtype=torch.int32,
            device=device,
        )
        + offset
        for offset in (0, 0, 100)
    ]
    expected = [table.clone() for table in tables]
    for table in expected[:2]:
        table[1][table[1] == 7] = -1
    invoke(
        pointers(tables[:2], device),
        torch.tensor([1, 0], dtype=torch.int32, device=device),
        torch.tensor([0, -1], dtype=torch.int32, device=device),
        torch.tensor([7, 7], dtype=torch.int64, device=device),
        2,
        8,
    )
    assert_tables(tables, expected)


@pytest.mark.parametrize("region_stride", [1, 7, 8, 255, 256, 257, 2048])
def test_shared_state_multiple_rewrites_and_row_boundaries(device, region_stride):
    generator = torch.Generator().manual_seed(20261102)
    tables = [
        torch.randint(-1, 17, (3, region_stride), generator=generator, dtype=torch.int32).to(device) for _ in range(2)
    ]
    states = torch.tensor([2, 0, 2, -1, 3], dtype=torch.int32, device=device)
    requests = torch.tensor([0, 2, 0, 1, 3, 4, -1, 5, 0, 2, 1], dtype=torch.int32, device=device)
    slots = torch.tensor([3, 4, 5, 3, 3, 3, 3, 3, 6, 4, 16], dtype=torch.int64, device=device)
    expected = reference(tables, states, requests, slots)
    invoke(pointers(tables, device), states, requests, slots, 3, region_stride)
    assert_tables(tables, expected)


def test_int64_slots_are_checked_before_narrowing(device):
    large = 2**31 - 1
    tables = [torch.tensor([[7, 16777217, large, -1], [7, 16777217, large, -1]], dtype=torch.int32, device=device)]
    states = torch.tensor([1, -1, 2, 2147483647], dtype=torch.int32, device=device)
    requests = torch.tensor([0, 0, 0, 0, 0, 0, 1, 2, 3, -2147483648, 2147483647], dtype=torch.int32, device=device)
    slots = torch.tensor(
        [2**32 + 7, 2**31, -(2**63), 2**63 - 1, 16777217, large, 7, 7, 7, 7, 7], dtype=torch.int64, device=device
    )
    expected = reference(tables, states, requests, slots)
    invoke(pointers(tables, device), states, requests, slots, 2, 4)
    assert_tables(tables, expected)


@pytest.mark.parametrize("region_stride", [7, 8, 257])
def test_offset_table_preserves_surrounding_storage(device, region_stride):
    storage = torch.full((2 * region_stride + 2,), 91, dtype=torch.int32, device=device)
    table = storage[1:-1].view(2, region_stride)
    table.fill_(7)
    invoke(
        pointers([table], device),
        torch.tensor([1], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.tensor([7], dtype=torch.int64, device=device),
        2,
        region_stride,
    )
    expected = torch.full((2 * region_stride + 2,), 91, dtype=torch.int32)
    expected[1 : region_stride + 1].fill_(7)
    expected[region_stride + 1 : -1].fill_(-1)
    torch.testing.assert_close(storage.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("empty", ["tokens", "groups", "states", "region", "requests"])
def test_empty_inputs_leave_tables_unchanged(device, empty):
    tables = [torch.full((2, 8), 7, dtype=torch.int32, device=device)]
    ptrs = pointers(tables, device)
    states = torch.tensor([1], dtype=torch.int32, device=device)
    requests = torch.tensor([0], dtype=torch.int32, device=device)
    slots = torch.tensor([7], dtype=torch.int64, device=device)
    if empty == "tokens":
        requests, slots = requests[:0], slots[:0]
    if empty == "groups":
        ptrs = ptrs[:0]
    if empty == "requests":
        states = states[:0]
    invoke(ptrs, states, requests, slots, 0 if empty == "states" else 2, 0 if empty == "region" else 8)
    torch.testing.assert_close(tables[0].cpu(), torch.full((2, 8), 7, dtype=torch.int32), rtol=0, atol=0)


def test_public_wrapper_rejects_cpu_tensors():
    tensors = [torch.empty(0, dtype=dtype) for dtype in (torch.int64, torch.int32, torch.int32, torch.int64)]
    with pytest.raises(RuntimeError, match="same NPU device"):
        hisparse.hisparse_invalidate_written_slots(*tensors, 0, 0)


@pytest.mark.parametrize("invalid", ["dtype", "ndim", "contiguity", "length", "negative_shape", "cpu_tensor"])
def test_npu_input_validation(device, invalid):
    tables = [torch.full((2, 8), 7, dtype=torch.int32, device=device)]
    ptrs = pointers(tables, device)
    states = torch.tensor([1], dtype=torch.int32, device=device)
    requests = torch.tensor([0], dtype=torch.int32, device=device)
    slots = torch.tensor([7], dtype=torch.int64, device=device)
    rows, stride = 2, 8
    if invalid == "dtype":
        slots = slots.to(torch.int32)
    elif invalid == "ndim":
        states = states.view(1, 1)
    elif invalid == "contiguity":
        states = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device=device)[::2]
    elif invalid == "length":
        slots = slots[:0]
    elif invalid == "negative_shape":
        rows = -1
    else:
        slots = slots.cpu()
    with pytest.raises(RuntimeError):
        hisparse.hisparse_invalidate_written_slots(ptrs, states, requests, slots, rows, stride)
    torch.testing.assert_close(tables[0].cpu(), torch.full((2, 8), 7, dtype=torch.int32), rtol=0, atol=0)


def test_npu_graph_replay_reads_changed_inputs_and_table_addresses(device):
    tables = [torch.full((2, 257), slot, dtype=torch.int32, device=device) for slot in (7, 9)]
    ptrs = pointers(tables[:1], device)
    states = torch.tensor([1], dtype=torch.int32, device=device)
    requests = torch.tensor([0, -1], dtype=torch.int32, device=device)
    slots = torch.tensor([7, 7], dtype=torch.int64, device=device)

    def run():
        invoke(ptrs, states, requests, slots, 2, 257)

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for selected, row, slot in ((1, 0, 9), (0, 1, 7)):
        for table, value in zip(tables, (7, 9)):
            table.fill_(value)
        ptrs.copy_(pointers([tables[selected]], device))
        states.fill_(row)
        requests.copy_(torch.tensor([-1, 0], dtype=torch.int32, device=device))
        slots.fill_(slot)
        expected = reference([tables[selected]], states, requests, slots)
        graph.replay()
        assert_tables([tables[selected]], expected)
        other = tables[1 - selected]
        torch.testing.assert_close(
            other.cpu(), torch.full((2, 257), (7, 9)[1 - selected], dtype=torch.int32), rtol=0, atol=0
        )


def test_npu_current_stream_event_order(device):
    tables = [torch.zeros((2, 257), dtype=torch.int32, device=device)]
    ptrs = pointers(tables, device)
    states = torch.tensor([1], dtype=torch.int32, device=device)
    requests = torch.tensor([0, 0], dtype=torch.int32, device=device)
    slots = torch.tensor([7, 9], dtype=torch.int64, device=device)
    producer = torch.npu.Stream()
    producer.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(producer):
        tables[0].fill_(7)
        tables[0][1, 128:].fill_(9)
        invoke(ptrs, states, requests, slots, 2, 257)
        ready = torch.npu.Event()
        ready.record()
    torch.npu.current_stream().wait_event(ready)
    expected = torch.full((2, 257), 7, dtype=torch.int32)
    expected[1].fill_(-1)
    assert_tables(tables, [expected])
