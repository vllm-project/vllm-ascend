# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.fused_moe.moe_utils import cumsum_group_list
from vllm_ascend.utils import enable_custom_op


def independent_scan(x):
    result = x.clone()
    step = 1
    while step < x.numel():
        result[step:] = result[step:] + result[:-step]
        step *= 2
    return result


@pytest.fixture(scope="module", autouse=True)
def require_extension():
    assert enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "moe_counts_cumsum")


@pytest.mark.parametrize(
    ("n", "offset"),
    [
        (1, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (7, 0),
        (8, 0),
        (9, 1),
        (4095, 0),
        (4096, 0),
    ],
)
def test_exact_int64_nondefault_stream(n, offset):
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        base = torch.arange(n + offset, dtype=torch.int64, device="npu") + (1 << 40)
        x = base[offset:] if offset else base
        assert x.is_contiguous() and x.storage_offset() == offset
        if offset:
            assert x.data_ptr() % 32 != 0
        before = x.clone()
        expected = independent_scan(x)
        result = torch.ops._C_ascend.moe_counts_cumsum(x)
    stream.synchronize()
    assert result.dtype == torch.int64 and result.shape == x.shape
    assert result.data_ptr() != x.data_ptr()
    assert torch.equal(result, expected)
    assert torch.equal(x, before)


@pytest.mark.parametrize("n", [32, 256, 512])
def test_graph_replay_uses_updated_counts(n):
    x = torch.ones(n, dtype=torch.int64, device="npu")
    for _ in range(3):
        torch.ops._C_ascend.moe_counts_cumsum(x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = torch.ops._C_ascend.moe_counts_cumsum(x)
    for value in [2, 7, 1 << 40, (1 << 53) + 17]:
        x.fill_(value)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(result, independent_scan(x))


def test_int64_extreme_and_zero_prefixes():
    max_int64 = (1 << 63) - 1
    zeros = torch.zeros(256, dtype=torch.int64, device="npu")
    assert torch.equal(torch.ops._C_ascend.moe_counts_cumsum(zeros), zeros)

    at_end = torch.zeros(256, dtype=torch.int64, device="npu")
    at_end[-1] = max_int64
    assert torch.equal(torch.ops._C_ascend.moe_counts_cumsum(at_end), independent_scan(at_end))

    # Values above FP64 exact integer range (2**53) and sums reaching INT64_MAX.
    above_fp64 = torch.ones(64, dtype=torch.int64, device="npu")
    above_fp64[0] = (1 << 53) + 17
    assert torch.equal(torch.ops._C_ascend.moe_counts_cumsum(above_fp64), independent_scan(above_fp64))

    q, r = divmod(max_int64, 257)
    sum_max = torch.full((257,), q, dtype=torch.int64, device="npu")
    sum_max[-1] += r
    assert torch.equal(torch.ops._C_ascend.moe_counts_cumsum(sum_max), independent_scan(sum_max))


def test_temporary_prefix_under_async_allocator_pressure():
    # The model passes prefix directly to the next operator and releases the
    # Python tensor. Retaining results until synchronize hides that lifecycle.
    x = torch.arange(32, dtype=torch.int64, device="npu")
    expected = independent_scan(x) + 1
    results = []
    for _ in range(2048):
        results.append(torch.ops._C_ascend.moe_counts_cumsum(x) + 1)
    torch.npu.synchronize()
    assert all(torch.equal(result, expected) for result in results)


def test_cross_stream_producer_and_temporary_consumer():
    main = torch.npu.current_stream()
    auxiliary = torch.npu.Stream()
    results = []
    for value in range(128):
        with torch.npu.stream(auxiliary):
            x = torch.arange(32, dtype=torch.int64, device="npu") + value
        main.wait_stream(auxiliary)
        x.record_stream(main)
        results.append(torch.ops._C_ascend.moe_counts_cumsum(x) + 1)
    torch.npu.synchronize()
    base = torch.arange(32, dtype=torch.int64, device="npu")
    for value, result in enumerate(results):
        assert torch.equal(result, independent_scan(base + value) + 1)


def test_wrapper_dispatches_to_custom_op():
    x = torch.arange(32, dtype=torch.int64, device="npu") + (1 << 40)
    orig = torch.ops._C_ascend.moe_counts_cumsum
    with patch.object(torch.ops._C_ascend, "moe_counts_cumsum", wraps=orig) as spy:
        result = cumsum_group_list(x, 1, 0)
    spy.assert_called_once_with(x)
    assert torch.equal(result, independent_scan(x))


@pytest.mark.parametrize(
    "shape,dtype",
    [((0,), torch.int64), ((4097,), torch.int64), ((32,), torch.int32), ((2, 16), torch.int64)],
)
def test_custom_entry_rejects_unsupported_metadata(shape, dtype):
    x = torch.zeros(shape, device="npu", dtype=dtype)
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.moe_counts_cumsum(x)


def test_custom_entry_rejects_noncontiguous_counts():
    x = torch.arange(64, device="npu", dtype=torch.int64)[::2]
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.moe_counts_cumsum(x)


def test_oversized_counts_fall_back_to_native():
    x = torch.arange(4097, dtype=torch.int64, device="npu") + (1 << 40)
    assert torch.equal(cumsum_group_list(x, 1, 0), independent_scan(x))
