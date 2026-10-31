# SPDX-License-Identifier: Apache-2.0
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


@pytest.mark.parametrize("n", [1, 7, 8, 9, 31, 32, 33, 255, 256, 257, 1023, 1024, 4095, 4096])
def test_exact_int64_nondefault_stream(n):
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        x = torch.arange(n, dtype=torch.int64, device="npu") + (1 << 40)
        before = x.clone()
        expected = independent_scan(x)
        result = cumsum_group_list(x, 1, 0)
    stream.synchronize()
    assert result.dtype == torch.int64 and result.shape == x.shape
    assert result.data_ptr() != x.data_ptr()
    assert torch.equal(result, expected)
    assert torch.equal(x, before)


@pytest.mark.parametrize("n", [32, 256])
def test_graph_replay_uses_updated_counts(n):
    x = torch.ones(n, dtype=torch.int64, device="npu")
    for _ in range(3):
        cumsum_group_list(x, 1, 0)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = cumsum_group_list(x, 1, 0)
    for value in [2, 7, 1 << 40]:
        x.fill_(value)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(result, independent_scan(x))


def test_int64_maximum_prefix():
    x = torch.zeros(256, dtype=torch.int64, device="npu")
    x[-1] = (1 << 63) - 1
    assert torch.equal(cumsum_group_list(x, 1, 0), independent_scan(x))


def test_temporary_prefix_under_async_allocator_pressure():
    # The model passes prefix directly to the next operator and releases the
    # Python tensor. Retaining results until synchronize hides that lifecycle.
    x = torch.arange(32, dtype=torch.int64, device="npu")
    expected = independent_scan(x) + 1
    results = []
    for _ in range(2048):
        results.append(cumsum_group_list(x, 1, 0) + 1)
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
        results.append(cumsum_group_list(x, 1, 0) + 1)
    torch.npu.synchronize()
    base = torch.arange(32, dtype=torch.int64, device="npu")
    for value, result in enumerate(results):
        assert torch.equal(result, independent_scan(base + value) + 1)


@pytest.mark.parametrize(
    "shape,dtype",
    [((0,), torch.int64), ((4097,), torch.int64), ((32,), torch.int32), ((2, 16), torch.int64)],
)
def test_custom_entry_rejects_unsupported_metadata(shape, dtype):
    x = torch.zeros(shape, device="npu", dtype=dtype)
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.moe_counts_cumsum(x)


def test_oversized_counts_fall_back_to_native():
    x = torch.arange(4097, dtype=torch.int64, device="npu") + (1 << 40)
    assert torch.equal(cumsum_group_list(x, 1, 0), independent_scan(x))
