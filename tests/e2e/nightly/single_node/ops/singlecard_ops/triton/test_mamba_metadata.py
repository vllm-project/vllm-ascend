# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch

from vllm_ascend.models.glm5next.ops.metadata import aligned_mamba_block_table

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Mamba metadata tests require an Ascend NPU",
)


def _torch_reference(table, lengths, block_size, width):
    start = ((lengths - 1) // block_size).clamp_min(0)
    columns = start[:, None] + torch.arange(width, dtype=torch.int32, device=table.device)
    return torch.gather(table, 1, columns.to(torch.int64))


@pytest.mark.parametrize("num_requests", [0, 1, 7, 128])
@pytest.mark.parametrize("num_speculative_blocks", [0, 1, 3, 5])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_aligned_mamba_block_table(num_requests, num_speculative_blocks, dtype, strided):
    table = torch.arange(num_requests * 32, dtype=dtype, device="npu").reshape(num_requests, 32)
    lengths = torch.tensor([0, 1, 16, 17, 32, 33, 48] * 19, dtype=dtype, device="npu")[:num_requests]
    if strided:
        table = table[:, ::2]
        lengths = torch.stack((lengths, lengths), dim=1)[:, 0]
    expected = _torch_reference(table, lengths, 16, num_speculative_blocks + 1)

    actual = aligned_mamba_block_table(table, lengths, 16, num_speculative_blocks)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_aligned_mamba_block_table_graph_replay_uses_live_seq_lens():
    table = torch.arange(32, dtype=torch.int32, device="npu").reshape(2, 16)
    lengths = torch.tensor([1, 17], dtype=torch.int32, device="npu")
    aligned_mamba_block_table(table, lengths, 16, 3)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = aligned_mamba_block_table(table, lengths, 16, 3)
    lengths.copy_(torch.tensor([33, 0], dtype=torch.int32, device="npu"))
    table.add_(100)
    graph.replay()
    expected = _torch_reference(table, lengths, 16, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_requests", [1, 8, 128])
def test_aligned_mamba_block_table_benchmark(num_requests, record_property):
    table = torch.arange(num_requests * 32, dtype=torch.int32, device="npu").reshape(num_requests, 32)
    lengths = torch.full((num_requests,), 17, dtype=torch.int32, device="npu")

    def measure(operation):
        # Include host submission overhead: these tiny kernels otherwise leave
        # the decode stream waiting for Python to submit the next operation.
        for _ in range(10):
            operation()
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            operation()
        torch.npu.synchronize()
        return (time.perf_counter() - start) * 1000 / 100

    baseline_ms = measure(lambda: _torch_reference(table, lengths, 16, 4))
    fused_ms = measure(lambda: aligned_mamba_block_table(table, lengths, 16, 3))
    record_property("torch_metadata_ms", baseline_ms)
    record_property("fused_metadata_ms", fused_ms)
