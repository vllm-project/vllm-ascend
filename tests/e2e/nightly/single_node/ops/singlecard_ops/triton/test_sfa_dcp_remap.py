# SPDX-License-Identifier: Apache-2.0
"""Two-kernel remap: independent integer oracle and changed graph inputs."""

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton  # noqa: E402


def oracle(indices, rank, ranks=8, interleave=128):
    expected = torch.full_like(indices, -1)
    # CPU int64 arithmetic is independent of NPU vector integer division.
    count = indices.shape[-1]
    for source, target in zip(indices.long().reshape(-1, count), expected.reshape(-1, count)):
        owned = source[(source >= 0) & ((source // interleave) % ranks == rank)]
        target[: owned.numel()] = (owned // (ranks * interleave) * interleave + owned % interleave).to(target.dtype)
    return expected


@pytest.mark.parametrize("rows", [1, 6, 12])
@pytest.mark.parametrize("rank", range(8))
def test_exact_integer_boundaries_and_changed_graph_inputs(rows, rank):
    torch.npu.set_device(0)
    rng = torch.Generator().manual_seed(20260911 + rows + rank)
    base = torch.randint(0, 2**31 - 1, (rows, 2048), generator=rng, dtype=torch.int32)
    boundaries = torch.tensor(
        [-1, 0, 127, 128, 1023, 1024, 2**24 - 1, 2**24, 2**24 + 127, 2**31 - 1],
        dtype=torch.int32,
    )
    device_input = base.npu()
    for _ in range(5):
        remap_sparse_indices_triton(device_input, 8, rank, 128)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = remap_sparse_indices_triton(device_input, 8, rank, 128)
    for step in range(8):
        current = torch.roll(base, step, dims=-1)
        current[:, : boundaries.numel()] = boundaries
        if step == 0:
            current.fill_(-1)
        device_input.copy_(current)
        graph.replay()
        assert torch.equal(output.cpu(), oracle(current, rank))
    torch.npu.synchronize()


@pytest.mark.parametrize("shape", [(0, 2048), (1, 0), (2, 3, 0)])
def test_empty_input(shape):
    value = torch.empty(shape, dtype=torch.int32, device="npu")
    assert remap_sparse_indices_triton(value, 8, 0, 128) is value


@pytest.mark.parametrize("ranks,interleave", [(4, 128), (8, 64), (2, 1), (3, 127)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_generic_integer_fallback_boundaries_and_graph(ranks, interleave, dtype):
    torch.npu.set_device(0)
    rng = torch.Generator().manual_seed(16350)
    base = torch.randint(0, 2**31 - 1, (2, 3, 257), generator=rng, dtype=dtype)
    boundary = torch.tensor([-1, 0, 127, 128, 2**24 - 1, 2**24, 2**24 + 1, 2**31 - 1], dtype=dtype)
    base[..., : boundary.numel()] = boundary
    for rank in range(ranks):
        # Exercise noncontiguous input and arbitrary leading dimensions.
        storage = base.transpose(-1, -2).contiguous().npu()
        value = storage.transpose(-1, -2)
        for _ in range(3):
            remap_sparse_indices_triton(value, ranks, rank, interleave)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = remap_sparse_indices_triton(value, ranks, rank, interleave)
        for step in range(3):
            current = torch.roll(base, step, dims=-1) if step else torch.full_like(base, -1)
            storage.copy_(current.transpose(-1, -2).contiguous())
            graph.replay()
            assert torch.equal(result.cpu(), oracle(current, rank, ranks, interleave))
    torch.npu.synchronize()
