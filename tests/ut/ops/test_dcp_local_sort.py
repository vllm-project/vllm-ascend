# SPDX-License-Identifier: Apache-2.0
"""Check exact local ordering and NPU packed transport, including graph replay."""

import ast
from functools import partial
from pathlib import Path

import pytest
import torch


@pytest.fixture
def sort_candidates():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/sharded_indexer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_sort_local_candidates"
    ]
    assert len(functions) == 1

    def prepare(indices, scores):
        # The kernel preparation is checked separately against NumPy and on NPU.
        valid = (indices >= 0) & torch.isfinite(scores)
        return (
            torch.where(valid, indices.float(), torch.inf),
            torch.where(valid, scores.float(), -torch.inf),
        )

    scope = {"torch": torch, "_local_sort_inputs": prepare}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    return scope["_sort_local_candidates"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("rows,k", [(0, 7), (3, 0), (3, 7), (5, 2048)])
def test_exact_local_sort(sort_candidates, dtype, strided, rows, k):
    generator = torch.Generator().manual_seed(40)
    ids = torch.randint(0, 1 << 24, (rows, 1, k * 2), generator=generator, dtype=torch.int32)
    scores = torch.randint(-2, 3, ids.shape, generator=generator).to(dtype)
    if k:
        ids[..., 0] = 1 << 24
        ids[..., 2:4] = -1
        scores[..., :3] = torch.tensor([torch.nan, torch.inf, -torch.inf]).to(dtype)
        if dtype == torch.float32 and k > 4:
            scores[..., 4] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
        if k > 3:
            ids[..., 6] = 1 << 24
            scores[..., 6] = 2
        if rows > 1:
            ids[0] = -1
    ids, scores = ids[..., ::2], scores[..., ::2]
    if not strided:
        ids, scores = ids.contiguous(), scores.contiguous()
    actual_ids, actual_scores = sort_candidates(ids, scores)
    for row in range(rows):
        source_ids, source_scores = ids[row, 0], scores[row, 0].float()
        valid = (source_ids >= 0) & torch.isfinite(source_scores)
        order = sorted(
            range(k),
            key=lambda i: (-float(source_scores[i]), int(source_ids[i])) if valid[i] else (float("inf"), float("inf")),
        )
        expected_scores = torch.where(valid, source_scores, -torch.inf)[order]
        expected_ids = torch.where(valid, source_ids.float(), torch.inf)[order]
        assert actual_ids.dtype == torch.float32
        assert torch.equal(actual_ids[row, 0], expected_ids)
        assert torch.equal(actual_scores[row, 0], expected_scores)


@pytest.fixture
def npu_pack():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU hardware is required")
    torch.npu.set_device(0)
    from vllm_ascend.ops.triton import sharded_indexer

    expected = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/sharded_indexer.py"
    assert Path(sharded_indexer.__file__).resolve() == expected
    return partial(sharded_indexer.pack_local_candidates, fuse_sort=True)


@pytest.mark.parametrize("world", [4, 16])
@pytest.mark.parametrize("k", [7, 2048])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("rows", [3, 48])
def test_npu_sorted_pack_preserves_exact_pairs(npu_pack, world, k, dtype, strided, rows):
    generator = torch.Generator().manual_seed(41)
    ids = torch.randint(0, (1 << 24) // world, (rows, 1, k * 2), generator=generator, dtype=torch.int32)
    scores = torch.randint(-2, 3, ids.shape, generator=generator).to(dtype)
    ids[0] = -1
    scores[..., :6:2] = torch.tensor([torch.nan, torch.inf, -torch.inf]).to(dtype)
    scores[..., 6] = 0.0
    scores[..., 8] = -0.0
    ids[..., 12] = (1 << 24) // world - 1
    if dtype == torch.float32:
        scores[..., 10] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    device_ids, device_scores = ids.npu()[..., ::2], scores.npu()[..., ::2]
    if not strided:
        device_ids, device_scores = device_ids.contiguous(), device_scores.contiguous()
    ids, scores = ids[..., ::2], scores[..., ::2].float()
    expected = torch.empty((rows, k, 2), dtype=torch.int32)
    for row in range(rows):
        local, values = ids[row, 0], scores[row, 0]
        valid = (local >= 0) & torch.isfinite(values)
        global_ids = local // 768 * (world * 768) + (world - 1) * 768 + local % 768
        global_ids = torch.where(valid, global_ids, -1)
        values = torch.where(valid, values, -torch.inf)
        order = sorted(
            range(k), key=lambda i: (-float(values[i]), int(global_ids[i])) if valid[i] else (torch.inf, torch.inf)
        )
        expected[row] = torch.stack((global_ids[order], values[order].view(torch.int32)), -1)
    actual = npu_pack(device_ids, device_scores, world, world - 1, 768, sort_candidates=True)
    assert torch.equal(actual.cpu(), expected)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            captured = npu_pack(device_ids, device_scores, world, world - 1, 768, sort_candidates=True)
        for _ in range(10):
            graph.replay()
        torch.npu.synchronize()
        assert torch.equal(captured.cpu(), expected)
        # Refresh both ordering and validity without changing captured addresses.
        ids = ids.flip(-1).clone()
        scores = -scores.flip(-1)
        ids[1] = -1
        device_ids.copy_(ids)
        device_scores.copy_(scores.to(dtype))
        for row in range(rows):
            local, values = ids[row, 0], scores[row, 0]
            valid = (local >= 0) & torch.isfinite(values)
            global_ids = local // 768 * (world * 768) + (world - 1) * 768 + local % 768
            global_ids = torch.where(valid, global_ids, -1)
            values = torch.where(valid, values, -torch.inf)
            order = sorted(
                range(k), key=lambda i: (-float(values[i]), int(global_ids[i])) if valid[i] else (torch.inf, torch.inf)
            )
            expected[row] = torch.stack((global_ids[order], values[order].view(torch.int32)), -1)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(captured.cpu(), expected)
    finally:
        graph.reset()
