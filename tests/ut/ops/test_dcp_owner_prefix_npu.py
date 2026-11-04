# SPDX-License-Identifier: Apache-2.0
"""Validate owner-prefix selection for skew, exact ties and graph mutation."""

import pytest
import torch

pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("NPU required", allow_module_level=True)

from vllm_ascend.ops.triton.sharded_indexer import merge_packed_candidates


def packed_and_expected(world, rows, dtype, pattern, interleave=768):
    owner, k = world - 1, 2048
    generator = torch.Generator().manual_seed(5303)
    local = torch.randint(0, (1 << 24) // world - interleave, (world, rows, k), generator=generator, dtype=torch.int32)
    ranks = torch.arange(world, dtype=torch.int32)[:, None, None]
    ids = (local // interleave * world + ranks) * interleave + local % interleave
    scores = torch.randn(ids.shape, generator=generator).to(dtype).float()
    if pattern == "all_ties":
        scores.fill_(1)
    elif pattern == "ties":
        scores = torch.randint(-2, 3, ids.shape, generator=generator).float()
        ids[..., :32] = ranks * interleave
    elif pattern == "owner_wins":
        scores[owner] += 100
    elif pattern == "peer_wins" and world > 1:
        scores[0] += 100
    elif pattern == "invalid":
        ids[:] = -1
    ids[:, 0] = -1
    scores[..., :3] = torch.tensor([torch.nan, torch.inf, -torch.inf])
    if pattern != "all_ties":
        scores[..., 3:5] = torch.tensor([0.0, -0.0])
    if dtype == torch.float32 and pattern != "all_ties":
        scores[..., 5:7] = torch.tensor([1.0, torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))])
        scores[..., 7:9] = torch.tensor([torch.finfo(torch.float32).tiny, -torch.finfo(torch.float32).tiny])
    if pattern == "owner_ties":
        scores[owner].fill_(1)
    valid = (ids >= 0) & torch.isfinite(scores)
    values = torch.where(valid, scores, -torch.inf)
    keys = torch.where(valid, ids, torch.iinfo(torch.int32).max)
    by_id = keys.argsort(stable=True)
    order = by_id.gather(-1, values.gather(-1, by_id).argsort(descending=True, stable=True))
    ordered_ids = torch.where(valid, ids, -1).gather(-1, order)
    ordered_scores = values.gather(-1, order).contiguous()
    packed = torch.stack((ordered_ids, ordered_scores.view(torch.int32)), -1).reshape(world * rows, k, 2)
    expected = torch.full((rows, 1, k), -1, dtype=torch.int32)
    for row in range(rows):
        row_ids, row_values = ids[:, row].reshape(-1), scores[:, row].reshape(-1)
        keep = (row_ids >= 0) & torch.isfinite(row_values)
        selected_ids, selected_values = row_ids[keep], row_values[keep]
        order = selected_ids.argsort(stable=True)
        order = order[selected_values[order].argsort(descending=True, stable=True)[:k]]
        selected = selected_ids[order]
        selected = selected[(selected // interleave) % world == owner]
        expected[row, 0, : selected.numel()] = selected // (world * interleave) * interleave + selected % interleave
    return packed, expected


@pytest.mark.parametrize("world,rows", [(1, 3), (2, 5), (3, 17), (8, 128), (16, 48), (16, 129), (17, 49)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pattern", ["random", "ties", "all_ties", "owner_ties", "owner_wins", "peer_wins"])
def test_owner_prefix_exact_and_graph(world, rows, dtype, pattern):
    torch.npu.set_device(0)
    packed_cpu, expected = packed_and_expected(world, rows, dtype, pattern)
    packed = packed_cpu.npu()
    output = torch.full(expected.shape, -123, dtype=torch.int32, device="npu")

    def run():
        return merge_packed_candidates(
            packed,
            rows,
            world,
            2048,
            local_rank=world - 1,
            interleave=768,
            presorted=True,
            owner_sharded=True,
            out=output,
        )

    for _ in range(3):
        run()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            run()
        for changed in ("peer_wins" if pattern != "peer_wins" else "owner_wins", "invalid"):
            changed_packed, changed_expected = packed_and_expected(world, rows, dtype, changed)
            packed.copy_(changed_packed)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), changed_expected, rtol=0, atol=0)
    finally:
        graph.reset()
