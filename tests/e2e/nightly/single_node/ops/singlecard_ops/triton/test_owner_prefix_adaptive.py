# SPDX-License-Identifier: Apache-2.0
"""Exact owner prefixes against CPU global sorting, including changed graph inputs."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.sharded_indexer import merge_packed_candidates

CASES = [
    (144, 8, 0, 768, "random"),
    (152, 8, 0, 768, "random"),
    (1, 8, 3, 768, "random"),
    (39, 8, 3, 768, "random"),
    (40, 8, 3, 768, "random"),
    (41, 8, 3, 768, "random"),
    (257, 8, 3, 768, "random"),
    (48, 8, 3, 768, "random"),
    (49, 8, 3, 768, "random"),
    (41, 4, 1, 768, "random"),
    (41, 16, 15, 768, "random"),
    (41, 3, 2, 768, "random"),
    (41, 8, 0, 768, "ties"),
    (41, 8, 0, 768, "own_high"),
    (41, 8, 0, 768, "peer_high"),
    (41, 8, 0, 768, "short"),
    (41, 8, 0, 768, "mixed"),
    (49, 8, 0, 768, "ties"),
    (49, 8, 0, 768, "own_high"),
    (49, 8, 0, 768, "peer_high"),
    (49, 8, 0, 768, "short"),
    (49, 8, 0, 768, "mixed"),
    (1, 1, 0, 1, "ties"),
    (49, 17, 16, 1, "ties"),
]


def make_inputs(rows, world, owner, interleave, distribution, edge):
    k = 2048
    generator = torch.Generator().manual_seed(21584 + rows)
    local = torch.arange(k, dtype=torch.int64)
    ids = (
        torch.stack([local + ((local // interleave) * (world - 1) + rank) * interleave for rank in range(world)])[
            :, None, :
        ]
        .expand(world, rows, k)
        .clone()
    )
    scores = torch.randn(ids.shape, generator=generator)
    if distribution == "ties":
        scores = scores.round()
    elif distribution == "own_high":
        scores[owner] += 20
    elif distribution == "peer_high":
        scores[(owner + 1) % world] += 20
    elif distribution in ("short", "mixed"):
        for rank in range(world):
            count = 97 + 23 * rank if distribution == "short" else rank * 307 % k
            ids[rank, :, count:] = -1
            scores[rank, :, count:] = -torch.inf
    if edge:
        scores[..., :8] = torch.tensor([0.0, -0.0, torch.nan, torch.inf, -torch.inf, 1.0, 1.0, 1.0])
        ids[..., 9] = -1
        ids[:, 0] = -1
    valid = (ids >= 0) & scores.isfinite()
    ids = torch.where(valid, ids, -1)
    scores = torch.where(valid, scores, -torch.inf)
    by_id = torch.where(valid, ids, torch.iinfo(torch.int64).max).argsort(stable=True)
    order = by_id.gather(-1, scores.gather(-1, by_id).argsort(descending=True, stable=True))
    ids, scores = ids.gather(-1, order), scores.gather(-1, order)
    packed = torch.stack((ids.int(), scores.contiguous().view(torch.int32)), -1)
    flat_ids = ids.transpose(0, 1).reshape(rows, -1)
    flat_scores = scores.transpose(0, 1).reshape(rows, -1)
    by_id = torch.where(flat_ids >= 0, flat_ids, torch.iinfo(torch.int64).max).argsort(stable=True)
    order = by_id.gather(-1, flat_scores.gather(-1, by_id).argsort(descending=True, stable=True))
    winners = flat_ids.gather(-1, order)[:, :k]
    keep = (winners >= 0) & ((winners // interleave) % world == owner)
    expected = torch.full((rows, 1, k), -1, dtype=torch.int32)
    for row in range(rows):
        selected = winners[row, keep[row]]
        blocks = selected // interleave
        local = blocks // world * interleave + selected - blocks * interleave
        expected[row, 0, : local.numel()] = local.int()
    # The public API receives concatenated rank rows: [WORLD*rows,K,2].
    return packed.reshape(world * rows, k, 2), expected


@pytest.mark.parametrize("rows,world,owner,interleave,distribution", CASES)
@pytest.mark.parametrize("edge", [False, True])
def test_owner_prefix_exact(rows, world, owner, interleave, distribution, edge):
    torch.npu.set_device(0)
    packed, expected = make_inputs(rows, world, owner, interleave, distribution, edge)
    storage = torch.full((rows + 2, 1, 2048), -123, device="npu", dtype=torch.int32)
    actual = merge_packed_candidates(
        packed.npu(),
        rows,
        world,
        2048,
        local_rank=owner,
        interleave=interleave,
        out=storage[1:-1],
        presorted=True,
        owner_sharded=True,
    )
    assert torch.equal(actual.cpu(), expected)
    assert (storage[[0, -1]].cpu() == -123).all()


def test_owner_prefix_changed_graph_input():
    torch.npu.set_device(0)
    rows, world, owner, interleave = 49, 8, 3, 768
    packed, expected = make_inputs(rows, world, owner, interleave, "random", False)
    device = packed.npu()
    output = torch.empty((rows, 1, 2048), device="npu", dtype=torch.int32)

    def run():
        return merge_packed_candidates(
            device,
            rows,
            world,
            2048,
            local_rank=owner,
            interleave=interleave,
            out=output,
            presorted=True,
            owner_sharded=True,
        )

    for _ in range(3):
        run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for distribution, edge in (("peer_high", False), ("ties", True), ("mixed", True)):
        changed, expected = make_inputs(rows, world, owner, interleave, distribution, edge)
        device.copy_(changed)
        graph.replay()
        assert torch.equal(output.cpu(), expected)
