# SPDX-License-Identifier: Apache-2.0
"""Exact candidate transport and merge through the shared public NPU APIs."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.dcp_test_utils import (
    local_to_global,
    merge_topk_candidates,
)
from vllm_ascend.ops.triton.sharded_indexer import merge_packed_candidates, pack_local_candidates


@pytest.mark.parametrize(
    "world,rows,topk,local_k,dtype,local_rank",
    [
        (1, 1, 2048, 2048, torch.bfloat16, 0),
        (3, 17, 31, 33, torch.float32, None),
        (8, 128, 2048, 2048, torch.bfloat16, 7),
        (16, 48, 2048, 2048, torch.float32, 15),
        (17, 49, 2048, 2048, torch.bfloat16, 16),
        (8, 65, 2048, 2051, torch.float32, None),
    ],
)
@pytest.mark.parametrize("ties", [False, True])
def test_candidate_pipeline_and_changed_graph_inputs(world, rows, topk, local_k, dtype, local_rank, ties):
    torch.npu.set_device(0)
    interleave = 768
    generator = torch.Generator().manual_seed(20883)
    ids = torch.randint(0, 131072, (world, rows, 1, local_k), dtype=torch.int32, generator=generator)
    scores = torch.randn(ids.shape, generator=generator).to(dtype)
    if ties:
        scores.fill_(1)
    ids[:, 0] = -1
    scores[..., :5] = torch.tensor([torch.nan, torch.inf, -torch.inf, 0.0, -0.0], dtype=dtype)
    if dtype == torch.float32:
        scores[..., 5:7] = torch.tensor([1.0, torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))])
    device_ids = torch.stack((ids, ids), -1).npu()[..., 0]
    device_scores = torch.stack((scores, scores), -1).npu()[..., 1]
    storage = torch.full((rows + 2, 1, topk), -123, dtype=torch.int32, device="npu")
    result = storage[1:-1]
    presorted = topk == 2048 and local_k <= 2048

    def run():
        packed = torch.cat(
            [
                pack_local_candidates(
                    device_ids[r],
                    device_scores[r],
                    world,
                    r,
                    interleave,
                    sort_candidates=presorted,
                    fuse_sort=local_rank is not None,
                )
                for r in range(world)
            ],
            0,
        )
        return merge_packed_candidates(
            packed,
            rows,
            world,
            topk,
            local_rank=local_rank,
            interleave=interleave,
            out=result,
            presorted=presorted,
            owner_sharded=local_rank is not None,
        )

    def expected():
        global_ids = torch.cat([local_to_global(ids[r], world, r, interleave) for r in range(world)], -1)
        return merge_topk_candidates(
            global_ids,
            torch.cat(list(scores.float()), -1),
            topk,
            local_rank=local_rank,
            world_size=world,
            interleave=interleave,
        )

    for _ in range(3):
        run()
    assert torch.equal(result.cpu(), expected())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for invalid in (False, True):
        if invalid:
            ids.fill_(-1)
        else:
            ids.copy_(ids.flip(-1))
            scores[0].fill_(100)
        device_ids.copy_(ids)
        device_scores.copy_(scores)
        graph.replay()
        assert torch.equal(result.cpu(), expected())
        assert (storage[[0, -1]].cpu() == -123).all()
    graph.reset()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("world,rows", [(8, 128), (16, 48)])
def test_sorted_pack_preserves_score_bits_and_large_integer_payloads(dtype, world, rows):
    torch.npu.set_device(0)
    k, rank, interleave = 2048, world - 1, 768
    ids = torch.arange(k, dtype=torch.int32).repeat(rows, 1).view(rows, 1, k)
    ids[..., -1] = 1 << 24
    scores = torch.randn(ids.shape).to(dtype)
    scores[..., :7] = torch.tensor([torch.nan, torch.inf, -torch.inf, 0.0, -0.0, 1.0, -1.0], dtype=dtype)
    valid = (ids >= 0) & scores.isfinite()
    values = torch.where(valid, scores.float(), -torch.inf)
    mapped = torch.where(valid, local_to_global(ids, world, rank, interleave), -1)
    by_id = torch.where(valid, ids, torch.iinfo(torch.int32).max).argsort(stable=True)
    order = by_id.gather(-1, values.gather(-1, by_id).argsort(descending=True, stable=True))
    expected = torch.stack((mapped.gather(-1, order), values.gather(-1, order).contiguous().view(torch.int32)), -1)[
        :, 0
    ]
    actual = pack_local_candidates(
        ids.npu(), scores.npu(), world, rank, interleave, sort_candidates=True, fuse_sort=True
    )
    assert torch.equal(actual.cpu(), expected)
