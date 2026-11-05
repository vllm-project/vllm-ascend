# SPDX-License-Identifier: Apache-2.0
"""Eager/graph validation of candidate transport and exact merge."""

import pytest
import torch

from vllm_ascend.attention.context_parallel.sharded_indexer import local_to_global, merge_topk_candidates
from vllm_ascend.ops.triton.sharded_indexer import merge_packed_candidates, pack_local_candidates


@pytest.mark.parametrize("return_local", [False, True])
@torch.inference_mode()
def test_dcp128_uses_one_candidate_exchange(monkeypatch, return_local):
    from vllm_ascend.attention.context_parallel.sharded_indexer import distributed_topk
    from vllm_ascend.ops.triton import sharded_indexer as kernels

    calls = []

    class LaunchRecorder:
        def __init__(self, kernel, name):
            self.kernel, self.name = kernel, name

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def run(*args, **kwargs):
                calls.append(self.name)
                return launch(*args, **kwargs)

            return run

    class RepeatedCandidates:
        world_size, rank_in_group = 8, 7

        def all_gather(self, tensor, dim):
            calls.append("all_gather")
            peers = []
            for rank in range(self.world_size):
                peer = tensor.clone()
                peer[..., 0] += (rank - self.rank_in_group) * interleave
                peers.append(peer)
            return torch.cat(peers, dim=dim)

    rows, topk, interleave = 128, 2048, 768
    ids = torch.arange(topk, dtype=torch.int32).repeat(rows, 1, 1)
    scores = torch.ones(ids.shape, dtype=torch.bfloat16)
    global_ids = torch.cat([local_to_global(ids, 8, rank, interleave) for rank in range(8)], -1)
    expected = merge_topk_candidates(global_ids, torch.cat([scores.float()] * 8, -1), topk)
    if return_local:
        selected = expected[0, 0]
        selected = selected[(selected // interleave) % 8 == 7]
        selected = selected // (8 * interleave) * interleave + selected % interleave
        expected.fill_(-1)
        expected[:, 0, : selected.numel()] = selected
    ids, scores = ids.npu(), scores.npu()
    monkeypatch.setattr(kernels, "_pack_candidates", LaunchRecorder(kernels._pack_candidates, "pack"))
    monkeypatch.setattr(kernels, "_pack_sorted_candidates", LaunchRecorder(kernels._pack_sorted_candidates, "pack"))
    monkeypatch.setattr(kernels, "_merge_candidates", LaunchRecorder(kernels._merge_candidates, "merge"))
    monkeypatch.setattr(kernels, "_merge_owner_prefix", LaunchRecorder(kernels._merge_owner_prefix, "merge"))

    actual = distributed_topk(ids, scores, RepeatedCandidates(), topk, interleave, return_local=return_local)
    # The owner path has one fused pack, one collective and one fused merge.
    assert calls[:2] == ["pack", "all_gather"]
    assert calls[2:] and all(name == "merge" for name in calls[2:])
    if return_local:
        assert calls == ["pack", "all_gather", "merge"]
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows,world,topk", [(1, 1, 7), (65, 2, 31), (128, 8, 2048), (256, 8, 2048), (129, 16, 2048)])
@pytest.mark.parametrize("interleave", [128, 768])
@pytest.mark.parametrize("local_rank", [None, 0, 7])
@pytest.mark.parametrize("presorted", [False, True])
@torch.inference_mode()
def test_merge_matches_cpu_reference_and_graph(rows, world, topk, interleave, local_rank, presorted):
    torch.manual_seed(20261028)
    ids = torch.randint(0, 16384, (world, rows, 1, topk), dtype=torch.int32)
    scores = torch.randint(-2, 3, ids.shape).bfloat16()
    scores[..., :3] = torch.tensor([torch.nan, torch.inf, -torch.inf]).bfloat16()
    if rows > 1:
        ids[:, 0] = -1
    global_ids = torch.cat([local_to_global(ids[r], world, r, interleave) for r in range(world)], -1)
    expected = merge_topk_candidates(global_ids, torch.cat(list(scores.float()), -1), topk)
    if local_rank is not None:
        selected = expected.clone()
        expected.fill_(-1)
        for source, target in zip(selected[:, 0], expected[:, 0]):
            keep = source[(source >= 0) & ((source // interleave) % world == local_rank)]
            target[: keep.numel()] = keep // (world * interleave) * interleave + keep % interleave
    packed = torch.cat(
        [
            pack_local_candidates(ids[r].npu(), scores[r].npu(), world, r, interleave, sort_candidates=presorted)
            for r in range(world)
        ],
        0,
    )

    def run():
        return merge_packed_candidates(
            packed, rows, world, topk, local_rank=local_rank, interleave=interleave, presorted=presorted
        )

    actual = run()
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = run()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured.cpu(), expected, atol=0, rtol=0)
    # Replay an empty batch, preserving pack's (-1, -inf) invalid pair.
    packed[..., 0].fill_(-1)
    packed[..., 1].fill_(torch.tensor(-torch.inf).view(torch.int32).item())
    graph.replay()
    torch.npu.synchronize()
    assert (captured == -1).all().item()


@pytest.mark.parametrize("near_ties", [False, True])
@pytest.mark.parametrize("presorted", [False, True])
@pytest.mark.parametrize("local_k,topk", [(33, 31), (2048, 2048)])
@pytest.mark.parametrize("world", [8, 16])
@torch.inference_mode()
def test_merge_preserves_fp32_scores_and_full_id_bits(near_ties, presorted, local_k, topk, world):
    ids = ((1 << 24) - torch.arange(world * local_k)).int().reshape(world, 1, local_k)
    scores = torch.ones(ids.shape)
    if near_ties:
        scores[0, 0, 0] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
        scores[0, 0, 1:3] = torch.tensor([-0.0, 0.0])
    packed = torch.stack((ids, scores.view(torch.int32)), -1).reshape(world, local_k, 2).npu()
    if presorted:
        packed = pack_local_candidates(ids.npu(), scores.npu(), 1, 0, 1, sort_candidates=True)
    expected = merge_topk_candidates(ids.reshape(1, 1, -1), scores.reshape(1, 1, -1), topk)
    actual = merge_packed_candidates(packed, 1, world, topk, presorted=presorted)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
