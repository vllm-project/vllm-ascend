# SPDX-License-Identifier: Apache-2.0
"""Global Top-K is exact under the native comparator without KV replication."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.context_parallel.sharded_indexer import (
    distributed_topk,
    gathered_candidate_row_tile,
    local_to_global,
    local_visible_lengths,
    merge_topk_candidates,
    query_visible_lengths,
    use_local_topk_output,
)


@pytest.mark.parametrize("interleave", [1, 128, 768])
@pytest.mark.parametrize("length", [0, 1, 767, 768, 1536, 6143, 6144, 12345])
def test_local_visibility_and_interleaved_indices(interleave, length):
    world = 8
    total = 0
    for rank in range(world):
        visible = local_visible_lengths(torch.tensor([length]), world, rank, interleave).item()
        ids = local_to_global(torch.arange(visible), world, rank, interleave)
        expected = [i for i in range(length) if (i // interleave) % world == rank]
        assert ids.tolist() == expected
        assert local_to_global(torch.tensor([-1]), world, rank, interleave).item() == -1
        total += visible
    assert total == length


@pytest.mark.parametrize("equal_scores", [False, True])
@pytest.mark.parametrize("length", [0, 1, 767, 2048, 6144, 17000])
def test_local_k_union_matches_full_history(equal_scores, length):
    torch.manual_seed(7)
    topk, world, interleave = 2048, 8, 768
    scores = torch.randn(length).bfloat16().float()
    if equal_scores:
        scores.fill_(1)
    candidates, candidate_scores = [], []
    for rank in range(world):
        local_count = local_visible_lengths(torch.tensor([length]), world, rank, interleave).item()
        global_ids = local_to_global(torch.arange(local_count), world, rank, interleave)
        local_scores = scores[global_ids.long()]
        # Native QLI uses the same score comparator and ascending local IDs.
        chosen = torch.argsort(local_scores, descending=True, stable=True)[:topk]
        ids = torch.full((topk,), -1, dtype=torch.int32)
        vals = torch.full((topk,), -torch.inf)
        ids[: len(chosen)] = global_ids[chosen]
        vals[: len(chosen)] = local_scores[chosen]
        candidates.append(ids)
        candidate_scores.append(vals)
    merged = merge_topk_candidates(torch.cat(candidates), torch.cat(candidate_scores), topk)
    expected = torch.argsort(scores, descending=True, stable=True)[:topk].to(torch.int32)
    assert torch.equal(merged[: len(expected)], expected)
    assert (merged[len(expected) :] == -1).all()


def test_all_global_winners_can_belong_to_one_rank():
    topk = 16
    ids = torch.arange(8 * topk, dtype=torch.int32).reshape(8, topk)
    scores = torch.zeros((8, topk))
    scores[5] = 10
    result = merge_topk_candidates(ids.flatten(), scores.flatten(), topk)
    assert torch.equal(result, ids[5])


def test_varlen_global_causality_and_graph_padding_without_positions():
    visible = query_visible_lengths(
        torch.tensor([10, 20]), torch.tensor([2, 5]), torch.arange(8), torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
    )
    assert visible.tolist() == [9, 10, 18, 19, 20, 0, 0, 0]


def _reference_local(global_ids, world, rank, interleave):
    expected = torch.full_like(global_ids, -1)
    for source, target in zip(global_ids.reshape(-1, global_ids.shape[-1]), expected.reshape(-1, expected.shape[-1])):
        selected = source[(source >= 0) & ((source // interleave) % world == rank)]
        target[: selected.numel()] = selected // (world * interleave) * interleave + selected % interleave
    return expected


@pytest.fixture
def local_selector(monkeypatch):
    # Keep CPU reference execution in tests; production always calls Triton.
    def reference(indices, scores, world, rank, interleave):
        return _reference_local(torch.where(torch.isfinite(scores), indices, -1), world, rank, interleave)

    selector = Mock(side_effect=reference)
    module = ModuleType("vllm_ascend.ops.triton.sharded_indexer")
    module.select_local_candidates = selector
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return selector


@pytest.mark.parametrize("interleave", [1, 128, 768])
@pytest.mark.parametrize("world", [1, 2, 8])
@pytest.mark.parametrize("equal_scores", [False, True])
def test_local_merge_matches_global_selection_then_remap(interleave, world, equal_scores, local_selector):
    torch.manual_seed(29)
    topk = 31
    ids = torch.cat(
        [
            local_to_global(torch.randint(0, 4096, (7, 1, topk), dtype=torch.int32), world, r, interleave)
            for r in range(world)
        ],
        dim=-1,
    )
    scores = torch.ones_like(ids, dtype=torch.float32) if equal_scores else torch.randn(ids.shape).bfloat16().float()
    ids[0] = -1
    scores[1, :, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf])
    scores[-1] = -torch.inf
    scores[-1, :, :topk] = 10  # All winners on rank zero; every other owner is empty.
    global_ids = merge_topk_candidates(ids, scores, topk)
    for rank in range(world):
        actual = merge_topk_candidates(ids, scores, topk, local_rank=rank, world_size=world, interleave=interleave)
        assert actual.dtype == torch.int32
        assert torch.equal(actual, _reference_local(global_ids, world, rank, interleave))
    assert local_selector.call_count == world


def test_local_merge_propagates_triton_failure(local_selector):
    local_selector.side_effect = RuntimeError("Triton finalizer failed")
    ids = torch.tensor([[[0, 1]]], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="Triton finalizer failed"):
        merge_topk_candidates(ids, torch.ones_like(ids, dtype=torch.float32), 2, local_rank=0, world_size=2)


@pytest.fixture
def packed_candidate_ops(local_selector):
    def pack(indices, scores, world, rank, interleave, *, sort_candidates=False, fuse_sort=False):
        ids = local_to_global(indices, world, rank, interleave)
        valid = (ids >= 0) & torch.isfinite(scores)
        return torch.stack(
            (
                torch.where(valid, ids, -1)[:, 0],
                torch.where(valid, scores.float(), -torch.inf)[:, 0].view(torch.int32),
            ),
            -1,
        )

    def merge(packed, rows, world, topk, *, local_rank, interleave, out, presorted=False, owner_sharded=False):
        candidates = packed.view(world, rows, -1, 2).permute(1, 0, 2, 3).reshape(rows, 1, -1, 2)
        ids = candidates[..., 0]
        scores = candidates[..., 1].contiguous().view(torch.float32)
        out.copy_(
            merge_topk_candidates(ids, scores, topk, local_rank=local_rank, world_size=world, interleave=interleave)
        )
        return out

    module = sys.modules["vllm_ascend.ops.triton.sharded_indexer"]
    module.MAX_FUSED_TOPK = 2048
    module.pack_local_candidates = Mock(side_effect=pack)
    module.merge_packed_candidates = Mock(side_effect=merge)
    return module


@pytest.mark.parametrize("rows", [0, 1, 16, 17, 64, 65, 128, 129, 256])
@pytest.mark.parametrize("return_local", [False, True])
def test_distributed_output_uses_packed_exchange_and_query_tiles(rows, return_local, packed_candidate_ops):
    world, topk, interleave = 2, 7, 128
    indices = torch.arange(topk, dtype=torch.int32)[None, None] + torch.arange(rows)[:, None, None] * interleave
    indices = indices.to(torch.int32)
    scores = torch.ones_like(indices, dtype=torch.float32)
    for rank in range(world):
        calls = []
        offset = 0

        def all_gather(tensor, dim, calls=calls):
            nonlocal offset
            assert dim == 0
            start, offset = offset, offset + tensor.shape[0]
            calls.append(tensor.shape[0])
            return torch.cat(
                [
                    packed_candidate_ops.pack_local_candidates.side_effect(
                        indices[start:offset], scores[start:offset], world, r, interleave
                    )
                    for r in range(world)
                ],
                0,
            )

        group = SimpleNamespace(world_size=world, rank_in_group=rank, all_gather=all_gather)
        actual = distributed_topk(indices, scores, group, topk, interleave, return_local=return_local)
        expected = (
            (indices if rank == 0 else torch.full_like(indices, -1))
            if return_local
            else local_to_global(indices, world, 0, interleave)
        )
        assert torch.equal(actual, expected)
        assert calls == ([rows] if rows else [])
    assert packed_candidate_ops.pack_local_candidates.call_count == world * bool(rows)
    assert packed_candidate_ops.merge_packed_candidates.call_count == world * bool(rows)
    if rows:
        assert (
            packed_candidate_ops.pack_local_candidates.call_args.kwargs["sort_candidates"]
            == packed_candidate_ops.merge_packed_candidates.call_args.kwargs["presorted"]
        )


@pytest.mark.parametrize("rows", [0, 1, 128, 256, 2048, 2049])
def test_decode_ceiling_and_long_batch_budget(rows):
    tile = gathered_candidate_row_tile(rows, 8, 2048)
    assert tile == (2048 if rows else 0)


@pytest.mark.parametrize("return_local", [False, True])
def test_distributed_output_tiles_across_budget_boundary(monkeypatch, return_local, packed_candidate_ops):
    import vllm_ascend.attention.context_parallel.sharded_indexer as module

    # Keep PR35's guaranteed 256-row floor; force the first split at row 256.
    rows, world, topk, interleave = 257, 2, 7, 128
    monkeypatch.setattr(module, "MAX_GATHERED_CANDIDATE_BYTES", 32 * world * topk * module.PACKED_PAIR_BYTES)
    indices = torch.arange(topk, dtype=torch.int32)[None, None].expand(rows, 1, -1)
    scores = torch.ones_like(indices, dtype=torch.bfloat16)
    calls = []

    def all_gather(packed, dim):
        calls.append(packed.shape[0])
        n = packed.shape[0]
        return torch.cat(
            [
                packed_candidate_ops.pack_local_candidates.side_effect(indices[:n], scores[:n], world, r, interleave)
                for r in range(world)
            ],
            0,
        )

    group = SimpleNamespace(world_size=world, rank_in_group=0, all_gather=all_gather)
    actual = distributed_topk(indices, scores, group, topk, interleave, return_local=return_local)
    assert calls == [256, 1]
    assert torch.equal(actual, indices)


@pytest.mark.parametrize(
    "sharded,pcp_size,dsa_cp,turboquant,expected",
    [
        (True, 1, False, False, True),
        (False, 1, False, False, False),
        (True, 2, False, False, False),
        (True, 8, False, False, False),
        (True, 1, True, False, False),
        (True, 1, False, True, False),
    ],
)
def test_local_output_layout_is_fixed_for_shared_index_layers(sharded, pcp_size, dsa_cp, turboquant, expected):
    assert use_local_topk_output(sharded, pcp_size, dsa_cp, turboquant=turboquant) == expected
