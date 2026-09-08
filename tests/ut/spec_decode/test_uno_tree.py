# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Proposal ranking, sibling isolation, and accepted-path tests for UNO."""

import heapq

import pytest
import torch

from vllm_ascend.spec_decode.uno_tree import (
    UnoTreeProposal,
    build_uno_tree,
    compact_uno_tree_kv,
    fill_uno_tree_mask,
    traverse_uno_tree,
)


def _heap_reference(root, logits, nodes, top_k, temperature):
    scaled = logits.float() / (temperature if temperature > 0 else 1)
    values, ids = scaled.topk(top_k, dim=-1, sorted=True)
    scores = values - scaled.logsumexp(dim=-1, keepdim=True)
    result = [(root, -1, 0)]
    frontier = []

    def expand(parent, depth, mass):
        if depth < logits.shape[0]:
            for rank in range(top_k):
                token = int(ids[depth, rank])
                next_mass = mass + float(scores[depth, rank])
                heapq.heappush(frontier, (-next_mass, depth + 1, rank, token, parent))

    expand(0, 0, 0.0)
    while len(result) < nodes:
        cost, depth, _, token, parent = heapq.heappop(frontier)
        result.append((token, parent, depth))
        expand(len(result) - 1, depth, -cost)
    return result


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0, 1.5])
def test_tree_matches_best_first_probability_mass_and_ancestor_mask(temperature):
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(2, 4, 19, generator=generator)
    roots = torch.tensor([7, 13])
    tree = build_uno_tree(roots, logits, max_nodes=12, candidate_top_k=4, temperature=torch.full((2,), temperature))
    for batch in range(2):
        reference = _heap_reference(int(roots[batch]), logits[batch], 12, 4, temperature)
        assert tree.tokens[batch].tolist() == [row[0] for row in reference]
        assert tree.parents[batch].tolist() == [row[1] for row in reference]
        assert tree.depths[batch].tolist() == [row[2] for row in reference]
        for node in range(12):
            ancestors = set()
            current = node
            while current >= 0:
                ancestors.add(current)
                current = reference[current][1]
            assert set(tree.allowed_attention[batch, node].nonzero().flatten().tolist()) == ancestors


def test_tree_path_follows_second_sibling_and_stops_at_missing_target_token():
    tree = UnoTreeProposal(
        tokens=torch.tensor([[10, 11, 12, 13]]),
        parents=torch.tensor([[-1, 0, 0, 2]]),
        depths=torch.tensor([[0, 1, 1, 2]]),
        allowed_attention=torch.empty(0),
    )
    output, kv_rows = traverse_uno_tree(tree, torch.tensor([[12, 77, 13, 99]]), 5)
    assert output.tolist() == [[10, 12, 13, 99, -1]]
    assert kv_rows.tolist() == [[0, 2, 3, -1, -1]]
    changed_sibling = torch.tensor([[12, 3, 13, 99]])
    assert torch.equal(traverse_uno_tree(tree, changed_sibling, 5)[0], output)


def test_tree_selection_does_not_replace_target_distribution_with_proposal_scores():
    # Enumerate a uniform target draw. Tokens outside the tree are still emitted.
    count = 5
    tree = UnoTreeProposal(
        tokens=torch.tensor([[0, 1, 2]]).expand(count, -1),
        parents=torch.tensor([[-1, 0, 0]]).expand(count, -1),
        depths=torch.tensor([[0, 1, 1]]).expand(count, -1),
        allowed_attention=torch.empty(0),
    )
    target = torch.zeros(count, 3, dtype=torch.int64)
    target[:, 0] = torch.arange(count)
    result, _ = traverse_uno_tree(tree, target, 3)
    assert result[:, 1].tolist() == list(range(count))


def test_tree_budget_rejects_insufficient_candidate_capacity():
    with pytest.raises(ValueError, match="cannot fill"):
        build_uno_tree(
            torch.tensor([1]), torch.zeros(1, 2, 7), max_nodes=4, candidate_top_k=1, temperature=torch.ones(1)
        )


def test_tree_masks_remain_valid_when_all_candidate_masses_are_negative_infinity():
    tree = build_uno_tree(
        torch.tensor([1]), torch.full((1, 3, 7), -torch.inf), max_nodes=8, candidate_top_k=3, temperature=torch.ones(1)
    )
    assert (tree.parents[0, 1:] < torch.arange(1, 8)).all()
    assert (tree.depths < 4).all()
    assert tree.allowed_attention.diagonal(dim1=1, dim2=2).all()


@pytest.mark.parametrize("frontier", [0, 3, 127, 128, 251])
def test_physical_mask_tracks_prefix_and_excludes_siblings(frontier):
    allowed = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1]], dtype=torch.bool)
    mask = torch.ones(1, 1, 4, 256, dtype=torch.bool)
    fill_uno_tree_mask(mask, torch.arange(256), torch.tensor(frontier), allowed)
    assert not mask[..., :frontier].any()
    assert torch.equal(~mask[0, 0, :, frontier : frontier + 4], allowed)
    assert mask[..., frontier + 4 :].all()


def test_kv_compaction_gathers_before_overwriting_across_nonadjacent_pages():
    cache = torch.arange(24).reshape(6, 4, 1, 1).clone()
    original = cache.clone()
    # Seed at logical 3; the next page is physical page 3, not page 1.
    slots = torch.tensor([3, 12, 13, 14, 15, 4, 5], dtype=torch.int32)
    # Root then branch rows 3 and 4; destinations alias another source row.
    compact_uno_tree_kv(cache, slots, torch.tensor([0, 3, 4, -1]))
    flat = cache.flatten()
    assert flat[slots[:5]].tolist() == [3, 12, 15, 4, 15]
    untouched = torch.ones(24, dtype=torch.bool)
    untouched[slots[1:4]] = False
    assert torch.equal(flat[untouched], original.flatten()[untouched])


def test_clipped_tree_remains_prefix_closed_and_emits_uncached_bonus():
    tree = build_uno_tree(
        torch.tensor([1]), torch.randn(1, 4, 9), max_nodes=12, candidate_top_k=4, temperature=torch.ones(1)
    )
    for nodes in range(1, 13):
        clipped = UnoTreeProposal(
            tree.tokens[:, :nodes],
            tree.parents[:, :nodes],
            tree.depths[:, :nodes],
            tree.allowed_attention[:, :nodes, :nodes],
        )
        output, rows = traverse_uno_tree(clipped, torch.full((1, nodes), 20), min(6, nodes + 1))
        assert output[0, :2].tolist() == [1, 20]
        assert rows[0, :2].tolist() == [0, -1]


def test_compactor_keeps_unique_caches_and_rejects_incompatible_storage():
    from vllm_ascend.spec_decode.uno_tree import UnoTreeKVCompactor

    cache = torch.arange(24).reshape(6, 4, 1, 1).clone()
    compactor = UnoTreeKVCompactor([cache, cache])
    assert compactor.addresses == (cache.data_ptr(),)
    compactor.compact(torch.tensor([3, 12, 13, 14, 15, 4, 5]), torch.tensor([0, 3, 4, -1]))
    assert cache.flatten()[torch.tensor([3, 12, 13, 14, 15])].tolist() == [3, 12, 15, 4, 15]
    with pytest.raises(ValueError, match="populated"):
        UnoTreeKVCompactor([])
    with pytest.raises(ValueError, match="homogeneous"):
        UnoTreeKVCompactor([cache, cache.float()])
    with pytest.raises(ValueError, match="homogeneous"):
        UnoTreeKVCompactor([cache.transpose(0, 1)])
