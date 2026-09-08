# SPDX-License-Identifier: Apache-2.0
"""NPU tree-selection and graph regression against the tensor reference."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.spec_decode.uno_tree import (
    UnoTreeKVCompactor,
    UnoTreeProposal,
    build_uno_tree,
    compact_uno_tree_kv,
    traverse_uno_tree,
)


@pytest.mark.parametrize("depth,top_k,nodes,vocabulary", [(3, 4, 8, 257), (15, 32, 32, 151936), (15, 128, 32, 257)])
def test_fused_tree_and_graph_match_tensor_reference(depth, top_k, nodes, vocabulary):
    torch.npu.set_device(0)
    torch.manual_seed(42)
    roots = torch.tensor([41], device="npu")
    logits = torch.randn(1, depth, vocabulary, device="npu")
    temperature = torch.ones(1, device="npu")

    def build(fused):
        return build_uno_tree(
            roots, logits, max_nodes=nodes, candidate_top_k=top_k, temperature=temperature, use_fused_kernel=fused
        )

    with torch.inference_mode():
        build(True)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replay = build(True)
        for index, temp in enumerate([0.0, 0.7, 1.0, 1.5]):
            roots.fill_(42 + index)
            logits.normal_()
            temperature.fill_(temp)
            reference = build(False)
            eager = build(True)
            graph.replay()
            for field in ("tokens", "parents", "depths", "allowed_attention"):
                assert torch.equal(getattr(eager, field), getattr(reference, field)), field
                assert torch.equal(getattr(replay, field), getattr(reference, field)), field


@pytest.mark.parametrize("nodes", [1, 2, 9, 32])
def test_fused_traversal_matches_reference_for_clipped_and_branching_trees(nodes):
    torch.npu.set_device(0)
    tokens = torch.arange(100, 100 + nodes, device="npu")[None]
    parents = ((torch.arange(nodes, device="npu") - 1) // 2)[None]
    proposal = UnoTreeProposal(tokens, parents, torch.empty(0), torch.empty(0))
    samples = torch.empty((1, nodes), dtype=torch.int64, device="npu")
    width = min(17, nodes + 1)
    with torch.inference_mode():
        samples.fill_(999)
        traverse_uno_tree(proposal, samples, width)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replay = traverse_uno_tree(proposal, samples, width)
        for branch in (1, 2, 99):
            # Left child, right child, and a target token outside the tree.
            next_node = torch.arange(nodes, device="npu") * 2 + branch
            samples.copy_((100 + next_node)[None])
            reference = traverse_uno_tree(proposal, samples, width, use_fused_kernel=False)
            eager = traverse_uno_tree(proposal, samples, width)
            graph.replay()
            assert all(torch.equal(a, b) for a, b in zip(eager, reference))
            assert all(torch.equal(a, b) for a, b in zip(replay, reference))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("row_count", [1, 4, 9, 16])
def test_fused_kv_compaction_preserves_aliases_pages_padding_and_graph_replay(dtype, row_count):
    torch.npu.set_device(0)
    torch.manual_seed(43)
    originals = [torch.randn((6, 128, 8, 128), device="npu", dtype=dtype) for _ in range(4)]
    caches = [cache.clone() for cache in originals]
    # The physical pages are not contiguous. Some destinations alias later sources.
    slots = torch.tensor([127, *range(384, 400), *range(128, 144)], device="npu", dtype=torch.int32)
    accepted = torch.full((row_count,), -1, device="npu", dtype=torch.int64)
    compactor = UnoTreeKVCompactor([*caches, caches[0]])
    assert len(compactor.caches) == 4
    with torch.inference_mode():
        compactor.compact(slots, accepted)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            compactor.compact(slots, accepted)
        branch_rows = ([0, 3, 4, 9, 31] + [-1] * 11)[:row_count]
        for rows in ([-1] * row_count, branch_rows, list(range(row_count))):
            accepted.copy_(torch.tensor(rows, device="npu"))
            reference = [cache.clone() for cache in originals]
            for cache in reference:
                compact_uno_tree_kv(cache, slots, accepted)
            for cache, original in zip(caches, originals):
                cache.copy_(original)
            compactor.compact(slots, accepted)
            assert all(torch.equal(a, b) for a, b in zip(caches, reference))
            for cache, original in zip(caches, originals):
                cache.copy_(original)
            graph.replay()
            assert all(torch.equal(a, b) for a, b in zip(caches, reference))
