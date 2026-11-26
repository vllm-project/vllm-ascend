# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops import bf16_sharded_indexer


@pytest.mark.parametrize("topk", [2, 20])
@pytest.mark.parametrize("full_graph", [False, True])
def test_bf16_paged_candidates_match_independent_causal_reference(monkeypatch, topk, full_graph):
    monkeypatch.setattr(bf16_sharded_indexer, "_KEY_TOKENS_PER_TILE", 4)
    generator = torch.Generator().manual_seed(23)
    query = torch.randn((8, 3, 6), generator=generator).bfloat16()
    weights = torch.randn((8, 3), generator=generator).bfloat16()
    query[2].zero_()  # Tied scores must be ordered by actual token ID.
    key = torch.randn((10, 4, 1, 6), generator=generator).bfloat16()
    key[:2].fill_(999)
    # Pages 0 and 1 belong to the physical null block. Page 10 is out of range.
    table = torch.tensor([[4, 2, 3], [2, 3, 4], [7, 1, 5], [0, 6, 10]], dtype=torch.int32)
    ends = torch.tensor([2, 2, 5, 7], dtype=torch.int32)
    visible = torch.tensor([[1], [7], [5], [6], [9], [0], [10], [12]], dtype=torch.int32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual_ids, actual_scores = bf16_sharded_indexer.bf16_paged_indexer_candidates(
            query,
            weights,
            key,
            table,
            ends,
            visible,
            topk,
            first_valid_page=2,
            full_graph=full_graph,
            query_rows_per_tile=2,
        )

    expected_ids = torch.full_like(actual_ids, -1)
    expected_scores = torch.full_like(actual_scores, -torch.inf)
    for row in range(8):
        request = next((req for req, end in enumerate(ends) if row < end), None)
        if request is None:
            continue
        candidates = []
        for token in range(int(visible[row, 0])):
            page, offset = divmod(token, 4)
            physical = int(table[request, page])
            if not 2 <= physical < key.shape[0]:
                continue
            dots = (query[row].float() * key[physical, offset, 0].float()).sum(dim=-1)
            score = (dots.relu() * weights[row].float()).sum().item()
            candidates.append((score, token))
        for index, (score, token) in enumerate(sorted(candidates, key=lambda pair: (-pair[0], pair[1]))[:topk]):
            expected_ids[row, 0, index] = token
            expected_scores[row, 0, index] = score

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_scores, expected_scores)


@pytest.mark.parametrize("full_graph", [False, True])
def test_16_row_tile_preserves_exact_scores_and_ties_across_tiles(monkeypatch, full_graph):
    monkeypatch.setattr(bf16_sharded_indexer, "_KEY_TOKENS_PER_TILE", 4)
    generator = torch.Generator().manual_seed(51)
    query = torch.randint(-2, 3, (37, 3, 8), generator=generator).bfloat16()
    query[15:18].zero_()
    weights = torch.ones((37, 3), dtype=torch.bfloat16)
    key = torch.randint(-2, 3, (5, 4, 1, 8), generator=generator).bfloat16()
    args = (query, weights, key, torch.tensor([[4, 2, 3]]), torch.tensor([37]), torch.full((37,), 12), 7)
    reference = bf16_sharded_indexer.bf16_paged_indexer_candidates(*args, full_graph=full_graph)
    ids, scores = bf16_sharded_indexer.bf16_paged_indexer_candidates(
        *args, full_graph=full_graph, query_rows_per_tile=16
    )
    torch.testing.assert_close(ids, reference[0], rtol=0, atol=0)
    torch.testing.assert_close(scores, reference[1], rtol=0, atol=0)
    assert ids[15:18, 0].tolist() == [list(range(7))] * 3


def test_bf16_paged_candidates_handle_ranks_with_no_visible_pages():
    query = torch.ones((2, 1, 4), dtype=torch.bfloat16)
    ids, scores = bf16_sharded_indexer.bf16_paged_indexer_candidates(
        query,
        query[..., 0],
        torch.ones((2, 4, 1, 4), dtype=query.dtype),
        torch.empty((1, 0), dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        3,
    )
    torch.testing.assert_close(ids, torch.full((2, 1, 3), -1, dtype=torch.int32))
    torch.testing.assert_close(scores, torch.full((2, 1, 3), -torch.inf))


@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("topk", [4, 6])
def test_paged_candidates_preserve_nonfinite_cutoff_and_finite_ties(monkeypatch, full_graph, topk):
    monkeypatch.setattr(bf16_sharded_indexer, "_KEY_TOKENS_PER_TILE", 2)
    query = torch.ones((2, 1, 1), dtype=torch.bfloat16)
    weights = torch.tensor([[1], [-1]], dtype=torch.bfloat16)
    key = torch.tensor([0, 0, torch.nan, 2, 4, torch.inf, 4, 3], dtype=torch.bfloat16).view(4, 2, 1, 1)
    ids, scores = bf16_sharded_indexer.bf16_paged_indexer_candidates(
        query,
        weights,
        key,
        torch.tensor([[1, 2, 3]]),
        torch.tensor([2]),
        torch.tensor([6, 6]),
        topk,
        full_graph=full_graph,
    )
    expected_ids = torch.tensor([[0, 3, 2, 4, 5, 1], [0, 1, 5, 2, 4, -1]], dtype=torch.int32)
    expected_scores = torch.tensor([[torch.nan, torch.inf, 4, 4, 3, 2], [torch.nan, -2, -3, -4, -4, -torch.inf]])
    torch.testing.assert_close(ids[:, 0], expected_ids[:, :topk])
    torch.testing.assert_close(scores[:, 0], expected_scores[:, :topk], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("full_graph", [False, True])
def test_paged_candidates_follow_replaced_table_and_shorter_causal_lengths(monkeypatch, full_graph):
    # A captured step can change both its page mapping and active rows while
    # keeping the input addresses and logical page-table extent fixed.
    from torch.fx.experimental.proxy_tensor import make_fx

    monkeypatch.setattr(bf16_sharded_indexer, "_KEY_TOKENS_PER_TILE", 4)
    query = torch.ones((3, 1, 1), dtype=torch.bfloat16)
    weights = query[..., 0]
    key = torch.arange(8, dtype=torch.bfloat16).view(4, 2, 1, 1)
    table = torch.tensor([[2, 1, 3], [3, 2, 1]], dtype=torch.int32)
    ends = torch.tensor([2, 3], dtype=torch.int32)
    visible = torch.full((3,), 6, dtype=torch.int32)

    def select(table, ends, visible):
        return bf16_sharded_indexer.bf16_paged_indexer_candidates(
            query, weights, key, table, ends, visible, 8, full_graph=full_graph, query_rows_per_tile=2
        )

    captured = make_fx(select)(table, ends, visible)
    table.copy_(torch.tensor([[0, 3, 1], [1, 2, 3]]))
    ends.copy_(torch.tensor([1, 2]))
    visible.copy_(torch.tensor([5, 3, 6]))
    ids, scores = captured(table, ends, visible)
    assert ids[:, 0].tolist() == [[3, 2, 4] + [-1] * 5, [2, 1, 0] + [-1] * 5, [-1] * 8]
    assert scores[:, 0].tolist() == [[7, 6, 2] + [-torch.inf] * 5, [4, 3, 2] + [-torch.inf] * 5, [-torch.inf] * 8]
