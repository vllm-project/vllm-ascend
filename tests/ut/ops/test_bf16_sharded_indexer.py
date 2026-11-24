# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops import bf16_sharded_indexer


@pytest.mark.parametrize("topk", [2, 20])
def test_bf16_paged_candidates_match_independent_causal_reference(monkeypatch, topk):
    monkeypatch.setattr(bf16_sharded_indexer, "_QUERY_ROWS_PER_TILE", 2)
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
            query, weights, key, table, ends, visible, topk, first_valid_page=2
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
