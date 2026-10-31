# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm_ascend.worker.v2.sample.apply_top_k_top_p as topk_topp
from vllm_ascend.worker.v2.sample.apply_top_k_top_p import (
    _apply_top_k_top_p_chunk,
    _apply_top_k_top_p_pytorch,
    _get_max_sampling_rows,
)


@pytest.mark.parametrize(
    "k,p",
    [
        (torch.tensor([4, 32, 8, 32, 16, 2]), torch.tensor([0.95, 1.0, 0.9, 0.8, 0.95, 1.0])),
        (None, torch.tensor([0.95, 1.0, 0.9, 0.8, 0.95, 1.0])),
        (torch.tensor([4, 32, 8, 32, 16, 2]), None),
    ],
)
def test_chunked_top_k_top_p_matches_full_batch(monkeypatch, k, p):
    torch.manual_seed(0)
    logits = torch.randn(6, 32, dtype=torch.float32)

    expected = _apply_top_k_top_p_chunk(logits.clone(), k, p)

    monkeypatch.setattr(topk_topp, "_get_max_sampling_rows", lambda _: 2)
    actual = _apply_top_k_top_p_pytorch(logits.clone(), k, p)

    assert torch.equal(actual, expected)


def test_chunked_top_k_top_p_is_noop_when_both_disabled():
    logits = torch.randn(6, 32, dtype=torch.float32)

    assert _apply_top_k_top_p_pytorch(logits, None, None) is logits


def test_top_k_top_p_splits_large_batches(monkeypatch):
    logits = torch.randn(5, 32, dtype=torch.float32)
    k = torch.full((5,), 4, dtype=torch.int64)
    p = torch.full((5,), 0.95, dtype=torch.float32)
    chunk_rows = []
    chunk_fn = topk_topp._apply_top_k_top_p_chunk

    def tracked_chunk(chunk, chunk_k, chunk_p):
        chunk_rows.append(chunk.shape[0])
        return chunk_fn(chunk, chunk_k, chunk_p)

    monkeypatch.setattr(topk_topp, "_apply_top_k_top_p_chunk", tracked_chunk)
    monkeypatch.setattr(topk_topp, "_get_max_sampling_rows", lambda _: 2)

    result = _apply_top_k_top_p_pytorch(logits, k, p)

    assert chunk_rows == [2, 2, 1]
    assert result is logits


def test_sampling_chunk_budget_accounts_for_sort_temporaries():
    vocab_size = 151936
    max_rows = _get_max_sampling_rows(vocab_size)
    bytes_per_element = topk_topp._SAMPLING_TEMP_BYTES_PER_ELEMENT
    chunk_bytes = topk_topp._MAX_SAMPLING_CHUNK_BYTES

    assert max_rows * vocab_size * bytes_per_element <= chunk_bytes
    assert (max_rows + 1) * vocab_size * bytes_per_element > chunk_bytes
