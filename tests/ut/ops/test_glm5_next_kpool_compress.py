# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the kpool compress fallback's paged-cache writes.

The eager fallback must honor the same slot contract as the triton kernel:
``loc == -1`` (graph-captured padded rows) means "no write" -- a raw ``-1``
would index the LAST cache block via negative indexing and silently corrupt
it. CPU tensors keep these tests on the fallback path (``_can_use_triton``
requires an NPU tensor), on any host.
"""

from __future__ import annotations

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")  # noqa: F401  (registers the NPU backend)

from vllm_ascend.ops.glm5_next_kpool_compress import (  # noqa: E402
    glm5_next_kpool_compress_and_write_cache,
)

CACHE_BLOCKS, BLOCK_SIZE, HEAD_DIM = 4, 128, 64


def _reference_compress(slot_k: torch.Tensor, slot_score: torch.Tensor, ape: torch.Tensor):
    scores = slot_score.float() + ape.float().unsqueeze(0)
    return (torch.softmax(scores, dim=1) * slot_k.float()).sum(dim=1).to(torch.bfloat16)


def _inputs(num_rows: int, pool: int = 2, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    slot_k = torch.randn(num_rows, pool, HEAD_DIM, generator=gen).to(torch.bfloat16)
    slot_score = torch.randn(num_rows, pool, HEAD_DIM, generator=gen).to(torch.bfloat16)
    ape = torch.randn(pool, HEAD_DIM, generator=gen)
    return slot_k, slot_score, ape


def _rand_cache(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM, generator=gen).to(
        torch.bfloat16
    )


def test_fallback_skips_invalid_slots():
    kv_cache = torch.zeros(CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    before = kv_cache.clone()
    slot_k, slot_score, ape = _inputs(3)
    # Rows 0/1 address valid slots; row 2 is a padded graph row (-1).
    loc = torch.tensor([5, 2 * BLOCK_SIZE + 44, -1], dtype=torch.int64)

    glm5_next_kpool_compress_and_write_cache(kv_cache, slot_k, slot_score, ape, loc)

    expected = _reference_compress(slot_k, slot_score, ape)
    assert torch.equal(kv_cache[0, 5, 0, :], expected[0])
    assert torch.equal(kv_cache[2, 44, 0, :], expected[1])
    # The -1 row must not write anywhere: a raw negative index would have
    # landed in the last block (kv_cache[-1, BLOCK_SIZE - 1]).
    assert torch.equal(kv_cache[3], before[3])
    # And nothing outside the two valid slots changed.
    untouched = before.clone()
    untouched[0, 5] = kv_cache[0, 5]
    untouched[2, 44] = kv_cache[2, 44]
    assert torch.equal(kv_cache, untouched)


def test_fallback_all_invalid_slots_is_noop():
    kv_cache = _rand_cache(1)
    before = kv_cache.clone()
    slot_k, slot_score, ape = _inputs(2, seed=2)
    loc = torch.full((2,), -1, dtype=torch.int64)

    glm5_next_kpool_compress_and_write_cache(kv_cache, slot_k, slot_score, ape, loc)
    assert torch.equal(kv_cache, before)


def test_fallback_empty_batch_is_noop():
    kv_cache = _rand_cache(3)
    before = kv_cache.clone()
    slot_k = torch.zeros(0, 2, HEAD_DIM, dtype=torch.bfloat16)
    slot_score = torch.zeros(0, 2, HEAD_DIM, dtype=torch.bfloat16)
    ape = torch.randn(2, HEAD_DIM)
    loc = torch.zeros(0, dtype=torch.int64)

    glm5_next_kpool_compress_and_write_cache(kv_cache, slot_k, slot_score, ape, loc)
    assert torch.equal(kv_cache, before)
