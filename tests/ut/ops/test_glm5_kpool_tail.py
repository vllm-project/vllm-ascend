# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the kpool causal-tail append and the pool budget arithmetic.

The checks compare against an independent reference, not the
implementation:

1. ``_append_causal_tail``: each query's tail slots hold exactly the
   causal tokens not covered by any completed pool (floor semantics),
   ``-1`` elsewhere.
2. Pass-through: the already-selected columns are returned unmodified and
   the tail columns close the gap up to the query's own position.
3. Budget arithmetic: ``pool_budget = topk_tokens // kpool`` (512 pools
   for 2048/4), expand width 2048, ``+kpool-1`` tail columns, padded
   buffer width 2176.
"""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")  # noqa: F401  (registers the NPU backend)

from vllm_ascend.ops.glm5_kpool_indexer_ascend import (  # noqa: E402
    AscendGlm5KpoolIndexerOp,
)


class _FakeIndexerMetadata:
    def __init__(self, pool_lens: list[int]):
        self.seq_lens = torch.tensor(pool_lens, dtype=torch.int32)


def _reference_tail(
    positions: torch.Tensor,
    pool_lens_per_req: list[int],
    cum_query_lens: list[int],
    kpool: int,
) -> torch.Tensor:
    """Independent per-query expectation: [T, kpool-1] of token ids / -1."""
    out = torch.full((positions.shape[0], kpool - 1), -1, dtype=torch.int64)
    req_of = torch.bucketize(
        torch.arange(positions.shape[0]), torch.tensor(cum_query_lens), right=True
    )
    for i, p in enumerate(positions.tolist()):
        cached = pool_lens_per_req[req_of[i].item()]
        pools = min((p + 1) // kpool, cached)
        tail_start = pools * kpool
        tail_count = (p + 1) - tail_start
        for j in range(min(tail_count, kpool - 1)):
            out[i, j] = tail_start + j
    return out


@pytest.mark.parametrize(
    "positions,pool_lens,cum_query_lens",
    [
        pytest.param([9], [2], [1], id="decode-p9-two-tail-tokens"),
        pytest.param([8], [2], [1], id="decode-p8-one-tail-token"),
        pytest.param([7], [2], [1], id="decode-p7-no-tail"),
        pytest.param([0], [0], [1], id="first-token-p0"),
        pytest.param([2], [0], [1], id="short-seq-p2-three-tail-tokens"),
        pytest.param([12, 13, 14, 15], [4], [4], id="prefill-queries-12-to-15"),
        pytest.param([3, 4, 9, 10], [5, 1], [3, 5], id="two-requests"),
        pytest.param([9], [1], [1], id="clamped-to-cached-pools"),
    ],
)
def test_append_causal_tail_matches_reference(
    positions: list[int], pool_lens: list[int], cum_query_lens: list[int]
):
    kpool = 4
    positions_t = torch.tensor(positions, dtype=torch.int64)
    indexer_meta = _FakeIndexerMetadata(pool_lens)
    cum = torch.tensor(cum_query_lens, dtype=torch.int32)
    num_tokens = positions_t.shape[0]

    # Arbitrary non-tail content in the already-selected columns.
    expanded = (
        torch.arange(num_tokens * 2048, dtype=torch.int32)
        .reshape(num_tokens, -1)
        % 100000
    )

    got = AscendGlm5KpoolIndexerOp._append_causal_tail(
        expanded, positions_t, indexer_meta, cum, kpool
    )

    assert got.shape == (num_tokens, expanded.shape[1] + kpool - 1)
    # The selected-pool region must pass through unmodified.
    assert torch.equal(got[:, : expanded.shape[1]], expanded)

    ref = _reference_tail(positions_t, pool_lens, cum_query_lens, kpool)
    assert torch.equal(got[:, expanded.shape[1] :].to(torch.int64), ref)


def test_append_causal_tail_kpool1_passthrough():
    positions = torch.tensor([5], dtype=torch.int64)
    expanded = torch.zeros(1, 8, dtype=torch.int32)
    got = AscendGlm5KpoolIndexerOp._append_causal_tail(
        expanded, positions, _FakeIndexerMetadata([5]), torch.tensor([1]), 1
    )
    assert got.shape == (1, 8)


def test_pool_budget_arithmetic():
    """The GLM-5.3-Flash configuration must divide evenly."""
    topk_tokens, kpool = 2048, 4
    pool_budget = topk_tokens // kpool if topk_tokens % kpool == 0 else topk_tokens
    buffer_width = math.ceil((topk_tokens + kpool - 1) / 128) * 128
    assert pool_budget == 512
    assert pool_budget * kpool == 2048
    assert buffer_width == 2176
