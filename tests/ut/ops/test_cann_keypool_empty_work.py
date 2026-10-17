# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm_ascend.ops.cann_keypool import pool_key_indexer_nonempty


@pytest.mark.parametrize(
    "ends,lengths,tails",
    [
        ([3, 3], [0, 0], [3, 0]),  # CP rank 0, no completed pool.
        ([0, 3], [0, 0], [0, 3]),  # CP rank 2, leading empty request.
        ([0, 0], [0, 0], [0, 0]),  # Entirely padded graph replay.
        ([2, 3], [800, 900], [1, 2]),  # Real selection must remain unchanged.
    ],
)
def test_empty_work_guard_preserves_live_requests_and_hides_dummy(ends, lengths, tails):
    query = torch.randn(5, 2, 8)
    weights = torch.randn(5, 2)
    cache = torch.randn(16, 32, 1, 8)
    table = torch.arange(8, dtype=torch.int32).view(2, 4)
    q_ends = torch.tensor(ends, dtype=torch.int64)
    k_lengths = torch.tensor(lengths, dtype=torch.int64)
    tail = torch.tensor(tails, dtype=torch.int64)
    originals = [x.clone() for x in (query, weights, table, q_ends, k_lengths, tail)]

    def checked_kernel(q, k, w, tk, **kwargs):
        # Model the unsafe precondition: no usable query/key work. The adapter
        # must prevent it even when all real requests are padding.
        qlen = torch.diff(kwargs["actual_seq_q"], prepend=torch.zeros(1, dtype=torch.int64))
        assert torch.any((qlen > 0) & (kwargs["actual_seq_k"] > 0))
        assert torch.all(kwargs["actual_seq_k"] > 0)
        torch.testing.assert_close(q[:-1], query)
        torch.testing.assert_close(w[:-1], weights)
        torch.testing.assert_close(kwargs["actual_seq_q"][:-1], q_ends)
        torch.testing.assert_close(kwargs["block_table"][:-1], table)
        torch.testing.assert_close(tk[:-1], tail)
        nonempty = k_lengths > 0
        torch.testing.assert_close(kwargs["actual_seq_k"][:-1][nonempty], k_lengths[nonempty])
        assert k is cache
        # A dummy result must never become one of the real query rows.
        indices = torch.arange(q.shape[0], dtype=torch.int32)[:, None].expand(-1, 7).clone()
        indices[-1] = 99999
        return indices, torch.empty(0)

    result, _ = pool_key_indexer_nonempty(
        checked_kernel,
        query,
        cache,
        weights,
        tail,
        actual_seq_q=q_ends,
        actual_seq_k=k_lengths,
        block_table=table,
        topk=4,
        pool_size=4,
    )
    torch.testing.assert_close(result, torch.arange(5, dtype=torch.int32)[:, None].expand(-1, 7))
    for actual, original in zip((query, weights, table, q_ends, k_lengths, tail), originals):
        torch.testing.assert_close(actual, original)
