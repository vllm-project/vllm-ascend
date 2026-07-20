# SPDX-License-Identifier: Apache-2.0

import math

import torch

from vllm_ascend.attention.context_parallel.common_cp import _update_out_and_lse


def test_mla_dcp_partial_softmax_precision() -> None:
    torch.manual_seed(19)
    query_nope = torch.randn(4, 2, 6, dtype=torch.float64)
    query_rope = torch.randn(4, 2, 2, dtype=torch.float64)
    query = torch.cat([query_nope, query_rope], dim=-1)
    key = torch.randn(10, 2, 8, dtype=torch.float64)
    value = torch.randn(10, 2, 5, dtype=torch.float64)

    full_scores = torch.einsum("thd,shd->ths", query, key) / math.sqrt(8)
    reference = torch.einsum(
        "ths,shd->thd",
        torch.softmax(full_scores, dim=-1),
        value,
    )

    partial_outputs = []
    partial_lse = []
    for key_shard, value_shard in zip(key.chunk(2), value.chunk(2)):
        scores = torch.einsum("thd,shd->ths", query, key_shard) / math.sqrt(8)
        partial_outputs.append(
            torch.einsum("ths,shd->thd", torch.softmax(scores, dim=-1), value_shard)
        )
        partial_lse.append(torch.logsumexp(scores, dim=-1, keepdim=True))

    output, _ = _update_out_and_lse(
        torch.stack(partial_outputs),
        torch.stack(partial_lse),
    )
    torch.testing.assert_close(output, reference, rtol=1e-10, atol=1e-10)
