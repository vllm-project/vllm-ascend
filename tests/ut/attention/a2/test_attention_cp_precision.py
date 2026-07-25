# SPDX-License-Identifier: Apache-2.0

import math

import torch

from vllm_ascend.attention.context_parallel.common_cp import _update_out_and_lse


def _attention_partial(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.einsum("thd,shd->ths", query, key) / math.sqrt(query.shape[-1])
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    output = torch.einsum("ths,shd->thd", torch.softmax(scores, dim=-1), value)
    return output, lse


def test_dcp_sharded_kv_precision_matches_unsharded_attention() -> None:
    torch.manual_seed(7)
    query = torch.randn(5, 3, 8, dtype=torch.float64)
    key = torch.randn(12, 3, 8, dtype=torch.float64)
    value = torch.randn(12, 3, 8, dtype=torch.float64)

    reference, _ = _attention_partial(query, key, value)
    partials = [
        _attention_partial(query, key_shard, value_shard)
        for key_shard, value_shard in zip(key.chunk(3), value.chunk(3))
    ]
    output, _ = _update_out_and_lse(
        torch.stack([partial[0] for partial in partials]),
        torch.stack([partial[1] for partial in partials]),
    )

    torch.testing.assert_close(output, reference, rtol=1e-10, atol=1e-10)
