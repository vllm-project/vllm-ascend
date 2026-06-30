# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v4_dspark import (
    _copy_last_input_ids,
    _dspark_cache_capacity,
    _headwise_scores,
    _headwise_weighted_sum,
)


def _dspark_attention_loop(
    q: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    rows = []
    for token_idx in range(q.shape[0]):
        scores = _headwise_scores(q[token_idx], k_ctx) * scale
        sink = attn_sink[: q.shape[1]].float().unsqueeze(-1)
        scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
        exp_scores = torch.exp(scores - scores_max)
        probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
        rows.append(_headwise_weighted_sum(probs, v_ctx, q.dtype))
    return torch.stack(rows, dim=0)


def _dspark_attention_reference(
    q: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    scores = torch.einsum("qhd,khd->qhk", q.float(), k_ctx.float()) * scale
    sink = attn_sink[: q.shape[1]].float().view(1, q.shape[1], 1)
    scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
    exp_scores = torch.exp(scores - scores_max)
    probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
    return torch.einsum("qhk,khd->qhd", probs, v_ctx.float()).to(q.dtype)


def test_dspark_attention_loop_matches_vectorized_reference():
    torch.manual_seed(0)
    q = torch.randn(4, 3, 8, dtype=torch.float32)
    k_ctx = torch.randn(9, 3, 8, dtype=torch.float32)
    v_ctx = torch.randn(9, 3, 8, dtype=torch.float32)
    attn_sink = torch.randn(3, dtype=torch.float32)

    actual = _dspark_attention_loop(q, k_ctx, v_ctx, attn_sink, scale=0.125)
    expected = _dspark_attention_reference(q, k_ctx, v_ctx, attn_sink, scale=0.125)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_dspark_attention_is_noncausal_within_draft_block():
    ctx_len = 2
    block = 4
    heads = 1
    dim = 4
    scale = 1.0

    q = torch.ones(block, heads, dim, dtype=torch.float32)
    k_ctx = torch.zeros(ctx_len + block, heads, dim, dtype=torch.float32)
    v_ctx = torch.zeros(ctx_len + block, heads, dim, dtype=torch.float32)

    # The last draft token is a future token for the first query. Make it
    # dominate softmax if the implementation truly attends non-causally.
    k_ctx[-1].fill_(4.0)
    v_ctx[-1].fill_(10.0)
    attn_sink = torch.full((heads,), -100.0, dtype=torch.float32)

    noncausal = _dspark_attention_loop(q, k_ctx, v_ctx, attn_sink, scale)
    causal_first = _dspark_attention_reference(
        q[:1],
        k_ctx[: ctx_len + 1],
        v_ctx[: ctx_len + 1],
        attn_sink,
        scale,
    )

    assert noncausal[0].mean().item() > 9.0
    assert causal_first[0].mean().item() == 0.0
    assert (noncausal[0] - causal_first[0]).abs().max().item() > 9.0


def test_dspark_last_input_ids_copy_keeps_buffer_storage():
    buffer = torch.empty(8, dtype=torch.int32)
    storage_ptr = buffer.data_ptr()

    copied = _copy_last_input_ids(buffer, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    assert copied == 4
    assert buffer.data_ptr() == storage_ptr
    torch.testing.assert_close(buffer[:copied], torch.tensor([1, 2, 3, 4], dtype=torch.int32))

    copied = _copy_last_input_ids(buffer, torch.tensor([9, 8], dtype=torch.int64))
    assert copied == 2
    assert buffer.data_ptr() == storage_ptr
    torch.testing.assert_close(buffer[:copied], torch.tensor([9, 8], dtype=torch.int32))


def test_dspark_last_input_ids_copy_rejects_oversized_input():
    buffer = torch.empty(2, dtype=torch.int32)

    with pytest.raises(ValueError, match="preallocated buffer capacity"):
        _copy_last_input_ids(buffer, torch.tensor([1, 2, 3], dtype=torch.int32))


def test_dspark_attention_cache_capacity_includes_draft_block():
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096))

    assert _dspark_cache_capacity(vllm_config, block_size=5) == 4101
    assert _dspark_cache_capacity(SimpleNamespace(model_config=None), block_size=5) == 5
