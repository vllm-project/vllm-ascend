# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.ops.dspark_attention as dspark_attention_module
from vllm_ascend.models.deepseek_v4_dspark import (
    _dspark_cache_capacity,
)
from vllm_ascend.ops.dspark_attention import (
    _dspark_sas_window,
    _gather_context_kv,
    _validate_query_block_slots,
    dspark_attention,
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
        scores = torch.einsum("hd,khd->hk", q[token_idx].float(), k_ctx.float()) * scale
        sink = attn_sink[: q.shape[1]].float().unsqueeze(-1)
        scores_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
        exp_scores = torch.exp(scores - scores_max)
        probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink - scores_max))
        rows.append(torch.einsum("hk,khd->hd", probs, v_ctx.float()).to(q.dtype))
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


def _band_visible_indices(
    query_idx: int,
    s1_size: int,
    s2_size: int,
    ori_win_left: int,
    ori_win_right: int,
) -> list[int]:
    left = max(s2_size - s1_size + query_idx - ori_win_left, 0)
    right = min(s2_size - s1_size + query_idx + ori_win_right, s2_size - 1)
    if right < left:
        return []
    return list(range(left, right + 1))


def test_dspark_attention_loop_matches_vectorized_reference():
    torch.manual_seed(0)
    q = torch.randn(4, 3, 8, dtype=torch.float32)
    k_ctx = torch.randn(9, 3, 8, dtype=torch.float32)
    v_ctx = torch.randn(9, 3, 8, dtype=torch.float32)
    attn_sink = torch.randn(3, dtype=torch.float32)

    actual = _dspark_attention_loop(q, k_ctx, v_ctx, attn_sink, scale=0.125)
    expected = _dspark_attention_reference(q, k_ctx, v_ctx, attn_sink, scale=0.125)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_dspark_sas_window_covers_context_and_full_draft_block():
    block_size = 5
    window_size = 7
    context_len = window_size
    s2_size = context_len + block_size

    mask_mode, ori_win_left, ori_win_right = _dspark_sas_window(block_size, window_size)

    assert mask_mode == 4
    assert ori_win_left == window_size + block_size - 1
    assert ori_win_right == block_size - 1
    for query_idx in range(block_size):
        assert _band_visible_indices(
            query_idx,
            block_size,
            s2_size,
            ori_win_left,
            ori_win_right,
        ) == list(range(s2_size))


def test_dspark_sas_window_rejects_plain_sliding_window_formula():
    block_size = 5
    window_size = 7
    context_len = window_size
    s2_size = context_len + block_size

    bad_left = window_size - 1
    right = block_size - 1

    assert _band_visible_indices(
        block_size - 1,
        block_size,
        s2_size,
        bad_left,
        right,
    ) != list(range(s2_size))


def test_dspark_attention_entry_uses_custom_op_gateway(monkeypatch):
    q = torch.zeros(2, 1, 4, dtype=torch.float32)
    expected = torch.full_like(q, 7.0)

    def fake_custom_op(*args):
        assert args[0] is q
        assert args[-3:] == (2, 3, 0.125)
        return expected

    monkeypatch.setattr(
        dspark_attention_module,
        "_get_dspark_attention_custom_op",
        lambda _q: fake_custom_op,
    )
    actual = dspark_attention(
        q,
        torch.empty(1, 4, 1, 4),
        torch.empty(1, 4, 1, 4),
        torch.empty(1, 4, dtype=torch.int32),
        torch.empty(1, 4, dtype=torch.bool),
        torch.empty(2, 1, 4),
        torch.empty(2, 1, 4),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([3, 4], dtype=torch.int32),
        torch.zeros(1),
        2,
        3,
        0.125,
    )

    assert actual is expected


def test_dspark_attention_entry_uses_sas_gateway(monkeypatch):
    q = torch.ones(2, 4, 4, dtype=torch.float32)
    draft_k = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 4, 4)
    block_size = 2
    window_size = 3
    calls = []

    def fake_metadata_op(**kwargs):
        assert kwargs["num_heads_q"] == 4
        assert kwargs["num_heads_kv"] == 1
        assert kwargs["ori_win_left"] == window_size + block_size - 1
        assert kwargs["ori_win_right"] == block_size - 1
        assert kwargs["layout_q"] == "TND"
        assert kwargs["layout_kv"] == "TND"
        assert kwargs["has_cmp_kv"] is False
        return torch.empty(1024, dtype=torch.int32)

    def fake_attn_op(q_block, **kwargs):
        assert kwargs["ori_kv"].shape == (2, 1, 4)
        assert kwargs["sinks"].dtype == torch.float32
        assert kwargs["sinks"].shape == (4,)
        assert kwargs["ori_win_left"] == window_size + block_size - 1
        assert kwargs["ori_win_right"] == block_size - 1
        calls.append(kwargs)
        return q_block + 2, torch.empty(0)

    monkeypatch.setattr(
        dspark_attention_module,
        "_get_dspark_attention_custom_op",
        lambda _q: None,
    )
    monkeypatch.setattr(
        dspark_attention_module,
        "_get_dspark_sas_ops",
        lambda _q: (fake_metadata_op, fake_attn_op),
    )

    actual = dspark_attention(
        q,
        torch.empty(1, 4, 4, 4),
        torch.empty(1, 4, 4, 4),
        torch.empty(1, 4, dtype=torch.int32),
        torch.empty(1, 4, dtype=torch.bool),
        draft_k,
        draft_k,
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.zeros(4, dtype=torch.bfloat16),
        block_size,
        window_size,
        0.125,
        shared_kv=True,
    )

    assert len(calls) == 1
    torch.testing.assert_close(actual, q + 2)


def test_dspark_attention_entry_does_not_use_sas_without_shared_kv(monkeypatch):
    q = torch.ones(2, 1, 4, dtype=torch.float32)
    draft_k = torch.ones(2, 1, 4, dtype=torch.float32)
    draft_v = draft_k + 1

    monkeypatch.setattr(
        dspark_attention_module,
        "_get_dspark_attention_custom_op",
        lambda _q: None,
    )

    def fail_if_sas_is_used(_q):
        raise AssertionError("SAS gateway must be gated by shared_kv=True")

    monkeypatch.setattr(
        dspark_attention_module,
        "_get_dspark_sas_ops",
        fail_if_sas_is_used,
    )

    actual = dspark_attention(
        q,
        torch.empty(1, 4, 1, 4),
        torch.empty(1, 4, 1, 4),
        torch.empty(1, 4, dtype=torch.int32),
        torch.empty(1, 4, dtype=torch.bool),
        draft_k,
        draft_v,
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.zeros(1),
        2,
        3,
        0.125,
    )

    torch.testing.assert_close(
        actual,
        _dspark_attention_reference(q, draft_k, draft_v, torch.zeros(1), 0.125),
    )


def test_dspark_attention_warns_when_sas_falls_back(monkeypatch):
    q = torch.ones(2, 1, 4, dtype=torch.float32)
    draft_k = torch.ones(2, 1, 4, dtype=torch.float32)
    warnings = []

    def fake_metadata_op(**kwargs):
        return torch.empty(1, dtype=torch.int32)

    def fake_attn_op(*args, **kwargs):
        raise RuntimeError("synthetic sas failure")

    monkeypatch.setattr(
        dspark_attention_module,
        "_get_dspark_attention_custom_op",
        lambda _q: None,
    )
    monkeypatch.setattr(
        dspark_attention_module,
        "_get_dspark_sas_ops",
        lambda _q: (fake_metadata_op, fake_attn_op),
    )
    monkeypatch.setattr(
        dspark_attention_module.logger,
        "warning_once",
        lambda *args, **kwargs: warnings.append(args),
    )

    actual = dspark_attention(
        q,
        torch.empty(1, 4, 1, 4),
        torch.empty(1, 4, 1, 4),
        torch.empty(1, 4, dtype=torch.int32),
        torch.empty(1, 4, dtype=torch.bool),
        draft_k,
        draft_k,
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.zeros(1),
        2,
        3,
        0.125,
        shared_kv=True,
    )

    assert warnings
    assert "DSpark SAS attention failed" in warnings[0][0]
    torch.testing.assert_close(
        actual,
        _dspark_attention_reference(q, draft_k, draft_k, torch.zeros(1), 0.125),
    )


def test_dspark_attention_shared_kv_fallback_uses_k_as_v(monkeypatch):
    q = torch.ones(2, 1, 4, dtype=torch.float32)
    draft_k = torch.ones(2, 1, 4, dtype=torch.float32)
    draft_v = torch.full_like(draft_k, 100.0)

    def fake_metadata_op(**kwargs):
        return torch.empty(1, dtype=torch.int32)

    def fake_attn_op(*args, **kwargs):
        raise RuntimeError("synthetic sas failure")

    monkeypatch.setattr(
        dspark_attention_module,
        "_get_dspark_attention_custom_op",
        lambda _q: None,
    )
    monkeypatch.setattr(
        dspark_attention_module,
        "_get_dspark_sas_ops",
        lambda _q: (fake_metadata_op, fake_attn_op),
    )

    actual = dspark_attention(
        q,
        torch.empty(1, 4, 1, 4),
        torch.empty(1, 4, 1, 4).fill_(100.0),
        torch.empty(1, 4, dtype=torch.int32),
        torch.empty(1, 4, dtype=torch.bool),
        draft_k,
        draft_v,
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.zeros(1),
        2,
        3,
        0.125,
        shared_kv=True,
    )

    torch.testing.assert_close(
        actual,
        _dspark_attention_reference(q, draft_k, draft_k, torch.zeros(1), 0.125),
    )


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


def test_dspark_attention_entry_matches_reference_with_request_cache():
    torch.manual_seed(1)
    block_size = 2
    window_size = 3
    heads = 2
    dim = 4
    q = torch.randn(4, heads, dim, dtype=torch.float32)
    draft_k = torch.randn(4, heads, dim, dtype=torch.float32)
    draft_v = torch.randn(4, heads, dim, dtype=torch.float32)
    positions = torch.tensor([5, 6, 9, 10], dtype=torch.int32)
    request_slots = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    attn_sink = torch.tensor([0.25, -0.5], dtype=torch.float32)
    scale = 0.125

    cache_k = torch.zeros(2, 8, heads, dim, dtype=torch.float32)
    cache_v = torch.zeros(2, 8, heads, dim, dtype=torch.float32)
    cache_positions = torch.full((2, 8), -1, dtype=torch.int32)
    cache_valid = torch.zeros(2, 8, dtype=torch.bool)

    for slot, ctx_positions in [(0, torch.tensor([3, 4, 5])), (1, torch.tensor([7, 8, 9]))]:
        values = torch.randn(ctx_positions.numel(), heads, dim, dtype=torch.float32)
        indices = ctx_positions % cache_k.shape[1]
        cache_k[slot, indices] = values
        cache_v[slot, indices] = values + (slot + 1)
        cache_positions[slot, indices] = ctx_positions.to(torch.int32)
        cache_valid[slot, indices] = True

    actual = dspark_attention(
        q,
        cache_k,
        cache_v,
        cache_positions,
        cache_valid,
        draft_k,
        draft_v,
        request_slots,
        positions,
        attn_sink,
        block_size,
        window_size,
        scale,
    )

    expected_blocks = []
    for block_offset, slot, ctx_start, ctx_end in [(0, 0, 3, 4), (2, 1, 7, 8)]:
        ctx_positions = torch.arange(ctx_start, ctx_end + 1)
        ctx_indices = ctx_positions % cache_k.shape[1]
        k_ctx = torch.cat([cache_k[slot, ctx_indices], draft_k[block_offset : block_offset + block_size]], dim=0)
        v_ctx = torch.cat([cache_v[slot, ctx_indices], draft_v[block_offset : block_offset + block_size]], dim=0)
        expected_blocks.append(
            _dspark_attention_reference(
                q[block_offset : block_offset + block_size],
                k_ctx,
                v_ctx,
                attn_sink,
                scale,
            )
        )
    expected = torch.cat(expected_blocks, dim=0)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_dspark_attention_cache_capacity_includes_draft_block():
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096))

    assert _dspark_cache_capacity(vllm_config, block_size=5) == 4101
    assert _dspark_cache_capacity(vllm_config, block_size=5, window_size=128) == 133
    assert _dspark_cache_capacity(SimpleNamespace(model_config=None), block_size=5) == 5


def test_dspark_context_cache_is_request_local_and_rolling():
    cache_k = torch.zeros(2, 4, 1, 1, dtype=torch.float32)
    cache_v = torch.zeros(2, 4, 1, 1, dtype=torch.float32)
    cache_positions = torch.full((2, 4), -1, dtype=torch.int32)
    cache_valid = torch.zeros(2, 4, dtype=torch.bool)

    # Both requests write the same absolute positions. They alias on rolling
    # indices, but must stay isolated by request slot.
    for slot, value in [(0, 10.0), (1, 20.0)]:
        positions = torch.tensor([4, 5], dtype=torch.long)
        indices = positions % cache_k.shape[1]
        cache_k[slot, indices] = value
        cache_v[slot, indices] = value + 1
        cache_positions[slot, indices] = positions.to(torch.int32)
        cache_valid[slot, indices] = True

    k0, v0 = _gather_context_kv(cache_k, cache_v, cache_positions, cache_valid, 0, 4, 5)
    k1, v1 = _gather_context_kv(cache_k, cache_v, cache_positions, cache_valid, 1, 4, 5)

    torch.testing.assert_close(k0.flatten(), torch.tensor([10.0, 10.0]))
    torch.testing.assert_close(v0.flatten(), torch.tensor([11.0, 11.0]))
    torch.testing.assert_close(k1.flatten(), torch.tensor([20.0, 20.0]))
    torch.testing.assert_close(v1.flatten(), torch.tensor([21.0, 21.0]))

    # Position 0 shares rolling index with position 4, but the stored absolute
    # position prevents stale context from being reused.
    k_stale, v_stale = _gather_context_kv(cache_k, cache_v, cache_positions, cache_valid, 0, 0, 1)
    assert k_stale.numel() == 0
    assert v_stale.numel() == 0


def test_dspark_query_block_slots_must_not_mix_requests():
    _validate_query_block_slots(torch.tensor([0, 0, 1, 1], dtype=torch.int32), block_size=2)

    with pytest.raises(ValueError, match="constant within each draft block"):
        _validate_query_block_slots(torch.tensor([0, 1, 1, 1], dtype=torch.int32), block_size=2)
