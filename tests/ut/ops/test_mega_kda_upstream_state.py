# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.mega_kda import UpstreamMegaKDA, run_upstream_mega_kda
from vllm_ascend.ops.mega_kda_upstream_state import (
    prepare_token_layout,
    prepare_upstream_state_plan,
    run_upstream_state_batch,
)


@pytest.mark.parametrize("padded_pages", [False, True])
def test_upstream_state_writes_only_real_request_rows(padded_pages):
    window, dim = 3, 2
    hidden = torch.arange(5 * dim, dtype=torch.bfloat16).reshape(5, dim)
    prefix = torch.tensor([0, 2, 2, 3, 4], dtype=torch.int32)
    conv_slots = torch.tensor([6, -1, 2, 0], dtype=torch.int32)
    slots = torch.tensor([[3, 4, 5], [-1, -1, -1], [7, 8, 9], [0, 0, 0]], dtype=torch.int32)
    accepted = torch.tensor([2, 1, 1, 1], dtype=torch.int32)
    initial = torch.tensor([1, 1, 0, 1])
    conv = torch.arange(10 * (window + 2) * dim, dtype=torch.bfloat16).reshape(10, window + 2, dim)
    state = torch.arange(10 * dim**2, dtype=torch.float32).reshape(10, 1, dim, dim)
    if padded_pages:
        conv_storage = torch.full((10, (window + 3) * dim + 3), -1, dtype=torch.bfloat16)
        state_storage = torch.full((10, dim**2 + 5), -1, dtype=torch.float32)
        conv_view = conv_storage[:, : (window + 3) * dim].view(10, window + 3, dim)
        state_view = state_storage[:, : dim**2].view(10, 1, dim, dim)
        conv_view[:, : window + 2].copy_(conv)
        state_view.copy_(state)
        conv, state = conv_view, state_view
    old_conv, old_state = conv.clone(), state.clone()

    def launch(x, history, checkpoint, u, k, decay, ids, counts, *, cu_seqlens, metadata):
        # The public ABI supplies a read-only checkpoint and three separate
        # [request,head,step,channel] replay factors.
        assert torch.equal(ids, torch.arange(4, dtype=torch.int32))
        assert counts.tolist() == [1] * 4
        assert (cu_seqlens[1:] > cu_seqlens[:-1]).all()
        assert metadata is plan.metadata
        assert torch.equal(checkpoint[0], old_state[4])
        assert checkpoint[2].count_nonzero() == 0
        assert history[2, :3].count_nonzero() == 0
        u.fill_(2.0)
        k.fill_(3.0)
        decay.fill_(0.5)
        for row, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
            history[row, :2] = history[row, 1:3].clone()
            history[row, 2 : 2 + end - start] = x[start:end]
        return x + 10, history, u, k, decay

    plan = prepare_upstream_state_plan(
        prepare_token_layout(prefix, 5, window, lambda cu, tokens: cu.clone()),
        slots,
        conv_slots,
        accepted,
        initial,
        max_query_len=window,
        num_state_slots=10,
        num_conv_slots=10,
    )
    out = torch.zeros_like(hidden)
    run_upstream_state_batch(hidden, conv, state, plan, launch, out)
    expected_out = torch.zeros_like(hidden)
    expected_out[:3] = hidden[:3] + 10
    assert torch.equal(out, expected_out)
    expected_conv = old_conv.clone()
    expected_conv[6, :2] = old_conv[6, 2:4]
    expected_conv[6, 2:4] = hidden[:2]
    expected_conv[2, :2].zero_()
    expected_conv[2, 2] = hidden[2]
    assert torch.equal(conv, expected_conv)
    expected_state = old_state.clone()
    expected_state[3] = 0.5 * old_state[4] + 6
    expected_state[4] = 0.5 * expected_state[3] + 6
    expected_state[7].fill_(6)
    assert torch.equal(state, expected_state)
    if padded_pages:
        assert (conv_storage[:, (window + 3) * dim :] == -1).all()
        assert (state_storage[:, dim**2 :] == -1).all()


def test_full_speculative_window_preserves_official_th_plan():
    prefix = torch.arange(0, 121, 4, dtype=torch.int32)
    slots = torch.arange(1, 31, dtype=torch.int32)
    ones = torch.ones_like(slots)
    plan = prepare_upstream_state_plan(
        prepare_token_layout(prefix, 120, 4, lambda cu, tokens: torch.tensor(tokens)),
        slots[:, None].expand(-1, 4),
        slots,
        ones,
        ones,
        max_query_len=4,
        num_state_slots=31,
        num_conv_slots=31,
    )
    assert plan.metadata.item() == 120
    assert torch.equal(plan.cu_seqlens, prefix)
    assert torch.equal(plan.token_indices, torch.arange(120))
    assert plan.real_tokens.all()


@torch.inference_mode()
def test_forward_reuses_layout_but_keeps_layer_pages_and_output_rows_separate(monkeypatch):
    import vllm.forward_context as forward_context

    context = SimpleNamespace(additional_kwargs={})
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    build = Mock(side_effect=lambda prefix, tokens: prefix.clone())
    prefix = torch.tensor([0, 2, 2], dtype=torch.int32)
    slots = torch.tensor([[1, 2], [0, 0], [3, 4]], dtype=torch.int32)
    conv_slots = torch.tensor([1, 0], dtype=torch.int32)
    accepted = torch.ones(2, dtype=torch.int32)
    spec_rows = torch.tensor([2, 0], dtype=torch.int32)
    ordinary = torch.tensor([3, 0], dtype=torch.int32)
    ordinary_rows = torch.tensor([1, 3], dtype=torch.int32)
    ordinary_prefix = torch.arange(3, dtype=torch.int32)
    hidden = torch.arange(10, dtype=torch.bfloat16).reshape(5, 2)

    def forward(**kw):
        x = kw["hidden_states"]
        return x + 1, kw["conv_state"], kw["replay_u"], kw["replay_k"], kw["replay_decay"]

    def run(table, cu):
        # A new wrapper and new views, as can happen between attention layers.
        meta = SimpleNamespace(
            num_actual_tokens=4,
            num_spec_decodes=2,
            num_prefills=0,
            num_decodes=2,
            spec_token_indx=spec_rows[:],
            non_spec_token_indx=ordinary_rows[:],
            spec_state_indices_tensor=table,
            non_spec_state_indices_tensor=ordinary[:],
            spec_decode_metadata=SimpleNamespace(
                spec_causal_conv1d=SimpleNamespace(
                    query_start_loc=cu,
                    cache_indices=conv_slots[:],
                    num_accepted_tokens=accepted[:],
                )
            ),
            non_spec_decode_metadata=SimpleNamespace(
                causal_conv1d=SimpleNamespace(
                    query_start_loc=ordinary_prefix[:],
                    cache_indices=ordinary[:],
                )
            ),
        )
        state = torch.ones(5, 1, 2, 2)
        layer = SimpleNamespace(
            kv_cache=(torch.zeros(5, 4, 2, dtype=torch.bfloat16), state),
            num_spec=1,
            head_dim=2,
            gate_lower_bound=-1,
            o_norm=SimpleNamespace(eps=1e-5),
        )
        out = torch.full_like(hidden, -99)
        run_upstream_mega_kda(layer, UpstreamMegaKDA(forward, build), meta, hidden, {}, output=out)
        return out, state

    expected = torch.zeros_like(hidden)
    expected[:3] = hidden[:3] + 1
    for _ in range(2):
        out, state = run(slots[:2], prefix[:])
        torch.testing.assert_close(out, expected)
        assert state[1:4].count_nonzero() == 0
        assert state[0].min() == 1
    assert build.call_count == 1
    assert len(context.additional_kwargs["mega_kda_state_plans"]) == 2

    # Same shape and storage, different offset: spec now targets an inactive page.
    out, state = run(slots[1:], prefix[:])
    expected[[0, 2]] = 0
    torch.testing.assert_close(out, expected)
    assert state[1:3].min() == 1
    assert build.call_count == 1
    assert len(context.additional_kwargs["mega_kda_state_plans"]) == 3

    run(slots[:2], prefix.clone())
    assert build.call_count == 2
    context.additional_kwargs.clear()
    run(slots[:2], prefix[:])
    assert build.call_count == 3
