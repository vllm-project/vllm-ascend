# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.mega_kda_upstream_state import prepare_upstream_state_plan, run_upstream_state_batch


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
        prefix,
        slots,
        conv_slots,
        accepted,
        initial,
        num_tokens=5,
        max_query_len=window,
        num_state_slots=10,
        num_conv_slots=10,
        build_metadata=lambda cu, tokens: cu.clone(),
    )
    out = run_upstream_state_batch(
        hidden,
        conv,
        state,
        prefix,
        slots,
        conv_slots,
        accepted,
        initial,
        launch,
        max_query_len=window,
        plan=plan,
    )
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
        prefix,
        slots[:, None].expand(-1, 4),
        slots,
        ones,
        ones,
        num_tokens=120,
        max_query_len=4,
        num_state_slots=31,
        num_conv_slots=31,
        build_metadata=lambda cu, tokens: torch.tensor(tokens),
    )
    assert plan.metadata.item() == 120
    assert torch.equal(plan.cu_seqlens, prefix)
    assert torch.equal(plan.token_indices, torch.arange(120))
    assert plan.real_tokens.all()
