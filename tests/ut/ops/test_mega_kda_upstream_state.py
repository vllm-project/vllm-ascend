# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.mega_kda_upstream_state import prepare_upstream_state_plan, run_upstream_state_batch


@pytest.mark.parametrize("recover", [False, True])
def test_upstream_state_writes_only_real_request_rows(recover):
    window, dim = 3, 2
    hidden = torch.arange(5 * dim, dtype=torch.bfloat16).reshape(5, dim)
    prefix = torch.tensor([0, 2, 2, 3, 4], dtype=torch.int32)
    conv_slots = torch.tensor([6, -1, 2, 0], dtype=torch.int32)
    slots = torch.tensor([[3, 4, 5], [-1, -1, -1], [7, 8, 9], [0, 0, 0]], dtype=torch.int32)
    accepted = torch.tensor([2, 1, 1, 1], dtype=torch.int32)
    if recover:
        slots = slots[:, :1]
        accepted.fill_(1)
    initial = torch.tensor([1, 1, 0, 1])
    conv = torch.arange(10 * (window + 2) * dim, dtype=torch.bfloat16).reshape(10, window + 2, dim)
    state = torch.arange(10 * dim**2, dtype=torch.float32).reshape(10, 1, dim, dim)
    old_conv, old_state = conv.clone(), state.clone()
    correction = torch.full((4, 1, window, dim), -1.0)
    normalized_key = torch.full_like(correction, -1.0)
    decay_record = torch.full_like(correction, -1.0)

    def launch(x, history, checkpoint, u, k, decay, ids, counts, *, cu_seqlens, metadata):
        # The public ABI supplies a read-only checkpoint and three separate
        # [request,head,step,channel] replay factors.
        assert torch.equal(ids, torch.arange(4, dtype=torch.int32))
        assert counts.tolist() == [1] * 4
        assert (cu_seqlens[1:] > cu_seqlens[:-1]).all()
        assert metadata is plan.metadata
        assert torch.equal(checkpoint[0], old_state[3 if recover else 4])
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
        recover=recover,
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
        records=(correction, normalized_key, decay_record) if recover else None,
        plan=plan,
    )
    expected_out = torch.zeros_like(hidden)
    expected_out[:3] = hidden[:3] + 10
    assert torch.equal(out, expected_out)
    expected_conv = old_conv.clone()
    history_start = 0 if recover else 1
    expected_conv[6, :2] = old_conv[6, history_start + 1 : history_start + 3]
    expected_conv[6, 2:4] = hidden[:2]
    expected_conv[2, :2].zero_()
    expected_conv[2, 2] = hidden[2]
    assert torch.equal(conv, expected_conv)
    if recover:
        assert torch.equal(state, old_state)
        valid = torch.tensor([[1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool)
        assert (correction[:, 0][valid] == 2).all()
        assert (normalized_key[:, 0][valid] == 3).all()
        assert (decay_record[:, 0][valid] == 0.5).all()
        assert (correction[:, 0][~valid] == -1).all()
        assert (normalized_key[:, 0][~valid] == -1).all()
        assert (decay_record[:, 0][~valid] == -1).all()
    else:
        expected_state = old_state.clone()
        expected_state[3] = 0.5 * old_state[4] + 6
        expected_state[4] = 0.5 * expected_state[3] + 6
        expected_state[7].fill_(6)
        assert torch.equal(state, expected_state)


def test_full_speculative_window_preserves_official_th_plan():
    prefix = torch.arange(0, 121, 4, dtype=torch.int32)
    slots = torch.arange(1, 31, dtype=torch.int32)
    ones = torch.ones_like(slots)
    plan = prepare_upstream_state_plan(
        prefix,
        slots,
        slots,
        ones,
        ones,
        num_tokens=120,
        max_query_len=4,
        num_state_slots=31,
        num_conv_slots=31,
        recover=True,
        build_metadata=lambda cu, tokens: torch.tensor(tokens),
    )
    assert plan.metadata.item() == 120
    assert torch.equal(plan.cu_seqlens, prefix)
    assert torch.equal(plan.token_indices, torch.arange(120))
    assert plan.real_tokens.all()
