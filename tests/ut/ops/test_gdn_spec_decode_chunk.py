#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Speculative decode GDN over a draft window wider than the kernel token cap.

``recurrent_gated_delta_rule`` silently returns without writing any output when
a sequence carries more than ``_MAX_RECURRENT_GDN_TOKENS`` tokens, which is
exactly what a draft window wider than 8 hits. These tests pin the chunked
replay added in ``vllm_ascend.ops.gdn`` against an unbounded CPU reference.
"""

import pytest
import torch

from vllm_ascend.ops import gdn

HEADS = 2
HEAD_DIM = 4
SCALE = HEAD_DIM**-0.5
# Hard cap of the fla_npu AscendC kernel (MAX_MTP in
# recurrent_gated_delta_rule.h). Kept literal so the counter-proof does not
# follow the constant it is meant to guard.
KERNEL_CAP = 8


def _actual_seq_lengths(lengths: list[int]) -> torch.Tensor:
    """fla layout: entry 0 is unused, entry i + 1 is the token count of seq i."""
    return torch.tensor([0] + lengths, dtype=torch.int32)


def _query_start_loc(lengths: list[int]) -> torch.Tensor:
    return torch.tensor([0] + list(torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist()), dtype=torch.int32)


def _offsets(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(lengths, 0) - lengths


def _sequential_block(hidden: torch.Tensor, k_t: torch.Tensor) -> torch.Tensor:
    return (hidden * k_t.unsqueeze(-2)).sum(-1)


def _reference_step(
    hidden: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    g_t: torch.Tensor,
    beta_t: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One delta-rule step, matching the fla recurrence used by the 310P path."""
    hidden = hidden * torch.exp(g_t).view(-1, 1, 1)
    v_t = v_t - _sequential_block(hidden, k_t)
    v_t = v_t * beta_t.view(-1, 1)
    hidden = hidden + v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
    out_t = _sequential_block(hidden, q_t * scale)
    return hidden, out_t


def _run_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    lengths: list[int],
    slots: torch.Tensor,
    accepted: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unbounded CPU recurrence: what the kernel would compute without the cap."""
    out = torch.zeros_like(value)
    offsets = _offsets(torch.tensor(lengths, dtype=torch.int64))
    for seq, seq_len in enumerate(lengths):
        if seq_len == 0:
            continue
        start = int(offsets[seq])
        # ``num_accepted_tokens`` never exceeds the tokens the sequence carries.
        resume = start + min(max(accepted[seq], 1), seq_len) - 1
        hidden = state[int(slots[resume])].float().clone()
        for step in range(seq_len):
            token = start + step
            hidden, out_t = _reference_step(
                hidden,
                query[token].float(),
                key[token].float(),
                value[token].float(),
                gate[token].float(),
                beta[token].float(),
                SCALE,
            )
            out[token] = out_t.to(out.dtype)
            state[int(slots[token])] = hidden.to(state.dtype)
    return out, state


def _capped_kernel(max_tokens: list[int] | None = None):
    """Stand-in for ``fla_npu``'s kernel, including its >MAX_MTP bail-out.

    A sequence longer than ``gdn._MAX_RECURRENT_GDN_TOKENS`` leaves the output
    buffer untouched and its state slots stale, exactly like
    ``recurrent_gated_delta_rule.h`` returning early.
    """

    def kernel(
        query,
        key,
        value,
        state,
        *,
        g,
        beta,
        scale,
        actual_seq_lengths,
        ssm_state_indices,
        num_accepted_tokens,
    ):
        num_seqs = num_accepted_tokens.shape[0]
        lengths = [int(x) for x in actual_seq_lengths[1 : num_seqs + 1].tolist()]
        if max_tokens is not None:
            max_tokens.append(max(lengths) if lengths else 0)
        out = torch.zeros_like(value)
        offsets = _offsets(torch.tensor(lengths, dtype=torch.int64))
        for seq, seq_len in enumerate(lengths):
            if seq_len > KERNEL_CAP:
                continue
            if seq_len == 0:
                continue
            start = int(offsets[seq])
            resume = start + int(num_accepted_tokens[seq].item()) - 1
            hidden = state[int(ssm_state_indices[resume])].float().clone()
            for step in range(seq_len):
                token = start + step
                hidden, out_t = _reference_step(
                    hidden,
                    query[token].float(),
                    key[token].float(),
                    value[token].float(),
                    g[token].float(),
                    beta[token].float(),
                    float(scale),
                )
                out[token] = out_t.to(out.dtype)
                state[int(ssm_state_indices[token])] = hidden.to(state.dtype)
        return out

    return kernel


def _build_inputs(lengths: list[int], seed: int = 0):
    num_tokens = sum(lengths)
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(num_tokens, HEADS, HEAD_DIM, generator=generator)
    key = torch.randn(num_tokens, HEADS, HEAD_DIM, generator=generator)
    value = torch.randn(num_tokens, HEADS, HEAD_DIM, generator=generator)
    gate = -torch.rand(num_tokens, HEADS, generator=generator)
    beta = torch.rand(num_tokens, HEADS, generator=generator)
    # One dedicated state slot per recurrent token: the spec-decode block table.
    slots = torch.arange(num_tokens, dtype=torch.int64)
    state = torch.randn(num_tokens, HEADS, HEAD_DIM, HEAD_DIM, generator=generator)
    return query, key, value, gate, beta, slots, state


def _average_width_replay(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    slots: torch.Tensor,
    lengths: list[int],
):
    """Model the old rectangular plan: infer every boundary from average width."""
    num_seqs = len(lengths)
    width = value.shape[0] // num_seqs
    query_rows = query.view(num_seqs, width, *query.shape[1:])
    key_rows = key.view(num_seqs, width, *key.shape[1:])
    value_rows = value.view(num_seqs, width, *value.shape[1:])
    gate_rows = gate.view(num_seqs, width, *gate.shape[1:])
    beta_rows = beta.view(num_seqs, width, *beta.shape[1:])
    slot_rows = slots.view(num_seqs, width)
    out = torch.empty_like(value)
    out_rows = out.view(num_seqs, width, *value.shape[1:])
    resume_slots = slot_rows[:, 0]
    chunk_accepted = torch.ones(num_seqs, dtype=torch.int32)

    for start in range(0, width, KERNEL_CAP):
        chunk = min(KERNEL_CAP, width - start)
        state.index_copy_(0, slot_rows[:, start], state.index_select(0, resume_slots))
        out_rows[:, start : start + chunk] = _capped_kernel()(
            query_rows[:, start : start + chunk].reshape(-1, *query.shape[1:]),
            key_rows[:, start : start + chunk].reshape(-1, *key.shape[1:]),
            value_rows[:, start : start + chunk].reshape(-1, *value.shape[1:]),
            state,
            g=gate_rows[:, start : start + chunk].reshape(-1, *gate.shape[1:]),
            beta=beta_rows[:, start : start + chunk].reshape(-1, *beta.shape[1:]),
            scale=SCALE,
            actual_seq_lengths=torch.tensor([0] + [chunk] * num_seqs, dtype=torch.int32),
            ssm_state_indices=slot_rows[:, start : start + chunk].reshape(-1),
            num_accepted_tokens=chunk_accepted,
        ).view(num_seqs, chunk, *value.shape[1:])
        resume_slots = slot_rows[:, start + chunk - 1]
    return out


@pytest.mark.parametrize("tokens_per_seq", [1, 8, 9, 16, 17])
def test_chunked_replay_matches_unbounded_reference(tokens_per_seq, monkeypatch):
    lengths = [tokens_per_seq, tokens_per_seq]
    accepted = [1, tokens_per_seq]
    query, key, value, gate, beta, slots, state = _build_inputs(lengths)
    monkeypatch.setattr(gdn, "recurrent_gated_delta_rule", _capped_kernel())

    expected_out, expected_state = _run_reference(
        query, key, value, gate, beta, state.clone(), lengths, slots, accepted
    )
    out = gdn._spec_decode_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state.clone(),
        g=gate,
        beta=beta,
        scale=SCALE,
        actual_seq_lengths=_actual_seq_lengths(lengths),
        ssm_state_indices=slots,
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32),
        query_start_loc_cpu=_query_start_loc(lengths),
    )

    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("accepted", [[1, 5], [3, 7], [8, 8]])
def test_chunked_replay_resumes_from_accepted_slot(accepted, monkeypatch):
    tokens_per_seq = 11
    lengths = [tokens_per_seq, tokens_per_seq]
    query, key, value, gate, beta, slots, state = _build_inputs(lengths, seed=1)
    monkeypatch.setattr(gdn, "recurrent_gated_delta_rule", _capped_kernel())

    expected_out, expected_state = _run_reference(
        query, key, value, gate, beta, state.clone(), lengths, slots, accepted
    )
    working_state = state.clone()
    out = gdn._spec_decode_recurrent_gated_delta_rule(
        query,
        key,
        value,
        working_state,
        g=gate,
        beta=beta,
        scale=SCALE,
        actual_seq_lengths=_actual_seq_lengths(lengths),
        ssm_state_indices=slots,
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32),
        query_start_loc_cpu=_query_start_loc(lengths),
    )

    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(working_state, expected_state, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "lengths",
    [
        [9, 7],
        [9, 11],
        [17, 9, 1],
        [4, 16, 8, 3],
        [9, 0, 0],
        [0, 9, 0],
        [9, 0, 11, 0],
    ],
)
def test_chunked_replay_handles_ragged_windows(lengths, monkeypatch):
    """Dynamic verify windows must follow query_start_loc, not their average width."""
    accepted = [max(1, seq_len - 1) for seq_len in lengths]
    query, key, value, gate, beta, slots, state = _build_inputs(lengths, seed=2)
    monkeypatch.setattr(gdn, "recurrent_gated_delta_rule", _capped_kernel())

    expected_out, expected_state = _run_reference(
        query, key, value, gate, beta, state.clone(), lengths, slots, accepted
    )
    working_state = state.clone()
    out = gdn._spec_decode_recurrent_gated_delta_rule(
        query,
        key,
        value,
        working_state,
        g=gate,
        beta=beta,
        scale=SCALE,
        actual_seq_lengths=_actual_seq_lengths(lengths),
        ssm_state_indices=slots,
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32),
        query_start_loc_cpu=_query_start_loc(lengths),
    )

    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(working_state, expected_state, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("lengths", [[9, 7], [9, 11]])
def test_average_width_cannot_replace_packed_request_boundaries(lengths):
    """A rectangular width discards the real boundary for packed ragged requests."""
    accepted = [1, 1]
    query, key, value, gate, beta, slots, state = _build_inputs(lengths, seed=5)
    expected_out, _ = _run_reference(
        query,
        key,
        value,
        gate,
        beta,
        state.clone(),
        lengths,
        slots,
        accepted,
    )

    wrong_out = _average_width_replay(
        query,
        key,
        value,
        state.clone(),
        gate,
        beta,
        slots,
        lengths,
    )

    assert not torch.allclose(wrong_out, expected_out, rtol=1e-5, atol=1e-5)


def test_chunked_replay_requires_host_query_boundaries(monkeypatch):
    lengths = [9, 9]
    query, key, value, gate, beta, slots, state = _build_inputs(lengths)
    monkeypatch.setattr(gdn, "recurrent_gated_delta_rule", _capped_kernel())

    with pytest.raises(ValueError, match="query_start_loc_cpu is required"):
        gdn._spec_decode_recurrent_gated_delta_rule(
            query,
            key,
            value,
            state,
            g=gate,
            beta=beta,
            scale=SCALE,
            actual_seq_lengths=_actual_seq_lengths(lengths),
            ssm_state_indices=slots,
            num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32),
            query_start_loc_cpu=None,
        )


@pytest.mark.parametrize(
    ("query_start_loc_cpu", "num_tokens", "message"),
    [
        (torch.tensor([1, 9, 16]), 16, "must start at zero"),
        (torch.tensor([0, 9, 8]), 17, "must be non-decreasing"),
        (torch.tensor([0, 8, 16]), 17, "does not match the flat token count"),
    ],
)
def test_invalid_host_query_boundaries_fail_loudly(query_start_loc_cpu, num_tokens, message):
    with pytest.raises(ValueError, match=message):
        gdn._validate_spec_query_lengths(query_start_loc_cpu, num_seqs=2, num_tokens=num_tokens)


def test_host_query_boundaries_must_be_one_dimensional():
    with pytest.raises(ValueError, match="must be one-dimensional"):
        gdn._validate_spec_query_lengths(torch.tensor([[0, 9, 17]]), num_seqs=2, num_tokens=17)


def test_state_slot_count_must_match_flat_token_count():
    lengths = [9, 8]
    query, key, value, gate, beta, slots, state = _build_inputs(lengths)

    with pytest.raises(ValueError, match="one state slot per recurrent token"):
        gdn._spec_decode_recurrent_gated_delta_rule(
            query,
            key,
            value,
            state,
            g=gate,
            beta=beta,
            scale=SCALE,
            actual_seq_lengths=_actual_seq_lengths(lengths),
            ssm_state_indices=slots[:-1],
            num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32),
            query_start_loc_cpu=_query_start_loc(lengths),
        )


def test_all_zero_spec_windows_skip_the_recurrent_kernel(monkeypatch):
    lengths = [0, 0]
    query, key, value, gate, beta, slots, state = _build_inputs(lengths)

    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("zero-length speculative windows must not call the recurrent kernel")

    monkeypatch.setattr(gdn, "recurrent_gated_delta_rule", unexpected_kernel)
    out = gdn._spec_decode_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        g=gate,
        beta=beta,
        scale=SCALE,
        actual_seq_lengths=_actual_seq_lengths(lengths),
        ssm_state_indices=slots,
        num_accepted_tokens=torch.tensor([0, 0], dtype=torch.int32),
        query_start_loc_cpu=_query_start_loc(lengths),
    )

    assert out.shape == value.shape
    assert out.numel() == 0


@pytest.mark.parametrize("tokens_per_seq", [9, 16, 17])
def test_chunked_replay_never_exceeds_kernel_cap(tokens_per_seq, monkeypatch):
    lengths = [tokens_per_seq, tokens_per_seq]
    query, key, value, gate, beta, slots, state = _build_inputs(lengths, seed=3)
    seen: list[int] = []
    monkeypatch.setattr(gdn, "recurrent_gated_delta_rule", _capped_kernel(seen))

    gdn._spec_decode_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state.clone(),
        g=gate,
        beta=beta,
        scale=SCALE,
        actual_seq_lengths=_actual_seq_lengths(lengths),
        ssm_state_indices=slots,
        num_accepted_tokens=torch.tensor([1, 2], dtype=torch.int32),
        query_start_loc_cpu=_query_start_loc(lengths),
    )

    assert seen, "the kernel must be called at least once"
    assert max(seen) <= KERNEL_CAP
    assert len(seen) == -(-tokens_per_seq // gdn._MAX_RECURRENT_GDN_TOKENS)


def test_kernel_cap_is_crossed_without_chunking(monkeypatch):
    """Counter-proof: the unchunked call is exactly what the issue reports."""
    lengths = [9, 9]
    query, key, value, gate, beta, slots, state = _build_inputs(lengths, seed=4)
    kernel = _capped_kernel()
    monkeypatch.setattr(gdn, "recurrent_gated_delta_rule", kernel)
    expected_out, _ = _run_reference(query, key, value, gate, beta, state.clone(), lengths, slots, [1, 1])

    # What the call site did before the fix: hand the whole 9-token window to
    # the kernel, which bails out and leaves the output buffer untouched.
    unchunked = kernel(
        query,
        key,
        value,
        state.clone(),
        g=gate,
        beta=beta,
        scale=SCALE,
        actual_seq_lengths=_actual_seq_lengths(lengths),
        ssm_state_indices=slots,
        num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32),
    )
    assert torch.count_nonzero(unchunked) == 0
    assert not torch.allclose(unchunked, expected_out, rtol=1e-5, atol=1e-5)

    monkeypatch.setattr(gdn, "_MAX_RECURRENT_GDN_TOKENS", 64)
    no_fix = gdn._spec_decode_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state.clone(),
        g=gate,
        beta=beta,
        scale=SCALE,
        actual_seq_lengths=_actual_seq_lengths(lengths),
        ssm_state_indices=slots,
        num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32),
        query_start_loc_cpu=_query_start_loc(lengths),
    )
    # Without the chunked replay the kernel bails out and returns zeros, which
    # is what makes n=8 draft tokens degenerate into a repeated token.
    assert torch.count_nonzero(no_fix) == 0
