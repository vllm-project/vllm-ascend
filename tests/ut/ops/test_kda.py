# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import pytest
import torch
import torch.nn.functional as F

import vllm_ascend.ops.kda as kda_ops


def _normalize(tensor):
    value = tensor.float()
    return (value * torch.rsqrt(value.square().sum(-1, keepdim=True) + 1e-6)).to(tensor.dtype)


def _reference(q, k, v, raw_gate, beta, scale, initial_state, cu_seqlens, a_log, dt_bias, lower_bound):
    """Token-wise VK recurrence, independent of chunking and padding."""
    bias = dt_bias.reshape(q.shape[2], q.shape[3])
    rate = a_log.reshape(-1, 1).float().exp()
    state = initial_state.float().clone()
    output = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=torch.float32)
    for sequence, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for token in range(start, end):
            # Evaluate a fixed token shape so CPU vectorized transcendental
            # tails cannot change the reference rounding when padding grows T.
            gate_input = raw_gate[0, token].float() + bias
            gate = (
                -rate * F.softplus(gate_input)
                if lower_bound is None
                else lower_bound * torch.sigmoid(rate * gate_input)
            )
            state[sequence] *= gate.exp().unsqueeze(-2)
            key = k[0, token].float()
            residual = v[0, token].float() - torch.einsum("hvk,hk->hv", state[sequence], key)
            state[sequence] += (beta[0, token].float().unsqueeze(-1) * residual).unsqueeze(-1) * key.unsqueeze(-2)
            output[0, token] = torch.einsum("hvk,hk->hv", state[sequence], q[0, token].float()) * scale
    return output.to(v.dtype), state


@pytest.mark.parametrize("lengths", [(129,), (127, 129, 65), (0, 129, 0, 64, 1, 0)])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("has_initial_state", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("physical_padding", [0, 7])
def test_a5_tail_padding_matches_recurrence(
    monkeypatch, lengths, lower_bound, has_initial_state, dtype, physical_padding
):
    generator = torch.Generator().manual_seed(20261007)
    valid_tokens = sum(lengths)
    tokens, heads, key_dim, value_dim = valid_tokens + physical_padding, 2, 4, 5

    def values(dim):
        # Match the strided projection views used by model callers.
        return torch.randn(1, tokens, heads, dim * 2, generator=generator).to(dtype)[..., ::2]

    q, k, v, raw_gate = values(key_dim), values(key_dim), values(value_dim), values(key_dim)
    beta = torch.rand(1, tokens, heads, generator=generator)
    initial_state = torch.randn(len(lengths), heads, value_dim, key_dim, generator=generator)
    if not has_initial_state:
        initial_state.zero_()
    a_log = torch.tensor([-0.2, 0.3])
    dt_bias = torch.linspace(-2.0, 0.5, heads * key_dim)
    cu = tuple(itertools.accumulate(lengths, initial=0))
    chunks = tuple(
        value
        for sequence, length in enumerate(lengths)
        for chunk in range((length + 63) // 64)
        for value in (sequence, chunk)
    )
    inputs = (q, k, v, raw_gate, beta, initial_state)
    saved = tuple(tensor.clone() for tensor in inputs)
    expected = _reference(
        _normalize(q), _normalize(k), v, raw_gate, beta, key_dim**-0.5, initial_state, cu, a_log, dt_bias, lower_bound
    )
    returned_state = None

    def chunk(q_arg, k_arg, v_arg, gate_arg, beta_arg, scale, chunk_size, **kwargs):
        nonlocal returned_state
        assert chunk_size == 64
        assert kwargs["initial_state"] is initial_state
        assert kwargs["safe_gate"] == (lower_bound is not None)
        assert kwargs["state_v_first"] and kwargs["use_gate_in_kernel"]
        assert kwargs["disable_recompute"] is False
        padded_cu = kwargs["cu_seqlens"]
        expected_chunks = []
        for sequence, length in enumerate(lengths):
            start, end = padded_cu[sequence : sequence + 2]
            assert end - start == ((length + 63) // 64) * 64
            for chunk_index in range((end - start) // 64):
                expected_chunks.extend((sequence, chunk_index))
            for tensor in (q_arg, k_arg, v_arg, beta_arg):
                assert torch.count_nonzero(tensor[:, start + length : end]) == 0
            assert torch.isneginf(gate_arg[:, start + length : end]).all()
        assert kwargs["chunk_indices"] == tuple(expected_chunks)
        result, returned_state = _reference(
            q_arg, k_arg, v_arg, gate_arg, beta_arg, scale, initial_state, padded_cu, a_log, dt_bias, lower_bound
        )
        return result, returned_state

    monkeypatch.setattr(kda_ops, "is_950", lambda: True)
    monkeypatch.setattr(kda_ops, "l2norm_fwd", _normalize)
    monkeypatch.setattr(kda_ops, "chunk_kda_fwd", chunk)
    output, final_state = kda_ops.run_chunk_kda(
        q, k, v, raw_gate, beta, initial_state, cu, chunks, a_log, dt_bias, lower_bound=lower_bound
    )
    assert output.shape == (1, tokens, heads, value_dim)
    assert output.dtype == dtype
    assert final_state is returned_state
    torch.testing.assert_close(output[:, :valid_tokens], expected[0][:, :valid_tokens], rtol=0, atol=0)
    assert torch.count_nonzero(output[:, valid_tokens:]) == 0
    torch.testing.assert_close(final_state, expected[1], rtol=0, atol=0)
    for tensor, original in zip(inputs, saved):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


@pytest.mark.parametrize("a5,lengths", [(False, (129,)), (False, (64, 128)), (True, (64, 0, 128)), (True, (0, 0))])
def test_chunk_unaffected_paths_preserve_descriptors_and_output(monkeypatch, a5, lengths):
    q = torch.ones(1, sum(lengths), 2, 4)
    beta = torch.ones(1, sum(lengths), 2)
    initial_state = torch.zeros(len(lengths), 2, 4, 4)
    cu = tuple(itertools.accumulate(lengths, initial=0))
    chunks = tuple(
        value
        for sequence, length in enumerate(lengths)
        for chunk in range((length + 63) // 64)
        for value in (sequence, chunk)
    )
    expected_output, expected_state = torch.empty_like(q), torch.empty_like(initial_state)

    def chunk(q_arg, k_arg, v_arg, gate_arg, beta_arg, *args, **kwargs):
        assert q_arg is q and k_arg is q and v_arg is q and gate_arg is q
        assert beta_arg is beta
        assert kwargs["cu_seqlens"] is cu and kwargs["chunk_indices"] is chunks
        assert kwargs["initial_state"] is initial_state
        return expected_output, expected_state

    monkeypatch.setattr(kda_ops, "is_950", lambda: a5)
    monkeypatch.setattr(kda_ops, "l2norm_fwd", lambda tensor: tensor)
    monkeypatch.setattr(kda_ops, "chunk_kda_fwd", chunk)
    output, state = kda_ops.run_chunk_kda(
        q, q, q, q, beta, initial_state, cu, chunks, torch.zeros(2), torch.zeros(8), lower_bound=None
    )
    assert output is expected_output and state is expected_state
