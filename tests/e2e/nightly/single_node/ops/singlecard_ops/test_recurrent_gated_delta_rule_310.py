import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.set_compile_mode(jit_compile=False)


def npu_recurrent_gated_delta_rule_310(
    query,
    key,
    value,
    beta,
    state,
    actual_seq_lengths,
    ssm_state_indices,
    g=None,
    gk=None,
    num_accepted_tokens=None,
    scale=1.0,
):
    """Call RecurrentGatedDeltaRule."""
    out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
        query=query,
        key=key,
        value=value,
        g=g,
        gk=gk,
        beta=beta,
        state=state,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale_value=scale,
    )
    return out


def golden_recurrent_gated_delta_rule(
    query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, g, num_accepted_tokens
):
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    initial_state = state.clone().to(torch.float32)
    T, n_heads_v, Dv = v.shape
    n_heads_k = q.shape[-2]
    g = torch.ones(T, n_heads_v).to(torch.float32) if g is None else g.to(torch.float32).exp()

    beta = torch.ones(T, n_heads_v).to(torch.float32) if beta is None else beta.to(torch.float32)
    o = torch.empty_like(v).to(torch.float32)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    q = q * scale

    seq_start = 0
    for i in range(len(actual_seq_lengths)):
        if num_accepted_tokens is None:
            init_state = initial_state[ssm_state_indices[seq_start]]
        else:
            init_state = initial_state[ssm_state_indices[seq_start + num_accepted_tokens[i] - 1]]

        for head_id in range(n_heads_v):
            S = init_state[head_id]
            for slot_id in range(seq_start, seq_start + actual_seq_lengths[i]):
                q_i = q[slot_id][head_id // (n_heads_v // n_heads_k)]
                k_i = k[slot_id][head_id // (n_heads_v // n_heads_k)]
                v_i = v[slot_id][head_id]
                alpha_i = g[slot_id][head_id]
                beta_i = beta[slot_id][head_id]
                S = S * alpha_i
                x = (S * k_i.unsqueeze(-2)).sum(dim=-1)
                y = (v_i - x) * beta_i
                S_ = y[:, None] * k_i[None, :]
                S = S + S_
                initial_state[ssm_state_indices[slot_id]][head_id] = S
                o[slot_id][head_id] = (S * q_i.unsqueeze(-2)).sum(dim=-1)
        seq_start += actual_seq_lengths[i]

    return o.to(query.dtype), initial_state.to(query.dtype)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("headnum", [(4, 8), (8, 16), (16, 32)])
@pytest.mark.parametrize("headdim_k", [128])
@pytest.mark.parametrize("headdim_v", [128])
def test_fused_recurrent_gated_delta_rule_310(batch_size, mtp, headnum, headdim_k, headdim_v):
    enable_custom_op()
    dtype = torch.float16
    headnum_k, headnum_v = headnum
    actual_seq_lengths = torch.ones(batch_size, dtype=torch.int32) * mtp
    T = int(torch.sum(actual_seq_lengths))
    state = torch.rand((T, headnum_v, headdim_v, headdim_k)).to(dtype)
    query = torch.nn.functional.normalize(torch.rand((T, headnum_k, headdim_k)), p=2, dim=-1).to(dtype)
    key = torch.nn.functional.normalize(torch.rand((T, headnum_k, headdim_k)), p=2, dim=-1).to(dtype)
    value = torch.rand((T, headnum_v, headdim_v)).to(dtype)
    g = torch.rand((T, headnum_v), dtype=torch.float32)
    beta = torch.rand((T, headnum_v)).to(dtype)
    ssm_state_indices = torch.arange(T, dtype=torch.int32)
    num_accepted_tokens = torch.randint(1, mtp + 1, (batch_size,), dtype=torch.int32)
    scale = headdim_k**-0.5

    out_golden, state_golden = golden_recurrent_gated_delta_rule(
        query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, g, num_accepted_tokens
    )
    out_golden = out_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    state_npu = state.npu()
    out = npu_recurrent_gated_delta_rule_310(
        query.npu(),
        key.npu(),
        value.npu(),
        beta.npu(),
        state_npu,
        actual_seq_lengths.npu(),
        ssm_state_indices.npu(),
        g=g.npu(),
        num_accepted_tokens=num_accepted_tokens.npu(),
        scale=scale,
    )
    out = out.to(torch.float32).cpu()

    torch.testing.assert_close(
        out.to(torch.float32).cpu(),
        out_golden.to(torch.float32).cpu(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state_npu.to(torch.float32).cpu(),
        state_golden.to(torch.float32).cpu(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
