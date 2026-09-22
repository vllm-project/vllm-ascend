import gc
import random

import numpy as np
import pytest
import torch
import torch_npu

torch_npu.npu.set_compile_mode(jit_compile=False)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def golden_recurrent_gated_delta_rule(
    query,
    key,
    value,
    state,
    beta,
    scale,
    actual_seq_lengths,
    ssm_state_indices,
    g,
    num_accepted_tokens,
):
    """Pure torch/CPU golden implementation of recurrent gated delta rule.

    Args:
        query: [T, nk, dk]
        key: [T, nk, dk]
        value: [T, nv, dv]
        state: [S, nv, dv, dk]
        beta: [T, nv]
        scale: float
        actual_seq_lengths: [batch_size] per-sequence lengths
        ssm_state_indices: [T] token-packed indices or [B, slots] state rows
        g: [T, nv] or None
        num_accepted_tokens: [batch_size] or None

    Returns:
        (output [T, nv, dv], updated_state [S, nv, dv, dk])
    """
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
        if actual_seq_lengths[i] == 0:
            continue
        row = ssm_state_indices[i] if ssm_state_indices.ndim == 2 else ssm_state_indices[seq_start:]
        if num_accepted_tokens is None:
            init_state = initial_state[row[0]]
        else:
            init_state = initial_state[row[num_accepted_tokens[i] - 1]]
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
                initial_state[row[slot_id - seq_start]][head_id] = S
                o[slot_id][head_id] = (S * q_i.unsqueeze(-2)).sum(dim=-1)
        seq_start += actual_seq_lengths[i]

    return o.to(query.dtype), initial_state.to(state.dtype)


@pytest.mark.parametrize("lengths", [(2, 2), (2, 1), (1, 0), (3, 2), (4, 4)])
@pytest.mark.parametrize("previous_accepted", [None, 1, 4])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_recurrent_gated_delta_rule_dynamic_state_stride(lengths, previous_accepted, state_dtype):
    """Previous acceptance is independent of the current query length, including padding."""
    torch.manual_seed(42)
    tokens = sum(lengths)
    query = torch.nn.functional.normalize(torch.randn(tokens, 4, 128), dim=-1).bfloat16()
    key = torch.nn.functional.normalize(torch.randn(tokens, 4, 128), dim=-1).bfloat16()
    value = torch.randn(tokens, 8, 128).bfloat16()
    state = torch.randn(8, 8, 128, 128).to(state_dtype)
    beta = torch.rand(tokens, 8).bfloat16()
    g = -torch.rand(tokens, 8)
    indices = torch.tensor([[6, 1, 7, 3], [5, 0, 4, 2]], dtype=torch.int32)
    accepted = None if previous_accepted is None else torch.full((2,), previous_accepted, dtype=torch.int32)
    expected, expected_state = golden_recurrent_gated_delta_rule(
        query, key, value, state, beta, 128**-0.5, lengths, indices, g, accepted
    )
    actual_state = state.npu()
    actual = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=query.npu(),
        key=key.npu(),
        value=value.npu(),
        state=actual_state,
        beta=beta.npu(),
        scale=128**-0.5,
        actual_seq_lengths=torch.tensor([0, *lengths], dtype=torch.int32).npu(),
        ssm_state_indices=indices.npu(),
        num_accepted_tokens=None if accepted is None else accepted.npu(),
        g=g.npu(),
    )
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=3e-3, atol=1e-2)
    torch.testing.assert_close(actual_state.cpu().float(), expected_state.float(), rtol=3e-3, atol=1e-2)


@pytest.mark.parametrize("index_shape", [(2, 2, 2), (1, 4), (2, 0)])
def test_recurrent_gated_delta_rule_rejects_invalid_state_rows(index_shape):
    query = torch.ones(2, 4, 128, dtype=torch.bfloat16, device="npu")
    value = torch.ones(2, 8, 128, dtype=torch.bfloat16, device="npu")
    state = torch.zeros(8, 8, 128, 128, dtype=torch.float32, device="npu")
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=query,
            key=query,
            value=value,
            state=state,
            beta=torch.ones(2, 8, dtype=torch.bfloat16, device="npu"),
            scale=128**-0.5,
            actual_seq_lengths=torch.tensor([0, 1, 1], dtype=torch.int32, device="npu"),
            ssm_state_indices=torch.zeros(index_shape, dtype=torch.int32, device="npu"),
            num_accepted_tokens=torch.ones(2, dtype=torch.int32, device="npu"),
        )


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("headnum", [(4, 8), (8, 16), (16, 32)])
@pytest.mark.parametrize("headdim_k", [128])
@pytest.mark.parametrize("headdim_v", [128])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_recurrent_gated_delta_rule(
    batch_size,
    mtp,
    headnum,
    headdim_k,
    headdim_v,
    state_dtype,
):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    headnum_k, headnum_v = headnum
    seq_lengths = torch.ones(batch_size, dtype=torch.int32) * mtp
    T = int(torch.sum(seq_lengths))

    state = torch.rand((T, headnum_v, headdim_v, headdim_k)).to(state_dtype)
    query = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    value = torch.rand((T, headnum_v, headdim_v)).to(dtype)
    g = torch.rand((T, headnum_v), dtype=torch.float32)
    beta = torch.rand((T, headnum_v)).to(dtype)
    ssm_state_indices = torch.arange(T, dtype=torch.int32)
    num_accepted_tokens = torch.randint(1, mtp + 1, (batch_size,), dtype=torch.int32)
    scale = headdim_k**-0.5

    out_golden, state_golden = golden_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta,
        scale,
        seq_lengths,
        ssm_state_indices,
        g,
        num_accepted_tokens,
    )
    out_golden = out_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    # torch_npu op expects actual_seq_lengths = [start_pos, len1, len2, ..., lenB]
    actual_seq_lengths_npu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            seq_lengths,
        ]
    )

    state_npu = state.npu()
    npu_out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=query.npu(),
        key=key.npu(),
        value=value.npu(),
        g=g.npu(),
        beta=beta.npu(),
        state=state_npu,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths_npu.npu(),
        ssm_state_indices=ssm_state_indices.npu(),
        num_accepted_tokens=num_accepted_tokens.npu(),
    )

    torch.testing.assert_close(
        npu_out.to(torch.float32).cpu(),
        out_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state_npu.to(torch.float32).cpu(),
        state_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("headnum", [(4, 8), (8, 16)])
@pytest.mark.parametrize("headdim_k", [128])
@pytest.mark.parametrize("headdim_v", [128])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_recurrent_gated_delta_rule_no_accepted(
    batch_size,
    mtp,
    headnum,
    headdim_k,
    headdim_v,
    state_dtype,
):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    headnum_k, headnum_v = headnum
    seq_lengths = torch.ones(batch_size, dtype=torch.int32) * mtp
    T = int(torch.sum(seq_lengths))

    state = torch.rand((T, headnum_v, headdim_v, headdim_k)).to(state_dtype)
    query = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)),
        p=2,
        dim=-1,
    ).to(dtype)
    value = torch.rand((T, headnum_v, headdim_v)).to(dtype)
    g = torch.rand((T, headnum_v), dtype=torch.float32)
    beta = torch.rand((T, headnum_v)).to(dtype)
    ssm_state_indices = torch.arange(T, dtype=torch.int32)
    scale = headdim_k**-0.5

    out_golden, state_golden = golden_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta,
        scale,
        seq_lengths,
        ssm_state_indices,
        g,
        None,
    )
    out_golden = out_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    actual_seq_lengths_npu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            seq_lengths,
        ]
    )

    state_npu = state.npu()
    npu_out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=query.npu(),
        key=key.npu(),
        value=value.npu(),
        g=g.npu(),
        beta=beta.npu(),
        state=state_npu,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths_npu.npu(),
        ssm_state_indices=ssm_state_indices.npu(),
    )

    torch.testing.assert_close(
        npu_out.to(torch.float32).cpu(),
        out_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state_npu.to(torch.float32).cpu(),
        state_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
