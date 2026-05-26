import gc
import random

import numpy as np
import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op, is_310p

torch_npu.npu.set_compile_mode(jit_compile=False)

pytestmark = pytest.mark.skipif(
    is_310p(),
    reason="npu_fused_sigmoid_gating_delta_rule_update is not built for 310P",
)

if not is_310p():
    enable_custom_op()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def golden_fused_sigmoid_gating_delta_rule_update(
    A_log, a, b, dt_bias, query, key, value, state, scale,
    actual_seq_lengths, ssm_state_indices, num_accepted_tokens,
    softplus_beta=1.0, softplus_threshold=20.0,
):
    """Pure torch/CPU golden implementation of fused sigmoid gating delta rule.

    Args:
        A_log: [nv]
        a: [T, nv]
        b: [T, nv]
        dt_bias: [nv]
        query: [T, nk, dk]
        key: [T, nk, dk]
        value: [T, nv, dv]
        state: [S, nv, dv, dk]
        scale: float
        actual_seq_lengths: [batch_size] per-sequence lengths
        ssm_state_indices: [T] per-token state block index
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
    if scale is None:
        scale = k.shape[-1] ** -0.5
    q = q * scale

    gate = a.to(torch.float32) + dt_bias.to(torch.float32).unsqueeze(0)
    g = torch.exp(
        -torch.exp(A_log.to(torch.float32)).unsqueeze(0) *
        torch.nn.functional.softplus(
            gate,
            beta=softplus_beta,
            threshold=softplus_threshold,
        )
    )
    beta = torch.sigmoid(b.to(torch.float32))
    o = torch.empty_like(v).to(torch.float32)

    seq_start = 0
    for i in range(len(actual_seq_lengths)):
        if num_accepted_tokens is None:
            init_state = initial_state[ssm_state_indices[seq_start]]
        else:
            init_state = initial_state[
                ssm_state_indices[seq_start + num_accepted_tokens[i] - 1]
            ]
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

    return o.to(query.dtype), initial_state.to(state.dtype)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("mtp", [1, 2])
@pytest.mark.parametrize("headnum", [(4, 8), (8, 16), (16, 32)])
@pytest.mark.parametrize("headdim_k", [128])
@pytest.mark.parametrize("headdim_v", [128])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_fused_sigmoid_gating_delta_rule_update(
    batch_size, mtp, headnum, headdim_k, headdim_v, state_dtype,
):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    headnum_k, headnum_v = headnum
    seq_lengths = torch.ones(batch_size, dtype=torch.int32) * mtp
    T = int(torch.sum(seq_lengths))

    state = torch.rand((T, headnum_v, headdim_v, headdim_k)).to(state_dtype)
    query = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)), p=2, dim=-1,
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)), p=2, dim=-1,
    ).to(dtype)
    value = torch.rand((T, headnum_v, headdim_v)).to(dtype)
    A_log = torch.rand((headnum_v,), dtype=torch.float32)
    a = torch.rand((T, headnum_v)).to(dtype)
    b = torch.rand((T, headnum_v)).to(dtype)
    dt_bias = torch.rand((headnum_v,), dtype=torch.float32)
    ssm_state_indices = torch.arange(T, dtype=torch.int32)
    num_accepted_tokens = torch.randint(1, mtp + 1, (batch_size,), dtype=torch.int32)
    scale = headdim_k**-0.5

    out_golden, state_golden = golden_fused_sigmoid_gating_delta_rule_update(
        A_log, a, b, dt_bias, query, key, value, state, scale,
        seq_lengths, ssm_state_indices, num_accepted_tokens,
    )
    out_golden = out_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    actual_seq_lengths_npu = torch.cat([
        torch.zeros(1, dtype=torch.int32), seq_lengths,
    ])

    state_npu = state.npu()
    npu_out = torch.ops._C_ascend.npu_fused_sigmoid_gating_delta_rule_update(
        A_log.npu(),
        a.npu(),
        b.npu(),
        dt_bias.npu(),
        query.npu(),
        key.npu(),
        value.npu(),
        state_npu,
        actual_seq_lengths_npu.npu(),
        ssm_state_indices.npu(),
        num_accepted_tokens.npu(),
        scale,
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
def test_fused_sigmoid_gating_delta_rule_update_no_accepted(
    batch_size, mtp, headnum, headdim_k, headdim_v, state_dtype,
):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    headnum_k, headnum_v = headnum
    seq_lengths = torch.ones(batch_size, dtype=torch.int32) * mtp
    T = int(torch.sum(seq_lengths))

    state = torch.rand((T, headnum_v, headdim_v, headdim_k)).to(state_dtype)
    query = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)), p=2, dim=-1,
    ).to(dtype)
    key = torch.nn.functional.normalize(
        torch.rand((T, headnum_k, headdim_k)), p=2, dim=-1,
    ).to(dtype)
    value = torch.rand((T, headnum_v, headdim_v)).to(dtype)
    A_log = torch.rand((headnum_v,), dtype=torch.float32)
    a = torch.rand((T, headnum_v)).to(dtype)
    b = torch.rand((T, headnum_v)).to(dtype)
    dt_bias = torch.rand((headnum_v,), dtype=torch.float32)
    ssm_state_indices = torch.arange(T, dtype=torch.int32)
    scale = headdim_k**-0.5

    out_golden, state_golden = golden_fused_sigmoid_gating_delta_rule_update(
        A_log, a, b, dt_bias, query, key, value, state, scale,
        seq_lengths, ssm_state_indices, None,
    )
    out_golden = out_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    actual_seq_lengths_npu = torch.cat([
        torch.zeros(1, dtype=torch.int32), seq_lengths,
    ])

    state_npu = state.npu()
    npu_out = torch.ops._C_ascend.npu_fused_sigmoid_gating_delta_rule_update(
        A_log.npu(),
        a.npu(),
        b.npu(),
        dt_bias.npu(),
        query.npu(),
        key.npu(),
        value.npu(),
        state_npu,
        actual_seq_lengths_npu.npu(),
        ssm_state_indices.npu(),
        None,
        scale,
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
