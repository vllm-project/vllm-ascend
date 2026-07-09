# SPDX-License-Identifier: Apache-2.0
"""E2E correctness test for AscendC chunk_gated_delta_rule kernel.

Validates torch.ops._C_ascend.npu_chunk_gated_delta_rule against a CPU golden
reference (3-stage chunked gated delta rule) across batch / seqlen / head
number / GQA combinations.

Prerequisite: the AscendC kernel must be compiled and installed via
  bash csrc/build_aclnn.sh <ROOT_DIR> <SOC_VERSION>

Run:
  pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_chunk_gated_delta_rule.py -v
"""

import gc

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SEED = 42
CHUNK_SIZE = 64


# ---------------------------------------------------------------------------
# Golden reference helpers
# ---------------------------------------------------------------------------


def _get_chunk(tensor, C, start):
    """Extract a chunk of size C from tensor along dim-0, zero-padding if needed."""
    S = tensor.shape[0]
    end = start + C
    if end <= S:
        return tensor[start:end]
    pad_size = end - S
    if len(tensor.shape) > 1:
        return F.pad(tensor[start:], (0, 0, 0, pad_size))
    return F.pad(tensor[start:], (0, pad_size))


def _stage1_chunk(query, key, value, g, beta, scale):
    """Stage1: intra-chunk preprocessing via (I+L)^{-1} for gated delta rule."""
    device = query.device
    C = query.shape[0]

    kkt = key.float() @ key.transpose(-1, -2).float()
    kkt = kkt * beta.float().unsqueeze(-1)

    qkt = query.float() @ key.transpose(-1, -2).float()

    g_cum = g.cumsum(dim=-1)
    g_cum_exp = g_cum.exp()

    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=-1)
    attn = ((g_cum[:, None] - g_cum[None, :]) * lower).exp() * lower

    attn_1 = kkt * attn
    attn_1 *= -1.0
    for i in range(1, C):
        row = attn_1[i, :i].clone()
        sub = attn_1[:i, :i].clone()
        attn_1[i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn_1 = attn_1 + torch.eye(C, dtype=attn_1.dtype, device=attn_1.device)

    kg = key.float() * (g_cum[-1, None] - g_cum).exp()[..., None]
    k_cumdecay = (beta.unsqueeze(-1).float() * g_cum_exp[:, None]) * (-1) * key.float()

    v_beta = value.float() * beta.unsqueeze(-1).float()
    q_prime = query.float() * scale * g_cum_exp[:, None]

    v_inner = attn_1 @ v_beta
    k_cumdecay = attn_1 @ k_cumdecay

    return (g_cum, k_cumdecay.to(torch.bfloat16), v_inner, q_prime.to(torch.bfloat16), kg.to(torch.bfloat16), qkt)


def _stage1(query, key, value, g, beta, scale, C):
    """Stage1 outer loop: split by head and chunk, call _stage1_chunk."""
    S, Nk, Dk = query.shape
    _, Nv, Dv = value.shape
    device = query.device

    if Nv // Nk > 1:
        query = query.repeat_interleave(Nv // Nk, dim=1)
        key = key.repeat_interleave(Nv // Nk, dim=1)

    padded_seq_len = (S + C - 1) // C * C

    g_cum = torch.zeros((Nv, padded_seq_len), dtype=torch.float32, device=device)
    k_cumdecay = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.bfloat16, device=device)
    v_inner = torch.zeros((Nv, padded_seq_len, Dv), dtype=torch.float32, device=device)
    q_prime = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.bfloat16, device=device)
    kg = torch.zeros((Nv, padded_seq_len, Dk), dtype=torch.bfloat16, device=device)
    qkt = torch.zeros((Nv, padded_seq_len, C), dtype=torch.float32, device=device)

    loop_range = range(0, padded_seq_len, C)
    for nid in range(Nv):
        for idx in reversed(loop_range):
            q_chunk = _get_chunk(query[:, nid, :], C, idx)
            k_chunk = _get_chunk(key[:, nid, :], C, idx)
            v_chunk = _get_chunk(value[:, nid, :], C, idx)
            g_chunk = _get_chunk(g[:, nid], C, idx)
            beta_chunk = _get_chunk(beta[:, nid], C, idx)

            (g_cum_c, k_cumdecay_c, v_inner_c, qg_c, kg_c, qkt_c) = _stage1_chunk(
                q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, scale
            )

            g_cum[nid, idx : idx + C] = g_cum_c
            k_cumdecay[nid, idx : idx + C, :] = k_cumdecay_c
            v_inner[nid, idx : idx + C, :] = v_inner_c
            q_prime[nid, idx : idx + C, :] = qg_c
            kg[nid, idx : idx + C, :] = kg_c
            qkt[nid, idx : idx + C, :] = qkt_c

    return g_cum, k_cumdecay, v_inner, q_prime, kg, qkt


def _stage2_chunk(q_prime, v_inner, g_cum, k_cumdecay, state, kg):
    """Stage2: sequential inter-chunk state propagation."""
    bf16_state = state.to(torch.bfloat16)

    attn_inter = q_prime.float() @ bf16_state.float().transpose(0, 1)
    v_prime = k_cumdecay.float() @ bf16_state.float().transpose(0, 1)
    v_new = v_inner + v_prime

    state_out = v_new.transpose(0, 1).to(torch.bfloat16).float() @ kg.float()
    state_old = state.float() * g_cum.exp()[-1]
    state_old = state_old + state_out

    return state_old, attn_inter, v_new.to(torch.bfloat16)


def _stage2(q_prime, v_inner, g_cum, k_cumdecay, state, kg, C):
    """Stage2 outer loop: sequential chunk traversal for state updates."""
    Nv, Sp, Dv = v_inner.shape
    attn_inter = torch.zeros((Nv, Sp, Dv), dtype=torch.float32, device=q_prime.device)
    v_new = torch.zeros((Nv, Sp, Dv), dtype=torch.bfloat16, device=q_prime.device)
    final_state = torch.empty_like(state).to(torch.float32)

    for nid in range(Nv):
        cur_state = state[nid]
        for idx in range(0, Sp, C):
            cur_state, attn_inter_c, v_new_c = _stage2_chunk(
                q_prime[nid, idx : idx + C, :],
                v_inner[nid, idx : idx + C, :],
                g_cum[nid, idx : idx + C],
                k_cumdecay[nid, idx : idx + C, :],
                cur_state,
                kg[nid, idx : idx + C, :],
            )
            attn_inter[nid, idx : idx + C, :] = attn_inter_c
            v_new[nid, idx : idx + C, :] = v_new_c
        final_state[nid, ...] = cur_state
    return final_state, attn_inter, v_new


def _stage3_chunk(qkt, value, scale, g_cum, attn_inter, v_new):
    """Stage3: merge inter-chunk and intra-chunk attention for final output."""
    device = value.device
    C = value.shape[0]
    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device), diagonal=0)
    masked_qkt = qkt.float() * scale * ((g_cum[:, None] - g_cum[None, :]) * lower).exp() * lower.float()
    attn_inner = masked_qkt.to(torch.bfloat16).float() @ v_new.to(torch.bfloat16).float()
    return (attn_inter + attn_inner).to(torch.bfloat16)


def _stage3(qkt, value, scale, g_cum, attn_inter, v_new, C):
    """Stage3 outer loop: per-head per-chunk output assembly."""
    Nv, Sp, Dv = attn_inter.shape
    attn_out = torch.empty((Sp, Nv, Dv), dtype=torch.bfloat16, device=value.device)

    for nid in range(Nv):
        for idx in range(0, Sp, C):
            attn_out[idx : idx + C, nid, ...] = _stage3_chunk(
                qkt[nid, idx : idx + C, :],
                _get_chunk(value[:, nid, :], C, idx),
                scale,
                g_cum[nid, idx : idx + C],
                attn_inter[nid, idx : idx + C, :],
                v_new[nid, idx : idx + C, :],
            )
    return attn_out


def golden_chunk_gated_delta_rule(query, key, value, beta, scale, initial_state, actual_seq_lengths, g=None):
    """Full 3-stage chunked gated delta rule golden reference (CPU, pure PyTorch).

    Args:
        query:              (T, Nk, Dk), BF16
        key:                (T, Nk, Dk), BF16
        value:              (T, Nv, Dv), BF16
        beta:               (T, Nv), BF16
        scale:              float
        initial_state:      (B, Nv, Dv, Dk), FP32
        actual_seq_lengths: (B,), INT32 - per-batch sequence lengths
        g:                  (T, Nv), FP32 or None

    Returns:
        (output [T, Nv, Dv] BF16, final_state [B, Nv, Dv, Dk] FP32)
    """
    T = query.shape[0]
    B, Nv, Dv, _ = initial_state.shape
    device = query.device
    C = CHUNK_SIZE

    if g is None:
        g = torch.zeros((T, Nv), dtype=torch.float32, device=device)
    attn_out = torch.empty((T, Nv, Dv), dtype=torch.bfloat16, device=device)
    final_state = torch.empty_like(initial_state).to(torch.float32)

    start = 0
    for bid in range(B):
        cur_state = initial_state[bid].clone()
        S = actual_seq_lengths[bid]
        end = start + S

        g_cum, k_cum_decay, v_inner, q_prime, kg, qkt = _stage1(
            query[start:end], key[start:end], value[start:end], g[start:end], beta[start:end], scale, C
        )

        cur_state, attn_inter, v_new = _stage2(q_prime, v_inner, g_cum, k_cum_decay, cur_state, kg, C)
        final_state[bid] = cur_state

        attn_out_padded = _stage3(qkt, value[start:end], scale, g_cum, attn_inter, v_new, C)
        attn_out[start:end, ...] = attn_out_padded[:S]
        start = end

    return attn_out, final_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(batch_size, seqlen, nk, nv, dk, dv, seed=SEED):
    """Build random tensors on CPU for both golden and NPU execution."""
    torch.manual_seed(seed)
    dtype = torch.bfloat16

    T = batch_size * seqlen
    q = F.normalize(torch.rand((T, nk, dk)), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.rand((T, nk, dk)), p=2, dim=-1).to(dtype)
    v = torch.rand((T, nv, dv), dtype=dtype)
    g = torch.rand((T, nv), dtype=torch.float32) * -1.0
    beta = torch.rand((T, nv), dtype=dtype)
    scale = 1.0 / (dk**0.5)
    initial_state = torch.rand((batch_size, nv, dv, dk), dtype=torch.float32)
    seq_lengths = torch.tensor([seqlen] * batch_size, dtype=torch.int32)

    return q, k, v, g, beta, scale, initial_state, seq_lengths


def _npu_op_exec(q, k, v, beta, initial_state, seq_lengths, g, scale):
    """Execute npu_chunk_gated_delta_rule on NPU and return CPU tensors."""
    o_npu, state_npu = torch.ops._C_ascend.npu_chunk_gated_delta_rule(
        query=q.npu(),
        key=k.npu(),
        value=v.npu(),
        beta=beta.npu(),
        initial_state=initial_state.npu(),
        actual_seq_lengths=seq_lengths.npu(),
        g=g.npu(),
        scale=float(scale),
    )
    return o_npu.cpu(), state_npu.cpu()


def _assert_close(actual, expected, rtol=3e-3, atol=1e-2):
    torch.testing.assert_close(
        actual.to(torch.float32),
        expected.to(torch.float32),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )


def _cleanup():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Tests: core correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seqlen", [64, 128, 256, 512, 768, 1024, 1536])
@pytest.mark.parametrize("headnum", [(8, 24)])
def test_chunk_gated_delta_rule_basic(batch_size, seqlen, headnum):
    nk, nv = headnum
    dk, dv = 128, 128
    q, k, v, g, beta, scale, initial_state, seq_lengths = _make_inputs(batch_size, seqlen, nk, nv, dk, dv)

    ref_out, ref_state = golden_chunk_gated_delta_rule(q, k, v, beta, scale, initial_state, seq_lengths, g)

    npu_out, npu_state = _npu_op_exec(q, k, v, beta, initial_state, seq_lengths, g, scale)

    _assert_close(npu_out, ref_out)
    _assert_close(npu_state, ref_state)
    _cleanup()


def test_chunk_gated_delta_rule_base_case():
    """Baseline case matching original golden: (1, 1536, 8, 24, 128, 128)."""
    q, k, v, g, beta, scale, initial_state, seq_lengths = _make_inputs(1, 1536, 8, 24, 128, 128)

    ref_out, ref_state = golden_chunk_gated_delta_rule(q, k, v, beta, scale, initial_state, seq_lengths, g)

    npu_out, npu_state = _npu_op_exec(q, k, v, beta, initial_state, seq_lengths, g, scale)

    _assert_close(npu_out, ref_out)
    _assert_close(npu_state, ref_state)
    _cleanup()
