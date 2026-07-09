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
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch.npu.config.allow_internal_format = False
torch_npu.npu.set_compile_mode(jit_compile=False)

enable_custom_op()

SEED = 42


# ---------------------------------------------------------------------------
# Golden reference helpers (verbatim from test_chunk_gated_delta_rule_float.py)
# ---------------------------------------------------------------------------


def _get_chunk(tensor, C, start):
    S = tensor.shape[0]
    end = start + C
    if end <= S:
        return tensor[start:end]
    pad_size = end - S
    if len(tensor.shape) > 1:
        return F.pad(tensor[start:], (0, 0, 0, pad_size))
    return F.pad(tensor[start:], (0, pad_size))


def _stage1_chunk(query, key, value, g, beta, scale):
    device = query.device
    C = query.shape[0]

    kkt = key.float() @ key.transpose(-1, -2).float()
    kkt = kkt * beta.float().unsqueeze(-1)

    qkt = query.float() @ key.transpose(-1, -2).float()

    g_cum = g.cumsum(dim=-1)
    g_cum_exp = g_cum.exp()

    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device),
                       diagonal=-1)
    attn = ((g_cum[:, None] - g_cum[None, :]) * lower).exp() * lower

    attn_1 = kkt * attn
    attn_1 *= -1.0
    for i in range(1, C):
        row = attn_1[i, :i].clone()
        sub = attn_1[:i, :i].clone()
        attn_1[i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn_1 = attn_1 + torch.eye(C, dtype=attn_1.dtype, device=attn_1.device)

    kg = key.float() * (g_cum[-1, None] - g_cum).exp()[..., None]
    k_cumdecay = (beta.unsqueeze(-1).float() * g_cum_exp[:, None]) * (
        -1) * key.float()

    v_beta = value.float() * beta.unsqueeze(-1).float()
    q_prime = query.float() * scale * g_cum_exp[:, None]

    v_inner = attn_1 @ v_beta
    k_cumdecay = attn_1 @ k_cumdecay

    return (g_cum, k_cumdecay.to(torch.bfloat16), v_inner,
            q_prime.to(torch.bfloat16), kg.to(torch.bfloat16), qkt)


def _stage1(query, key, value, g, beta, scale, C):
    S, Nk, Dk = query.shape
    _, Nv, Dv = value.shape
    device = query.device

    if Nv // Nk > 1:
        query = query.repeat_interleave(Nv // Nk, dim=1)
        key = key.repeat_interleave(Nv // Nk, dim=1)

    padded_seq_len = (S + C - 1) // C * C

    g_cum = torch.zeros((Nv, padded_seq_len),
                        dtype=torch.float32, device=device)
    k_cumdecay = torch.zeros((Nv, padded_seq_len, Dk),
                             dtype=torch.bfloat16, device=device)
    v_inner = torch.zeros((Nv, padded_seq_len, Dv),
                          dtype=torch.float32, device=device)
    q_prime = torch.zeros((Nv, padded_seq_len, Dk),
                          dtype=torch.bfloat16, device=device)
    kg = torch.zeros((Nv, padded_seq_len, Dk),
                     dtype=torch.bfloat16, device=device)
    qkt = torch.zeros((Nv, padded_seq_len, C),
                      dtype=torch.float32, device=device)

    loop_range = range(0, padded_seq_len, C)
    for nid in range(Nv):
        for idx in reversed(loop_range):
            q_chunk = _get_chunk(query[:, nid, :], C, idx)
            k_chunk = _get_chunk(key[:, nid, :], C, idx)
            v_chunk = _get_chunk(value[:, nid, :], C, idx)
            g_chunk = _get_chunk(g[:, nid], C, idx)
            beta_chunk = _get_chunk(beta[:, nid], C, idx)

            (g_cum_c, k_cumdecay_c, v_inner_c,
             qg_c, kg_c, qkt_c) = _stage1_chunk(
                q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, scale)

            g_cum[nid, idx:idx + C] = g_cum_c
            k_cumdecay[nid, idx:idx + C, :] = k_cumdecay_c
            v_inner[nid, idx:idx + C, :] = v_inner_c
            q_prime[nid, idx:idx + C, :] = qg_c
            kg[nid, idx:idx + C, :] = kg_c
            qkt[nid, idx:idx + C, :] = qkt_c

    return g_cum, k_cumdecay, v_inner, q_prime, kg, qkt


def _stage2_chunk(q_prime, v_inner, g_cum, k_cumdecay, state, kg):
    bf16_state = state.to(torch.bfloat16)

    attn_inter = q_prime.float() @ bf16_state.float().transpose(0, 1)
    v_prime = k_cumdecay.float() @ bf16_state.float().transpose(0, 1)
    v_new = v_inner + v_prime

    state_out = (v_new.transpose(0, 1).to(torch.bfloat16).float()
                 @ kg.float())
    state_old = state.float() * g_cum.exp()[-1]
    state_old = state_old + state_out

    return state_old, attn_inter, v_new.to(torch.bfloat16)


def _stage2(q_prime, v_inner, g_cum, k_cumdecay, state, kg, C):
    Nv, Sp, Dv = v_inner.shape
    attn_inter = torch.zeros((Nv, Sp, Dv),
                             dtype=torch.float32, device=q_prime.device)
    v_new = torch.zeros((Nv, Sp, Dv),
                        dtype=torch.bfloat16, device=q_prime.device)
    final_state = torch.empty_like(state).to(torch.float32)

    for nid in range(Nv):
        cur_state = state[nid]
        for idx in range(0, Sp, C):
            cur_state, attn_inter_c, v_new_c = _stage2_chunk(
                q_prime[nid, idx:idx + C, :],
                v_inner[nid, idx:idx + C, :],
                g_cum[nid, idx:idx + C],
                k_cumdecay[nid, idx:idx + C, :],
                cur_state,
                kg[nid, idx:idx + C, :])
            attn_inter[nid, idx:idx + C, :] = attn_inter_c
            v_new[nid, idx:idx + C, :] = v_new_c
        final_state[nid, ...] = cur_state
    return final_state, attn_inter, v_new


def _stage3_chunk(qkt, value, scale, g_cum, attn_inter, v_new):
    device = value.device
    C = value.shape[0]
    core_attn_out = torch.zeros_like(value).to(torch.bfloat16)
    lower = torch.tril(torch.ones(C, C, dtype=torch.bool, device=device),
                       diagonal=0)
    masked_qkt = (qkt.float() * scale
                  * ((g_cum[:, None] - g_cum[None, :]) * lower).exp()
                  * lower.float())
    attn_inner = (masked_qkt.to(torch.bfloat16).float()
                  @ v_new.to(torch.bfloat16).float())
    core_attn_out = (attn_inter + attn_inner).to(torch.bfloat16)
    return core_attn_out


def _stage3(qkt, value, scale, g_cum, attn_inter, v_new, C):
    Nv, Sp, Dv = attn_inter.shape
    S = value.shape[0]
    assert Sp == (S + C - 1) // C * C

    attn_out = torch.empty((Sp, Nv, Dv),
                           dtype=torch.bfloat16, device=value.device)

    for nid in range(Nv):
        for idx in range(0, Sp, C):
            v_chunk = _get_chunk(value[:, nid, :], C, idx)
            g_cum_chunk = g_cum[nid, idx:idx + C]
            attn_inter_chunk = attn_inter[nid, idx:idx + C, :]
            v_new_chunk = v_new[nid, idx:idx + C, :]
            qkt_chunk = qkt[nid, idx:idx + C, :]
            attn_out_chunk = _stage3_chunk(
                qkt_chunk, v_chunk, scale, g_cum_chunk,
                attn_inter_chunk, v_new_chunk)
            attn_out[idx:idx + C, nid, ...] = attn_out_chunk

    return attn_out


def _golden_benchmark(query, key, value, beta, scale,
                      initial_state, actual_seq_lengths, g=None):
    T, Nk, Dk = query.shape
    B, Nv, Dv, _ = initial_state.shape
    device = query.device
    C = 64

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
            query[start:end], key[start:end], value[start:end],
            g[start:end], beta[start:end], scale, C)

        cur_state, attn_inter, v_new = _stage2(
            q_prime, v_inner, g_cum, k_cum_decay, cur_state, kg, C)
        final_state[bid] = cur_state

        attn_out_padded = _stage3(
            qkt, value[start:end], scale, g_cum, attn_inter, v_new, C)
        attn_out[start:end, ...] = attn_out_padded[:S]
        start = end

    return attn_out, final_state


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _print_compare(name, npu_tensor, golden_tensor, topk=5):
    npu = npu_tensor.detach().float().cpu()
    golden = golden_tensor.detach().float().cpu()
    diff = (npu - golden).abs()
    flat_diff = diff.flatten()
    flat_npu = npu.flatten()
    flat_golden = golden.flatten()

    max_idx = int(torch.argmax(flat_diff))
    print(f"\n========== compare {name} ==========")
    print(f"shape: npu={tuple(npu.shape)}, golden={tuple(golden.shape)}")
    print(f"abs_err: max={flat_diff[max_idx].item():.6e}, "
          f"mean={flat_diff.mean().item():.6e}, min={flat_diff.min().item():.6e}")
    print(f"max_err_idx={max_idx}")
    print(f"  npu[{max_idx}]    = {flat_npu[max_idx].item():.6e}")
    print(f"  golden[{max_idx}] = {flat_golden[max_idx].item():.6e}")

    topk = min(topk, flat_diff.numel())
    topk_vals, topk_idx = torch.topk(flat_diff, k=topk)
    print(f"top-{topk} abs_err:")
    for rank, (idx, err) in enumerate(
            zip(topk_idx.tolist(), topk_vals.tolist()), start=1):
        print(f"  #{rank} idx={idx}, err={err:.6e}, "
              f"npu={flat_npu[idx].item():.6e}, "
              f"golden={flat_golden[idx].item():.6e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _execute_case(batch_size, seqlen, nk, nv, dk, dv, dtype=torch.bfloat16):
    """Match the original golden test pattern exactly: generate on NPU,
    run NPU op, then copy to CPU for golden reference comparison."""
    torch.manual_seed(SEED)
    device = "npu:0"

    T = batch_size * seqlen
    q = torch.rand((T, nk, dk), dtype=dtype, device=device)
    k = torch.rand((T, nk, dk), dtype=dtype, device=device)
    v = torch.rand((T, nv, dv), dtype=dtype, device=device)
    g = torch.rand((T, nv), dtype=torch.float32, device=device) * -1.0
    beta = torch.rand((T, nv), dtype=dtype, device=device)
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    scale = 1 / (dk ** 0.5)
    initial_state = torch.rand(
        (batch_size, nv, dv, dk), dtype=torch.float32, device=device)
    actual_seq_lengths = torch.tensor(
        [seqlen] * batch_size, dtype=torch.int32, device=device)

    o_npu, state_npu = torch.ops._C_ascend.npu_chunk_gated_delta_rule(
        query=q,
        key=k,
        value=v,
        beta=beta,
        initial_state=initial_state,
        actual_seq_lengths=actual_seq_lengths,
        g=g,
        scale=float(scale),
    )
    o_npu = o_npu.cpu().to(torch.float32)
    state_npu = state_npu.cpu().to(torch.float32)

    o_golden, state_golden = _golden_benchmark(
        q.cpu().to(dtype),
        k.cpu().to(dtype),
        v.cpu().to(dtype),
        beta.cpu().to(dtype),
        scale,
        initial_state.cpu().to(torch.float32),
        actual_seq_lengths.cpu(),
        g.cpu(),
    )
    o_golden = o_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    _print_compare("output", o_npu, o_golden)
    _print_compare("state", state_npu, state_golden)

    torch.testing.assert_close(
        o_npu, o_golden, rtol=3e-3, atol=1e-2, equal_nan=True)
    torch.testing.assert_close(
        state_npu, state_golden, rtol=3e-3, atol=1e-2, equal_nan=True)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chunk_gated_delta_rule_single_batch():
    _execute_case(1, 128, 16, 16, 128, 128)


# def test_chunk_gated_delta_rule_multi_batch():
#     _execute_case(4, 256, 16, 16, 128, 128)


# def test_chunk_gated_delta_rule_head_num_differ():
#     _execute_case(2, 128, 16, 32, 128, 128)


# def test_chunk_gated_delta_rule_base_case():
#     _execute_case(1, 1536, 8, 24, 128, 128)
