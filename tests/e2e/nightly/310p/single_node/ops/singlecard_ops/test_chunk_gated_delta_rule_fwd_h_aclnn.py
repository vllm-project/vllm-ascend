"""
chunk_gated_delta_rule_fwd_h correctness tests on Ascend 310P
via torch.ops._C_ascend binding.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

CHUNK_SIZE = 64


def npu_chunk_gdr_fwd_h(k, w, u, g, initial_state=None, chunk_size=64):
    enable_custom_op()
    return torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        chunk_size=chunk_size,
        save_new_value=True,
    )


def cpu_reference(k, w, u, g, initial_state=None, chunk_size=64):
    """CPU fp32 reference matching kernel semantics."""
    k, w, u, g = k.float(), w.float(), u.float(), g.float()
    B, Hg, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    NT = T // chunk_size
    h = initial_state.float().clone() if initial_state is not None else torch.zeros(B, HV, K, V)
    h_chunks = [h.clone()]
    v_new = torch.zeros_like(u)

    for c in range(NT):
        t0 = c * chunk_size
        W_chunk = w[:, :, t0 : t0 + chunk_size, :]
        ws = torch.einsum("bhik,bhkv->bhiv", W_chunk, h)
        g_chunk = g[:, :, t0 : t0 + chunk_size]
        v_update = torch.zeros(B, HV, chunk_size, V)
        for i in range(chunk_size):
            gi_cum = g_chunk[:, :, -1] - g_chunk[:, :, i]
            vn = u[:, :, t0 + i, :] - ws[:, :, i, :]
            v_new[:, :, t0 + i, :] = vn
            v_update[:, :, i, :] = gi_cum.unsqueeze(-1).exp() * vn
        K_chunk = k[:, :, t0 : t0 + chunk_size, :]
        h_work = torch.einsum("bhik,bhiv->bhkv", K_chunk, v_update)
        h = h * g_chunk[:, :, -1:].unsqueeze(-1).exp() + h_work
        h_chunks.append(h.clone())
    return h_chunks, v_new


def cosine(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


class TestChunkGatedDeltaRuleFwdH310:
    """chunk_gated_delta_rule_fwd_h kernel correctness on Ascend 310P."""

    @pytest.mark.parametrize(
        "B,Hg,HV,T,K,V",
        [
            (1, 1, 1, 128, 128, 128),
            (1, 2, 2, 128, 128, 128),
        ],
    )
    def test_h_state_correctness(self, B, Hg, HV, T, K, V):
        torch.manual_seed(42)
        DTYPE = torch.float16
        k = torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1
        w = torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1
        u = torch.randn(B, HV, T, V, dtype=DTYPE) * 0.1
        g = (-torch.rand(B, HV, T) * 0.1).float()
        init = torch.randn(B, HV, K, V, dtype=DTYPE) * 0.01

        h_ref, _ = cpu_reference(k, w, u, g, init, CHUNK_SIZE)
        h_out, _, _ = npu_chunk_gdr_fwd_h(
            k.npu(),
            w.npu(),
            u.npu(),
            g.npu(),
            initial_state=init.npu(),
            chunk_size=CHUNK_SIZE,
        )
        h_npu = h_out.cpu().float()
        NT = T // CHUNK_SIZE

        for c in range(min(NT + 1, h_npu.shape[2])):
            ref = h_ref[c].flatten()
            npu = h_npu[0, :, c].flatten()
            cos = cosine(npu, ref)
            assert cos >= 0.99, f"h[{c}] cos={cos:.6f} too low"

    @pytest.mark.parametrize(
        "B,Hg,HV,T,K,V",
        [
            (1, 1, 1, 128, 128, 128),
            (1, 2, 2, 128, 128, 128),
        ],
    )
    def test_v_new_correctness(self, B, Hg, HV, T, K, V):
        torch.manual_seed(42)
        DTYPE = torch.float16
        k = torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1
        w = torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1
        u = torch.randn(B, HV, T, V, dtype=DTYPE) * 0.1
        g = (-torch.rand(B, HV, T) * 0.1).float()
        init = torch.randn(B, HV, K, V, dtype=DTYPE) * 0.01

        _, vn_ref = cpu_reference(k, w, u, g, init, CHUNK_SIZE)
        _, vn_out, _ = npu_chunk_gdr_fwd_h(
            k.npu(),
            w.npu(),
            u.npu(),
            g.npu(),
            initial_state=init.npu(),
            chunk_size=CHUNK_SIZE,
        )
        vn_npu = vn_out.cpu().float()
        NT = T // CHUNK_SIZE

        for c in range(NT):
            t0, t1 = c * CHUNK_SIZE, (c + 1) * CHUNK_SIZE
            ref = vn_ref[:, :, t0:t1].flatten()
            npu = vn_npu[:, :, t0:t1].flatten()
            cos = cosine(npu, ref)
            assert cos >= 0.99, f"v_new chunk {c} cos={cos:.6f} too low"

    def test_no_nan(self):
        torch.manual_seed(42)
        B, Hg, HV, T, K, V = 1, 1, 1, 128, 128, 128
        DTYPE = torch.float16
        k = (torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1).npu()
        w = (torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1).npu()
        u = (torch.randn(B, HV, T, V, dtype=DTYPE) * 0.1).npu()
        g = (-torch.rand(B, HV, T).float() * 0.1).npu()
        init = (torch.randn(B, HV, K, V, dtype=DTYPE) * 0.01).npu()

        h_out, vn_out, _ = npu_chunk_gdr_fwd_h(
            k,
            w,
            u,
            g,
            initial_state=init,
            chunk_size=CHUNK_SIZE,
        )

        assert torch.isnan(h_out.cpu()).sum() == 0, "h_out has NaN"
        assert torch.isnan(vn_out.cpu()).sum() == 0, "vn_out has NaN"
        assert torch.isinf(h_out.cpu()).sum() == 0, "h_out has Inf"
        assert torch.isinf(vn_out.cpu()).sum() == 0, "vn_out has Inf"
