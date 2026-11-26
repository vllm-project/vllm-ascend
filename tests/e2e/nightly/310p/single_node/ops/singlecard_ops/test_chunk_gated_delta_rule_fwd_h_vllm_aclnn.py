"""
chunk_gated_delta_rule_fwd_h_vllm correctness tests on Ascend 310P
via torch.ops._C_ascend binding.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

CHUNK_SIZE = 64


def npu_chunk_gdr_fwd_h(k, w, u, g, initial_state=None, chunk_size=64, output_final_state=False):
    enable_custom_op()
    return torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h_vllm(
        k,
        w,
        u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
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
        head_groups = HV // Hg
        W_chunk = w[:, :, t0 : t0 + chunk_size, :]
        ws = torch.einsum("bhik,bhkv->bhiv", W_chunk, h)
        g_chunk = g[:, :, t0 : t0 + chunk_size]
        v_update = torch.zeros(B, HV, chunk_size, V)
        for i in range(chunk_size):
            gi_cum = g_chunk[:, :, -1] - g_chunk[:, :, i]
            vn = u[:, :, t0 + i, :] - ws[:, :, i, :]
            v_new[:, :, t0 + i, :] = vn
            v_update[:, :, i, :] = gi_cum.unsqueeze(-1).exp() * vn
        K_chunk = k[:, :, t0 : t0 + chunk_size, :].repeat_interleave(head_groups, dim=1)
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


class TestChunkGatedDeltaRuleFwdHVllm310:
    """chunk_gated_delta_rule_fwd_h_vllm kernel correctness on Ascend 310P."""

    @pytest.mark.parametrize("B,HV", [(1, 4), (1, 8), (1, 12), (1, 16), (4, 4)])
    def test_final_state_decay_across_interleaved_heads(self, B, HV):
        """The final pipeline drain must preserve every head's g_last."""
        DTYPE = torch.float16
        T = K = V = 128
        init = (torch.arange(B * HV).reshape(B, HV, 1, 1) + 1).div(128).expand(B, HV, K, V).contiguous().to(DTYPE)
        factors = 1 + (torch.arange(B * HV).reshape(B, HV) % 5).float() / 4
        g_chunk = (
            -torch.log(torch.tensor(2.0))
            * factors[:, :, None, None]
            * (torch.arange(CHUNK_SIZE) + 1)[None, None, None, :]
            / CHUNK_SIZE
        )
        g = g_chunk.expand(B, HV, 2, CHUNK_SIZE).reshape(B, HV, T).contiguous()
        zeros = torch.zeros(B, HV, T, K, dtype=DTYPE)

        _, v_new, final_state = npu_chunk_gdr_fwd_h(
            zeros.npu(),
            zeros.npu(),
            zeros.npu(),
            g.npu(),
            initial_state=init.npu(),
            chunk_size=CHUNK_SIZE,
            output_final_state=True,
        )
        expected = init.double() * (2 * g[:, :, -1].double()).exp()[:, :, None, None]
        assert torch.count_nonzero(v_new.cpu()) == 0
        assert cosine(final_state.cpu(), expected) >= 0.999999

    @pytest.mark.parametrize(
        "B,Hg,HV,T,K,V",
        [
            (1, 1, 1, 128, 128, 128),
            (1, 2, 2, 128, 128, 128),
            (1, 16, 32, 128, 128, 128),
        ],
    )
    def test_h_state_correctness(self, B, Hg, HV, T, K, V):
        torch.manual_seed(42)
        DTYPE = torch.float16
        k = torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1
        w = torch.randn(B, HV, T, K, dtype=DTYPE) * 0.1
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
        # The 310P producer and FwdO consumer share zN(K,V) state images.
        # Decode that physical layout before comparing with the ND reference.
        h_npu = (
            h_out.cpu()
            .reshape(B, HV, -1, V // 16, K // 16, 16, 16)
            .permute(0, 1, 2, 4, 5, 3, 6)
            .reshape(B, HV, -1, K, V)
            .float()
        )
        NT = T // CHUNK_SIZE

        for c in range(min(NT + 1, h_npu.shape[2])):
            ref = h_ref[c].flatten()
            npu = h_npu[0, :, c].flatten()
            cos = cosine(npu, ref)
            finite_by_head = torch.isfinite(h_npu[0, :, c]).flatten(1).all(1)
            assert cos >= 0.99, (
                f"h[{c}] cos={cos:.6f} too low; nonfinite_heads={(~finite_by_head).nonzero().flatten().tolist()}"
            )

    @pytest.mark.parametrize(
        "B,Hg,HV,T,K,V",
        [
            (1, 1, 1, 128, 128, 128),
            (1, 2, 2, 128, 128, 128),
            (1, 16, 32, 128, 128, 128),
        ],
    )
    def test_v_new_correctness(self, B, Hg, HV, T, K, V):
        torch.manual_seed(42)
        DTYPE = torch.float16
        k = torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1
        w = torch.randn(B, HV, T, K, dtype=DTYPE) * 0.1
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
            finite_by_head = torch.isfinite(vn_npu[:, :, t0:t1]).flatten(2).all(2)
            assert cos >= 0.99, (
                f"v_new chunk {c} cos={cos:.6f} too low; "
                f"nonfinite_heads={(~finite_by_head[0]).nonzero().flatten().tolist()}"
            )

    @pytest.mark.parametrize(
        "batch,key_heads,value_heads,key_dim,value_dim,chunk_size,state_dtype",
        [
            (1, 1, 1, 128, 256, 64, torch.float16),
            (2, 2, 4, 192, 256, 64, torch.float32),
            (2, 1, 3, 128, 144, 64, torch.float32),
            (1, 1, 2, 128, 256, 32, torch.float32),
        ],
    )
    def test_value_tiles_match_cpu_reference(
        self, batch, key_heads, value_heads, key_dim, value_dim, chunk_size, state_dtype
    ):
        torch.manual_seed(312)
        tokens = 3 * chunk_size
        k = (torch.randn(batch, key_heads, tokens, key_dim) * 0.08).half()
        w = (torch.randn(batch, value_heads, tokens, key_dim) * 0.06).half()
        u = (torch.randn(batch, value_heads, tokens, value_dim) * 0.1).half()
        g = (-torch.rand(batch, value_heads, 3, chunk_size) * 0.015).cumsum(-1).flatten(2)
        state = (torch.randn(batch, value_heads, key_dim, value_dim) * 0.03).to(state_dtype)

        h, v_new, final = npu_chunk_gdr_fwd_h(
            k.npu(),
            w.npu(),
            u.npu(),
            g.npu(),
            initial_state=state.npu(),
            chunk_size=chunk_size,
            output_final_state=True,
        )
        expected_h, expected_v = cpu_reference(k, w, u, g, state, chunk_size)
        # h is stored as zN panels, while final_state is row major.
        actual_h = (
            h.cpu()
            .reshape(batch, value_heads, 3, value_dim // 16, key_dim // 16, 16, 16)
            .permute(0, 1, 2, 4, 5, 3, 6)
            .reshape(batch, value_heads, 3, key_dim, value_dim)
            .float()
        )
        torch.testing.assert_close(actual_h, torch.stack(expected_h[:-1], dim=2), atol=0.001, rtol=0.01)
        torch.testing.assert_close(v_new.cpu().float(), expected_v, atol=0.001, rtol=0.01)
        torch.testing.assert_close(final.cpu().float(), expected_h[-1], atol=0.001, rtol=0.01)

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


@pytest.mark.parametrize("packed", [False, True])
def test_gdn_prefill_with_wide_values(packed):
    from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import (
        chunk_gated_delta_rule_310,
        chunk_gated_delta_rule_pytorch,
    )

    torch.manual_seed(313)
    tokens, key_heads, value_heads, key_dim, value_dim = 128, 1, 2, 128, 256
    q = torch.nn.functional.normalize(torch.randn(1, tokens, key_heads, key_dim), dim=-1).half()
    k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).half()
    v = (torch.randn(1, tokens, value_heads, value_dim) * 0.2).half()
    g = -torch.rand(1, tokens, value_heads) * 0.025
    beta = (0.1 + torch.rand_like(g) * 0.8).half()
    cu_seqlens = torch.tensor([0, 65, tokens], dtype=torch.int64) if packed else None
    state = torch.randn(2 if packed else 1, value_heads, value_dim, key_dim) * 0.02
    npu_inputs = [x.npu() for x in (q, k, v, g, beta)]
    output, final = chunk_gated_delta_rule_310(
        *npu_inputs, initial_state=state.npu(), output_final_state=True, cu_seqlens=cu_seqlens
    )
    expected, expected_final = chunk_gated_delta_rule_pytorch(
        q, k, v, g, beta, initial_state=state, output_final_state=True, cu_seqlens=cu_seqlens
    )
    assert final is not None and expected_final is not None
    torch.testing.assert_close(output.cpu().float(), expected.float(), atol=0.001, rtol=0.01)
    torch.testing.assert_close(final.cpu().float(), expected_final, atol=0.001, rtol=0.01)
