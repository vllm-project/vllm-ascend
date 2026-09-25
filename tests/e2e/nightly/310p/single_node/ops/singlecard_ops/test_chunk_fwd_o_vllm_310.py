"""
chunk_fwd_o_vllm correctness tests on Ascend 310P via torch.ops._C_ascend binding.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

CHUNK_SIZE = 64


def test_golden_decay_sign():
    """A previous value decays to one half; the current value contributes one."""
    q = torch.ones(1, 1, CHUNK_SIZE, 1)
    h = torch.zeros(1, 1, 1, 1)
    g = torch.full((1, 1, CHUNK_SIZE), -torch.log(torch.tensor(2.0)).item())
    g[..., 0] = 0
    actual = golden_chunk_fwd_o(q, q, q, h, g, 1.0)
    assert actual[0, 0, 0, 0].item() == 1.0
    assert actual[0, 0, 1, 0].item() == 1.5


def npu_chunk_fwd_o(q, k, v, h, g, scale):
    enable_custom_op()
    return torch.ops._C_ascend.chunk_fwd_o_vllm(
        q,
        k,
        v,
        h,
        scale,
        g=g,
        g_gamma=None,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=CHUNK_SIZE,
        transpose_state_layout=False,
    )


def golden_chunk_fwd_o(q, k, v, h_state, g, scale):
    """CPU fp32 reference.

    Per chunk c (CS tokens starting at t0):
      attn = q[c] @ k[c].T                             [CS, CS]
      gate[i,j] = exp(min(0, g[i] - g[j])) * (j<=i)   [CS, CS]
      attn_masked = attn * gate
      h_work = q[c] @ h_state[c]                       [CS, Dv]
      v_work = attn_masked @ v[c]                       [CS, Dv]
      o[c] = scale * (v_work + exp(g[c]) * h_work)
    """
    q, k, v, g = q.float(), k.float(), v.float(), g.float()
    h_state = h_state.float()
    B, H_k, L, D_k = q.shape
    H_v, D_v = v.shape[1], v.shape[3]
    CS = CHUNK_SIZE
    NT = L // CS
    head_groups = H_v // H_k
    o = torch.zeros(B, H_v, L, D_v)
    for b in range(B):
        for hv in range(H_v):
            hk = hv // head_groups
            for c in range(NT):
                t0 = c * CS
                q_c = q[b, hk, t0 : t0 + CS]
                k_c = k[b, hk, t0 : t0 + CS]
                v_c = v[b, hv, t0 : t0 + CS]
                g_c = g[b, hv, t0 : t0 + CS]
                h_c = h_state[b, hv, c * D_k : (c + 1) * D_k]
                attn = q_c @ k_c.T
                g_row = g_c.unsqueeze(1)
                g_col = g_c.unsqueeze(0)
                gate = torch.exp(torch.clamp(g_row - g_col, max=0.0))
                causal = torch.tril(torch.ones(CS, CS))
                attn_masked = attn * gate * causal
                h_work = q_c @ h_c
                v_work = attn_masked @ v_c
                g_exp = torch.exp(g_c).unsqueeze(1)
                o[b, hv, t0 : t0 + CS] = scale * (v_work + g_exp * h_work)
    return o


class TestChunkFwdOVllm310:
    """chunk_fwd_o_vllm kernel correctness on Ascend 310P."""

    @pytest.mark.parametrize(
        "B,Hk,Hv,L,Dk,Dv",
        [
            (1, 2, 2, 128, 128, 128),
            (1, 4, 4, 256, 128, 128),
        ],
    )
    def test_constant_inputs(self, B, Hk, Hv, L, Dk, Dv):
        """Constant q=k=v, h=0, g=0 => analytically verifiable output."""
        scale = 1.0 / (Dk**0.5)
        NC = L // CHUNK_SIZE
        c = 0.01
        q = torch.full((B, Hk, L, Dk), c, dtype=torch.float16).npu()
        k = torch.full((B, Hk, L, Dk), c, dtype=torch.float16).npu()
        v = torch.full((B, Hv, L, Dv), c, dtype=torch.float16).npu()
        h = torch.zeros(B, Hv, NC * Dk, Dv, dtype=torch.float16).npu()
        g = torch.zeros(B, Hv, L, dtype=torch.float32).npu()

        o = npu_chunk_fwd_o(q, k, v, h, g, scale)
        oc = o.cpu().float()

        assert torch.isnan(oc).sum() == 0, "output has NaN"
        assert torch.isinf(oc).sum() == 0, "output has Inf"

        attn_val = c * c * Dk
        for i in range(CHUNK_SIZE):
            expected = scale * (i + 1) * attn_val * c
            actual = oc[0, 0, i, 0].item()
            rel_err = abs(actual - expected) / max(abs(expected), 1e-10)
            assert rel_err < 0.10, f"row {i}: actual={actual:.8f} expected={expected:.8f} rel_err={rel_err:.2f}"

    @pytest.mark.parametrize(
        "B,Hk,Hv,L,Dk,Dv",
        [
            (1, 2, 2, 128, 128, 128),
            (1, 4, 4, 256, 128, 128),
        ],
    )
    def test_random_inputs_no_nan(self, B, Hk, Hv, L, Dk, Dv):
        """Random small inputs: no NaN/Inf in output."""
        torch.manual_seed(42)
        scale = 1.0 / (Dk**0.5)
        NC = L // CHUNK_SIZE
        q = (torch.randn(B, Hk, L, Dk) * 0.01).half().npu()
        k = (torch.randn(B, Hk, L, Dk) * 0.01).half().npu()
        v = (torch.randn(B, Hv, L, Dv) * 0.01).half().npu()
        h = (torch.randn(B, Hv, NC * Dk, Dv) * 0.01).half().npu()
        g = torch.randn(B, Hv, L, dtype=torch.float32).npu() * 0.001

        o = npu_chunk_fwd_o(q, k, v, h, g, scale)
        oc = o.cpu().float()

        assert torch.isnan(oc).sum() == 0, "output has NaN"
        assert torch.isinf(oc).sum() == 0, "output has Inf"
        assert oc.abs().max() > 0, "output is all zeros"

    def test_g_zero_reduces_to_standard_attention(self):
        """g=0 => gate=1, so kernel = scale*(causal_attn@v + q@h)."""
        torch.manual_seed(123)
        B, Hk, Hv, L, Dk, Dv = 1, 2, 2, 128, 128, 128
        scale = 1.0 / (Dk**0.5)
        NC = L // CHUNK_SIZE
        q = (torch.randn(B, Hk, L, Dk) * 0.01).half()
        k = (torch.randn(B, Hk, L, Dk) * 0.01).half()
        v = (torch.randn(B, Hv, L, Dv) * 0.01).half()
        h = torch.zeros(B, Hv, NC * Dk, Dv, dtype=torch.float16)
        g = torch.zeros(B, Hv, L, dtype=torch.float32)

        o_npu = npu_chunk_fwd_o(q.npu(), k.npu(), v.npu(), h.npu(), g.npu(), scale)
        o_ref = golden_chunk_fwd_o(q, k, v, h, g, scale)

        cos = torch.nn.functional.cosine_similarity(o_npu.cpu().float().flatten(), o_ref.flatten(), dim=0).item()
        assert cos > 0.999, f"cosine {cos:.4f} too low for g=0 h=0 case"

    def test_chunk_boundary_independence(self):
        """Each chunk should produce the same output for identical data."""
        B, Hk, Hv, L, Dk, Dv = 1, 2, 2, 128, 128, 128
        scale = 1.0 / (Dk**0.5)
        NC = L // CHUNK_SIZE
        c = 0.02
        q = torch.full((B, Hk, L, Dk), c, dtype=torch.float16).npu()
        k = torch.full((B, Hk, L, Dk), c, dtype=torch.float16).npu()
        v = torch.full((B, Hv, L, Dv), c, dtype=torch.float16).npu()
        h = torch.zeros(B, Hv, NC * Dk, Dv, dtype=torch.float16).npu()
        g = torch.zeros(B, Hv, L, dtype=torch.float32).npu()

        o = npu_chunk_fwd_o(q, k, v, h, g, scale).cpu().float()

        chunk0 = o[0, 0, :CHUNK_SIZE, :]
        chunk1 = o[0, 0, CHUNK_SIZE:, :]
        cos = torch.nn.functional.cosine_similarity(chunk0.flatten(), chunk1.flatten(), dim=0).item()
        assert cos > 0.999, f"chunks differ: cosine={cos:.6f}"

    @pytest.mark.parametrize("future_value", [1.0, 2.0])
    def test_real_shape_full_causal_mask(self, future_value):
        """Catch future-value leakage, including the last eight rows of every chunk."""
        heads_qk, heads_v, tokens, dim = 8, 16, 1920, 128
        q = torch.ones(1, heads_qk, tokens, dim, dtype=torch.float16).npu()
        k = torch.ones_like(q)
        block = torch.zeros(CHUNK_SIZE, dim, dtype=torch.float16)
        block[:, :CHUNK_SIZE] = torch.eye(CHUNK_SIZE, dtype=torch.float16)
        v_cpu = block.repeat(tokens // CHUNK_SIZE, 1).expand(1, heads_v, tokens, dim).clone()
        expected_block = torch.zeros_like(block)
        expected_block[:, :CHUNK_SIZE] = torch.ones(CHUNK_SIZE, CHUNK_SIZE).tril()
        expected = expected_block.repeat(tokens // CHUNK_SIZE, 1).expand_as(v_cpu).clone()
        # Perturb one head and one chunk only. Earlier rows and all other
        # heads/chunks must remain identical to their independently known answer.
        v_cpu[0, 1, CHUNK_SIZE - 1] *= future_value
        expected[0, 1, CHUNK_SIZE - 1, CHUNK_SIZE - 1] = future_value
        v = v_cpu.npu()
        h = torch.zeros(1, heads_v, tokens // CHUNK_SIZE, dim, dim, dtype=torch.float16).npu()
        g = torch.zeros(1, heads_v, tokens, dtype=torch.float32).npu()
        for repeat in range(5):
            actual = npu_chunk_fwd_o(q, k, v, h, g, 1.0 / dim).cpu()
            bad = actual != expected
            assert not bad.any(), (
                f"repeat={repeat}, wrong_elements={bad.sum().item()}, "
                f"first_bad={bad.nonzero()[:8].tolist()}"
            )
        assert torch.equal(v.cpu(), v_cpu), "FwdO modified its value input"

    @pytest.mark.parametrize("g_dtype", [torch.float16, torch.float32])
    @pytest.mark.parametrize("decay_multiplier", [1, 4])
    def test_real_shape_distinct_gated_heads_and_chunks(self, g_dtype, decay_multiplier):
        """Catch lost/reversed decay and wrong head, chunk or second-stage g indices."""
        heads, chunks, dim = 16, 30, 128
        tokens = chunks * CHUNK_SIZE
        q = torch.ones(1, 8, tokens, dim, dtype=torch.float16).npu()
        k = torch.ones_like(q)
        block = torch.zeros(CHUNK_SIZE, dim, dtype=torch.float16)
        block[:, :CHUNK_SIZE] = torch.eye(CHUNK_SIZE, dtype=torch.float16)
        v = block.repeat(chunks, 1).expand(1, heads, tokens, dim).contiguous().npu()
        h = torch.zeros(1, heads, chunks, dim, dim, dtype=torch.float16).npu()
        # Integer multiples of 1/512, rounded once to the selected input dtype.
        # Reversing head/block order must change a measurable number of outputs.
        # The stronger decay crosses the FP32 exp(-g) overflow boundary;
        # exp(g_i - g_j) remains finite on every causal element.
        slopes = decay_multiplier * (torch.arange(heads * chunks).reshape(1, heads, chunks, 1) + 1) / 512
        g_cpu = (-slopes * torch.arange(CHUNK_SIZE)).to(g_dtype).contiguous()
        gv = g_cpu.double()
        gate = torch.exp((gv.unsqueeze(-1) - gv.unsqueeze(-2)).clamp(max=0)).tril()
        expected = torch.zeros(1, heads, chunks, CHUNK_SIZE, dim, dtype=torch.float64)
        expected[..., :CHUNK_SIZE] = gate
        # Validate that this fixture rejects the specific addressing mutations.
        for mutant in (gate.flip(1), gate.flip(2), torch.ones_like(gate).tril()):
            assert ((mutant - gate).abs() > .002 + .005 * gate.abs()).any()
        g = g_cpu.reshape(1, heads, tokens).npu()
        before = g.cpu().clone()
        expected = expected.reshape(1, heads, tokens, dim)
        first = None
        for repeat in range(3):
            actual = npu_chunk_fwd_o(q, k, v, h, g, 1.0 / dim).cpu()
            error = (actual.double() - expected).abs()
            bad = ~torch.isfinite(actual) | (error > .002 + .005 * expected.abs())
            assert not bad.any(), f"repeat={repeat}, wrong={bad.sum().item()}, max_abs={error.max().item()}"
            if first is None:
                first = actual.clone()
            else:
                assert torch.equal(actual, first), "nonzero-g output changed across repeats"
        assert torch.equal(g.cpu(), before), "FwdO modified g"

    @pytest.mark.parametrize("tail", [1, 7, 8, 11, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63])
    @pytest.mark.parametrize("g_dtype", [torch.float16, torch.float32])
    def test_packed_partial_chunks(self, tail, g_dtype):
        """NZ strides, exact DMA tails, nonzero states and packed boundaries."""
        heads_qk, heads_v, dim = 16, 32, 128
        lengths = [CHUNK_SIZE + tail, CHUNK_SIZE, tail]
        tokens = sum(lengths)
        chunks = 4
        q = torch.ones(1, heads_qk, tokens, dim, dtype=torch.float16)
        v = torch.zeros(1, heads_v, tokens, dim, dtype=torch.float16)
        g = torch.zeros(1, heads_v, tokens, dtype=g_dtype)
        h = torch.zeros(1, heads_v, chunks, dim, dim, dtype=torch.float16)
        expected = torch.zeros_like(v, dtype=torch.float64)
        cu, indices, chunk_index = [0], [], 0
        for seq, length in enumerate(lengths):
            start = cu[-1]
            for local_chunk, pos in enumerate(range(0, length, CHUNK_SIZE)):
                n = min(CHUNK_SIZE, length - pos)
                token_slice = slice(start + pos, start + pos + n)
                v[0, :, token_slice, :n] = torch.eye(n)
                # Distinct heads/states also expose incorrect h-layout contracts.
                h[0, :, chunk_index, 0, :] = (
                    (torch.arange(heads_v)[:, None] + chunk_index + 1)
                    * (torch.arange(dim)[None, :] + 1) / 1024
                )
                slopes = (torch.arange(heads_v)[:, None] + 1) / 16
                gate_log = (-slopes * torch.arange(n)).to(g_dtype)
                g[0, :, token_slice] = gate_log
                gd = gate_log.double()
                expected[0, :, token_slice, :n] = (
                    gd.unsqueeze(-1) - gd.unsqueeze(-2)
                ).clamp(max=0).exp().tril()
                expected[0, :, token_slice] += (
                    gd.exp().unsqueeze(-1) * h[0, :, chunk_index, 0].double().unsqueeze(1) / dim
                )
                indices.extend((seq, local_chunk))
                chunk_index += 1
            cu.append(start + length)
        # FwdH -> FwdO uses packed zN state tiles on 310P.
        h_nz = h.reshape(1, heads_v, chunks, dim // 16, 16, dim // 16, 16)
        h_nz = h_nz.permute(0, 1, 2, 5, 3, 4, 6).contiguous().reshape_as(h)
        qn, vn, hn, gn = (x.npu() for x in (q, v, h_nz, g))
        enable_custom_op()
        first = None
        for repeat in range(3):
            actual = torch.ops._C_ascend.chunk_fwd_o(
                qn, qn, vn, hn, 1.0 / dim, g=gn, cu_seqlens=cu,
                chunk_indices=indices, chunk_size=CHUNK_SIZE,
                transpose_state_layout=False,
            ).cpu()
            error = (actual.double() - expected).abs()
            bad = ~torch.isfinite(actual) | (error > .002 + .005 * expected.abs())
            assert not bad.any(), (
                f"tail={tail}, repeat={repeat}, wrong={bad.sum().item()}, "
                f"first_bad={bad.nonzero()[:5].tolist()}, max_abs={error.max().item()}"
            )
            if first is None:
                first = actual.clone()
            else:
                assert torch.equal(first, actual)
