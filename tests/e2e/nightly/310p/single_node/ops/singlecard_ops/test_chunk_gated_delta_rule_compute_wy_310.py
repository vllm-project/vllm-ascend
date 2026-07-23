"""
chunk_gated_delta_rule_compute_wy correctness tests on Ascend 310P.
"""

import torch
import torch_npu  # noqa: F401

import vllm_ascend._310p.ops.fla.chunk_gated_delta_rule as chunk_mod
from vllm_ascend.utils import enable_custom_op

CHUNK_SIZE = 64


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _make_inputs_cpu(batch=1, tokens=128, q_heads=2, v_heads=4, k_dim=128, v_dim=128, g_scale=0.01):
    torch.manual_seed(123)
    q = (torch.randn(batch, tokens, q_heads, k_dim) * 0.01).half()
    k = (torch.randn(batch, tokens, q_heads, k_dim) * 0.01).half()
    v = (torch.randn(batch, tokens, v_heads, v_dim) * 0.01).half()
    g = -torch.rand(batch, tokens, v_heads, dtype=torch.float32) * g_scale
    beta = (0.1 + 0.2 * torch.rand(batch, tokens, v_heads)).half()
    return q, k, v, g, beta


def test_doubling_wy_matches_torch_reference_cpu():
    """Phase-0 gate: nilpotent doubling matches the scalar-loop torch WY reference."""
    for tokens, g_scale in ((128, 0.01), (512, 0.01), (128, 1.0)):
        q, k, v, g, beta = _make_inputs_cpu(tokens=tokens, q_heads=16, v_heads=32, g_scale=g_scale)
        ref = chunk_mod._compute_kernel_inputs_from_torch_wy(q, k, v, g, beta, CHUNK_SIZE)
        out = chunk_mod._compute_kernel_inputs_from_doubling_wy(q, k, v, g, beta, CHUNK_SIZE)
        torch.testing.assert_close(out[0], ref[0], rtol=0, atol=0)
        torch.testing.assert_close(out[1], ref[1], rtol=0, atol=0)
        torch.testing.assert_close(out[4], ref[4], rtol=1e-5, atol=1e-5)
        assert _cosine(out[2].float(), ref[2].float()) > 0.99
        assert _cosine(out[3].float(), ref[3].float()) > 0.99

        a_mat, rhs, _, _ = chunk_mod._wy_build_A_and_R(k, v, g, beta, CHUNK_SIZE)
        blocked = chunk_mod._wy_blocked_fs_apply(a_mat, rhs)
        doubled = chunk_mod._wy_doubling_apply(a_mat, rhs)
        assert _cosine(doubled, blocked) > 0.99


def _make_inputs(batch=1, tokens=128, q_heads=2, v_heads=4, k_dim=128, v_dim=128):
    torch.manual_seed(123)
    q = (torch.randn(batch, tokens, q_heads, k_dim) * 0.01).half().npu()
    k = (torch.randn(batch, tokens, q_heads, k_dim) * 0.01).half().npu()
    v = (torch.randn(batch, tokens, v_heads, v_dim) * 0.01).half().npu()
    g = (-torch.rand(batch, tokens, v_heads, dtype=torch.float32) * 0.01).npu()
    beta = (0.1 + 0.2 * torch.rand(batch, tokens, v_heads)).half().npu()
    return q, k, v, g, beta


def _assert_compute_wy_close(out, ref):
    torch.testing.assert_close(out[0].cpu(), ref[0].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(out[1].cpu(), ref[1].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(out[4].cpu(), ref[4].cpu(), rtol=1e-5, atol=1e-5)
    assert _cosine(out[2].cpu().float(), ref[2].cpu().float()) > 0.99
    assert _cosine(out[3].cpu().float(), ref[3].cpu().float()) > 0.99


def test_compute_wy_matches_torch_reference_grouped_heads():
    enable_custom_op()
    q, k, v, g, beta = _make_inputs(q_heads=2, v_heads=4)

    ref = chunk_mod._compute_kernel_inputs_from_torch_wy(q, k, v, g, beta, CHUNK_SIZE)
    out = torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, CHUNK_SIZE)
    _assert_compute_wy_close(out, ref)


def test_compute_wy_matches_torch_reference_qwen35_heads():
    enable_custom_op()
    q, k, v, g, beta = _make_inputs(batch=1, tokens=64, q_heads=8, v_heads=16, k_dim=64, v_dim=64)
    assert chunk_mod._can_use_npu_compute_wy(q, k, v, g, beta, CHUNK_SIZE)

    ref = chunk_mod._compute_kernel_inputs_from_torch_wy(q, k, v, g, beta, CHUNK_SIZE)
    out = torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, CHUNK_SIZE)
    _assert_compute_wy_close(out, ref)
    print("✓ test_compute_wy_matches_torch_reference_qwen35_heads passed")


def test_chunk_gated_delta_rule_310_uses_npu_wy(monkeypatch):
    enable_custom_op()
    q, k, v, g, beta = _make_inputs(batch=1, tokens=128, q_heads=2, v_heads=2)
    initial_state = torch.zeros(1, v.shape[2], v.shape[-1], k.shape[-1], dtype=torch.float32, device=q.device)

    out_npu, state_npu = chunk_mod.chunk_gated_delta_rule_310(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )

    monkeypatch.setattr(chunk_mod, "_can_use_npu_compute_wy", lambda *args, **kwargs: False)
    out_ref, state_ref = chunk_mod.chunk_gated_delta_rule_310(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )

    assert state_npu is not None
    assert state_ref is not None
    assert _cosine(out_npu.cpu().float(), out_ref.cpu().float()) > 0.99
    assert _cosine(state_npu.cpu().float(), state_ref.cpu().float()) > 0.99


if __name__ == "__main__":
    print("Running tests...")
    test_doubling_wy_matches_torch_reference_cpu()
    test_compute_wy_matches_torch_reference_grouped_heads()
    test_compute_wy_matches_torch_reference_qwen35_heads()
    test_chunk_gated_delta_rule_310_uses_npu_wy()
    print("\n✓ All tests passed!")