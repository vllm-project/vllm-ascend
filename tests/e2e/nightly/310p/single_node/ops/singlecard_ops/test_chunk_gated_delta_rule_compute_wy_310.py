"""
chunk_gated_delta_rule_compute_wy correctness tests on Ascend 310P.

Also compares end-to-end chunk_gated_delta_rule_310 with:
  - NPU AscendC compute_wy
  - torch WY fallback (same fwd_h / fwd_o)
"""

import pytest
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


def _make_inputs(batch=1, tokens=128, q_heads=2, v_heads=4, k_dim=64, v_dim=64, seed=123):
    torch.manual_seed(seed)
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


def _run_cgdr_310(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    *,
    output_final_state=True,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
):
    return chunk_mod.chunk_gated_delta_rule_310(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def test_compute_wy_matches_torch_reference_grouped_heads():
    enable_custom_op()
    q, k, v, g, beta = _make_inputs(q_heads=2, v_heads=4)
    assert chunk_mod._can_use_npu_compute_wy(q, k, v, g, beta, CHUNK_SIZE)

    ref = chunk_mod._compute_kernel_inputs_from_torch_wy(q, k, v, g, beta, CHUNK_SIZE)
    out = torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, CHUNK_SIZE)
    _assert_compute_wy_close(out, ref)


def test_compute_wy_rejects_head_dim_over_ub_budget():
    """K=V=128 is a legal model shape but overflows 310P UB; NPU WY must skip it."""
    enable_custom_op()
    q, k, v, g, beta = _make_inputs(k_dim=128, v_dim=128)
    assert not chunk_mod._can_use_npu_compute_wy(q, k, v, g, beta, CHUNK_SIZE)


def test_compute_wy_matches_torch_reference_qwen35_heads():
    enable_custom_op()
    q, k, v, g, beta = _make_inputs(batch=1, tokens=64, q_heads=8, v_heads=16, k_dim=64, v_dim=64)
    assert chunk_mod._can_use_npu_compute_wy(q, k, v, g, beta, CHUNK_SIZE)

    ref = chunk_mod._compute_kernel_inputs_from_torch_wy(q, k, v, g, beta, CHUNK_SIZE)
    out = torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, CHUNK_SIZE)
    _assert_compute_wy_close(out, ref)
    expected = tuple(tensor.cpu() for tensor in out)
    for _ in range(20):
        repeated = torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, CHUNK_SIZE)
        for actual, expected_tensor in zip(repeated, expected):
            torch.testing.assert_close(actual.cpu(), expected_tensor, rtol=0, atol=0)


def test_compute_wy_qwen35_production_shape_is_deterministic():
    """Regression: production-like long prefill must be bitwise deterministic."""
    enable_custom_op()
    q, k, v, g, beta = _make_inputs(batch=1, tokens=1536, q_heads=8, v_heads=16, k_dim=64, v_dim=64)
    assert chunk_mod._can_use_npu_compute_wy(q, k, v, g, beta, CHUNK_SIZE)

    baseline = torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, CHUNK_SIZE)
    expected = tuple(tensor.cpu() for tensor in baseline)
    for _ in range(5):
        repeated = torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, CHUNK_SIZE)
        for actual, expected_tensor in zip(repeated, expected):
            torch.testing.assert_close(actual.cpu(), expected_tensor, rtol=0, atol=0)


@pytest.mark.parametrize(
    "batch,tokens,q_heads,v_heads,k_dim,v_dim",
    [
        (1, 128, 2, 2, 64, 64),  # equal heads
        (1, 128, 2, 4, 64, 64),  # grouped heads
        (1, 64, 8, 16, 64, 64),  # qwen-like short
        (1, 256, 8, 16, 64, 64),  # qwen-like medium
        (2, 128, 4, 8, 64, 64),  # batch>1
    ],
)
def test_cgdr_310_npu_wy_matches_torch_wy_e2e(monkeypatch, batch, tokens, q_heads, v_heads, k_dim, v_dim):
    """End-to-end: same fwd_h/fwd_o, only compute_wy path differs (AscendC vs torch)."""
    enable_custom_op()
    q, k, v, g, beta = _make_inputs(
        batch=batch, tokens=tokens, q_heads=q_heads, v_heads=v_heads, k_dim=k_dim, v_dim=v_dim
    )
    assert chunk_mod._can_use_npu_compute_wy(q, k, v, g, beta, CHUNK_SIZE)
    initial_state = torch.zeros(batch, v.shape[2], v.shape[-1], k.shape[-1], dtype=torch.float32, device=q.device)

    out_npu, state_npu = _run_cgdr_310(q, k, v, g, beta, initial_state)

    monkeypatch.setattr(chunk_mod, "_can_use_npu_compute_wy", lambda *args, **kwargs: False)
    out_ref, state_ref = _run_cgdr_310(q, k, v, g, beta, initial_state)

    assert state_npu is not None and state_ref is not None
    assert out_npu.shape == out_ref.shape == v.shape
    cos_o = _cosine(out_npu.cpu().float(), out_ref.cpu().float())
    cos_s = _cosine(state_npu.cpu().float(), state_ref.cpu().float())
    assert cos_o > 0.99, f"out cosine={cos_o}"
    assert cos_s > 0.99, f"state cosine={cos_s}"


def test_chunk_gated_delta_rule_310_uses_npu_wy(monkeypatch):
    """Kept for nightly compatibility; covered more thoroughly by parametrized e2e above."""
    test_cgdr_310_npu_wy_matches_torch_wy_e2e(monkeypatch, 1, 128, 2, 2, 64, 64)


def _colleague_precision_inputs(seed=42, g_scale=1.0, dim=64):
    """Qwen3.5-2B-like GDN prefill, with head dim capped at 64 for NPU WY.

    Production Qwen3.5-2B uses dim=128, but 310P UB cannot hold K=V=128
    InitBuffers, so NPU compute_wy is limited to dim<=64. Default here
    exercises the NPU path; pass dim=128 to cover torch fallback.

    token=111, q/k/v [1, 111, 16, dim] fp16, g [1, 111, 16] fp32,
    beta [1, 111, 16] fp16, initial_state [1, 16, dim, dim] fp16,
    scale=None, output_final_state=False, head_first=False,
    use_qk_l2norm_in_kernel=True.

    g_scale defaults to O(1) (production-like). Old *0.01 gates hide gram/WY errors
    because A≈0 and doubling is nearly a no-op.
    """
    torch.manual_seed(seed)
    batch, tokens, heads = 1, 111, 16
    # Unit-scale q/k: after l2norm the gram is O(1), matching serve.
    q = torch.randn(batch, tokens, heads, dim).half().npu()
    k = torch.randn(batch, tokens, heads, dim).half().npu()
    v = (torch.randn(batch, tokens, heads, dim) * 0.1).half().npu()
    g = (-torch.rand(batch, tokens, heads, dtype=torch.float32) * g_scale).npu()
    beta = (0.1 + 0.8 * torch.rand(batch, tokens, heads)).half().npu()
    initial_state = (torch.randn(batch, heads, dim, dim) * 0.01).half().npu()
    return q, k, v, g, beta, initial_state


def test_compute_wy_qwen35_2b_l2norm_matches_torch():
    """Op-level: padded Qwen3.5-2B shape + l2norm, NPU w/u vs torch WY / doubling."""
    from vllm_ascend._310p.ops.fla.l2norm import l2norm_310p

    enable_custom_op()
    q, k, v, g, beta, _ = _colleague_precision_inputs()
    q = l2norm_310p(q)
    k = l2norm_310p(k)
    q_pad, k_pad, v_pad, g_pad, beta_pad, _, _ = chunk_mod._pad_bthd_to_chunk(q, k, v, g, beta, CHUNK_SIZE)
    assert chunk_mod._can_use_npu_compute_wy(q_pad, k_pad, v_pad, g_pad, beta_pad, CHUNK_SIZE)

    ref = chunk_mod._compute_kernel_inputs_from_torch_wy(q_pad, k_pad, v_pad, g_pad, beta_pad, CHUNK_SIZE)
    dbl = chunk_mod._compute_kernel_inputs_from_doubling_wy(q_pad, k_pad, v_pad, g_pad, beta_pad, CHUNK_SIZE)
    out = torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(q_pad, k_pad, v_pad, g_pad, beta_pad, CHUNK_SIZE)

    # g cumsum must match exactly (fp32 path).
    torch.testing.assert_close(out[4].cpu(), ref[4].cpu(), rtol=1e-5, atol=1e-5)
    # Doubling is what AscendC implements; must track torch FS closely under l2norm.
    assert _cosine(dbl[2].cpu().float(), ref[2].cpu().float()) > 0.99, "doubling w vs torch"
    assert _cosine(dbl[3].cpu().float(), ref[3].cpu().float()) > 0.99, "doubling u vs torch"
    cos_w = _cosine(out[2].cpu().float(), ref[2].cpu().float())
    cos_u = _cosine(out[3].cpu().float(), ref[3].cpu().float())
    assert cos_w > 0.99, f"NPU w cosine vs torch={cos_w}"
    assert cos_u > 0.99, f"NPU u cosine vs torch={cos_u}"


def test_compute_wy_correlated_qk_uses_stable_fp32_path():
    """Serve regression: correlated K makes ||A|| large and must remain finite."""
    from vllm_ascend._310p.ops.fla.l2norm import l2norm_310p

    enable_custom_op()
    torch.manual_seed(2026)
    batch, tokens, heads, dim = 1, 47, 16, 64
    q_base = torch.randn(batch, 1, heads, dim)
    k_base = torch.randn(batch, 1, heads, dim)
    q = (q_base + 0.02 * torch.randn(batch, tokens, heads, dim)).half().npu()
    k = (k_base + 0.02 * torch.randn(batch, tokens, heads, dim)).half().npu()
    v = (0.05 * torch.randn(batch, tokens, heads, dim)).half().npu()
    g = torch.full((batch, tokens, heads), -0.001, dtype=torch.float32).npu()
    beta = torch.full((batch, tokens, heads), 0.8, dtype=torch.float16).npu()

    q = l2norm_310p(q)
    k = l2norm_310p(k)
    q, k, v, g, beta, _, _ = chunk_mod._pad_bthd_to_chunk(q, k, v, g, beta, CHUNK_SIZE)
    a_mat, _, _, _ = chunk_mod._wy_build_A_and_R(k, v, g, beta, CHUNK_SIZE)
    max_row_sum = a_mat.abs().sum(dim=-1).amax().item()
    assert max_row_sum >= 0.75, f"test did not select stable path: ||A||_inf={max_row_sum}"

    ref = chunk_mod._compute_kernel_inputs_from_torch_wy(q, k, v, g, beta, CHUNK_SIZE)
    out = torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, CHUNK_SIZE)

    assert torch.isfinite(out[2]).all().item(), "w_kernel contains NaN/Inf"
    assert torch.isfinite(out[3]).all().item(), "u_kernel contains NaN/Inf"
    _assert_compute_wy_close(out, ref)


def test_cgdr_310_colleague_shape_npu_wy_vs_torch_wy(monkeypatch):
    """T=111 (pad to 128) + l2norm=True: NPU compute_wy vs torch WY fallback."""
    enable_custom_op()
    q, k, v, g, beta, initial_state = _colleague_precision_inputs()

    # After pad_bthd, T becomes 128 and NPU compute_wy must be eligible.
    q_pad, k_pad, v_pad, g_pad, beta_pad, _, _ = chunk_mod._pad_bthd_to_chunk(q, k, v, g, beta, CHUNK_SIZE)
    assert q_pad.shape[1] == 128
    assert chunk_mod._can_use_npu_compute_wy(q_pad, k_pad, v_pad, g_pad, beta_pad, CHUNK_SIZE)

    out_npu, state_npu = _run_cgdr_310(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    assert state_npu is None
    assert out_npu.shape == v.shape

    monkeypatch.setattr(chunk_mod, "_can_use_npu_compute_wy", lambda *args, **kwargs: False)
    out_ref, state_ref = _run_cgdr_310(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    assert state_ref is None
    cos_o = _cosine(out_npu.cpu().float(), out_ref.cpu().float())
    assert cos_o > 0.99, f"out cosine={cos_o}"


def test_cgdr_310_colleague_shape_vs_pytorch_reference():
    """Absolute accuracy vs torch reference for the production-like shape (incl. l2norm).

    dim=128 exceeds 310P UB, so this path uses the torch WY fallback.
    """
    enable_custom_op()
    q, k, v, g, beta, initial_state = _colleague_precision_inputs(dim=128)
    q_pad, k_pad, v_pad, g_pad, beta_pad, _, _ = chunk_mod._pad_bthd_to_chunk(q, k, v, g, beta, CHUNK_SIZE)
    assert not chunk_mod._can_use_npu_compute_wy(q_pad, k_pad, v_pad, g_pad, beta_pad, CHUNK_SIZE)

    out_npu, _ = _run_cgdr_310(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    out_ref, _ = chunk_mod.chunk_gated_delta_rule_pytorch(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=None,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=None,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

    assert out_npu.shape == out_ref.shape == v.shape
    cos_o = _cosine(out_npu.cpu().float(), out_ref.cpu().float())
    assert cos_o > 0.99, f"out cosine vs pytorch ref={cos_o}"


def test_cgdr_310_colleague_shape_with_cu_seqlens(monkeypatch):
    """Same shape via varlen path: cu_seqlens int32 [0, 111]."""
    enable_custom_op()
    q, k, v, g, beta, initial_state = _colleague_precision_inputs()
    cu_seqlens = torch.tensor([0, 111], dtype=torch.int32, device=q.device)

    out_npu, _ = _run_cgdr_310(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    monkeypatch.setattr(chunk_mod, "_can_use_npu_compute_wy", lambda *args, **kwargs: False)
    out_ref, _ = _run_cgdr_310(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )

    cos_o = _cosine(out_npu.cpu().float(), out_ref.cpu().float())
    assert cos_o > 0.99, f"out cosine (cu_seqlens)={cos_o}"
