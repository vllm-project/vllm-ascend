# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's fused_recurrent_kda Triton operator on NPU.

Tests the recurrent-mode (decode) kernel against a naive PyTorch recurrent
reference implementation. Both ref and kernel use the same token-by-token
recurrence algorithm, so errors are purely from FP accumulation differences
on NPU triton-ascend.
"""

import pytest
import torch
import torch.nn.functional as F

import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.fla.kda import fused_recurrent_kda
from vllm_ascend.ops.triton.fla.l2norm_kda import l2norm_fwd

DEVICE = "npu"

# Both ref and kernel use the same recurrent algorithm; errors come from
# FP accumulation differences on NPU triton-ascend.
NPU_RMSE_RATIO_O = 0.005
NPU_RMSE_RATIO_HT = 0.005


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive recurrent KDA reference (pure PyTorch, runs on any device).

    Ported from flash-linear-attention/fla/ops/kda/naive.py.
    """
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q * scale

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "bhk,bhv->bhkv",
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),
        )
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def assert_close(
    name: str,
    ref: torch.Tensor,
    tri: torch.Tensor,
    ratio: float,
    err_atol: float = 1e-6,
):
    """RMSE-based relative error comparison (same logic as FLA's assert_close)."""
    abs_err = (ref.detach() - tri.detach()).flatten().abs().max().item()
    rmse_diff = (
        (ref.detach() - tri.detach()).flatten().square().mean().sqrt().item()
    )
    rmse_base = ref.detach().flatten().square().mean().sqrt().item()
    rel_err = rmse_diff / (rmse_base + 1e-8)
    print(
        f"{name:>8} | max abs err: {abs_err:.6f}"
        f" | rmse ratio: {rel_err:.6f} | threshold: {ratio}"
    )
    if abs_err <= err_atol:
        return
    assert not torch.isnan(ref).any(), f"{name}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{name}: NaN detected in tri"
    assert rel_err < ratio, (
        f"{name}: max abs err {abs_err:.6f},"
        f" rmse ratio {rel_err:.6f} >= {ratio}"
    )


# ---------------------------------------------------------------------------
# Test 1: Non-inplace varlen (clean output / state comparison)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("H", "D", "cu_seqlens", "dtype"),
    [
        pytest.param(
            *test,
            id="H{}-D{}-cu{}-{}".format(*test),
        )
        for test in [
            # Decode: single token per sequence
            (32, 128, [0, 1], torch.float16),
            (32, 128, [0, 1, 2, 3, 4], torch.float16),
            (32, 128, [0, 1, 2, 3, 4, 5, 6, 7, 8], torch.float16),
            # Short sequences (multi-token recurrent)
            (32, 128, [0, 16], torch.float16),
            (32, 128, [0, 8, 24], torch.float16),
            (32, 128, [0, 4, 8, 16], torch.float16),
            (32, 128, [0, 64], torch.float16),
            # Different head count
            (64, 128, [0, 1, 2, 3, 4], torch.float16),
            # BFloat16
            (32, 128, [0, 1, 2, 3, 4], torch.bfloat16),
            (32, 128, [0, 8, 24], torch.bfloat16),
        ]
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_fused_recurrent_kda(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    """Non-inplace varlen mode — easiest to verify output and per-token state."""
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    B = 1

    torch.manual_seed(42)
    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(DEVICE)

    q = torch.randn(B, T, H, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, T, H, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, T, H, D, dtype=dtype, device=DEVICE)
    g = F.logsigmoid(
        torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE)
    ).to(dtype)
    beta = torch.rand(B, T, H, dtype=dtype, device=DEVICE).sigmoid()
    # Kernel layout: [T, H, V, K] = [T, H, D, D].
    # For varlen without ssm_state_indices, seq i reads from h0[cu_seqlens[i]].
    h0 = torch.randn(T, H, D, D, dtype=torch.float32, device=DEVICE)

    # --- naive reference per sequence ---
    ref_outputs = []
    ref_states = []
    for i in range(N):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        q_i = l2norm_fwd(q[:, s:e].contiguous())
        k_i = l2norm_fwd(k[:, s:e].contiguous())
        # Kernel state [H, V, K] -> naive [H, K, V]
        init_state_i = h0[s].transpose(-1, -2).unsqueeze(0)
        o_i, ht_i = naive_recurrent_kda(
            q_i,
            k_i,
            v[:, s:e],
            g[:, s:e],
            beta[:, s:e],
            initial_state=init_state_i,
            output_final_state=True,
        )
        ref_outputs.append(o_i)
        ref_states.append(ht_i)
    ref_o = torch.cat(ref_outputs, dim=1)

    # --- Triton kernel ---
    tri_o, tri_ht = fused_recurrent_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens_t,
    )

    assert not torch.isnan(tri_o).any(), "Triton output o contains NaN"
    assert not torch.isnan(tri_ht).any(), "Triton output ht contains NaN"

    assert_close("o", ref_o, tri_o, NPU_RMSE_RATIO_O)
    # Compare final state per sequence: tri_ht[eos-1] in kernel layout [H,V,K]
    for i in range(N):
        e = cu_seqlens[i + 1]
        tri_state = tri_ht[e - 1].transpose(-1, -2).unsqueeze(0)
        assert_close(f"ht_{i}", ref_states[i], tri_state, NPU_RMSE_RATIO_HT)


# ---------------------------------------------------------------------------
# Test 2: Inplace decode with ssm_state_indices (vllm actual pattern)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("H", "D", "N", "dtype"),
    [
        pytest.param(
            *test,
            id="H{}-D{}-N{}-{}".format(*test),
        )
        for test in [
            (32, 128, 1, torch.float16),
            (32, 128, 4, torch.float16),
            (32, 128, 16, torch.float16),
            (64, 128, 4, torch.float16),
            (32, 128, 4, torch.bfloat16),
        ]
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_fused_recurrent_kda_decode_inplace(
    H: int,
    D: int,
    N: int,
    dtype: torch.dtype,
):
    """Decode with inplace state update + ssm_state_indices — vllm usage."""
    B = 1
    T = N  # one token per sequence
    cu_seqlens = list(range(N + 1))  # [0, 1, 2, ..., N]

    torch.manual_seed(42)
    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(DEVICE)

    q = torch.randn(B, T, H, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, T, H, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, T, H, D, dtype=dtype, device=DEVICE)
    g = F.logsigmoid(
        torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE)
    ).to(dtype)
    beta = torch.rand(B, T, H, dtype=dtype, device=DEVICE).sigmoid()

    # State buffer: slot 0 is NULL (invalid), slots 1..N are valid
    max_slots = N + 1
    state_buf = torch.randn(
        max_slots, H, D, D, dtype=torch.float32, device=DEVICE
    )
    state_buf[0] = 0  # NULL slot

    # ssm_state_indices: seq i -> slot (i + 1), all valid (> 0)
    ssm_state_indices = torch.arange(
        1, N + 1, dtype=torch.long, device=DEVICE
    )

    # --- naive reference per sequence ---
    ref_outputs = []
    ref_states = []
    for i in range(N):
        slot = i + 1
        q_i = l2norm_fwd(q[:, i : i + 1].contiguous())
        k_i = l2norm_fwd(k[:, i : i + 1].contiguous())
        init_state_i = state_buf[slot].transpose(-1, -2).unsqueeze(0)
        o_i, ht_i = naive_recurrent_kda(
            q_i,
            k_i,
            v[:, i : i + 1],
            g[:, i : i + 1],
            beta[:, i : i + 1],
            initial_state=init_state_i,
            output_final_state=True,
        )
        ref_outputs.append(o_i)
        ref_states.append(ht_i)
    ref_o = torch.cat(ref_outputs, dim=1)

    # --- Triton kernel — inplace updates state_buf ---
    state_buf_tri = state_buf.clone()
    tri_o, _ = fused_recurrent_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=state_buf_tri,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens_t,
        ssm_state_indices=ssm_state_indices,
    )

    assert not torch.isnan(tri_o).any(), "Triton output o contains NaN"
    assert_close("o", ref_o, tri_o, NPU_RMSE_RATIO_O)

    # Verify inplace state update at each slot
    for i in range(N):
        slot = i + 1
        tri_state = state_buf_tri[slot].transpose(-1, -2).unsqueeze(0)
        assert_close(f"ht_{i}", ref_states[i], tri_state, NPU_RMSE_RATIO_HT)

    # Verify NULL slot was not modified
    assert torch.all(state_buf_tri[0] == 0), (
        "NULL slot (0) should not be modified"
    )


# ---------------------------------------------------------------------------
# Test 3: Float32 — isolate algorithmic error from dtype precision
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("H", "D", "cu_seqlens"),
    [
        pytest.param(
            *test,
            id="H{}-D{}-cu{}".format(*test),
        )
        for test in [
            (32, 128, [0, 1]),
            (32, 128, [0, 1, 2, 3, 4]),
            (32, 128, [0, 16]),
            (32, 128, [0, 8, 24]),
        ]
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_fused_recurrent_kda_fp32(
    H: int,
    D: int,
    cu_seqlens: list[int],
):
    """Float32 test to isolate algorithmic error from dtype precision."""
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    B = 1

    torch.manual_seed(42)
    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(DEVICE)

    q = torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE)
    k = torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE)
    v = torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE)
    g = F.logsigmoid(
        torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE)
    )
    beta = torch.rand(B, T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    h0 = torch.randn(T, H, D, D, dtype=torch.float32, device=DEVICE)

    ref_outputs = []
    ref_states = []
    for i in range(N):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        q_i = l2norm_fwd(q[:, s:e].contiguous())
        k_i = l2norm_fwd(k[:, s:e].contiguous())
        init_state_i = h0[s].transpose(-1, -2).unsqueeze(0)
        o_i, ht_i = naive_recurrent_kda(
            q_i,
            k_i,
            v[:, s:e],
            g[:, s:e],
            beta[:, s:e],
            initial_state=init_state_i,
            output_final_state=True,
        )
        ref_outputs.append(o_i)
        ref_states.append(ht_i)
    ref_o = torch.cat(ref_outputs, dim=1)

    tri_o, tri_ht = fused_recurrent_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens_t,
    )

    assert not torch.isnan(tri_o).any(), "Triton output o contains NaN"
    assert not torch.isnan(tri_ht).any(), "Triton output ht contains NaN"

    assert_close("o", ref_o, tri_o, NPU_RMSE_RATIO_O)
    for i in range(N):
        e = cu_seqlens[i + 1]
        tri_state = tri_ht[e - 1].transpose(-1, -2).unsqueeze(0)
        assert_close(f"ht_{i}", ref_states[i], tri_state, NPU_RMSE_RATIO_HT)
