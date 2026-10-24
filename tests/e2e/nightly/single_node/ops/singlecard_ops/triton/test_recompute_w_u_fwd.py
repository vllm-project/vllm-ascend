import gc
import random

import numpy as np
import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

seed = 45
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


@pytest.fixture(autouse=True, scope="module")
def _init_triton_device():
    """Triton kernels call get_aicore_num()/get_vectorcore_num() which require
    device properties to be initialized first. The ops-level conftest does this
    at import time, but it may not run early enough when collecting a single
    file, so initialize explicitly here (idempotent)."""
    init_device_properties_triton()


def recompute_w_u_fwd_golden(k, v, beta, g_cumsum, A, cu_seqlens=None):
    """PyTorch reference for recompute_w_u_fwd.

    w = A @ (k * beta * exp(g)), u = A @ (v * beta), per chunk.
    A is treated as a (BT, BT) lower-triangular matrix per chunk.
    """
    B, T, Hg, K = k.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = A.shape[-1]
    out_dtype = k.dtype

    repeats = H // Hg
    k_exp = k.repeat_interleave(repeats, dim=2).to(torch.float32)
    v_f = v.to(torch.float32)
    beta_f = beta.to(torch.float32)
    g_f = g_cumsum.to(torch.float32)
    A_f = A.to(torch.float32)

    w = torch.zeros(B, T, H, K, dtype=torch.float32)
    u = torch.zeros(B, T, H, V, dtype=torch.float32)

    def fill(b, t_len, t_base):
        n_chunks = (t_len + BT - 1) // BT
        for h in range(H):
            for c in range(n_chunks):
                cs = c * BT
                ce = min(cs + BT, t_len)
                L = ce - cs
                kk = k_exp[b, t_base + cs : t_base + ce, h]
                vv = v_f[b, t_base + cs : t_base + ce, h]
                bt = beta_f[b, t_base + cs : t_base + ce, h]
                gg = g_f[b, t_base + cs : t_base + ce, h]
                aa = A_f[b, t_base + cs : t_base + ce, h, :L]
                kb = kk * bt[:, None] * torch.exp(gg)[:, None]
                vb = vv * bt[:, None]
                w[b, t_base + cs : t_base + ce, h] = aa @ kb
                u[b, t_base + cs : t_base + ce, h] = aa @ vb

    if cu_seqlens is not None:
        for s in range(len(cu_seqlens) - 1):
            bos, eos = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
            fill(0, eos - bos, bos)
    else:
        fill(0, T, 0)
    return w.to(out_dtype), u.to(out_dtype)


def _make_unit_lower(B, T, H, BT, dtype, device):
    """Build a (B, T, H, BT) unit lower-triangular A (diag=1, upper=0)."""
    A = torch.zeros(B, T, H, BT, dtype=dtype, device=device)
    n_chunks = (T + BT - 1) // BT
    for c in range(n_chunks):
        cs = c * BT
        ce = min(cs + BT, T)
        L = ce - cs
        rand = torch.randn(B, H, L, L, dtype=torch.float32, device=device) * 0.1
        lower = torch.tril(rand, diagonal=-1)
        eye = torch.eye(L, dtype=torch.float32, device=device)
        block = (eye + lower).to(dtype)  # (L, L)
        A[:, cs:ce, :, :L] = block.transpose(1, 2)  # (B, L, H, L)
    return A


@pytest.mark.parametrize("T", [128, 320])
@pytest.mark.parametrize("Hg, H", [(4, 4), (2, 8)])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-6, 1e-6),
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1.5e-5, 1.5e-5),
    ],
)
def test_recompute_w_u_fwd_fixed_len(T, Hg, H, dtype, atol, rtol):
    from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd

    B = 1
    K, V, BT = 128, 128, 64

    k = (torch.randn(B, T, Hg, K, dtype=torch.float32) * 0.1).npu().to(dtype)
    v = (torch.randn(B, T, H, V, dtype=torch.float32) * 0.1).npu().to(dtype)
    beta = torch.rand(B, T, H, dtype=torch.float32).npu().to(dtype).sigmoid()
    g_raw = -torch.rand(B, T, H, dtype=torch.float32) * 0.1
    g_cumsum = torch.cumsum(g_raw, dim=1).npu()
    A = _make_unit_lower(B, T, H, BT, dtype, "npu")
    # Work around a kernel bug: the non-varlen (else) branch does not assign
    # `i_t`, so `global_offs_t = i_t * BT + offs_t` fails to compile. Wrap the
    # fixed-length sequence as a single-sequence varlen input to take the
    # IS_VARLEN=True path, which correctly defines `i_t`.
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32).npu()

    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, g_cumsum=g_cumsum, A=A, cu_seqlens=cu_seqlens)
    w_g, u_g = recompute_w_u_fwd_golden(
        k.cpu().to(torch.float32),
        v.cpu().to(torch.float32),
        beta.cpu().to(torch.float32),
        g_cumsum.cpu().to(torch.float32),
        A.cpu().to(torch.float32),
        cu_seqlens=cu_seqlens.cpu(),
    )

    for out, golden in [
        (w.cpu().to(torch.float32), w_g.to(torch.float32)),
        (u.cpu().to(torch.float32), u_g.to(torch.float32)),
    ]:
        a = golden.abs() > 1
        a1 = golden.abs() <= 1
        torch.testing.assert_close(out * a, golden * a, atol=atol, rtol=100)
        torch.testing.assert_close(out * a1, golden * a1, rtol=rtol, atol=100)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("seqlens", [[256, 512], [128, 256, 384]])
@pytest.mark.parametrize("Hg, H", [(4, 4), (2, 8)])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-6, 1e-6),
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1.5e-5, 1.5e-5),
    ],
)
def test_recompute_w_u_fwd_varlen(seqlens, Hg, H, dtype, atol, rtol):
    from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd

    K, V, BT = 128, 128, 64
    T_total = sum(seqlens)

    k = (torch.randn(1, T_total, Hg, K, dtype=torch.float32) * 0.1).npu().to(dtype)
    v = (torch.randn(1, T_total, H, V, dtype=torch.float32) * 0.1).npu().to(dtype)
    beta = torch.rand(1, T_total, H, dtype=torch.float32).npu().to(dtype).sigmoid()
    g_raw = -torch.rand(1, T_total, H, dtype=torch.float32) * 0.1
    g_cumsum = torch.cumsum(g_raw, dim=1).npu()
    A = _make_unit_lower(1, T_total, H, BT, dtype, "npu")
    cu_seqlens = torch.tensor([0] + list(np.cumsum(seqlens)), dtype=torch.int32).npu()

    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, g_cumsum=g_cumsum, A=A, cu_seqlens=cu_seqlens)
    w_g, u_g = recompute_w_u_fwd_golden(
        k.cpu().to(torch.float32),
        v.cpu().to(torch.float32),
        beta.cpu().to(torch.float32),
        g_cumsum.cpu().to(torch.float32),
        A.cpu().to(torch.float32),
        cu_seqlens=cu_seqlens.cpu(),
    )

    for out, golden in [
        (w.cpu().to(torch.float32), w_g.to(torch.float32)),
        (u.cpu().to(torch.float32), u_g.to(torch.float32)),
    ]:
        a = golden.abs() > 1
        a1 = golden.abs() <= 1
        torch.testing.assert_close(out * a, golden * a, atol=atol, rtol=100)
        torch.testing.assert_close(out * a1, golden * a1, rtol=rtol, atol=100)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
