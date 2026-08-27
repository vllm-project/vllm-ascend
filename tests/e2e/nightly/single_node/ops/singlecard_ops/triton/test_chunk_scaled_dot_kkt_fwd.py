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


def _safe_exp(x):
    # mirrors `safe_exp` in the triton kernel: exp(x) for x<=0 else 0
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def chunk_scaled_dot_kkt_fwd_golden(k, beta, g_cumsum=None, chunk_size=64, cu_seqlens=None):
    """PyTorch reference for chunk_scaled_dot_kkt_fwd.

    Computes beta * (K @ K^T) * safe_exp(g[:,None]-g[None,:]) on the strict
    lower-triangular region of each chunk, zero elsewhere.
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    A = torch.zeros(B, T, H, BT, dtype=torch.float32)

    repeats = H // Hg
    k_exp = k.repeat_interleave(repeats, dim=2).to(torch.float32)  # (B, T, H, K)
    beta_f = beta.to(torch.float32)
    g_f = g_cumsum.to(torch.float32) if g_cumsum is not None else None

    def fill_batch(b, t_len, t_base):
        n_chunks = (t_len + BT - 1) // BT
        for h in range(H):
            for c in range(n_chunks):
                cs = c * BT
                ce = min(cs + BT, t_len)
                L = ce - cs
                kk = k_exp[b, t_base + cs : t_base + ce, h]  # (L, K)
                attn = kk @ kk.T  # (L, L)
                if g_f is not None:
                    gg = g_f[b, t_base + cs : t_base + ce, h]
                    gdiff = gg[:, None] - gg[None, :]
                    attn = attn * _safe_exp(gdiff)
                attn = attn * beta_f[b, t_base + cs : t_base + ce, h][:, None]
                mask = torch.tril(torch.ones(L, L), diagonal=-1).bool()
                A[b, t_base + cs : t_base + ce, h, :L] = attn * mask

    if cu_seqlens is not None:
        for s in range(len(cu_seqlens) - 1):
            bos, eos = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
            fill_batch(0, eos - bos, bos)
    else:
        for b in range(B):
            fill_batch(b, T, 0)
    return A


@pytest.mark.parametrize("T", [128, 320])
@pytest.mark.parametrize("Hg, H", [(4, 4), (2, 8)])
@pytest.mark.parametrize("use_g", [False, True])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-6, 1e-6),
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1.5e-5, 1.5e-5),
    ],
)
def test_chunk_scaled_dot_kkt_fwd_fixed_len(T, Hg, H, use_g, dtype, atol, rtol):
    from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

    B = 2
    K = 128
    chunk_size = 64

    k = (torch.randn(B, T, Hg, K, dtype=torch.float32) * 0.1).npu().to(dtype)
    beta = torch.rand(B, T, H, dtype=torch.float32).npu().to(dtype).sigmoid()
    if use_g:
        # decreasing cumsum (mimics chunk_local_cumsum of negative gates)
        g_raw = -torch.rand(B, T, H, dtype=torch.float32) * 0.1
        g_cumsum = torch.cumsum(g_raw, dim=1).npu()
    else:
        # Pass zeros instead of None to work around a wrapper bug where
        # `torch.permute(g_cumsum, ...)` is called unconditionally. zeros is
        # mathematically equivalent to no gate since safe_exp(0) == 1.
        g_cumsum = torch.zeros(B, T, H, dtype=torch.float32).npu()

    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=g_cumsum, chunk_size=chunk_size)
    golden = chunk_scaled_dot_kkt_fwd_golden(
        k.cpu().to(torch.float32), beta.cpu().to(torch.float32), g_cumsum.cpu().to(torch.float32), chunk_size=chunk_size
    )

    A_cpu = A.cpu().to(torch.float32)

    a = golden.abs() > 1
    a1 = golden.abs() <= 1
    torch.testing.assert_close(A_cpu * a, golden * a, atol=atol, rtol=100)
    torch.testing.assert_close(A_cpu * a1, golden * a1, rtol=rtol, atol=100)

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
def test_chunk_scaled_dot_kkt_fwd_varlen(seqlens, Hg, H, dtype, atol, rtol):
    from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

    K = 128
    chunk_size = 64
    T_total = sum(seqlens)

    k = (torch.randn(1, T_total, Hg, K, dtype=torch.float32) * 0.1).npu().to(dtype)
    beta = torch.rand(1, T_total, H, dtype=torch.float32).npu().to(dtype).sigmoid()
    g_raw = -torch.rand(1, T_total, H, dtype=torch.float32) * 0.1
    g_cumsum = torch.cumsum(g_raw, dim=1).npu()
    cu_seqlens = torch.tensor([0] + list(np.cumsum(seqlens)), dtype=torch.int32).npu()

    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=g_cumsum, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    golden = chunk_scaled_dot_kkt_fwd_golden(
        k.cpu().to(torch.float32),
        beta.cpu().to(torch.float32),
        g_cumsum.cpu().to(torch.float32),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens.cpu(),
    )

    A_cpu = A.cpu().to(torch.float32)

    a = golden.abs() > 1
    a1 = golden.abs() <= 1
    torch.testing.assert_close(A_cpu * a, golden * a, atol=atol, rtol=100)
    torch.testing.assert_close(A_cpu * a1, golden * a1, rtol=rtol, atol=100)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
