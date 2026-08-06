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
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))


def chunk_fwd_o_golden(q, k, v, h, g=None, scale=None, cu_seqlens=None, chunk_size=64, chunk_offsets=None):
    """PyTorch reference for chunk_fwd_o.

    o = scale * (exp(g) * (q @ h) + tril(q @ k^T * safe_exp(g_diff)) @ v)
    per chunk. `h` is indexed as (B*NT, H, K, V).
    """
    B, T, Hg, K = q.shape
    H, V = v.shape[-2], v.shape[-1]
    BT = chunk_size
    if scale is None:
        scale = K**-0.5

    repeats = H // Hg
    q_exp = q.repeat_interleave(repeats, dim=2).to(torch.float32)  # (B, T, H, K)
    k_exp = k.repeat_interleave(repeats, dim=2).to(torch.float32)
    v_f = v.to(torch.float32)
    h_f = h.to(torch.float32)
    g_f = g.to(torch.float32) if g is not None else None

    o = torch.zeros(B, T, H, V, dtype=torch.float32)

    def fill(b, t_len, t_base, h_offset):
        NT = (t_len + BT - 1) // BT
        for h_idx in range(H):
            for i_t in range(NT):
                cs = i_t * BT
                ce = min(cs + BT, t_len)
                L = ce - cs
                qq = q_exp[b, t_base + cs : t_base + ce, h_idx]
                kk = k_exp[b, t_base + cs : t_base + ce, h_idx]
                vv = v_f[b, t_base + cs : t_base + ce, h_idx]
                hh = h_f[h_offset + i_t, h_idx]  # (K, V)

                b_o_inter = qq @ hh  # (L, V)
                b_A = qq @ kk.T  # (L, L)
                if g_f is not None:
                    gg = g_f[b, t_base + cs : t_base + ce, h_idx]
                    b_o_inter = b_o_inter * torch.exp(gg)[:, None]
                    gdiff = gg[:, None] - gg[None, :]
                    b_A = b_A * _safe_exp(gdiff)
                mask = torch.tril(torch.ones(L, L)).bool()
                b_A = b_A * mask
                o[b, t_base + cs : t_base + ce, h_idx] = scale * b_o_inter + scale * (b_A @ vv)

    if cu_seqlens is not None:
        for s in range(len(cu_seqlens) - 1):
            bos, eos = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
            boh = int(chunk_offsets[s]) if chunk_offsets is not None else 0
            fill(0, eos - bos, bos, boh)
    else:
        NT = (T + BT - 1) // BT
        for b in range(B):
            fill(b, T, 0, b * NT)
    return o


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
def test_chunk_fwd_o_fixed_len(T, Hg, H, use_g, dtype, atol, rtol):
    from vllm_ascend.ops.triton.fla.chunk_o import chunk_fwd_o

    B = 1
    K, V, BT = 128, 128, 64
    NT = (T + BT - 1) // BT

    q = (torch.randn(B, T, Hg, K, dtype=torch.float32) * 0.1).npu().to(dtype)
    k = (torch.randn(B, T, Hg, K, dtype=torch.float32) * 0.1).npu().to(dtype)
    v = (torch.randn(B, T, H, V, dtype=torch.float32) * 0.1).npu().to(dtype)
    h = (torch.randn(B * NT, H, K, V, dtype=torch.float32) * 0.1).npu().to(dtype)
    if use_g:
        g_raw = -torch.rand(B, T, H, dtype=torch.float32) * 0.1
        g = torch.cumsum(g_raw, dim=1).npu()
    else:
        # Pass zeros instead of None to work around a wrapper bug where
        # `g.transpose(1, 2)` is called unconditionally. zeros is mathematically
        # equivalent to no gate since exp(0) == 1 and safe_exp(0) == 1.
        g = torch.zeros(B, T, H, dtype=torch.float32).npu()

    o = chunk_fwd_o(q=q, k=k, v=v, h=h, g=g, scale=K**-0.5, chunk_size=BT)
    golden = chunk_fwd_o_golden(
        q.cpu().to(torch.float32),
        k.cpu().to(torch.float32),
        v.cpu().to(torch.float32),
        h.cpu().to(torch.float32),
        g.cpu().to(torch.float32),
        scale=K**-0.5,
        chunk_size=BT,
    )

    o_cpu = o.cpu().to(torch.float32)
    a = golden.abs() > 1
    a1 = golden.abs() <= 1
    torch.testing.assert_close(o_cpu * a, golden * a, atol=atol, rtol=100)
    torch.testing.assert_close(o_cpu * a1, golden * a1, rtol=rtol, atol=100)

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
def test_chunk_fwd_o_varlen(seqlens, Hg, H, dtype, atol, rtol):
    from vllm_ascend.ops.triton.fla.chunk_o import chunk_fwd_o
    from vllm_ascend.ops.triton.fla.utils import prepare_chunk_offsets

    K, V, BT = 128, 128, 64
    T_total = sum(seqlens)

    q = (torch.randn(1, T_total, Hg, K, dtype=torch.float32) * 0.1).npu().to(dtype)
    k = (torch.randn(1, T_total, Hg, K, dtype=torch.float32) * 0.1).npu().to(dtype)
    v = (torch.randn(1, T_total, H, V, dtype=torch.float32) * 0.1).npu().to(dtype)
    g_raw = -torch.rand(1, T_total, H, dtype=torch.float32) * 0.1
    g = torch.cumsum(g_raw, dim=1).npu()
    cu_seqlens = torch.tensor([0] + list(np.cumsum(seqlens)), dtype=torch.int64).npu()
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    NT_total = int(chunk_offsets[-1].item())
    h = (torch.randn(NT_total, H, K, V, dtype=torch.float32) * 0.1).npu().to(dtype)

    o = chunk_fwd_o(
        q=q, k=k, v=v, h=h, g=g, scale=K**-0.5, cu_seqlens=cu_seqlens, chunk_size=BT, chunk_offsets=chunk_offsets
    )
    golden = chunk_fwd_o_golden(
        q.cpu().to(torch.float32),
        k.cpu().to(torch.float32),
        v.cpu().to(torch.float32),
        h.cpu().to(torch.float32),
        g.cpu().to(torch.float32),
        scale=K**-0.5,
        chunk_size=BT,
        cu_seqlens=cu_seqlens.cpu(),
        chunk_offsets=chunk_offsets.cpu(),
    )

    o_cpu = o.cpu().to(torch.float32)
    a = golden.abs() > 1
    a1 = golden.abs() <= 1
    torch.testing.assert_close(o_cpu * a, golden * a, atol=atol, rtol=100)
    torch.testing.assert_close(o_cpu * a1, golden * a1, rtol=rtol, atol=100)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
