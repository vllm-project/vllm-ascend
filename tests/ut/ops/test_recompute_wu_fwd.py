# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401,E402


def _has_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="recompute_wu_fwd requires an NPU device")


def _make_inputs(*, dtype=torch.float16, chunk_size=64, seq_len=64):
    torch.manual_seed(1)
    batch, key_heads, value_heads = 1, 1, 2
    key_dim, value_dim = 128, 128

    k = torch.randn(batch, key_heads, seq_len, key_dim, dtype=dtype) * 0.02
    v = torch.randn(batch, value_heads, seq_len, value_dim, dtype=dtype) * 0.02
    beta = torch.sigmoid(torch.randn(batch, value_heads, seq_len, dtype=torch.float32))
    g = torch.randn(batch, value_heads, seq_len, dtype=torch.float32) * 0.01
    a = torch.randn(batch, value_heads, seq_len, chunk_size, dtype=dtype) * 0.02
    for b in range(batch):
        for h in range(value_heads):
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                a[b, h, start:end, : end - start] = torch.tril(a[b, h, start:end, : end - start])
    return k, v, beta, a, g


def _recompute_wu_golden(k, v, beta, a, g, chunk_size):
    batch, key_heads, seq_len, key_dim = k.shape
    value_heads = v.shape[1]
    value_dim = v.shape[3]
    key_group_size = value_heads // key_heads

    w = torch.zeros((batch, value_heads, seq_len, key_dim), dtype=torch.float32)
    u = torch.zeros((batch, value_heads, seq_len, value_dim), dtype=torch.float32)

    for b in range(batch):
        for hv in range(value_heads):
            hk = hv // key_group_size
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                block_len = end - start
                a_block = a[b, hv, start:end, :block_len].float()
                beta_block = beta[b, hv, start:end].float().unsqueeze(-1)
                g_block = g[b, hv, start:end].float().unsqueeze(-1)
                vb = v[b, hv, start:end, :].float() * beta_block
                kbg_exp = k[b, hk, start:end, :].float() * beta_block * torch.exp(g_block)
                u[b, hv, start:end, :] = a_block @ vb
                w[b, hv, start:end, :] = a_block @ kbg_exp
    return w.to(k.dtype), u.to(v.dtype)


def _recompute_wu_varlen_golden(k, v, beta, a, g, cu_seqlens, chunk_size):
    batch, key_heads, seq_len, key_dim = k.shape
    value_heads = v.shape[1]
    value_dim = v.shape[3]
    key_group_size = value_heads // key_heads

    w = torch.zeros((batch, value_heads, seq_len, key_dim), dtype=torch.float32)
    u = torch.zeros((batch, value_heads, seq_len, value_dim), dtype=torch.float32)

    for seq_idx in range(len(cu_seqlens) - 1):
        seq_start = cu_seqlens[seq_idx]
        seq_end = cu_seqlens[seq_idx + 1]
        for b in range(batch):
            for hv in range(value_heads):
                hk = hv // key_group_size
                for start in range(seq_start, seq_end, chunk_size):
                    end = min(start + chunk_size, seq_end)
                    block_len = end - start
                    a_block = a[b, hv, start:end, :block_len].float()
                    beta_block = beta[b, hv, start:end].float().unsqueeze(-1)
                    g_block = g[b, hv, start:end].float().unsqueeze(-1)
                    vb = v[b, hv, start:end, :].float() * beta_block
                    kbg_exp = k[b, hk, start:end, :].float() * beta_block * torch.exp(g_block)
                    u[b, hv, start:end, :] = a_block @ vb
                    w[b, hv, start:end, :] = a_block @ kbg_exp
    return w.to(k.dtype), u.to(v.dtype)


def test_npu_recompute_wu_fwd_matches_cpu_golden():
    chunk_size = 64
    k, v, beta, a, g = _make_inputs(chunk_size=chunk_size)
    expected_w, expected_u = _recompute_wu_golden(k, v, beta, a, g, chunk_size)

    actual_w, actual_u = torch.ops._C_ascend.npu_recompute_wu_fwd(
        k.npu(), v.npu(), beta.npu(), a.npu(), g.npu(), chunk_size=chunk_size
    )

    np.testing.assert_allclose(
        actual_w.cpu().float().numpy(),
        expected_w.float().numpy(),
        rtol=3e-2,
        atol=3e-2,
    )
    np.testing.assert_allclose(
        actual_u.cpu().float().numpy(),
        expected_u.float().numpy(),
        rtol=3e-2,
        atol=3e-2,
    )


def test_npu_recompute_wu_fwd_allows_trailing_empty_sequence():
    chunk_size = 64
    cu_seqlens = [0, 1, 65, 65]
    chunk_indices = [0, 0, 1, 0]
    k, v, beta, a, g = _make_inputs(dtype=torch.bfloat16, chunk_size=chunk_size, seq_len=cu_seqlens[-1])
    beta = beta.to(torch.bfloat16)
    g = g.to(torch.bfloat16)
    expected_w, expected_u = _recompute_wu_varlen_golden(k, v, beta, a, g, cu_seqlens, chunk_size)

    actual_w, actual_u = torch.ops._C_ascend.npu_recompute_wu_fwd(
        k.npu(),
        v.npu(),
        beta.npu(),
        a.npu(),
        g.npu(),
        cu_seqlens,
        chunk_indices,
        chunk_size=chunk_size,
    )

    np.testing.assert_allclose(
        actual_w.cpu().float().numpy(),
        expected_w.float().numpy(),
        rtol=4e-2,
        atol=4e-2,
    )
    np.testing.assert_allclose(
        actual_u.cpu().float().numpy(),
        expected_u.float().numpy(),
        rtol=4e-2,
        atol=4e-2,
    )


def test_npu_recompute_wu_fwd_meta_shape():
    k = torch.empty((2, 1, 64, 128), dtype=torch.float16, device="meta")
    v = torch.empty((2, 2, 64, 256), dtype=torch.float16, device="meta")
    beta = torch.empty((2, 2, 64), dtype=torch.float32, device="meta")
    a = torch.empty((2, 2, 64, 64), dtype=torch.float16, device="meta")
    g = torch.empty((2, 2, 64), dtype=torch.float32, device="meta")

    w, u = torch.ops._C_ascend.npu_recompute_wu_fwd(k, v, beta, a, g, chunk_size=64)

    assert w.shape == (2, 2, 64, 128)
    assert w.dtype == k.dtype
    assert w.device.type == "meta"
    assert u.shape == v.shape
    assert u.dtype == v.dtype
    assert u.device.type == "meta"
