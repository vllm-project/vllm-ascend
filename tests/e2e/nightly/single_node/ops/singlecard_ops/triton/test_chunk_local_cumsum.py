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


def chunk_local_cumsum_golden(g, chunk_size, reverse=False, scale=None, cu_seqlens=None, head_first=False):
    """PyTorch reference for chunk_local_cumsum.

    Splits the sequence dim into chunks of `chunk_size` and performs an
    independent cumsum inside each chunk.
    """
    if head_first:
        B, H, T = g.shape
        g_t = g.transpose(1, 2).contiguous()
    else:
        B, T, H = g.shape
        g_t = g

    out = torch.zeros(B, T, H, dtype=torch.float32)

    def _cumsum_block(block):
        # block: (L, H)
        if reverse:
            return torch.cumsum(block.flip(0), dim=0).flip(0)
        return torch.cumsum(block, dim=0)

    if cu_seqlens is not None:
        for s in range(len(cu_seqlens) - 1):
            bos, eos = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
            t_s = eos - bos
            n_chunks = (t_s + chunk_size - 1) // chunk_size
            for c in range(n_chunks):
                cs = c * chunk_size
                ce = min(cs + chunk_size, t_s)
                out[0, bos + cs : bos + ce] = _cumsum_block(g_t[0, bos + cs : bos + ce].to(torch.float32))
    else:
        for b in range(B):
            n_chunks = (T + chunk_size - 1) // chunk_size
            for c in range(n_chunks):
                cs = c * chunk_size
                ce = min(cs + chunk_size, T)
                out[b, cs:ce] = _cumsum_block(g_t[b, cs:ce].to(torch.float32))

    if scale is not None:
        out = out * scale

    if head_first:
        out = out.transpose(1, 2).contiguous()
    return out


@pytest.mark.parametrize("T", [64, 128, 320])
@pytest.mark.parametrize("H", [8, 64])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
    ],
)
def test_chunk_local_cumsum_fixed_len(T, H, reverse, dtype, atol, rtol):
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

    B = 2
    chunk_size = 64
    scale = 0.5

    g = (torch.rand(B, T, H, dtype=torch.float32) * 0.1 - 0.05).npu().to(dtype)

    out = chunk_local_cumsum(g, chunk_size, reverse=reverse, scale=scale, head_first=False)
    golden = chunk_local_cumsum_golden(
        g.cpu().to(torch.float32), chunk_size, reverse=reverse, scale=scale, head_first=False
    )

    out_cpu = out.cpu().to(torch.float32)

    a = golden > 1
    a1 = golden <= 1
    torch.testing.assert_close(out_cpu * a, golden * a, atol=atol, rtol=100)
    torch.testing.assert_close(out_cpu * a1, golden * a1, rtol=rtol, atol=100)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("seqlens", [[256, 512], [128, 256, 384]])
@pytest.mark.parametrize("H", [8, 32])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
    ],
)
def test_chunk_local_cumsum_varlen(seqlens, H, dtype, atol, rtol):
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

    chunk_size = 64
    T_total = sum(seqlens)
    g = (torch.rand(1, T_total, H, dtype=torch.float32) * 0.1 - 0.05).npu().to(dtype)

    cu_seqlens = torch.tensor([0] + list(np.cumsum(seqlens)), dtype=torch.int32).npu()

    out = chunk_local_cumsum(g, chunk_size, reverse=False, scale=None, cu_seqlens=cu_seqlens, head_first=False)
    golden = chunk_local_cumsum_golden(
        g.cpu().to(torch.float32), chunk_size, reverse=False, scale=None, cu_seqlens=cu_seqlens.cpu(), head_first=False
    )

    out_cpu = out.cpu().to(torch.float32)

    a = golden.abs() > 1
    a1 = golden.abs() <= 1
    torch.testing.assert_close(out_cpu * a, golden * a, atol=atol, rtol=100)
    torch.testing.assert_close(out_cpu * a1, golden * a1, rtol=rtol, atol=100)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
