import gc

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd, l2norm_qk_fwd
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 60, torch.float),
            (2, 500, 4, 64, torch.float),
            (2, 1000, 2, 100, torch.float),
            (3, 1024, 4, 128, torch.float),
        ]
    ],
)
def test_l2norm(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    x = torch.randn(B, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    x = x * 0.5 + 0.3

    ref = F.normalize(x, dim=-1, p=2)
    tri = l2norm_fwd(x)

    assert torch.allclose(tri, ref, rtol=rtol, atol=atol)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 60, torch.float),
            (2, 500, 4, 64, torch.float),
            (2, 1000, 2, 100, torch.float),
            (3, 1024, 4, 128, torch.float),
            (1, 4096, 8, 128, torch.bfloat16),
        ]
    ],
)
def test_l2norm_qk_fwd(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    q = torch.randn(B, T, H, D, dtype=dtype).to(device) * 0.5 + 0.3
    k = torch.randn(B, T, H, D, dtype=dtype).to(device) * 0.5 + 0.3

    q_ref = F.normalize(q, dim=-1, p=2)
    k_ref = F.normalize(k, dim=-1, p=2)
    q_tri, k_tri = l2norm_qk_fwd(q, k)

    assert torch.allclose(q_tri, q_ref, rtol=rtol, atol=atol)
    assert torch.allclose(k_tri, k_ref, rtol=rtol, atol=atol)
    # The fused kernel computes the same row-wise arithmetic as two separate
    # l2norm_fwd launches, so the outputs must match exactly.
    assert torch.equal(q_tri, l2norm_fwd(q))
    assert torch.equal(k_tri, l2norm_fwd(k))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_l2norm_qk_fwd_mismatched_shapes_fall_back():
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    q = torch.randn(1, 128, 4, 128, dtype=torch.bfloat16).to(device)
    k = torch.randn(1, 128, 8, 128, dtype=torch.bfloat16).to(device)

    q_tri, k_tri = l2norm_qk_fwd(q, k)

    assert q_tri.shape == q.shape and k_tri.shape == k.shape
    assert torch.equal(q_tri, l2norm_fwd(q))
    assert torch.equal(k_tri, l2norm_fwd(k))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_l2norm_qk_fwd_sliced_inputs():
    # Mirrors the GDN call sites, which l2-normalize time-sliced views of
    # q/k (e.g. ``query[:, num_decode_tokens:]``).
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    q = torch.randn(1, 600, 4, 128, dtype=torch.bfloat16).to(device)
    k = torch.randn(1, 600, 4, 128, dtype=torch.bfloat16).to(device)
    q_sliced, k_sliced = q[:, 100:], k[:, 100:]

    q_tri, k_tri = l2norm_qk_fwd(q_sliced, k_sliced)

    assert torch.equal(q_tri, l2norm_fwd(q_sliced))
    assert torch.equal(k_tri, l2norm_fwd(k_sliced))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
