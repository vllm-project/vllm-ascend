import pytest
import torch

from vllm_ascend.ops.triton.rms_norm import triton_q_rms
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

DEVICE = "npu"
TOLERANCES = {
    torch.float16: (2e-3, 2e-2),
    torch.bfloat16: (2e-2, 5e-2),
    torch.float32: (1e-4, 1e-4),
}


@pytest.fixture(scope="module", autouse=True)
def init_triton_device_properties():
    init_device_properties_triton()


def rms_norm_ref(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.cpu().float()
    out = x_fp32 * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return out.to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("shape", "eps"),
    [
        pytest.param((1, 1, 128), 1e-6, id="single-row"),
        pytest.param((2, 8, 512), 1e-5, id="multi-head"),
        pytest.param((1, 17, 2048), 1e-5, id="max-hidden-size"),
    ],
)
@torch.inference_mode()
def test_triton_q_rms_correctness(dtype, shape, eps):
    torch.manual_seed(42)
    q = (torch.randn(shape, dtype=torch.float32) * 0.5).to(dtype=dtype, device=DEVICE)

    actual = triton_q_rms(q, eps)
    expected = rms_norm_ref(q, eps)
    rtol, atol = TOLERANCES[dtype]

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(
        actual.float().cpu(),
        expected.float().cpu(),
        rtol=rtol,
        atol=atol,
    )


def test_triton_q_rms_rejects_unsupported_dim():
    q = torch.empty((1, 1, 2049), dtype=torch.float16, device=DEVICE)

    with pytest.raises(NotImplementedError, match="dim > 2048 not supported"):
        triton_q_rms(q, 1e-5)
