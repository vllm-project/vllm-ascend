import pytest
import torch

from vllm_ascend.ops.triton.muls_add import muls_add_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize(
    ("shape", "dtype", "scale"),
    [
        ((1, 2048), torch.float16, 1.25),
        ((4000, 2048), torch.float16, 0.75),
        ((4, 2048), torch.bfloat16, 1.0),
    ],
)
@torch.inference_mode()
def test_muls_add_triton_correctness(shape, dtype, scale):
    """compare the correctness of muls_add_triton with the PyTorch baseline implementation."""
    init_device_properties_triton()
    device = "npu"

    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype, device=device)
    y = torch.randn(*shape, dtype=dtype, device=device)

    out_triton = muls_add_triton(x, y, scale)
    out_ref = x * scale + y

    rtol, atol = 1e-3, 1e-3

    assert out_triton.shape == out_ref.shape
    assert out_triton.dtype == out_ref.dtype
    assert torch.allclose(out_triton, out_ref, rtol=rtol, atol=atol)

