import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'dtype'),
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
