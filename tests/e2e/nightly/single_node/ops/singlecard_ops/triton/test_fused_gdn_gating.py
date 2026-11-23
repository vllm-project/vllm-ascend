import pytest
import torch

from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
)
def test_fused_gdn_gating_310p_parity_precision():
    init_device_properties_triton()
    torch.manual_seed(0)
    device = "npu"

    num_tokens = 37
    num_heads = 8

    A_log = torch.randn(num_heads, dtype=torch.float16, device=device)
    dt_bias = torch.randn(num_heads, dtype=torch.float16, device=device)
    a = torch.randn(num_tokens, num_heads, dtype=torch.float16, device=device)
    b = torch.randn(num_tokens, num_heads, dtype=torch.float16, device=device)

    triton_g, triton_beta = fused_gdn_gating_patch(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        beta=1.0,
        threshold=20.0,
    )
    ref_g, ref_beta = fused_gdn_gating_pytorch(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        beta=1.0,
        threshold=20.0,
    )

    torch.testing.assert_close(
        triton_g.to(torch.float32).cpu(),
        ref_g.to(torch.float32).cpu(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        triton_beta.to(torch.float32).cpu(),
        ref_beta.to(torch.float32).cpu(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True,
    )


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend Triton",
)
@pytest.mark.parametrize("rows", [1, 7, 129, 4868])
@pytest.mark.parametrize("heads", [1, 8, 13, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["dense", "split", "transposed", "sliced", "expanded"])
@torch.inference_mode()
def test_gdn_gating_strided_inputs_match_contiguous(rows, heads, dtype, layout):
    init_device_properties_triton()
    a, b = torch.randn(rows, 2 * heads, device="npu", dtype=dtype).chunk(2, dim=-1)
    if layout == "dense":
        a, b = a.contiguous(), b.contiguous()
    elif layout == "transposed":
        b = torch.randn(heads, rows, device="npu", dtype=dtype).t()
    elif layout == "sliced":
        a = torch.randn(rows * 2, heads * 3, device="npu", dtype=dtype)[::2, ::3]
    elif layout == "expanded":
        a = torch.randn(1, heads, device="npu", dtype=dtype).expand(rows, -1)
    a_log = torch.randn(heads, device="npu", dtype=torch.float32)
    bias = torch.randn(heads, device="npu", dtype=torch.float32)
    before_a, before_b = a.clone(), b.clone()
    expected = fused_gdn_gating_patch(a_log, a.contiguous(), b.contiguous(), bias)
    actual = fused_gdn_gating_patch(a_log, a, b, bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(a, before_a, rtol=0, atol=0)
    torch.testing.assert_close(b, before_b, rtol=0, atol=0)
