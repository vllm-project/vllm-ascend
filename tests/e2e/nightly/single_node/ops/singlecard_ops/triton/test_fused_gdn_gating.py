import pytest
import torch

from vllm_ascend._310p.ops.fla.fused_gdn_gating import fused_gdn_gating_pytorch
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
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
    reason="Requires Ascend A2/A3/A5",
)
def test_fused_gdn_gating_head_major_raw_beta():
    init_device_properties_triton()
    torch.manual_seed(1)
    num_tokens, num_heads = 37, 8
    device = "npu"
    A_log = torch.randn(num_heads, dtype=torch.float16, device=device)
    dt_bias = torch.randn(num_heads, dtype=torch.float16, device=device)
    a = torch.randn(num_tokens, num_heads, dtype=torch.float16, device=device)
    b = torch.randn(num_tokens, num_heads, dtype=torch.float16, device=device)

    ref_g, _ = fused_gdn_gating_patch(A_log, a, b, dt_bias)
    opt_g, opt_beta = fused_gdn_gating_patch(
        A_log,
        a,
        b,
        dt_bias,
        output_head_major=True,
        output_raw_beta=True,
    )

    assert opt_g.shape == (1, num_heads, num_tokens)
    assert opt_beta.shape == (1, num_heads, num_tokens)
    assert opt_g.is_contiguous()
    assert opt_beta.is_contiguous()
    torch.testing.assert_close(
        opt_g.float().cpu(),
        ref_g.transpose(1, 2).contiguous().float().cpu(),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        opt_beta.float().cpu(),
        b.transpose(0, 1).unsqueeze(0).contiguous().float().cpu(),
        rtol=0,
        atol=0,
    )
