# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_heads", [8, 16, 24, 32])
@pytest.mark.parametrize("num_tokens", [1, 37, 65])
def test_fused_gdn_gating_triton(num_tokens, num_heads, dtype):
    init_device_properties_triton()
    torch.manual_seed(0)
    a = torch.randn(num_tokens, num_heads, dtype=dtype)
    b = torch.randn(num_tokens, num_heads, dtype=dtype)
    A_log = torch.randn(num_heads, dtype=dtype)
    dt_bias = torch.randn(num_heads, dtype=dtype)

    # Use an independent CPU reference with the same fp32 gating math.
    expected_g = (-torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float())).unsqueeze(0)
    expected_beta = torch.sigmoid(b.float()).to(dtype).unsqueeze(0)

    g, beta_output = fused_gdn_gating_patch(A_log.npu(), a.npu(), b.npu(), dt_bias.npu())

    assert g.shape == (1, num_tokens, num_heads)
    assert beta_output.shape == (1, num_tokens, num_heads)
    assert g.dtype == torch.float32
    assert beta_output.dtype == dtype
    assert g.device.type == "npu"
    assert beta_output.device == g.device
    torch.testing.assert_close(g.cpu(), expected_g, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(beta_output.cpu(), expected_beta, rtol=1e-2, atol=1e-3)
