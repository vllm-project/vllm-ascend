# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.activation import swiglu_clamp_and_mxfp8_quant


@pytest.mark.parametrize("rows", [1, 4])
def test_swiglu_clamp_and_mxfp8_quant_npu(rows):
    x = torch.randn((rows, 4096), device="npu", dtype=torch.bfloat16)

    y, scale = swiglu_clamp_and_mxfp8_quant(x, clamp_limit=10.0)
    torch.npu.synchronize()

    assert y.shape == (rows, 2048)
    assert y.dtype == torch.float8_e4m3fn
    assert scale.shape == (rows, 32, 2)
    assert scale.dtype == torch.uint8
