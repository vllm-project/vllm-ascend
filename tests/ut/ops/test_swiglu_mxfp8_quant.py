from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.activation import swiglu_clamp_and_mxfp8_quant


def test_swiglu_mxfp8_quant_applies_deepseek_clamp_before_quantization():
    x = torch.tensor([[-20.0, 20.0, -20.0, 20.0]], dtype=torch.bfloat16)
    quantized = torch.empty((1, 2), dtype=torch.float8_e4m3fn)
    scale = torch.empty((1, 1), dtype=torch.float8_e8m0fnu)

    with patch("vllm_ascend.ops.activation.torch_npu.npu_dynamic_mx_quant", return_value=(quantized, scale)) as quant:
        actual_quantized, actual_scale = swiglu_clamp_and_mxfp8_quant(x, clamp_limit=10.0)

    gate = x[:, :2].clamp(max=10.0)
    up = x[:, 2:].clamp(min=-10.0, max=10.0)
    expected = F.silu(gate) * up
    torch.testing.assert_close(quant.call_args.args[0], expected)
    assert quant.call_args.kwargs == {"dst_type": torch.float8_e4m3fn}
    assert actual_quantized is quantized
    assert actual_scale is scale


def test_swiglu_mxfp8_quant_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="last dimension must be even"):
        swiglu_clamp_and_mxfp8_quant(torch.empty((2, 3)), clamp_limit=10.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        swiglu_clamp_and_mxfp8_quant(torch.empty((2, 4)), clamp_limit=-1.0)
