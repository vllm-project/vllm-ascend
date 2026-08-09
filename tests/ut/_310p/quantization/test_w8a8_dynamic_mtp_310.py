# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import torch

from vllm_ascend._310p.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod310


@patch("torch_npu.npu_dynamic_quant", create=True)
@patch("torch_npu.npu_quant_matmul")
def test_apply_flattens_mtp_candidate_dimensions(mock_npu_quant_matmul, mock_npu_dynamic_quantize) -> None:
    method = AscendW8A8DynamicLinearMethod310()
    layer = MagicMock()
    layer.weight = torch.randn(128, 256, dtype=torch.float16)
    layer.weight_scale = torch.randn(128, dtype=torch.float32)

    x = torch.randn(3, 4, 128, dtype=torch.float16)
    flattened_x = x.reshape(12, 128)
    quantized_x = torch.randint(-128, 127, flattened_x.shape, dtype=torch.int8)
    pertoken_scale = torch.randn(12, 1, dtype=torch.float32)
    mock_npu_dynamic_quantize.return_value = quantized_x, pertoken_scale
    flattened_output = torch.randn(12, 256, dtype=torch.float16)
    mock_npu_quant_matmul.return_value = flattened_output

    output = method.apply(layer, x)

    torch.testing.assert_close(mock_npu_dynamic_quantize.call_args.args[0], flattened_x)
    assert mock_npu_quant_matmul.call_args.kwargs["pertoken_scale"].shape == (12,)
    assert output.shape == (3, 4, 256)
    torch.testing.assert_close(output, flattened_output.reshape(3, 4, 256))
