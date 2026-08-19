# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from unittest.mock import MagicMock, patch

import torch

from vllm_ascend._310p.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicLinearMethod310,
)


@patch("torch_npu.npu_dynamic_quant", create=True)
@patch("torch_npu.npu_quant_matmul")
def test_dynamic_linear_omits_bias_on_nonzero_tp_rank(
    mock_npu_quant_matmul,
    mock_npu_dynamic_quant,
) -> None:
    method = AscendW8A8DynamicLinearMethod310()
    layer = MagicMock()
    layer.weight = torch.randint(-127, 128, (128, 256), dtype=torch.int8)
    layer.weight_scale = torch.randn(128, dtype=torch.float32)
    inputs = torch.randn(2, 128, dtype=torch.float16)
    bias = torch.randn(256, dtype=torch.float16)
    mock_npu_dynamic_quant.return_value = (
        torch.randint(-128, 127, inputs.shape, dtype=torch.int8),
        torch.randn(inputs.shape[0], dtype=torch.float32),
    )
    mock_npu_quant_matmul.return_value = torch.randn(2, 256, dtype=torch.float16)

    method.apply(layer, inputs, bias=bias, tp_rank=1)

    assert mock_npu_quant_matmul.call_args.kwargs["bias"] is None
