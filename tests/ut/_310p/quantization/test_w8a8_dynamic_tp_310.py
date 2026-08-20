# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from unittest.mock import MagicMock, patch

import torch

from vllm_ascend._310p.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicLinearMethod310,
)


@patch("torch.nn.functional.linear")
def test_dynamic_linear_omits_bias_on_nonzero_tp_rank(mock_linear) -> None:
    method = AscendW8A8DynamicLinearMethod310()
    layer = MagicMock()
    layer.weight_fp = torch.randn(256, 128, dtype=torch.float16)
    inputs = torch.randn(2, 128, dtype=torch.float16)
    bias = torch.randn(256, dtype=torch.float16)
    mock_linear.return_value = torch.randn(2, 256, dtype=torch.float16)

    method.apply(layer, inputs, bias=bias, tp_rank=1)

    assert mock_linear.call_args.args[2] is None
