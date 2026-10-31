# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch
import torch.nn as nn

from vllm_ascend.quantization.methods.fp8 import AscendW4A8MXFPDSDynamicFusedMoEMethod


def test_catccos_preserves_original_mxfp4_expert_weights():
    layer = nn.Module()
    layer.w13_weight = nn.Parameter(torch.arange(128 * 32, dtype=torch.uint8).reshape(1, 128, 32), requires_grad=False)
    layer.w2_weight = nn.Parameter(torch.arange(64 * 32, dtype=torch.uint8).reshape(1, 64, 32), requires_grad=False)
    layer.w13_weight_scale = nn.Parameter(torch.ones((1, 128, 2), dtype=torch.uint8), requires_grad=False)
    layer.w2_weight_scale = nn.Parameter(torch.ones((1, 64, 2), dtype=torch.uint8), requires_grad=False)
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()
    method = object.__new__(AscendW4A8MXFPDSDynamicFusedMoEMethod)

    with (
        patch("vllm_ascend.ops.fused_moe.catccos_adapter.catccos_backend_enabled", return_value=True),
        patch("vllm_ascend.quantization.methods.fp8.torch_npu.npu_format_cast") as format_cast,
    ):
        method.process_weights_after_loading(layer)

    format_cast.assert_not_called()
    assert layer.w13_weight.dtype == torch.uint8
    assert layer.w2_weight.dtype == torch.uint8
    assert tuple(layer.w13_weight.shape) == (1, 32, 128)
    assert tuple(layer.w2_weight.shape) == (1, 32, 64)
    torch.testing.assert_close(layer.w13_weight, original_w13.transpose(1, 2))
    torch.testing.assert_close(layer.w2_weight, original_w2.transpose(1, 2))
    assert tuple(layer.w13_weight_scale.shape) == (1, 1, 128, 2)
    assert tuple(layer.w2_weight_scale.shape) == (1, 1, 64, 2)
