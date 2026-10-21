# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.fused_moe.shared_dequant_swiglu_quant import shared_dequant_swiglu_quant
from vllm_ascend.utils import enable_custom_op


@torch.inference_mode()
@pytest.mark.parametrize("num_tokens", [1, 2, 7, 8, 9, 64])
@pytest.mark.parametrize("width", [256, 384, 4096, 6144])
@pytest.mark.parametrize("limit", [1.0, 2.0, 10.0, 100.0])
def test_shared_clipped_swiglu_legacy_parity(num_tokens, width, limit):
    if not get_current_hardware_profile().supports(HardwareCapability.SHARED_SWIGLU_ONE_GROUP):
        pytest.skip("The official shared one-group adapter is validated only on A3.")
    torch_npu.npu.config.allow_internal_format = True
    enable_custom_op()
    generator = torch.Generator().manual_seed(20261021)
    x = torch.randint(-10000, 10000, (num_tokens, width), dtype=torch.int32, generator=generator).npu()
    weight_scale = (torch.rand(width, generator=generator) * 0.5 + 0.01).npu()
    activation_scale = (torch.rand(num_tokens, generator=generator) * 0.5 + 0.01).npu()

    def reference():
        return torch.ops._C_ascend.npu_dequant_swiglu_quant(
            x=x,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            group_index=None,
            activate_left=True,
            quant_mode=1,
            swiglu_mode=1,
            clamp_limit=limit,
            glu_alpha=1.0,
            glu_bias=0.0,
        )

    expected = reference()
    actual = shared_dequant_swiglu_quant(x, weight_scale, activation_scale, limit)
    torch.npu.synchronize()
    for output, golden in zip(actual, expected):
        assert torch.equal(output.cpu(), golden.cpu())

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured = shared_dequant_swiglu_quant(x, weight_scale, activation_scale, limit)
    # Replay with new data, not merely the values present during capture.
    x.neg_()
    activation_scale.mul_(0.5)
    expected = reference()
    graph.replay()
    torch.npu.synchronize()
    for output, golden in zip(captured, expected):
        assert torch.equal(output.cpu(), golden.cpu())
