# SPDX-License-Identifier: Apache-2.0
"""MXFP4 tail groups and TP group alignment on one A5 NPU.

TP ranks are evaluated sequentially to check their checkpoint slices and
partial projections without requiring a distributed worker group.
"""

import pytest
import torch
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import AscendW4A8MXFPDynamicLinearMethod
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="Requires A5 MXFP4 kernels.")


@pytest.mark.parametrize("input_size,tp_size", [(4304, 1), (4320, 1), (4352, 1), (4304, 2), (4304, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_bias", [False, True])
@torch.inference_mode()
def test_partial_groups_and_tp_projection(input_size, tp_size, dtype, with_bias):
    scheme = object.__new__(AscendW4A8MXFPDynamicLinearMethod)
    scheme.group_size = 32
    output_size = 128
    local_size = input_size // tp_size
    scale_count = (input_size + scheme.group_size - 1) // scheme.group_size
    # Each packed nibble 0x2 represents FP4 1.0; use different neighboring
    # group scales so a dropped tail or a misplaced group cannot pass.
    checkpoint_weight = torch.full((output_size, input_size // 2), 0x22, dtype=torch.uint8)
    checkpoint_scales = (127 + torch.arange(scale_count) % 3).to(torch.uint8).repeat(output_size, 1)
    weight_reference = torch.exp2(checkpoint_scales.float() - 127).repeat_interleave(32, dim=-1)[:, :input_size]
    # Exercise a non-contiguous batched input, with nonzero values at each
    # rank's trailing boundary, including the final incomplete MX group.
    x = torch.zeros(3, 2, input_size, dtype=dtype).transpose(0, 1)
    for rank in range(tp_size):
        end = (rank + 1) * local_size
        x[..., end - 16 : end] = 1
    bias = torch.arange(output_size, dtype=dtype) % 8 if with_bias else None
    expected = x.float() @ weight_reference.T
    if bias is not None:
        expected += bias
    actual = torch.zeros_like(expected)

    for rank in range(tp_size):
        layer = RowParallelLinear.__new__(RowParallelLinear)
        torch.nn.Module.__init__(layer)
        layer.input_size, layer.input_size_per_partition = input_size, local_size
        layer.output_size = layer.output_size_per_partition = output_size
        layer.tp_size, layer.tp_rank = tp_size, rank
        with torch.device("npu"):
            AscendLinearMethod(scheme).create_weights(
                layer,
                input_size_per_partition=local_size,
                output_partition_sizes=[output_size],
                input_size=input_size,
                output_size=output_size,
                params_dtype=dtype,
                weight_loader=layer.weight_loader,
            )
        layer.weight.weight_loader(layer.weight, checkpoint_weight)
        layer.weight_scale.weight_loader(layer.weight_scale, checkpoint_scales)
        start, end = rank * local_size, (rank + 1) * local_size
        local_x = x[..., start:end].npu()
        local_bias = bias.npu() if bias is not None and rank == 0 else None
        local_expected = x[..., start:end].float() @ weight_reference[:, start:end].T
        if bias is not None and rank == 0:
            local_expected += bias
        for _ in range(2):
            scheme.process_weights_after_loading(layer)
            output = scheme.apply(layer, local_x, local_bias).cpu()
            torch.testing.assert_close(output.float(), local_expected, rtol=0, atol=0)
            scheme.restore_weights_for_rl_loading(layer)
            torch.testing.assert_close(layer.weight.cpu(), checkpoint_weight[:, start // 2 : end // 2])
        actual += output.float()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
