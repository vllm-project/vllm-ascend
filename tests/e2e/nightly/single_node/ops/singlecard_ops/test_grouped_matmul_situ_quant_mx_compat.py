# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 native regression for all public interfaces shared with A3 MSD."""

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import enable_custom_op


def _nz(weight):
    # Preserve the ND source for the entry that performs its own conversion.
    result = weight.view(torch.uint8).clone().view(torch.float4_e2m1fn_x2)
    torch_npu.npu_format_cast_(result, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch.float4_e2m1fn_x2)
    return result


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Native MX interface regression requires Ascend A5",
)
@pytest.mark.parametrize("experts", [1, 2])
@pytest.mark.parametrize("capacity", [0, 8])
@pytest.mark.parametrize("expert_list", [False, True])
@pytest.mark.parametrize("nz", [False, True])
@torch.inference_mode()
def test_mx_shared_entry_compatibility(experts, capacity, expert_list, nz):
    enable_custom_op()
    torch.manual_seed(1729)
    k, n = 3584, 6144
    weights = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device="npu")
    weights = weights.view(torch.float4_e2m1fn_x2)
    scales = torch.randint(122, 126, (experts, n, k // 64, 2), dtype=torch.uint8, device="npu")
    scales = scales.view(torch.float8_e8m0fnu)
    # Quantize a nonempty source, then slice for the zero-capacity ABI case.
    x, x_scale = torch_npu.npu_dynamic_mx_quant(
        torch.randn(max(capacity, 1), k, dtype=torch.bfloat16, device="npu") * 0.1,
        dst_type=torch.float8_e4m3fn,
    )
    x, x_scale = x[:capacity], x_scale[:capacity].view(torch.float8_e8m0fnu)
    nz_list = [_nz(w) for w in weights.unbind()]
    scale_list = list(scales.unbind())
    groups = torch.zeros(experts, dtype=torch.int64, device="npu")
    op = (
        torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz if nz else torch.ops._C_ascend.grouped_matmul_situ_quant
    )
    if expert_list:
        op, weight_arg, scale_arg = op.list, nz_list if nz else list(weights.unbind()), scale_list
    else:
        weight_arg, scale_arg = _nz(weights) if nz else weights, scales

    def call(entry, weight, scale):
        return entry(
            x,
            weight,
            scale,
            None,
            None,
            x_scale,
            None,
            groups,
            dequant_mode=1,
            dequant_dtype=0,
            quant_mode=1,
            group_list_type=1,
            tuning_config=None,
            beta=4.0,
            linear_beta=25.0,
        )

    # The existing split-reference regression validates the NZ list kernel.
    # Compare every public wrapper against that path to detect ABI/format
    # regressions when the new INT8 branch is compiled into the same library.
    reference_op = torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz.list
    for expert in range(experts):
        counts = torch.zeros(experts, dtype=torch.int64)
        counts[expert] = capacity
        groups.copy_(counts)
        actual = call(op, weight_arg, scale_arg)
        reference = call(reference_op, nz_list, scale_list)
        assert actual[0].shape == (capacity, n // 2)
        assert actual[1].shape == (capacity, n // 128, 2)
        assert actual[0].dtype == torch.float8_e4m3fn
        assert actual[1].dtype == torch.float8_e8m0fnu
        for result, expected in zip(actual, reference):
            torch.testing.assert_close(result.view(torch.uint8).cpu(), expected.view(torch.uint8).cpu(), rtol=0, atol=0)
