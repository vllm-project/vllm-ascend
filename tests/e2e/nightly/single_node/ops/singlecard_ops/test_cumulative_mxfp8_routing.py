# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware import AscendDeviceType


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="MXFP8 routing requires A5")
@pytest.mark.parametrize("rows", [1, 7, 513])
@pytest.mark.parametrize("width", [128, 2048])
@pytest.mark.parametrize("experts", [8, 256])
@pytest.mark.parametrize("quant_mode", [-1, 3])
@pytest.mark.parametrize("skewed", [False, True])
@torch.inference_mode()
def test_cumulative_routing_matches_counts(rows, width, experts, quant_mode, skewed):
    x = torch.randn(rows, width, device="npu", dtype=torch.bfloat16)
    # Include unused experts and highly unbalanced routes.
    ids = (
        torch.zeros(rows, 1, device="npu", dtype=torch.int32)
        if skewed
        else torch.randint(experts, (rows, 2), device="npu", dtype=torch.int32)
    )
    kwargs = dict(
        active_num=ids.numel(),
        expert_num=experts,
        active_expert_range=[0, experts],
        quant_mode=quant_mode,
        act_quant_type=torch.float8_e4m3fn if quant_mode == 3 else None,
    )
    counted = DeviceOperator.npu_moe_init_routing(x, ids, expert_tokens_num_type=1, **kwargs)
    cumulative = DeviceOperator.npu_moe_init_routing(x, ids, expert_tokens_num_type=0, **kwargs)
    torch.testing.assert_close(cumulative[2].long(), counted[2].long().cumsum(0), rtol=0, atol=0)
    torch.testing.assert_close(cumulative[1], counted[1], rtol=0, atol=0)
    torch.testing.assert_close(cumulative[0].view(torch.uint8), counted[0].view(torch.uint8), rtol=0, atol=0)
    if quant_mode == 3:
        torch.testing.assert_close(cumulative[3].view(torch.uint8), counted[3].view(torch.uint8), rtol=0, atol=0)
