# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise W4A8 GMM hooks with empty and populated expert ranks on 950."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="Requires native 950 S8S4 GMM")
@pytest.mark.parametrize("rows", [0, 8])
@pytest.mark.parametrize("projection", [1, 2])
def test_w4a8_gmm_empty_expert_rank(rows, projection):
    method = object.__new__(AscendW4A8DynamicFusedMoEMethod)
    method.use_expert_weight_list = False
    layer = SimpleNamespace()
    # Two experts, all routed tokens owned by the first. Signed INT4 ones
    # packed into INT32 give a simple independent result: sum(x) == 256.
    for name, width in (("w13", 512), ("w2", 256)):
        setattr(layer, f"{name}_weight", torch.full((2, 256, width // 8), 0x11111111, device="npu", dtype=torch.int32))
        scale = torch.ones(2, 1, width, dtype=torch.float32).view(torch.int32).to(torch.int64).npu()
        setattr(layer, f"{name}_weight_scale", scale)
        setattr(layer, f"{name}_scale_bias", torch.zeros(2, width, device="npu"))
    hidden = torch.ones(rows, 256, device="npu", dtype=torch.int8)
    scale = torch.ones(rows, device="npu")
    inputs = SimpleNamespace(
        hidden_states=hidden,
        dynamic_scale=scale,
        layer=layer,
        group_list=torch.tensor([rows, 0], device="npu", dtype=torch.int64),
        group_list_type=1,
    )
    output = method.apply_gmm1(inputs) if projection == 1 else method.apply_gmm2(inputs, hidden, scale)
    torch.npu.synchronize()
    expected = torch.full((rows, 512 if projection == 1 else 256), 256, dtype=torch.bfloat16)
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
