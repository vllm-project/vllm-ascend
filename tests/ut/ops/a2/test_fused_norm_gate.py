# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import pytest
import torch

from vllm_ascend.ops.triton.fused_norm_gate import fused_norm_gate_fwd_npu
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize("num_rows", [1, 64, 129])
@pytest.mark.parametrize("norm_before_gate", [False, True])
@torch.inference_mode()
def test_kimi_k3_fused_rms_norm_sigmoid_gate(num_rows: int, norm_before_gate: bool):
    init_device_properties_triton()
    torch.manual_seed(42)
    head_dim = 128
    epsilon = 1e-5
    x = torch.randn(num_rows, head_dim, dtype=torch.bfloat16, device="npu")
    gate = torch.randn_like(x)
    weight = torch.randn(head_dim, dtype=torch.bfloat16, device="npu")

    actual, mean, rstd = fused_norm_gate_fwd_npu(
        x,
        gate,
        weight,
        None,
        activation="sigmoid",
        eps=epsilon,
        is_rms_norm=True,
        norm_before_gate=norm_before_gate,
    )

    x_fp32 = x.float()
    gate_fp32 = gate.float().sigmoid()
    if not norm_before_gate:
        x_fp32 *= gate_fp32
    expected = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + epsilon)
    expected *= weight.float()
    if norm_before_gate:
        expected *= gate_fp32

    assert mean is None
    assert not torch.isnan(actual).any()
    assert not torch.isnan(rstd).any()
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=2e-2, atol=2e-2)
