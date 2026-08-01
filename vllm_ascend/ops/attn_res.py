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

import torch
import torch_npu
from torch import nn
from vllm.model_executor.layers.layernorm import RMSNorm


def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Module,
    norm: RMSNorm,
    *,
    use_fused_kernel: bool = True,
) -> torch.Tensor:
    """Apply Kimi K3's learned residual mixture.

    The custom kernel is not numerically interchangeable with the reference
    decomposition for MLA DSpark acceptance, so callers can select the exact
    NPU reference path without removing the fused operator for other modes.
    """
    if use_fused_kernel:
        return torch.ops._C_ascend.attn_res_fwd(
            prefix_sum,
            block_residual,
            projection.weight,
            norm.weight,
            norm.variance_epsilon,
        )

    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_fp32 = values.float()
    normalized, _ = torch_npu.npu_rms_norm(
        values_fp32,
        norm.weight.float(),
        norm.variance_epsilon,
    )
    scores = torch.matmul(normalized, projection.weight.t().float()).squeeze(-1)
    probabilities = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probabilities, values_fp32).squeeze(1).to(values.dtype)
