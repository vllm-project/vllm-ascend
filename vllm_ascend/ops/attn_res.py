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
from torch import nn
from vllm.model_executor.layers.layernorm import RMSNorm


def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Module,
    norm: RMSNorm,
) -> torch.Tensor:
    """Apply Kimi K3's canonical learned residual mixture."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1).float()
    inverse_rms = torch.rsqrt(values.square().mean(-1, keepdim=True) + norm.variance_epsilon)
    normalized_without_gamma = values * inverse_rms
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    scores = (normalized_without_gamma * score_weight).sum(-1)
    probabilities = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probabilities, values).squeeze(1).to(prefix_sum.dtype)
