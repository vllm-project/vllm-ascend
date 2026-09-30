#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Host-side KDA gate used when the NPU recurrent kernel cannot fuse it."""

import torch
import torch.nn.functional as F

DEFAULT_KDA_LOWER_BOUND = -5.0


def apply_kda_gate(
    raw_g: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    *,
    safe_gate: bool,
    lower_bound: float = DEFAULT_KDA_LOWER_BOUND,
) -> torch.Tensor:
    """Compute the per-token KDA decay gate from a raw projection.

    GLM-5.3-Flash uses the bounded (safe) gate
    ``y = lower_bound * sigmoid(exp(A) * (g + bias))``.
    Kimi-style checkpoints use ``y = -exp(A) * softplus(g + bias)``.
    The NPU recurrent kernel does not fuse this arithmetic, so it is applied
    here before ``fused_recurrent_kda`` / ``chunk_kda``.
    """
    g = raw_g.float()
    heads = a_log.numel()
    if g.shape[-2] != heads:
        raise ValueError(f"KDA gate head dim mismatch: raw_g shape {tuple(raw_g.shape)} vs A_log numel {heads}.")
    a = a_log.reshape(*([1] * (g.dim() - 2)), heads, 1).to(device=g.device, dtype=torch.float32)
    if g_bias is not None:
        bias = g_bias.to(device=g.device, dtype=torch.float32).reshape(*([1] * (g.dim() - 2)), heads, g.shape[-1])
        g = g + bias
    if safe_gate:
        return (lower_bound * torch.sigmoid(torch.exp(a) * g)).to(dtype=torch.float32)
    return ((-torch.exp(a)) * F.softplus(g)).to(dtype=torch.float32)
