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
# mypy: ignore-errors

from __future__ import annotations

import math

import torch
import torch_npu
from vllm.model_executor.layers.fla.ops.utils import tensor_cache


@tensor_cache
def _l2norm_unit_weight(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # RMSNorm with weight 1/sqrt(dim) matches L2 norm: x / sqrt(sum(x^2)).
    return torch.full((dim,), 1.0 / math.sqrt(dim), dtype=dtype, device=device)


def l2norm_310p(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize the last dimension using the 310P NPU RMSNorm kernel."""
    orig_shape = x.shape
    dim = x.shape[-1]
    x_2d = x.reshape(-1, dim).contiguous()
    weight = _l2norm_unit_weight(dim, x.dtype, x.device)
    # RMSNorm: y = x / sqrt(mean(x^2) + eps_rms) * weight
    # With weight=1/sqrt(dim), this equals x / sqrt(sum(x^2) + dim * eps_rms).
    # L2 norm needs y = x / sqrt(sum(x^2) + eps), so eps_rms = eps / dim.
    y, _ = torch_npu.npu_rms_norm(x_2d, weight, eps / dim)
    return y.reshape(orig_shape)
