#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

import math

import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.activation import (
    QuickGELU,
    SiluAndMul,
    SiluAndMulWithClamp,
    SwigluOAIAndMul,
    SwigluStepAndMul,
)

from vllm_ascend.utils import get_weight_prefetch_method


class AscendQuickGELU(QuickGELU):
    def forward_oot(self, x: torch.tensor) -> torch.Tensor:
        out = torch_npu.npu_fast_gelu(x)
        return out


class AscendSiluAndMul(SiluAndMul):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mlp_weight_preprocess(weight_prefetch_method.MLP_DOWN, x)
        out = torch_npu.npu_swiglu(x)
        weight_prefetch_method.maybe_prefetch_mlp_weight_postprocess(out)
        return out


class AscendSiluAndMulWithClamp(SiluAndMulWithClamp):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mlp_weight_preprocess(weight_prefetch_method.MLP_DOWN, x)
        d = x.shape[-1] // 2
        gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
        up = torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)
        x = torch.cat([gate, up], dim=-1)
        out = torch_npu.npu_swiglu(x)
        weight_prefetch_method.maybe_prefetch_mlp_weight_postprocess(out)
        return out


def swiglu_clamp_and_mxfp8_quant(x: torch.Tensor, clamp_limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim < 1 or x.shape[-1] % 2:
        raise ValueError(f"SwiGLU input last dimension must be even, got {tuple(x.shape)}")
    if not math.isfinite(clamp_limit) or clamp_limit < 0:
        raise ValueError(f"SwiGLU clamp limit must be finite and nonnegative, got {clamp_limit}")

    gate, up = x.chunk(2, dim=-1)
    if clamp_limit > 0:
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(min=-clamp_limit, max=clamp_limit)
    activation = F.silu(gate) * up
    return torch_npu.npu_dynamic_mx_quant(activation.contiguous(), dst_type=torch.float8_e4m3fn)


class AscendSwigluOAIAndMul:
    def swiglu_oai_forward(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
        class MinimalSwigluOAIAndMul:
            def __init__(self):
                self.alpha = alpha
                self.limit = limit

        layer = MinimalSwigluOAIAndMul()
        return SwigluOAIAndMul.forward_native(layer, x)


class AscendSwigluStepAndMul:
    def swiglustep_forward(x: torch.Tensor, limit: float = 7.0) -> torch.Tensor:
        if limit is None:
            raise ValueError("SwigluStepAndMul requires limit to be set.")

        class MinimalSwigluStepAndMul:
            def __init__(self):
                self.limit = limit

        layer = MinimalSwigluStepAndMul()
        return SwigluStepAndMul.forward_native(layer, x)
