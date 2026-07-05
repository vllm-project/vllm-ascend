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

import torch
import torch_npu
from vllm.model_executor.layers.activation import (
    QuickGELU,
    SiluAndMul,
    SiluAndMulWithClamp,
    SwigluOAIAndMul,
    SwigluStepAndMul,
)

from vllm_ascend.utils import enable_custom_op, get_weight_prefetch_method


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

        if enable_custom_op():
            # Fused kernel takes the single row-major x[M, 2N] directly and splits
            # gate/up in UB, avoiding the host-side x.chunk(2, -1).contiguous()
            # GM->GM copies that the previous two-input form required.
            return torch.ops._C_ascend.npu_swiglustep(x, limit)

        # Fallback when custom ops are disabled (e.g. Ascend 950 / A5, where enable_custom_op()
        # returns False): vllm's SwigluStepAndMul.forward_native — same pattern as
        # AscendSwigluOAIAndMul.swiglu_oai_forward, numerically equivalent to the fused kernel.
        class MinimalSwigluStepAndMul:
            def __init__(self):
                self.limit = limit

        layer = MinimalSwigluStepAndMul()
        return SwigluStepAndMul.forward_native(layer, x)
