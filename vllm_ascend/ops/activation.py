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

from dataclasses import dataclass

import torch
import torch_npu
from torch import nn
from vllm.model_executor.layers.activation import (
    QuickGELU,
    SiluAndMul,
    SiluAndMulWithClamp,
    SwigluOAIAndMul,
    SwigluStepAndMul,
)
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm_ascend.ops.triton.kimi_k3.situ_and_mul import situ_and_mul as fused_situ_and_mul
else:
    fused_situ_and_mul = None


@dataclass(frozen=True, slots=True)
class SituActivationConfig:
    """Runtime parameters for Kimi's SiTU gated activation."""

    beta: float = 1.0
    linear_beta: float | None = None

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError(f"SiTU beta must be positive, got {self.beta}.")
        if self.linear_beta is not None and self.linear_beta <= 0:
            raise ValueError(f"SiTU linear_beta must be positive, got {self.linear_beta}.")


def situ_and_mul(
    x: torch.Tensor,
    *,
    beta: float = 1.0,
    linear_beta: float | None = None,
    group_list: torch.Tensor | None = None,
    group_list_type: int | None = None,
) -> torch.Tensor:
    """Apply Kimi SiTU, using the K3 Triton fusion on Ascend when available."""
    config = SituActivationConfig(beta=beta, linear_beta=linear_beta)
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"SiTU expects an even last dimension, got {x.shape[-1]}.")
    if group_list is not None and group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}.")

    if fused_situ_and_mul is not None and x.device.type == "npu" and x.numel() > 0:
        return fused_situ_and_mul(
            x,
            group_list=group_list,
            group_list_type=group_list_type,
            beta=config.beta,
            linear_beta=config.linear_beta,
        )

    gate, up = x.to(torch.float32).chunk(2, dim=-1)
    gate = config.beta * torch.tanh(gate / config.beta) * torch.sigmoid(gate)
    if config.linear_beta is not None:
        up = config.linear_beta * torch.tanh(up / config.linear_beta)
    return (gate * up).to(x.dtype)


class AscendSituAndMul(nn.Module):
    """Module form of Kimi SiTU used by dense and shared-expert MLPs."""

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None) -> None:
        super().__init__()
        self.config = SituActivationConfig(beta=beta, linear_beta=linear_beta)

    @property
    def beta(self) -> float:
        return self.config.beta

    @property
    def linear_beta(self) -> float | None:
        return self.config.linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return situ_and_mul(x, beta=self.beta, linear_beta=self.linear_beta)

    def extra_repr(self) -> str:
        return f"beta={self.beta}, linear_beta={self.linear_beta}"


class AscendQuickGELU(QuickGELU):
    def forward_oot(self, x: torch.tensor) -> torch.Tensor:
        out = torch_npu.npu_fast_gelu(x)
        return out


class AscendSiluAndMul(SiluAndMul):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        out = torch_npu.npu_swiglu(x)
        return out


class AscendSiluAndMulWithClamp(SiluAndMulWithClamp):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
        up = torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)
        x = torch.cat([gate, up], dim=-1)
        out = torch_npu.npu_swiglu(x)
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

        # Triton fused path: 1D-grid row-loop kernel that fuses
        # silu + clamp + mul into a single launch (see
        # vllm_ascend/ops/triton/activation/swiglustep.py). Numerically
        # equivalent to forward_native below.
        from vllm.triton_utils import HAS_TRITON

        if HAS_TRITON:
            from vllm_ascend.ops.triton.activation.swiglustep import (
                swiglustep_forward_triton,
            )

            return swiglustep_forward_triton(x, limit)

        # Fallback when triton is unavailable: vllm's native silu+clamp+mul.
        class MinimalSwigluStepAndMul:
            def __init__(self):
                self.limit = limit

        layer = MinimalSwigluStepAndMul()
        return SwigluStepAndMul.forward_native(layer, x)
