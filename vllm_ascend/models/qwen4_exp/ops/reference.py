# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Portable reference functions for Qwen3.8 HyperConnection tests."""

import torch
import torch.nn.functional as F


def grouped_gemma_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    """Apply Gemma RMSNorm independently to each HyperConnection stream."""
    if hidden_states.shape[-1] % num_groups:
        raise ValueError("hidden size must be divisible by the HC stream count")
    group_dim = hidden_states.shape[-1] // num_groups
    grouped = hidden_states.unflatten(-1, (num_groups, group_dim)).float()
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = grouped * torch.rsqrt(variance + eps)
    if weight.numel() == group_dim:
        affine = weight.reshape(1, 1, group_dim)
    elif weight.numel() == hidden_states.shape[-1]:
        affine = weight.reshape(1, num_groups, group_dim)
    else:
        raise ValueError("HC norm weight has an incompatible shape")
    return (normalized * (1.0 + affine.float())).flatten(-2).to(hidden_states.dtype)


def hc_silu(hidden_states: torch.Tensor, hc_count: int) -> torch.Tensor:
    return F.silu(hidden_states.float() / hc_count).to(hidden_states.dtype)


def hc_gate_mix(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if hidden_states.shape != gate.shape:
        raise ValueError("HC input and gate shapes must match")
    group_dim = hidden_states.shape[-1] // hc_count
    hidden_states = hidden_states.unflatten(-1, (hc_count, group_dim))
    gate = torch.sigmoid(gate.float()).unflatten(-1, (hc_count, group_dim))
    return (hidden_states.float() * gate).mean(dim=-2).to(hidden_states.dtype)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    group_dim = residual.shape[-1] // hc_count
    residual_grouped = residual.unflatten(-1, (hc_count, group_dim))
    injection = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
    output = residual_grouped.float() + block_output.float().unsqueeze(-2) * injection.unsqueeze(-1)
    return output.flatten(-2).to(residual.dtype)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = hc_combine(residual, block_output, injection_logits, hc_count)
    normalized = grouped_gemma_rmsnorm(output, norm_weight, eps, hc_count)
    return output, normalized
