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


def fused_gdn_gating_pytorch(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of fused_gdn_gating.
    This is a fallback implementation for 310P without Triton support.
    
    Args:
        A_log: Log of A parameter, shape [num_heads]
        a: a parameter, shape [batch, num_heads]
        b: b parameter, shape [batch, num_heads]
        dt_bias: dt bias, shape [num_heads]
        beta: softplus beta parameter
        threshold: softplus threshold parameter
    
    Returns:
        g: gating parameter, shape [1, batch, num_heads]
        beta_output: sigmoid(b), shape [1, batch, num_heads]
    """
    batch, num_heads = a.shape
    del num_heads
    # Keep nonlinear gating math in fp32 for stability.
    compute_dtype = torch.float32
    A_log_f = A_log.to(compute_dtype)
    a_f = a.to(compute_dtype)
    b_f = b.to(compute_dtype)
    dt_bias_f = dt_bias.to(compute_dtype)

    # Expand A_log and dt_bias to match a shape.
    A_log_expanded = A_log_f.unsqueeze(0).expand(batch, -1)
    dt_bias_expanded = dt_bias_f.unsqueeze(0).expand(batch, -1)

    # Compute x = a + dt_bias.
    x = a_f + dt_bias_expanded

    # Compute softplus(x).
    beta_x = beta * x
    softplus_x = torch.where(
        beta_x <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(beta_x)),
        x,
    )

    # Compute g = -exp(A_log) * softplus(x).
    g = -torch.exp(A_log_expanded) * softplus_x

    # Add sequence dimension.
    g = g.unsqueeze(0)

    # Compute beta_output = sigmoid(b).
    beta_output = torch.sigmoid(b_f)
    beta_output = beta_output.unsqueeze(0)

    return g, beta_output
