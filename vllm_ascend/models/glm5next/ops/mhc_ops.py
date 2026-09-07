# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hyper-connection width helpers used by the GLM-5.3-Flash decoder layers.

Upstream keeps these next to the MHC ops in ``vllm.model_executor.layers.mhc``.
They are plain shape ops with no backend-specific behavior, so vLLM Ascend
carries its own copy while the GLM-5.3-Flash architecture lives downstream.
"""

import torch


def hc_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    """[s, hidden_size] -> [s, n * hidden_size] by replication."""
    return x.unsqueeze(1).expand(-1, n, -1).contiguous()


def hc_contract(x: torch.Tensor, n: int) -> torch.Tensor:
    """[s, n * hidden_size] -> [s, hidden_size] by averaging."""
    return x.mean(dim=1)
