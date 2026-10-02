# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hyper-connection width helpers used by the GLM-5.3-Flash decoder layers.

Expansion dispatches to the Ascend implementation where supported.
"""

import torch

from vllm_ascend.ops.mhc import mhc_expand


def hc_expand(x: torch.Tensor, n: int) -> torch.Tensor:
    """[s, hidden_size] -> [s, n, hidden_size] by replication."""
    return mhc_expand(x, n)


def hc_contract(x: torch.Tensor, n: int) -> torch.Tensor:
    """[s, n, hidden_size] -> [s, hidden_size] by averaging."""
    return x.mean(dim=1)
