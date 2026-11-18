# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared-expert activation replay with a materialized BF16 SiLU result."""

import torch


def silu_bf16_then_mul(x):
    """Consume concatenated [gate, up] projections, without changing weights."""
    if x.ndim < 1 or x.shape[-1] % 2 or x.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 gate/up concatenation with an even last dimension")
    gate, up = x.chunk(2, dim=-1)
    gate = gate.float()
    silu = (gate * torch.sigmoid(gate)).to(torch.bfloat16)
    return (silu.float() * up.float()).to(torch.bfloat16)
