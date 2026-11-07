# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""LiLiCorr tensor fallbacks for vLLM 0.32.x.

Activated by the LiLiCorr worker before loading draft weights. Only the lattice
attention needs a fallback: LiLiCorr's grouped convolution is served by
``vllm_ascend.models.qwen3_dflash2.DFlashGroupedConv``, which already carries an
Ascend-safe implementation. Remove this once upstream offers Ascend-compatible
learned-bias attention.
"""

import torch
import torch.nn.functional as F
from vllm.model_executor.models.lilicorr import LiLiCorrLatticeAttention


def lattice_attention_forward(
    self: LiLiCorrLatticeAttention,
    hidden_states: torch.Tensor,
    attention_bias: torch.Tensor,
) -> torch.Tensor:
    """Preserve arbitrary per-head additive biases without fused SDPA."""
    batch_size, sequence_length, _ = hidden_states.shape
    projections = F.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)
    shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
    query, key, value = (projection.reshape(shape).transpose(1, 2) for projection in projections.chunk(3, dim=-1))
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * (self.head_dim**-0.5)
    probabilities = torch.softmax(scores + attention_bias.float(), dim=-1).to(value.dtype)
    output = torch.matmul(probabilities, value)
    return self.out_proj(output.transpose(1, 2).reshape(batch_size, sequence_length, self.hidden_size))


def apply_lilicorr_patch() -> None:
    """Install worker-local fallbacks only when a LiLiCorr model is loaded."""
    LiLiCorrLatticeAttention.forward = lattice_attention_forward
