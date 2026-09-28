# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram projection and gate for the rotated Ascend checkpoint."""

import torch
from torch import nn

from .common import engram_gate


class AscendEngram(nn.Module):
    """Consume rows prepared by the v1 runner using the existing rotated-checkpoint gate."""

    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.wkv = nn.Linear(
            (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim,
            (self.hc_mult + 1) * self.dim,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.q_weight = nn.Parameter(torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16))
        self.k_weight = nn.Parameter(torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16))

    def forward(
        self,
        hidden_states: torch.Tensor,
        rows: torch.Tensor,
        token_mask: torch.Tensor,
        rotation: torch.Tensor,
    ) -> torch.Tensor:
        kv = self.wkv(rows)
        key, value = kv.split([self.hc_mult * self.dim, self.dim], -1)
        return engram_gate(
            hidden_states,
            key.view(hidden_states.shape[0], self.hc_mult, self.dim),
            value,
            self.q_weight.float() * self.k_weight.float(),
            rotation,
            token_mask,
            self.eps,
        )
