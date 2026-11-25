# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[Role] Project Engram rows and inject gated values into residual streams.
[Invariants] Rotated checkpoints compute the gate in the original hidden basis.
[Timeouts & Locks] Current-stream tensor work; no host waits or locks.
[Side-Effect Policy] Return new hidden states; leave lookup rows/weights intact.
"""

import torch
from torch import nn
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.engram_gate import fused_engram_gate
from vllm_ascend.ops.triton.engram_gate_a3 import fused_engram_gate_a3

from .common import engram_gate


class AscendEngram(nn.Module):
    """Consume rows prepared by the v1 runner in the checkpoint residual basis."""

    def __init__(self, config, quant_config, prefix: str) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.wkv = ReplicatedLinear(
            (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim,
            (self.hc_mult + 1) * self.dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
        )
        self.q_weight = nn.Parameter(torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16))
        self.k_weight = nn.Parameter(torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16))

    def forward(
        self,
        hidden_states: torch.Tensor,
        rows: torch.Tensor,
        token_mask: torch.Tensor,
        rotation: torch.Tensor | None,
    ) -> torch.Tensor:
        kv = self.wkv(rows)
        if hidden_states.device.type == "npu":
            profile = get_current_hardware_profile()
            if rotation is None and profile.supports(HardwareCapability.ENGRAM_UNROTATED_GATE):
                return fused_engram_gate(hidden_states, kv, self.q_weight, self.k_weight, token_mask, self.eps)
            if rotation is not None and profile.supports(HardwareCapability.ENGRAM_ROTATED_GATE):
                return fused_engram_gate_a3(
                    hidden_states, kv, self.q_weight, self.k_weight, rotation, token_mask, self.eps
                )
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
