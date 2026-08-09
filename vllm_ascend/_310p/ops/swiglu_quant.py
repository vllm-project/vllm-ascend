# SPDX-License-Identifier: Apache-2.0
"""Composed clamped-SwiGLU plus dynamic quantization for Ascend 310P."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_npu


def swiglu_quant_310p(
    gate_up: torch.Tensor,
    *,
    clamp_limit: float = 0.0,
    glu_alpha: float = 1.0,
    glu_bias: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the DeepSeek V4 SwiGLU variant and dynamically quantize it."""
    if gate_up.shape[-1] % 2 != 0:
        raise ValueError(f"Gate/up width must be even, got {gate_up.shape[-1]}.")
    if glu_alpha != 1.0 or glu_bias != 0.0:
        raise NotImplementedError(
            "The 310P composed SwiGLU path currently supports only glu_alpha=1.0 and glu_bias=0.0."
        )

    gate, up = gate_up.chunk(2, dim=-1)
    if clamp_limit > 0.0:
        gate = torch.clamp(gate, max=clamp_limit)
        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    activated = F.silu(gate) * up
    return torch_npu.npu_dynamic_quant(activated)
