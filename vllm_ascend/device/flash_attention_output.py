# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.utils import supports_triton


def flash_attention_gate(projected: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply output gating without loading Triton on unsupported devices."""
    if supports_triton():
        from vllm_ascend.ops.triton.flash_attention_output import flash_attention_gate as triton_gate

        return triton_gate(projected, gate)
    return projected.mul_(torch.sigmoid(gate))


def register_flash_attention_gate_warmup(dtype: torch.dtype, hidden: int, gate_prefix: int = 0) -> None:
    if supports_triton():
        from vllm_ascend.ops.triton.flash_attention_output import register_flash_attention_gate_warmup as register

        register(dtype, hidden, gate_prefix=gate_prefix)
