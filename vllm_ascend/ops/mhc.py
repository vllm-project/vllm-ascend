# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference helpers for multi-stream hyper-connections."""

import torch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op

MHC_EXPAND_ALIGNMENT = 16  # 32-byte DMA blocks for both supported 16-bit dtypes.
MHC_EXPAND_MULT = 4  # The GLM stream count covered by the performance comparison.


def mhc_expand(x: torch.Tensor, mult: int) -> torch.Tensor:
    """Replicate ``[tokens, hidden]`` into contiguous ``[tokens, mult, hidden]``."""
    if (
        x.device.type == "npu"
        and x.ndim == 2
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.is_contiguous()
        and x.shape[-1] % MHC_EXPAND_ALIGNMENT == 0
        and not x.requires_grad
        and x.numel() > 0
        and mult == MHC_EXPAND_MULT
        and get_current_hardware_profile().supports(HardwareCapability.MHC_EXPAND)
        and enable_custom_op()
    ):
        return torch.ops._C_ascend.npu_mhc_expand(x, mult)
    return x.unsqueeze(1).expand(-1, mult, -1).contiguous()
