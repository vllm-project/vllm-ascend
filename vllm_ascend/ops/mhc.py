# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference helpers for multi-stream hyper-connections."""

import torch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op

# The hardware profile is immutable for the lifetime of this process. Reading
# its capability does not load the extension or initialize an NPU context.
MHC_EXPAND_SUPPORTED = get_current_hardware_profile().supports(HardwareCapability.MHC_EXPAND)


def mhc_expand(x: torch.Tensor, mult: int) -> torch.Tensor:
    """Replicate ``[tokens, hidden]`` into contiguous ``[tokens, mult, hidden]``."""
    if x.device.type == "npu" and not x.requires_grad and MHC_EXPAND_SUPPORTED and enable_custom_op():
        # Check tensor metadata within the C++ call to avoid repeated Python
        # dispatch. None leaves native aliasing and autograd behavior intact.
        result = torch.ops._C_ascend.npu_mhc_expand_if_supported(x, mult)
        if result is not None:
            return result
    return x.unsqueeze(1).expand(-1, mult, -1).contiguous()
