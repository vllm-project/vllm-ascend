# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""PLE residual replay after convolution and SiLU have already completed."""

import torch


def ple_residual_bf16(gated, conv_output, outer):
    """Return BF16(BF16(gated + conv_output) + outer), without mutation.

    conv_output is the already-activated BF16 convolution output. Do not pass
    the pre-activation accumulator or a result that already includes gated.
    Keep this eager: the intermediate tensor is a required memory boundary.
    """
    tensors = (gated, conv_output, outer)
    if any(t.dtype != torch.bfloat16 or t.shape != gated.shape or t.device != gated.device for t in tensors):
        raise ValueError("All three inputs must have identical BF16 shape and device")
    first_add = (gated.float() + conv_output.float()).to(torch.bfloat16)
    return (first_add.float() + outer.float()).to(torch.bfloat16)
