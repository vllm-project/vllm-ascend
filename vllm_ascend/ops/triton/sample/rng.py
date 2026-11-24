# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit
def spec_step_seed(seed, step):
    """Mix the draft step into the low 32 bits used by Ascend's RNG."""
    # Odd 32-bit multiplier gives every step a distinct mask, including zero
    # for step 0. The mask survives callers that convert the seed to int32.
    step_mask = step.to(tl.uint32) * 0x9E3779B9
    return seed ^ step_mask.to(tl.int64)
