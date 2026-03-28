# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# apply_all_penalties for AscendSampler - uses Triton-Ascend kernels.

import torch
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad

from vllm_ascend.ops.triton.penalty import apply_penalties_triton


def _convert_to_tensors(output_token_ids: list[list[int]], vocab_size: int, device: torch.device) -> torch.Tensor:
    """Convert output_token_ids (list of lists) to padded tensor."""
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    )
    return output_tokens_tensor.to(device, non_blocking=True)


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """Apply penalties to logits via Triton-Ascend."""
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    return apply_penalties_triton(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )
