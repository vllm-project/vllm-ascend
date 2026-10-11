# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-launch mask preparation for slotless Engram graphs."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _engram_prepare_masks(
    ids,
    lookback,
    valid_count,
    dead,
    keep,
    lookback_dead,
    tokens,
    output_tokens,
    lookback_elements,
    ids_stride,
    lookback_row_stride,
    lookback_col_stride,
    DEPTH: tl.constexpr,
    IMAGE_ID: tl.constexpr,
    IMAGE_PAD_ID: tl.constexpr,
    HAS_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(ids + i * ids_stride, i < tokens, other=0)
    is_dead = (token == IMAGE_ID) | (token == IMAGE_PAD_ID)
    if HAS_COUNT:
        count = tl.load(valid_count)
        is_dead |= i >= count
    tl.store(dead + i, is_dead, i < tokens)
    tl.store(keep + i, (i < tokens) & ~is_dead, i < output_tokens)
    row, col = i // DEPTH, i % DEPTH
    prior = tl.load(
        lookback + row * lookback_row_stride + col * lookback_col_stride,
        i < lookback_elements,
        other=-1,
    )
    tl.store(lookback_dead + i, (prior == IMAGE_ID) | (prior == IMAGE_PAD_ID), i < lookback_elements)


def prepare_engram_masks(
    input_ids: torch.Tensor,
    lookback_token_ids: torch.Tensor,
    *,
    image_token_id: int,
    image_pad_token_id: int,
    valid_token_count: torch.Tensor | None,
    mask_output_buffer: torch.Tensor,
    output_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write the mask directly, retaining replay-varying count and sentinels."""
    if input_ids.ndim != 1 or lookback_token_ids.ndim != 2:
        raise ValueError("Engram mask inputs require token vector and lookback matrix")
    if input_ids.device != lookback_token_ids.device or input_ids.device != mask_output_buffer.device:
        raise ValueError("Engram mask tensors must share a device")
    if (
        mask_output_buffer.dtype != torch.bool
        or mask_output_buffer.ndim != 1
        or not mask_output_buffer.is_contiguous()
        or not input_ids.numel() <= output_tokens <= mask_output_buffer.numel()
    ):
        raise ValueError("Invalid Engram mask output")
    if valid_token_count is not None and (
        valid_token_count.numel() != 1 or valid_token_count.device != input_ids.device
    ):
        raise ValueError("Engram valid count must be a scalar on the input device")
    dead = torch.empty(input_ids.shape, dtype=torch.bool, device=input_ids.device)
    lookback_dead = torch.empty(lookback_token_ids.shape, dtype=torch.bool, device=input_ids.device)
    work = max(output_tokens, lookback_token_ids.numel())
    if work:
        _engram_prepare_masks[(triton.cdiv(work, 256),)](
            input_ids,
            lookback_token_ids,
            valid_token_count,
            dead,
            mask_output_buffer,
            lookback_dead,
            input_ids.numel(),
            output_tokens,
            lookback_token_ids.numel(),
            input_ids.stride(0),
            lookback_token_ids.stride(0),
            lookback_token_ids.stride(1),
            DEPTH=max(1, lookback_token_ids.shape[1]),
            IMAGE_ID=image_token_id,
            IMAGE_PAD_ID=image_pad_token_id,
            HAS_COUNT=valid_token_count is not None,
            BLOCK=256,
        )
    return dead, mask_output_buffer[: input_ids.numel()], lookback_dead
