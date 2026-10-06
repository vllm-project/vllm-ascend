# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_next_token_ids_kernel(
    sampled_token_ids_ptr,
    backup_next_token_ids_ptr,
    discard_request_indices_ptr,
    next_token_ids_ptr,
    valid_sampled_tokens_count_ptr,
    sampled_row_stride,
    sampled_col_stride,
    num_discarded_requests,
    VOCAB_SIZE: tl.constexpr,
    MAX_NEW_TOKENS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DISCARDS: tl.constexpr,
):
    req_idx = tl.program_id(0)

    token_offsets = tl.arange(0, BLOCK_TOKENS)
    token_mask = token_offsets < MAX_NEW_TOKENS
    sampled = tl.load(
        sampled_token_ids_ptr + req_idx * sampled_row_stride + token_offsets * sampled_col_stride,
        mask=token_mask,
        other=-1,
    )
    valid = token_mask & (sampled != -1) & (sampled < VOCAB_SIZE)
    valid_count = tl.sum(valid.to(tl.int32), axis=0)

    discard_offsets = tl.arange(0, BLOCK_DISCARDS)
    discard_mask = discard_offsets < num_discarded_requests
    discarded_indices = tl.load(
        discard_request_indices_ptr + discard_offsets,
        mask=discard_mask,
        other=-1,
    )
    is_discarded = tl.sum(((discarded_indices == req_idx) & discard_mask).to(tl.int32), axis=0) > 0
    valid_count = tl.where(is_discarded, 0, valid_count)

    # Match the reference exactly: valid tokens form a prefix, and the selected
    # token is sampled[valid_count - 1]. The masked reduction avoids a dynamic
    # scalar load and keeps the whole operation in one kernel.
    selected_offset = tl.maximum(valid_count - 1, 0)
    selected_token = tl.sum(
        tl.where(token_offsets == selected_offset, sampled, 0),
        axis=0,
    )
    backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
    next_token = tl.where(valid_count > 0, selected_token, backup_token)

    tl.store(next_token_ids_ptr + req_idx, next_token)
    tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)


def prepare_next_token_ids(
    sampled_token_ids: torch.Tensor,
    backup_next_token_ids: torch.Tensor,
    discard_request_indices: torch.Tensor,
    num_discarded_requests: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse speculative-token filtering, counting, and next-token selection."""
    if sampled_token_ids.ndim != 2:
        raise ValueError("sampled_token_ids must be a 2D tensor")

    num_reqs, max_new_tokens = sampled_token_ids.shape
    next_token_ids = torch.empty(
        num_reqs,
        dtype=sampled_token_ids.dtype,
        device=sampled_token_ids.device,
    )
    valid_sampled_tokens_count = torch.empty(
        num_reqs,
        dtype=torch.int64,
        device=sampled_token_ids.device,
    )
    block_tokens = triton.next_power_of_2(max(1, max_new_tokens))
    block_discards = triton.next_power_of_2(max(1, discard_request_indices.numel()))
    _prepare_next_token_ids_kernel[(num_reqs,)](
        sampled_token_ids,
        backup_next_token_ids,
        discard_request_indices,
        next_token_ids,
        valid_sampled_tokens_count,
        sampled_token_ids.stride(0),
        sampled_token_ids.stride(1),
        num_discarded_requests,
        VOCAB_SIZE=vocab_size,
        MAX_NEW_TOKENS=max_new_tokens,
        BLOCK_TOKENS=block_tokens,
        BLOCK_DISCARDS=block_discards,
    )
    return next_token_ids, valid_sampled_tokens_count
