# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend backport of vLLM #58784's draft gather for MRV2 verification."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_draft_sampled_kernel(
    draft_sampled_ptr,
    pos_ptr,
    input_ids_ptr,
    positions_ptr,
    logits_indices_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    prefill_len_ptr,
    num_logits,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < num_logits
    token_idx = tl.load(logits_indices_ptr + block, mask=mask, other=0)
    token_id = tl.load(input_ids_ptr + token_idx, mask=mask, other=0)
    pos = tl.load(positions_ptr + token_idx, mask=mask, other=0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + block, mask=mask, other=0)
    local_pos = tl.load(expanded_local_pos_ptr + block, mask=mask, other=0)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx, mask=mask, other=0)
    # The drafter only proposes after the first token past prefill is sampled.
    # Draft slots on a step beginning in prefill are scheduler placeholders.
    is_placeholder = (local_pos > 0) & (pos - local_pos < prefill_len)
    tl.store(draft_sampled_ptr + block, tl.where(is_placeholder, -1, token_id), mask=mask)
    tl.store(pos_ptr + block, pos, mask=mask)


def gather_draft_sampled(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits_indices: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather verification rows without accepting unproposed draft slots."""
    num_logits = logits_indices.shape[0]
    draft_sampled = input_ids.new_empty(num_logits)
    pos = positions.new_empty(num_logits)
    block_size = 1024
    _gather_draft_sampled_kernel[(triton.cdiv(num_logits, block_size),)](
        draft_sampled,
        pos,
        input_ids,
        positions,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        prefill_len,
        num_logits,
        BLOCK_SIZE=block_size,
    )
    return draft_sampled, pos
