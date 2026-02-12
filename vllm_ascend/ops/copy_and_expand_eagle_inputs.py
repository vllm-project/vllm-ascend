from typing import Tuple

import torch

from vllm_ascend.utils import enable_custom_op


def npu_copy_and_expand_eagle_inputs(
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    next_token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_end_loc: torch.Tensor,
    padding_token_id: int,
    parallel_drafting_token_id: int,
    num_padding_slots_per_request: int,
    shift_input_ids: bool,
    total_draft_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Call the CopyAndExpandEagleInputs custom Ascend C operator.

    Args:
        target_token_ids: [total_input_tokens], int32
        target_positions: [total_input_tokens], int32
        next_token_ids: [num_reqs], int32
        query_start_loc: [num_reqs + 1], int32
        query_end_loc: [num_reqs], int32
        padding_token_id: padding token id
        parallel_drafting_token_id: parallel drafting token id
        num_padding_slots_per_request: number of padding slots per request
        shift_input_ids: whether to shift input ids
        total_draft_tokens: total number of output draft tokens

    Returns:
        Tuple of (out_input_ids, out_positions, out_is_rejected_token_mask,
                  out_is_masked_token_mask, out_new_token_indices,
                  out_hidden_state_mapping)
    """
    if not enable_custom_op():
        raise RuntimeError(
            "npu_copy_and_expand_eagle_inputs requires custom op to be "
            "enabled. Please call enable_custom_op() first.")

    return torch.ops._C_ascend.npu_copy_and_expand_eagle_inputs(
        target_token_ids,
        target_positions,
        next_token_ids,
        query_start_loc,
        query_end_loc,
        padding_token_id,
        parallel_drafting_token_id,
        num_padding_slots_per_request,
        shift_input_ids,
        total_draft_tokens,
    )
