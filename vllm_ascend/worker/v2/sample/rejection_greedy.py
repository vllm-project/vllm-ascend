# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Greedy verification of draft tokens, as plain PyTorch.

Hardware- and runner-neutral: the Triton variant used on standard hardware lives
in ``vllm_ascend.ops.triton.reject_sample``, while these PyTorch ones are the
fallback for 310P MRv2 and for the MRv1 rejection sampler. They live under the
MRv2 worker because MRv1 is scheduled for removal and must not own shared code.
"""

import torch


def rejection_greedy_sample_spec_len_1_pytorch(
    output_token_ids,  # [batch_size, 2]
    draft_token_ids,  # [num_tokens]
    target_argmax,  # [num_tokens]
    bonus_token_ids,  # [batch_size]
    uniform_probs=None,
    synthetic_conditional_rates=None,
    synthetic_mode=False,
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    assert batch_size == num_tokens
    if synthetic_mode:
        assert uniform_probs is not None and synthetic_conditional_rates is not None
        # spec_len == 1 => only position 0. Accept the draft token with prob
        # rate[0]; on accept emit the draft token (pos 0) + bonus (pos 1),
        # otherwise emit target_argmax (pos 0).
        rate = synthetic_conditional_rates[0]
        accept_req_mask = (uniform_probs < rate) & (draft_token_ids >= 0)
        output_token_ids[:, 0] = torch.where(accept_req_mask, draft_token_ids, target_argmax)
    else:
        accept_req_mask = draft_token_ids == target_argmax
        output_token_ids[:, 0] = target_argmax
    bonus_token_ids = bonus_token_ids.squeeze(1)
    output_token_ids[:, 1] = torch.where(accept_req_mask, bonus_token_ids, output_token_ids[:, 1])


def rejection_greedy_sample_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    target_argmax,  # [num_tokens]
    bonus_token_ids,  # [batch_size]
    draft_tokens_per_req,  # [batch_size], list
    max_spec_len,
    is_greedy=None,  # [batch_size] or None
    uniform_probs=None,
    synthetic_conditional_rates=None,
    synthetic_mode=False,
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    device = output_token_ids.device
    draft_tokens_per_req = torch.tensor(draft_tokens_per_req).to(device, non_blocking=True)
    if is_greedy is None:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)

    start_indices = cu_num_draft_tokens - draft_tokens_per_req
    req_ids = torch.arange(batch_size, device=device)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req)
    token_positions = torch.arange(num_tokens, device=device) - start_indices[token_req_ids]

    # Find the first mismatch position of each request.
    if synthetic_mode:
        assert uniform_probs is not None and synthetic_conditional_rates is not None
        # Synthetic: accept draft token i with prob conditional_rates[i],
        # regardless of target match. First rejection (= first mismatch) is the
        # first position not accepted.
        rates_per_token = synthetic_conditional_rates[token_positions]
        accept_per_token = (uniform_probs < rates_per_token) & (draft_token_ids >= 0)
        mismatch_global = ~accept_per_token
    else:
        mismatch_global = draft_token_ids != target_argmax
    if max_spec_len == 0:
        first_mismatch_pos_per_req = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        # [bs, max_spec_len]
        pos_matrix = torch.full((batch_size, max_spec_len), -1, dtype=torch.long, device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        mismatch_matrix = torch.full((batch_size, max_spec_len), False, dtype=torch.bool, device=device)
        mismatch_matrix[token_req_ids, token_positions] = mismatch_global
        mismatch_positions = torch.where(mismatch_matrix, pos_matrix, max_spec_len * 2)
        first_mismatch_pos_per_req, _ = torch.min(mismatch_positions, dim=1)
        no_mismatch_mask = first_mismatch_pos_per_req == max_spec_len * 2
        first_mismatch_pos_per_req[no_mismatch_mask] = draft_tokens_per_req[no_mismatch_mask]

    # Copy matched target tokens into output.
    copy_len = torch.minimum(first_mismatch_pos_per_req + 1, draft_tokens_per_req)
    copy_indices = torch.arange(max_spec_len + 1, device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    greedy_mask = is_greedy.unsqueeze(1)
    final_copy_mask = copy_mask & greedy_mask
    global_idx = start_indices.unsqueeze(1) + copy_indices
    if synthetic_mode:
        # Accepted positions emit the draft token; the first rejected position
        # emits target_argmax (greedy recovery).
        copy_tokens = torch.where(accept_per_token, draft_token_ids, target_argmax)
    else:
        copy_tokens = target_argmax
    output_token_ids[final_copy_mask] = copy_tokens[global_idx[final_copy_mask]].to(output_token_ids.dtype)
    # Fill bonus token.
    needs_bonus = is_greedy & (first_mismatch_pos_per_req >= draft_tokens_per_req)
    if torch.any(needs_bonus):
        bonus_rows = torch.where(needs_bonus)[0]
        bonus_cols = draft_tokens_per_req[bonus_rows]
        bonus_token_ids = bonus_token_ids.squeeze(1)
        output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids[bonus_rows]
