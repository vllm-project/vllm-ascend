#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# PyTorch reference implementations of the rejection sampling steps. They back
# the Triton kernel equivalence tests; the serving path does not call them.

import torch
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID


def rejection_random_sample_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
    IS_NGRAM=False,
    ENTROPY_VERIFY=False,
    POSTERIOR_THRESHOLD=0.95,
    POSTERIOR_ALPHA=0.4,
    EPSILON=1e-10,
    ori_target_probs=None,
    synthetic_mode=False,
    synthetic_conditional_rates=None,
):
    """
    This function implements the Speculative Decoding rejection sampling step.
    Instead of looping through each request and each token (which causes high
    overhead), it uses a fully vectorized approach:

    1.  **Index Mapping**: Converts the flattened 1D token arrays into a 2D
        [batch_size, max_draft_len] grid using 'cu_num_draft_tokens' to handle
        variable-length sequences in the batch.
    2.  **Parallel Validation**: Calculates the acceptance condition
        (target_prob / draft_prob >= uniform_sample) for ALL draft tokens
        simultaneously across the entire batch.
    3.  **Short-circuit Simulation**: In the loop version, once a token is rejected,
        subsequent tokens are ignored. Here, we simulate this by finding the
        'first_reject_pos' using argmax on the rejection mask and creating a
        'should_skip' mask for all indices after the first failure.
    4.  **Token Selection**: Uses 'torch.where' to select:
        - Draft tokens (if accepted)
        - Recovered tokens (at the point of first rejection)
        - Bonus tokens (if all tokens in a sequence were accepted)
    5.  **Masking**: Ensures operations only apply to non-greedy requests and
        within valid sequence lengths.
    """

    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    zero_cpu = torch.tensor([0], pin_memory=True)
    zero_device = zero_cpu.to(device, non_blocking=True)

    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = cu_end - cu_start

    max_draft_len = max_spec_len
    pos_indices_cpu = torch.arange(max_draft_len, pin_memory=True)
    pos_indices = pos_indices_cpu.to(device, non_blocking=True)[None, :]

    valid_mask = pos_indices < num_draft_per_batch[:, None]
    global_token_indices = cu_start[:, None] + pos_indices
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_token_indices]  # [batch_size, max_draft_len]
    placeholder_mask = draft_tokens == PLACEHOLDER_TOKEN_ID
    safe_draft_tokens = draft_tokens.masked_fill(placeholder_mask, 0)

    if IS_NGRAM:
        ones_cpu = torch.ones(1, pin_memory=True, dtype=torch.float32)
        draft_token_probs = ones_cpu.to(device, non_blocking=True).expand_as(draft_tokens)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = safe_draft_tokens.flatten()
        flat_draft_probs = draft_probs[flat_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_draft_len)

    # Get target token probs
    flat_indices = global_token_indices.flatten()
    flat_draft_tokens = safe_draft_tokens.flatten()
    flat_target_probs = target_probs[flat_indices, flat_draft_tokens]
    target_token_probs = flat_target_probs.view(batch_size, max_draft_len)

    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    zero_threshold_cpu = torch.tensor([0.0], pin_memory=True, dtype=torch.float32)
    zero_threshold = zero_threshold_cpu.to(device, non_blocking=True)

    if synthetic_mode:
        assert synthetic_conditional_rates is not None
        # Synthetic: accept draft token with per-position rate, bypassing the
        # draft/target prob comparison and the entropy threshold.
        rates = synthetic_conditional_rates[pos_indices]
        acceptance_condition = uniform_token_probs < rates
    elif ENTROPY_VERIFY:
        entropy_probs = ori_target_probs if ori_target_probs is not None else target_probs
        all_target_dist = entropy_probs[global_token_indices]
        entropy = -(all_target_dist * torch.log(all_target_dist + EPSILON)).sum(dim=-1)
        exp_neg_entropy = torch.exp(-entropy * POSTERIOR_ALPHA)
        posterior_threshold_device = torch.tensor(POSTERIOR_THRESHOLD, device=device, dtype=torch.float32)
        threshold = torch.minimum(exp_neg_entropy, posterior_threshold_device)
        modified_uniform_token_probs = threshold * uniform_token_probs
        acceptance_condition = (draft_token_probs > zero_threshold) & (
            target_token_probs / draft_token_probs >= modified_uniform_token_probs
        )
    else:
        acceptance_condition = (draft_token_probs > zero_threshold) & (
            target_token_probs / draft_token_probs >= uniform_token_probs
        )
    acceptance_condition = acceptance_condition & (~placeholder_mask)

    first_rejection = (~acceptance_condition) & valid_mask

    default_pos_cpu = torch.full([batch_size, 1], max_draft_len, pin_memory=True)
    default_pos = default_pos_cpu.to(device, non_blocking=True)

    first_reject_pos = torch.where(
        first_rejection.any(dim=1, keepdim=True), first_rejection.float().argmax(dim=1, keepdim=True), default_pos
    )
    pos_mask = pos_indices >= first_reject_pos
    should_skip = pos_mask & valid_mask

    final_acceptance = acceptance_condition & (~should_skip)
    non_greedy_mask = ~is_greedy
    update_mask = non_greedy_mask[:, None] & valid_mask & (~should_skip)

    first_reject_mask = (pos_indices == first_reject_pos) & valid_mask & non_greedy_mask[:, None]
    final_update_mask = update_mask | first_reject_mask
    final_tokens = torch.where(
        first_reject_mask,
        recovered_tokens,
        torch.where(final_acceptance, draft_tokens, output_token_ids[:, :max_draft_len]),
    )

    output_token_ids[:, :max_draft_len] = torch.where(
        final_update_mask, final_tokens, output_token_ids[:, :max_draft_len]
    )

    no_rejection = first_reject_pos.squeeze(1) >= num_draft_per_batch
    should_add_bonus = non_greedy_mask & no_rejection

    bonus_positions = num_draft_per_batch  # [batch_size]

    seq_len = output_token_ids.shape[1]
    all_positions_cpu = torch.arange(seq_len, pin_memory=True)
    all_positions = all_positions_cpu.to(device, non_blocking=True)[None, :]  # [1, seq_len]

    batch_bonus_positions = bonus_positions[:, None]  # [batch_size, 1]

    max_spec_len_cpu = torch.tensor([max_spec_len], pin_memory=True)
    max_spec_len_device = max_spec_len_cpu.to(device, non_blocking=True)

    valid_bonus_pos = bonus_positions < (max_spec_len_device + 1)
    final_bonus_mask = should_add_bonus & valid_bonus_pos

    bonus_pos_match = all_positions == batch_bonus_positions
    bonus_pos_mask = bonus_pos_match & final_bonus_mask[:, None]

    bonus_values_expanded = bonus_token_ids.view(-1, 1).expand(-1, seq_len)
    output_token_ids[:] = torch.where(bonus_pos_mask, bonus_values_expanded, output_token_ids)


def expand_pytorch(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS,
):
    """
    This function broadcasts batch-level values (input_ptr) to token-level
    positions (output_ptr) based on cumulative token offsets. It acts like
    a "scatter" or "repeat_interleave" operation but with custom logic:

    1.  **Range Broadcasting**: It creates a boolean matrix 'in_range' of size
        [num_tokens, batch_size] that identifies which batch index each token
        belongs to by checking if the token index falls between cu_start and cu_end.
    2.  **Conditional Replacement**: Before expansion, it replaces specific values
        (e.g., padding or special markers) in the input to prepare the data.
    3.  **Matrix-based Mapping**: It uses 'torch.einsum' to perform a weighted
        sum that effectively "picks" the correct batch value for every token position
        simultaneously, avoiding a Python loop over the batch.
    """
    device = cu_num_tokens_ptr.device
    batch_size = input_ptr.shape[0]
    num_tokens = output_ptr.shape[0]

    if batch_size == 0 or num_tokens == 0:
        return

    cu_start = torch.cat([torch.tensor([0], pin_memory=True).to(device, non_blocking=True), cu_num_tokens_ptr[:-1]])
    cu_end = cu_num_tokens_ptr

    token_indices = torch.arange(num_tokens, device=device)[:, None]  # [num_tokens, 1]
    cu_start_exp = cu_start[None, :]  # [1, batch_size]
    cu_end_exp = cu_end[None, :]  # [1, batch_size]

    in_range = (token_indices >= cu_start_exp) & (token_indices < cu_end_exp)

    replaced_input = torch.where(input_ptr == replace_from, replace_to, input_ptr).float()

    token_values = torch.einsum("tb,b->t", in_range.float(), replaced_input)

    needs_update = in_range.any(dim=1)

    output_ptr[:] = torch.where(needs_update, token_values, output_ptr)


def sample_recovered_tokens_pytorch(
    output_token_ids,  # [num_tokens]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    q,  # [batch_size, vocab_size]
    vocab_size,
    IS_NGRAM=False,
):
    """
    When a draft token is rejected, we must sample a "recovered" token from
    a modified distribution. This function calculates that distribution across
    the entire flattened batch.

    1.  **Token-to-Batch Mapping**: Using the cumulative draft token counts, it
        determines which request in the batch each token belongs to. This is
        necessary because 'q' (normalization factor) is stored per-request.
    2.  **Probability Adjustment**:
        - If N-GRAM: It zeroes out the draft token's probability in the target.
        - If Probabilistic: It calculates max(0, target_probs - draft_probs)
          as per the standard speculative decoding algorithm.
    3.  **Normalization & Sampling**: It divides the adjusted probabilities
        by the normalization distribution 'q'. To remain vectorized, it
        broadcasts 'q' from [batch_size, vocab] to [num_tokens, vocab].
    4.  **Argmax Selection**: It selects the best recovery token for every
        position in one pass using torch.argmax.
    """
    device = output_token_ids.device
    num_tokens = output_token_ids.shape[0]

    if num_tokens == 0:
        return

    cu_start = torch.cat(
        [
            torch.tensor([0], pin_memory=True).to(device, non_blocking=True),
            cu_num_draft_tokens[:-1],
        ]
    )
    cu_end = cu_num_draft_tokens

    token_indices = torch.arange(num_tokens, device=device)  # [num_tokens]

    token_indices_expanded = token_indices[:, None]  # [num_tokens, 1]
    cu_start_expanded = cu_start[None, :]  # [1, batch_size]
    cu_end_expanded = cu_end[None, :]  # [1, batch_size]

    in_range_mask = (token_indices_expanded >= cu_start_expanded) & (token_indices_expanded < cu_end_expanded)

    token_to_batch = torch.argmax(in_range_mask.int(), dim=1)

    has_match = in_range_mask.any(dim=1)
    token_to_batch = torch.where(has_match, token_to_batch, 0)

    if IS_NGRAM:
        token_indices = torch.arange(num_tokens, device=device)

        modified_target_probs = target_probs.clone()
        valid_draft_mask = draft_token_ids != PLACEHOLDER_TOKEN_ID
        modified_target_probs[
            token_indices[valid_draft_mask],
            draft_token_ids[valid_draft_mask],
        ] = 0
        prob = modified_target_probs

    else:
        prob = torch.maximum(
            target_probs - draft_probs,
            torch.tensor(0.0, pin_memory=True).to(device, non_blocking=True),
        )

    q_values = q[token_to_batch]  # [num_tokens, vocab_size]

    epsilon = 1e-10
    q_values_safe = torch.where(q_values == 0, epsilon, q_values)
    q_values_safe = torch.where(torch.isinf(q_values), epsilon, q_values_safe)

    prob_over_q = prob / q_values_safe

    prob_over_q = torch.where((q_values == 0) | torch.isinf(q_values), -1e10, prob_over_q)

    recovered_ids = torch.argmax(prob_over_q, dim=1)
    output_token_ids[:] = recovered_ids


def rejection_random_sample_block_verify_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    bonus_token_ids,  # [batch_size]
    recovered_token_ids,  # [num_tokens]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
    IS_NGRAM=False,
    ENTROPY_VERIFY=False,
    POSTERIOR_THRESHOLD=0.95,
    POSTERIOR_ALPHA=0.4,
    EPSILON=1e-10,
    ori_target_probs=None,
):
    batch_size = output_token_ids.shape[0]
    device = output_token_ids.device

    zero_cpu = torch.tensor([0], pin_memory=True)
    zero_device = zero_cpu.to(device, non_blocking=True)

    cu_start = torch.cat([zero_device, cu_num_draft_tokens[:-1]])
    cu_end = cu_num_draft_tokens
    num_draft_per_batch = (cu_end - cu_start)[:, None]
    pos_indices_cpu = torch.arange(max_spec_len, pin_memory=True)
    pos_indices = pos_indices_cpu.to(device, non_blocking=True)[None, :]
    valid_mask = pos_indices < num_draft_per_batch
    global_token_indices = cu_start[:, None] + pos_indices
    global_token_indices = global_token_indices.clamp(0, draft_token_ids.shape[0] - 1)
    draft_tokens = draft_token_ids[global_token_indices]
    placeholder_mask = draft_tokens == PLACEHOLDER_TOKEN_ID
    safe_draft_tokens = draft_tokens.masked_fill(placeholder_mask, 0)

    if IS_NGRAM:
        ones_cpu = torch.ones(1, pin_memory=True, dtype=torch.float32)
        draft_token_probs = ones_cpu.to(device, non_blocking=True).expand_as(draft_tokens)
    else:
        flat_indices = global_token_indices.flatten()
        flat_draft_tokens = safe_draft_tokens.flatten()
        flat_draft_probs = draft_probs[flat_indices, flat_draft_tokens]
        draft_token_probs = flat_draft_probs.view(batch_size, max_spec_len)

    # Get target token probs
    flat_indices = global_token_indices.flatten()
    flat_draft_tokens = safe_draft_tokens.flatten()
    flat_target_probs = target_probs[flat_indices, flat_draft_tokens]
    target_token_probs = flat_target_probs.view(batch_size, max_spec_len)

    uniform_token_probs = uniform_probs[global_token_indices]
    recovered_tokens = recovered_token_ids[global_token_indices]

    pi = target_token_probs / draft_token_probs
    pi = pi.clamp(max=1.0)
    pi = torch.cumprod(pi, dim=-1)
    cum_uniform_token_probs = torch.cumprod(uniform_token_probs, dim=-1)

    if ENTROPY_VERIFY:
        entropy_probs = ori_target_probs if ori_target_probs is not None else target_probs
        all_target_dist = entropy_probs[global_token_indices]
        entropy = -(all_target_dist * torch.log(all_target_dist + EPSILON)).sum(dim=-1)
        exp_neg_entropy = torch.exp(-entropy * POSTERIOR_ALPHA)
        posterior_threshold_device = torch.tensor(POSTERIOR_THRESHOLD, device=device, dtype=torch.float32)
        threshold = torch.minimum(exp_neg_entropy, posterior_threshold_device)
        modified_cum_uniform_token_probs = threshold * cum_uniform_token_probs
        legal_mask = (draft_token_probs > 0) & (pi >= modified_cum_uniform_token_probs)
    else:
        legal_mask = (draft_token_probs > 0) & (pi >= cum_uniform_token_probs)
    legal_mask = legal_mask & valid_mask & (~placeholder_mask)

    last_accept_pos = torch.where(
        legal_mask.any(dim=-1, keepdim=True),
        (max_spec_len - legal_mask.flip(dims=[-1]).float().argmax(dim=-1, keepdim=True) - 1),
        -1,
    )
    non_greedy_mask = (~is_greedy)[:, None]

    accept_mask = (pos_indices <= last_accept_pos) & valid_mask & non_greedy_mask
    output_token_ids[:, :max_spec_len] = torch.where(accept_mask, draft_tokens, output_token_ids[:, :max_spec_len])

    reject_mask = (pos_indices == last_accept_pos + 1) & valid_mask & non_greedy_mask
    output_token_ids[:, :max_spec_len] = torch.where(reject_mask, recovered_tokens, output_token_ids[:, :max_spec_len])

    bonus_mask = (last_accept_pos + 1 >= num_draft_per_batch) & non_greedy_mask
    all_positions_cpu = torch.arange(max_spec_len + 1, pin_memory=True)
    all_positions = all_positions_cpu.to(device, non_blocking=True)[None, :]
    bonus_pos_match = all_positions == num_draft_per_batch
    bonus_mask = bonus_mask & bonus_pos_match
    bonus_values_expanded = bonus_token_ids.view(-1, 1).expand(-1, max_spec_len + 1)
    output_token_ids[:] = torch.where(bonus_mask, bonus_values_expanded, output_token_ids)


def sample_recovered_tokens_blockwise_pytorch(
    output_token_ids,  # [num_tokens]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    q,  # [batch_size, vocab_size]
    vocab_size,
    IS_NGRAM=False,
):
    _ = vocab_size
    device = output_token_ids.device
    num_tokens = output_token_ids.shape[0]
    batch_size = cu_num_draft_tokens.shape[0]

    if num_tokens == 0:
        return

    cu_start = torch.cat(
        [
            torch.tensor([0], pin_memory=True).to(device, non_blocking=True),
            cu_num_draft_tokens[:-1],
        ]
    )
    cu_end = cu_num_draft_tokens

    token_indices = torch.arange(num_tokens, device=device)
    in_range_mask = (token_indices[:, None] >= cu_start[None, :]) & (token_indices[:, None] < cu_end[None, :])
    token_to_batch = torch.argmax(in_range_mask.int(), dim=1)
    token_to_batch = torch.where(in_range_mask.any(dim=1), token_to_batch, torch.zeros_like(token_to_batch))
    pos_in_seq = token_indices - cu_start[token_to_batch]

    max_spec_len = int((cu_end - cu_start).max().item())

    if IS_NGRAM:
        draft_token_scalar_probs = torch.ones(num_tokens, device=device, dtype=torch.float32)
    else:
        valid_draft_mask = draft_token_ids != PLACEHOLDER_TOKEN_ID
        safe_draft_token_ids = draft_token_ids.masked_fill(~valid_draft_mask, 0)
        draft_token_scalar_probs = draft_probs[token_indices, safe_draft_token_ids]
        draft_token_scalar_probs = torch.where(
            valid_draft_mask,
            draft_token_scalar_probs,
            torch.zeros_like(draft_token_scalar_probs),
        )

    # Get target probability for each draft token
    valid_draft_mask = draft_token_ids != PLACEHOLDER_TOKEN_ID
    safe_draft_token_ids = draft_token_ids.masked_fill(~valid_draft_mask, 0)
    target_token_scalar_probs = target_probs[token_indices, safe_draft_token_ids]
    target_token_scalar_probs = torch.where(
        valid_draft_mask,
        target_token_scalar_probs,
        torch.zeros_like(target_token_scalar_probs),
    )

    per_token_ratio = torch.where(
        draft_token_scalar_probs > 0,
        target_token_scalar_probs / draft_token_scalar_probs.clamp(min=1e-10),
        torch.zeros_like(target_token_scalar_probs),
    )

    ratio_grid = torch.ones(batch_size, max_spec_len, device=device, dtype=torch.float32)
    ratio_grid[token_to_batch, pos_in_seq] = per_token_ratio

    p_prefix = torch.ones(batch_size, max_spec_len + 1, device=device, dtype=torch.float32)
    for k in range(max_spec_len):
        p_prefix[:, k + 1] = (p_prefix[:, k] * ratio_grid[:, k]).clamp(max=1.0)

    p_i = p_prefix[token_to_batch, pos_in_seq]
    p_i_expanded = p_i[:, None]

    if IS_NGRAM:
        modified_target = target_probs.clone()
        valid_draft_mask = draft_token_ids != PLACEHOLDER_TOKEN_ID
        modified_target[
            token_indices[valid_draft_mask],
            draft_token_ids[valid_draft_mask],
        ] = 0.0
        residual = torch.clamp(p_i_expanded * modified_target, min=0.0)
    else:
        residual = torch.clamp(p_i_expanded * target_probs - draft_probs, min=0.0)

    q_values = q[token_to_batch]
    epsilon = 1e-10
    q_values_safe = torch.where(q_values == 0, epsilon, q_values)
    q_values_safe = torch.where(torch.isinf(q_values), epsilon, q_values_safe)
    prob_over_q = torch.where(
        (q_values == 0) | torch.isinf(q_values),
        torch.full_like(residual, -1e10),
        residual / q_values_safe,
    )

    output_token_ids[:] = torch.argmax(prob_over_q, dim=1)
