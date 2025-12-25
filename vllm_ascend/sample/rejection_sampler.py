# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import generate_uniform_probs

from vllm_ascend.sample.sampler import apply_top_k_top_p

PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = -1
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 32

vectorcore_num = None
device_properties = None

if HAS_TRITON:
    from triton.runtime import driver  # type: ignore
    device_properties = driver.active.utils.get_device_properties(
        torch.npu.current_device())
    vectorcore_num = device_properties['num_vectorcore']
#get vector core number in order for later tiling


def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Process logits based on sampling metadata.

    This function applies temperature scaling to the logits,
    as well as top-k and top-p. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be processed.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Processed logits if non-greedy sampling is used,
        otherwise returns the original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    return apply_top_k_top_p(logits, top_k, top_p)


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # When num_speculative_tokens>=3, using block verify.
    using_block_verify = max_spec_len >= 3

    # Create output buffer.
    output_token_ids = torch.empty(
        (batch_size, max_spec_len + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    output_token_ids.fill_(PLACEHOLDER_TOKEN_ID)

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_probs.argmax(dim=-1)
        if HAS_TRITON:
            vec_len = batch_size
            n = cu_num_draft_tokens.numel()
            BLOCK_SIZE = 2
            grid = triton.cdiv(n, BLOCK_SIZE)
            if n >= vectorcore_num:
                grid = vectorcore_num  # Empirically tuned value
                BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(n, grid))

            if min(num_draft_tokens) == 1 and max(
                    num_draft_tokens) == 1 and sampling_metadata.all_greedy:
                rejection_greedy_sample_spec_len_1_triton[(grid, )](
                    output_token_ids,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                    vec_len,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
            else:
                rejection_greedy_sample_triton[(grid, )](
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                    is_greedy,
                    vec_len,
                    max_spec_len,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
        else:
            if min(num_draft_tokens) == 1 and max(
                    num_draft_tokens) == 1 and sampling_metadata.all_greedy:
                rejection_greedy_sample_spec_len_1_pytorch(
                    output_token_ids,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                )
            else:
                rejection_greedy_sample_pytorch(
                    output_token_ids,
                    cu_num_draft_tokens,
                    draft_token_ids,
                    target_argmax,
                    bonus_token_ids,
                    num_draft_tokens,
                    max_spec_len,
                    is_greedy,
                )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Generate uniform probabilities for rejection sampling.
    # [num_tokens]
    uniform_probs = generate_uniform_probs(
        num_tokens,
        num_draft_tokens,
        sampling_metadata.generators,
        device,
    )
    if not using_block_verify:
        # Sample recovered tokens for each position.
        # [num_tokens]
        recovered_token_ids = sample_recovered_tokens(
            max_spec_len,
            num_draft_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            sampling_metadata,
            device,
        )

        # Rejection sampling for random sampling requests.
        if HAS_TRITON:
            rejection_random_sample_kernel[(batch_size, )](
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                bonus_token_ids,
                recovered_token_ids,
                uniform_probs.to(torch.float32),
                is_greedy,
                max_spec_len,
                vocab_size,
                NO_DRAFT_PROBS=draft_probs is None,
            )
        else:
            rejection_random_sample_pytorch(
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                bonus_token_ids,
                recovered_token_ids,
                uniform_probs,
                is_greedy,
                max_spec_len,
                vocab_size,
                IS_NGRAM=draft_probs is None,
            )
    else:
        # MagicMTP: Improving acceptance rate with Block Verify.
        if HAS_TRITON:
            rejection_random_sample_block_verify_kernel[(batch_size, )](
                output_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                bonus_token_ids,
                uniform_probs.to(torch.float32),
                is_greedy,
                max_spec_len,
                vocab_size,
                NO_DRAFT_PROBS=draft_probs is None,
                multibuffer=True,
            )
        else:
            rejection_random_sample_block_verify_pytorch(output_token_ids,
                                                         cu_num_draft_tokens,
                                                         draft_token_ids,
                                                         draft_probs,
                                                         target_probs,
                                                         bonus_token_ids,
                                                         uniform_probs,
                                                         is_greedy,
                                                         max_spec_len,
                                                         vocab_size,
                                                         IS_NGRAM=draft_probs
                                                         is None)
    return output_token_ids


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    if HAS_TRITON:
        vec_len = batch_size
        n = cu_num_tokens.numel()
        BLOCK_SIZE = 2
        grid = triton.cdiv(n, BLOCK_SIZE)
        if n >= vectorcore_num:
            grid = vectorcore_num
            BLOCK_SIZE = triton.next_power_of_2(triton.cdiv(n, grid))

        expand_kernel[(grid, )](
            expanded_x,
            x,
            cu_num_tokens,
            replace_from,
            replace_to,
            vec_len,
            MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        expand_pytorch(
            expanded_x,
            x,
            cu_num_tokens,
            replace_from,
            replace_to,
            MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
        )
    return expanded_x


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    recovered_token_ids = torch.empty_like(draft_token_ids)
    if HAS_TRITON:
        sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            triton.next_power_of_2(vocab_size),
            NO_DRAFT_PROBS=draft_probs is None,
            SUB_BLOCK=4 * 1024,
            # TODO: enable multibuffer when accuracy problem is solved.
            multibuffer=False,
        )
    else:
        sample_recovered_tokens_pytorch(
            recovered_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            q,
            vocab_size,
            IS_NGRAM=draft_probs is None,
        )
    return recovered_token_ids


def rejection_greedy_sample_spec_len_1_pytorch(
        output_token_ids,  # [batch_size, 2]
        draft_token_ids,  # [num_tokens]
        target_argmax,  # [num_tokens]
        bonus_token_ids,  # [batch_size]
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    assert batch_size == num_tokens
    accept_req_mask = draft_token_ids == target_argmax
    output_token_ids[:, 0] = target_argmax
    bonus_token_ids = bonus_token_ids.squeeze(1)
    output_token_ids[:, 1] = torch.where(accept_req_mask, bonus_token_ids,
                                         output_token_ids[:, 1])


def rejection_greedy_sample_pytorch(
        output_token_ids,  # [batch_size, max_spec_len + 1]
        cu_num_draft_tokens,  # [batch_size]
        draft_token_ids,  # [num_tokens]
        target_argmax,  # [num_tokens]
        bonus_token_ids,  # [batch_size]
        draft_tokens_per_req,  # [batch_size], list
        max_spec_len,
        is_greedy=None,  # [batch_size] or None
):
    batch_size = output_token_ids.size(0)
    num_tokens = draft_token_ids.size(0)
    device = output_token_ids.device
    draft_tokens_per_req = torch.tensor(draft_tokens_per_req).to(
        device, non_blocking=True)
    if is_greedy is None:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)

    start_indices = cu_num_draft_tokens - draft_tokens_per_req
    req_ids = torch.arange(batch_size, device=device)
    token_req_ids = torch.repeat_interleave(req_ids, draft_tokens_per_req)
    token_positions = torch.arange(
        num_tokens, device=device) - start_indices[token_req_ids]

    # Find the first mismatch position of each request.
    mismatch_global = (draft_token_ids != target_argmax)
    if max_spec_len == 0:
        first_mismatch_pos_per_req = torch.zeros(batch_size,
                                                 dtype=torch.long,
                                                 device=device)
    else:
        # [bs, max_spec_len]
        pos_matrix = torch.full((batch_size, max_spec_len),
                                -1,
                                dtype=torch.long,
                                device=device)
        pos_matrix[token_req_ids, token_positions] = token_positions
        mismatch_matrix = torch.full((batch_size, max_spec_len),
                                     False,
                                     dtype=torch.bool,
                                     device=device)
        mismatch_matrix[token_req_ids, token_positions] = mismatch_global
        mismatch_positions = torch.where(mismatch_matrix, pos_matrix,
                                         max_spec_len * 2)
        first_mismatch_pos_per_req, _ = torch.min(mismatch_positions, dim=1)
        no_mismatch_mask = (first_mismatch_pos_per_req == max_spec_len * 2)
        first_mismatch_pos_per_req[no_mismatch_mask] = draft_tokens_per_req[
            no_mismatch_mask]

    # Copy matched target tokens into output.
    copy_len = torch.minimum(first_mismatch_pos_per_req + 1,
                             draft_tokens_per_req)
    copy_indices = torch.arange(max_spec_len + 1,
                                device=device).expand(batch_size, -1)
    copy_mask = copy_indices < copy_len.unsqueeze(1)
    greedy_mask = is_greedy.unsqueeze(1)
    final_copy_mask = copy_mask & greedy_mask
    global_idx = start_indices.unsqueeze(1) + copy_indices
    output_token_ids[final_copy_mask] = target_argmax[
        global_idx[final_copy_mask]].to(output_token_ids.dtype)
    # Fill bonus token.
    needs_bonus = is_greedy & (first_mismatch_pos_per_req
                               >= draft_tokens_per_req)
    if torch.any(needs_bonus):
        bonus_rows = torch.where(needs_bonus)[0]
        bonus_cols = draft_tokens_per_req[bonus_rows]
        bonus_token_ids = bonus_token_ids.squeeze(1)
        output_token_ids[bonus_rows, bonus_cols] = bonus_token_ids[bonus_rows]


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
):
    batch_size = output_token_ids.shape[0]

    for req_idx in range(batch_size):
        if is_greedy[req_idx]:
            continue

        if req_idx == 0:
            start_idx = 0
        else:
            start_idx = cu_num_draft_tokens[req_idx - 1].item()
        end_idx = cu_num_draft_tokens[req_idx].item()
        num_draft_tokens = end_idx - start_idx

        rejected = False
        for pos in range(num_draft_tokens):
            if not rejected:
                draft_token_id = draft_token_ids[start_idx + pos].item()

                if IS_NGRAM:
                    draft_prob = 1.0
                else:
                    draft_prob = draft_probs[start_idx + pos,
                                             draft_token_id].item()

                target_prob = target_probs[start_idx + pos,
                                           draft_token_id].item()
                uniform_prob = uniform_probs[start_idx + pos].item()

                if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                    token_id = draft_token_id
                else:
                    rejected = True
                    token_id = recovered_token_ids[start_idx + pos].item()

                output_token_ids[req_idx, pos] = token_id

        if not rejected:
            bonus_token_id = bonus_token_ids[req_idx].item()
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_id


def rejection_random_sample_block_verify_pytorch(
    output_token_ids,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens,  # [batch_size]
    draft_token_ids,  # [num_tokens]
    draft_probs,  # [num_tokens, vocab_size] or None
    target_probs,  # [num_tokens, vocab_size]
    bonus_token_ids,  # [batch_size]
    uniform_probs,  # [num_tokens]
    is_greedy,  # [batch_size]
    max_spec_len,
    vocab_size,
    IS_NGRAM=False,
):
    batch_size = output_token_ids.shape[0]

    for req_idx in range(batch_size):
        if is_greedy[req_idx]:
            continue

        if req_idx == 0:
            start_idx = 0
        else:
            start_idx = cu_num_draft_tokens[req_idx - 1].item()
        end_idx = cu_num_draft_tokens[req_idx].item()
        num_draft_tokens = end_idx - start_idx

        rejected = False
        pi = 1.0
        uniform_prob = 1.0
        last_accepted_token_pos = -1
        for pos in range(num_draft_tokens):
            draft_token_id = draft_token_ids[start_idx + pos].item()

            target_prob = target_probs[start_idx + pos, draft_token_id].item()
            uniform_prob = uniform_prob * uniform_probs[start_idx + pos].item()

            if IS_NGRAM:
                draft_prob = 1.0
            else:
                draft_prob = draft_probs[start_idx + pos,
                                         draft_token_id].item()

            pi = min(pi * target_prob / draft_prob, 1.0)

            if draft_prob > 0 and pi >= uniform_prob:
                last_accepted_token_pos = pos
                rejected = False
            else:
                rejected = True

        if last_accepted_token_pos > -1:
            for pos in range(last_accepted_token_pos + 1):
                draft_token_id = draft_token_ids[start_idx + pos].item()
                output_token_ids[req_idx, pos] = draft_token_id

        if rejected:
            recovered_token_id = torch.argmax(
                target_probs[start_idx + last_accepted_token_pos + 1]).item()
            output_token_ids[req_idx,
                             last_accepted_token_pos + 1] = recovered_token_id
        else:
            bonus_token_id = bonus_token_ids[req_idx].item()
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_id


def expand_pytorch(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS,
):
    batch_size = len(input_ptr)

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else cu_num_tokens_ptr[req_idx - 1]
        end_idx = cu_num_tokens_ptr[req_idx]
        num_tokens = end_idx - start_idx

        src_val = input_ptr[req_idx]
        src_val = replace_to if src_val == replace_from else src_val

        offset = torch.arange(MAX_NUM_TOKENS, device=num_tokens.device)
        mask = offset < num_tokens

        output_slice = start_idx + offset[mask]
        output_ptr[output_slice] = src_val


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
    batch_size = len(cu_num_draft_tokens)

    for req_idx in range(batch_size):
        start_idx = 0 if req_idx == 0 else cu_num_draft_tokens[req_idx - 1]
        end_idx = cu_num_draft_tokens[req_idx]
        num_draft_tokens = end_idx - start_idx

        for pos in range(num_draft_tokens):
            token_idx = start_idx + pos

            if IS_NGRAM:
                draft_token_id = draft_token_ids[token_idx]
                orig_prob = target_probs[token_idx, draft_token_id].item()
                target_probs[token_idx, draft_token_id] = 0
                prob = target_probs[token_idx].clone()
            else:
                draft_p = draft_probs[token_idx].clone()
                target_p = target_probs[token_idx].clone()
                prob = torch.maximum(target_p - draft_p,
                                     torch.tensor(0.0, device=target_p.device))

            q_values = torch.full((vocab_size, ),
                                  float('-inf'),
                                  device=q.device)
            q_values[:vocab_size] = q[req_idx, :vocab_size]

            recovered_id = torch.argmax(prob / q_values).item()
            output_token_ids[token_idx] = recovered_id

            if IS_NGRAM:
                target_probs[token_idx, draft_token_id] = orig_prob


@triton.jit(do_not_specialize=["max_spec_len"])
def bonus_renew_1(
    bonus_token_ids_ptr,
    position,
    output_token_ids_ptr,
):
    bonus_token_id = tl.load(bonus_token_ids_ptr + position)
    tl.store(output_token_ids_ptr + position * 2 + 1, bonus_token_id)


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_spec_len_1_triton(
    output_token_ids_ptr,  # [batch_size, 2]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,
    vec_len,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < vec_len

    draft_token_id = tl.load(draft_token_ids_ptr + offset, mask)
    target_argmax_id = tl.load(target_argmax_ptr + offset, mask)
    tl.store(output_token_ids_ptr + offset * 2, target_argmax_id, mask)

    for pos in tl.range(0, BLOCK_SIZE):
        draft_token_id1 = tl.get_element(draft_token_id, (pos, ))
        target_argmax1 = tl.get_element(target_argmax_id, (pos, ))
        position = block_idx * BLOCK_SIZE + pos
        if draft_token_id1 == target_argmax1:
            bonus_renew_1(
                bonus_token_ids_ptr,
                position,
                output_token_ids_ptr,
            )


@triton.jit(do_not_specialize=["max_spec_len"])
def bonus_renew(
    bonus_token_ids_ptr,
    position,
    output_token_ids_ptr,
    max_spec_len,
    num_tokens1,
):
    bonus_token_id = tl.load(bonus_token_ids_ptr + position)
    tl.store(
        output_token_ids_ptr + position * (max_spec_len + 1) + num_tokens1,
        bonus_token_id)


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_triton(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    vec_len,
    max_spec_len,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < vec_len

    if is_greedy_ptr is None:
        is_greedy_mask = mask
    else:
        is_greedy = tl.load(is_greedy_ptr + offset, mask=mask, other=0)
        is_greedy_mask = mask & (is_greedy != 0)

    start_idx = tl.where(
        offset == 0, 0,
        tl.load(cu_num_draft_tokens_ptr + offset - 1, is_greedy_mask))
    end_idx = tl.load(cu_num_draft_tokens_ptr + offset, is_greedy_mask)
    num_draft_tokens = end_idx - start_idx

    for pos in tl.range(0, BLOCK_SIZE):
        num_tokens1 = tl.get_element(num_draft_tokens, (pos, ))
        rejected = False
        start_idx1 = tl.get_element(start_idx, (pos, ))
        is_greedy_mask1 = tl.get_element(is_greedy_mask, (pos, ))
        position = block_idx * BLOCK_SIZE + pos
        for i in range(num_tokens1):
            if not rejected:
                draft_token_id = tl.load(draft_token_ids_ptr + start_idx1 + i)
                target_argmax_id = tl.load(target_argmax_ptr + start_idx1 + i)
                tl.store(
                    output_token_ids_ptr + position * (max_spec_len + 1) + i,
                    target_argmax_id,
                )
                if draft_token_id != target_argmax_id:
                    # Reject.
                    rejected = True

        if not rejected and is_greedy_mask1:
            bonus_renew(
                bonus_token_ids_ptr,
                position,
                output_token_ids_ptr,
                max_spec_len,
                num_tokens1,
            )


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exost for greedy sampling requests
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr +
                                               req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            if NO_DRAFT_PROBS:
                draft_prob = 1
            else:
                draft_prob = tl.load(draft_probs_ptr +
                                     (start_idx + pos) * vocab_size +
                                     draft_token_id)
            target_prob = tl.load(target_probs_ptr +
                                  (start_idx + pos) * vocab_size +
                                  draft_token_id)
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                # Accept
                token_id = draft_token_id
            else:
                # Reject. Use recovered token
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                     token_id)

    if not rejected:
        # If all tokens are accepted, append the bonus token
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) +
            num_draft_tokens,
            bonus_token_id,
        )


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_block_verify_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
    SUB_BLOCK: tl.constexpr = 1500,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr +
                                               req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    pi = 1.0
    uniform_prob = 1.0
    last_accepted_token_pos = -1

    for pos in range(num_draft_tokens):
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        target_prob = tl.load(target_probs_ptr +
                              (start_idx + pos) * vocab_size + draft_token_id)
        tmp_uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
        uniform_prob = uniform_prob * tmp_uniform_prob

        if NO_DRAFT_PROBS:
            draft_prob = 1
        else:
            draft_prob = tl.load(draft_probs_ptr +
                                 (start_idx + pos) * vocab_size +
                                 draft_token_id)

        pi = min(pi * target_prob / draft_prob, 1.0)
        if draft_prob > 0 and pi >= uniform_prob:
            last_accepted_token_pos = pos
            rejected = False
        else:
            rejected = True

    if last_accepted_token_pos > -1:
        for pos in range(last_accepted_token_pos + 1):
            token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                     token_id)

    if rejected:
        loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
        global_recovered_id = -1
        global_max_p = -1.0
        for loop_i in range(loop):
            vocab_start = loop_i * SUB_BLOCK
            vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
            tmp_target_prob = tl.load(
                target_probs_ptr +
                (start_idx + last_accepted_token_pos + 1) * vocab_size +
                vocab_offset,
                mask=vocab_offset < vocab_size,
                other=0)
            recovered_id = tl.argmax(tmp_target_prob, axis=-1)
            max_p = tl.get_element(tmp_target_prob, (recovered_id, ))
            if max_p > global_max_p:
                global_max_p = max_p
                global_recovered_id = vocab_start + recovered_id
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) +
            last_accepted_token_pos + 1, global_recovered_id)
    else:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) +
            num_draft_tokens, bonus_token_id)


@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    vec_len,
    MAX_NUM_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    offset = req_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    len_mask = offset < vec_len

    start_idx = tl.where(offset == 0, 0,
                         tl.load(cu_num_tokens_ptr + offset - 1, len_mask))
    end_idx = tl.load(cu_num_tokens_ptr + offset, len_mask)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + offset, len_mask)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)

    for i in tl.range(0, BLOCK_SIZE):
        num_tokens1 = tl.get_element(num_tokens, (i, ))
        start_idx1 = tl.get_element(start_idx, (i, ))
        src_val1 = tl.get_element(src_val, (i, ))
        offset1 = tl.arange(0, MAX_NUM_TOKENS)
        tl.store(output_ptr + start_idx1 + offset1,
                 src_val1,
                 mask=offset1 < num_tokens1)


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr +
                                               req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK
    global_recovered_id = -1
    global_max_p = -1.0
    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
        orig_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size +
                            draft_token_id)
        # Temporarily zero out the probability of the draft token.
        # This is essentially the same as target_prob - draft_prob, except that
        # n-gram does not have draft_prob. We regard it as 1.
        tl.store(
            target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id,
            0)
        for loop_i in range(loop):
            vocab_start = loop_i * SUB_BLOCK
            vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
            prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size +
                           vocab_offset,
                           mask=vocab_offset < vocab_size,
                           other=0)
            q = tl.load(q_ptr + req_idx * vocab_size + vocab_offset,
                        mask=vocab_offset < vocab_size,
                        other=float("-inf"))
            new_p = prob / q
            recovered_id = tl.argmax(new_p, axis=-1)
            max_p = tl.get_element(new_p, (recovered_id, ))
            if max_p > global_max_p:
                global_max_p = max_p
                global_recovered_id = vocab_start + recovered_id
    else:
        for loop_i in range(loop):
            vocab_start = loop_i * SUB_BLOCK
            vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
            draft_prob = tl.load(draft_probs_ptr +
                                 (start_idx + pos) * vocab_size + vocab_offset,
                                 mask=vocab_offset < vocab_size,
                                 other=0)
            target_prob = tl.load(target_probs_ptr +
                                  (start_idx + pos) * vocab_size +
                                  vocab_offset,
                                  mask=vocab_offset < vocab_size,
                                  other=0)
            prob = tl.maximum(target_prob - draft_prob, 0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

            q = tl.load(q_ptr + req_idx * vocab_size + vocab_offset,
                        mask=vocab_offset < vocab_size,
                        other=float("-inf"))
            new_p = prob / q
            recovered_id = tl.argmax(new_p, axis=-1)
            max_p = tl.get_element(new_p, (recovered_id, ))
            if max_p > global_max_p:
                global_max_p = max_p
                global_recovered_id = vocab_start + recovered_id

    tl.store(output_token_ids_ptr + start_idx + pos, global_recovered_id)

    if NO_DRAFT_PROBS:
        # Restore the original probability.
        tl.store(
            target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id,
            orig_prob)
