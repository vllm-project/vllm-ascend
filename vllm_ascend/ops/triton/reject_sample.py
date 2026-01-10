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

from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


def cal_grid_and_block_size(batch_size: int):
    vectorcore_num = get_vectorcore_num()
    if batch_size <= vectorcore_num:
        grid = batch_size
        block_size = 1
    else:
        grid = vectorcore_num
        block_size = triton.next_power_of_2(triton.cdiv(batch_size, grid))
    return grid, block_size


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
        vec_len,
        NO_DRAFT_PROBS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0
    start_idxs = tl.where(
        offsets == 0, 0,
        tl.load(cu_num_draft_tokens_ptr + offsets - 1, not_greedy_mask))
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask)
    n_num_draft_tokens = end_idxs - start_idxs
    for req_i in range(BLOCK_SIZE):
        not_greedy = tl.get_element(not_greedy_mask, (req_i, ))
        if not_greedy:
            rejected = False
            start_idx = tl.get_element(start_idxs, (req_i, ))
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = tl.get_element(n_num_draft_tokens, (req_i, ))
            for pos in range(num_draft_tokens):
                if not rejected:
                    draft_token_id = tl.load(draft_token_ids_ptr + start_idx +
                                             pos)
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
                    # NOTE(woosuk): While the draft probability should never be 0,
                    # we check it to avoid NaNs. If it happens to be 0, we reject.
                    if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                        # Accept.
                        token_id = draft_token_id
                    else:
                        # Reject. Use recovered token.
                        rejected = True
                        token_id = tl.load(recovered_token_ids_ptr +
                                           start_idx + pos)
                    tl.store(
                        output_token_ids_ptr + req_idx * (max_spec_len + 1) +
                        pos, token_id)
            if not rejected:
                # If all tokens are accepted, append the bonus token.
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) +
                    num_draft_tokens,
                    bonus_token_id,
                )


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


def rejection_greedy_sample_with_triton(output_token_ids, num_draft_tokens,
                                        cu_num_draft_tokens, draft_token_ids,
                                        target_argmax, bonus_token_ids,
                                        is_greedy, max_spec_len, grid,
                                        block_size):
    vec_len = output_token_ids.shape[0]

    if min(num_draft_tokens) == 1 and max(
            num_draft_tokens) == 1 and is_greedy is None:
        rejection_greedy_sample_spec_len_1_triton[(grid, )](
            output_token_ids,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            vec_len,
            BLOCK_SIZE=block_size,
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
            BLOCK_SIZE=block_size,
        )


def expand_triton(batch_size, expanded_x, x, cu_num_tokens, replace_from,
                  replace_to, max_num_tokens):
    vec_len = batch_size
    grid, block_size = cal_grid_and_block_size(batch_size)

    expand_kernel[(grid, )](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        vec_len,
        MAX_NUM_TOKENS=max_num_tokens,  # To avoid recompilation.
        BLOCK_SIZE=block_size,
    )


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_block_verify_kernel(
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
        vec_len,
        NO_DRAFT_PROBS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0
    start_idxs = tl.where(
        offsets == 0, 0,
        tl.load(cu_num_draft_tokens_ptr + offsets - 1, not_greedy_mask))
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask)
    n_num_draft_tokens = end_idxs - start_idxs
    for req_i in range(BLOCK_SIZE):
        not_greedy = tl.get_element(not_greedy_mask, (req_i, ))
        if not_greedy:

            rejected = False
            pi = 1.0
            uniform_prob = 1.0
            last_accepted_token_pos = -1
            start_idx = tl.get_element(start_idxs, (req_i, ))
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = tl.get_element(n_num_draft_tokens, (req_i, ))

            for pos in range(num_draft_tokens):
                draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                target_prob = tl.load(target_probs_ptr +
                                      (start_idx + pos) * vocab_size +
                                      draft_token_id)
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
                    tl.store(
                        output_token_ids_ptr + req_idx * (max_spec_len + 1) +
                        pos, token_id)

            if rejected:
                recovered_token_id = tl.load(recovered_token_ids_ptr +
                                             start_idx +
                                             last_accepted_token_pos + 1)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) +
                    last_accepted_token_pos + 1, recovered_token_id)
            else:
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) +
                    num_draft_tokens, bonus_token_id)
