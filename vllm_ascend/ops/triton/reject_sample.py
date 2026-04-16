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

from vllm_ascend.ops.triton.triton_utils import get_element, get_vectorcore_num


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

    # Add validity check for pos within the loop
    for pos in tl.range(0, BLOCK_SIZE):
        # Calculate the global position of the current token
        global_pos = block_idx * BLOCK_SIZE + pos
        if global_pos < vec_len:
            draft_token_id1 = get_element(draft_token_id, (pos,))
            target_argmax1 = get_element(target_argmax_id, (pos,))
            if draft_token_id1 == target_argmax1:
                bonus_renew_1(
                    bonus_token_ids_ptr,
                    global_pos,
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
    tl.store(output_token_ids_ptr + position * (max_spec_len + 1) + num_tokens1, bonus_token_id)


@triton.jit(do_not_specialize=["vec_len", "max_spec_len"])
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

    start_idx = tl.where(offset == 0, 0, tl.load(cu_num_draft_tokens_ptr + offset - 1, is_greedy_mask))
    end_idx = tl.load(cu_num_draft_tokens_ptr + offset, is_greedy_mask)
    num_draft_tokens = end_idx - start_idx

    for pos in tl.range(0, BLOCK_SIZE):
        num_tokens1 = get_element(num_draft_tokens, (pos,))
        rejected = False
        start_idx1 = get_element(start_idx, (pos,))
        is_greedy_mask1 = get_element(is_greedy_mask, (pos,))
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
    target_probs_ptr,  # [num_tokens, vocab_size] or [num_tokens, compressed_vocab_size] if COMPRESSED_MODE
    target_indices_ptr,  # [num_tokens, compressed_vocab_size] global vocab indices, only used if COMPRESSED_MODE
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,  # vocab_size or compressed_vocab_size if COMPRESSED_MODE
    global_vocab_size,  # global vocab size for draft_probs indexing (only used if COMPRESSED_MODE)
    vec_len,
    NO_DRAFT_PROBS: tl.constexpr,
    COMPRESSED_MODE: tl.constexpr,  # Whether using compressed logits
    BLOCK_SIZE: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr = 512, 
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0
    start_idxs = tl.where(offsets == 0, 0, tl.load(cu_num_draft_tokens_ptr + offsets - 1, not_greedy_mask))
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask)
    n_num_draft_tokens = end_idxs - start_idxs

    for req_i in range(BLOCK_SIZE):
        not_greedy = get_element(not_greedy_mask, (req_i,))
        if not_greedy:
            rejected = False
            start_idx = get_element(start_idxs, (req_i,))
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = get_element(n_num_draft_tokens, (req_i,))

            for pos in range(num_draft_tokens):
                if not rejected:
                    token_idx = start_idx + pos
                    draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

                    if COMPRESSED_MODE:
                        target_prob = 0.0
                        found = False
                        
                        for v_offset in range(0, vocab_size, VOCAB_BLOCK_SIZE):
                            if not found:
                                vocab_offsets = v_offset + tl.arange(0, VOCAB_BLOCK_SIZE)
                                vocab_mask = vocab_offsets < vocab_size
                                
                                candidate_indices = tl.load(
                                    target_indices_ptr + token_idx * vocab_size + vocab_offsets,
                                    mask=vocab_mask,
                                    other=-1  
                                )
                                
                                match_mask = candidate_indices == draft_token_id
                                
                                candidate_probs = tl.load(
                                    target_probs_ptr + token_idx * vocab_size + vocab_offsets,
                                    mask=vocab_mask,
                                    other=0.0
                                )
                                
                                current_match_prob = tl.sum(candidate_probs * match_mask, axis=0)
                                if current_match_prob > 0.0:
                                    target_prob = current_match_prob
                                    found = True 
                    else:
                        target_prob = tl.load(target_probs_ptr + token_idx * vocab_size + draft_token_id)

                    if NO_DRAFT_PROBS:
                        draft_prob = 1.0
                    else:
                        vocab_for_draft = global_vocab_size if COMPRESSED_MODE else vocab_size
                        draft_prob = tl.load(draft_probs_ptr + token_idx * vocab_for_draft + draft_token_id)

                    uniform_prob = tl.load(uniform_probs_ptr + token_idx)

                    # Acceptance condition
                    if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                        # Accept
                        token_id = draft_token_id
                    else:
                        # Reject - use recovered token
                        rejected = True
                        token_id = tl.load(recovered_token_ids_ptr + token_idx)

                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)

            if not rejected:
                # All tokens accepted - append bonus token
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
                    bonus_token_id,
                )


@triton.jit(do_not_specialize=["replace_from", "replace_to", "vec_len"])
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

    start_idx = tl.where(offset == 0, 0, tl.load(cu_num_tokens_ptr + offset - 1, len_mask))
    end_idx = tl.load(cu_num_tokens_ptr + offset, len_mask)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + offset, len_mask)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)

    for i in tl.range(0, BLOCK_SIZE):
        num_tokens1 = get_element(num_tokens, (i,))
        start_idx1 = get_element(start_idx, (i,))
        src_val1 = get_element(src_val, (i,))
        offset1 = tl.arange(0, MAX_NUM_TOKENS)
        tl.store(output_ptr + start_idx1 + offset1, src_val1, mask=offset1 < num_tokens1)

@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    target_indices_ptr,
    q_ptr,
    compressed_vocab_size,
    global_vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    BLOCK_VERIFY: tl.constexpr,
    COMPRESSED_MODE: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    pos = tl.program_id(1)

    # Compute token index
    start_idx = tl.where(req_idx == 0, 0, tl.load(cu_num_draft_tokens_ptr + req_idx - 1))
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if COMPRESSED_MODE:
        C = compressed_vocab_size
        n_loop = tl.cdiv(C, SUB_BLOCK)

        global_max_p = tl.full((), -float("inf"), tl.float32)
        global_recovered_id = tl.full((), -1, tl.int64)
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx).to(tl.int64)

        for li in range(n_loop):
            c_start = li * SUB_BLOCK
            offs = c_start + tl.arange(0, SUB_BLOCK)
            mask = offs < C

            # Load target prob and global index
            tprob = tl.load(target_probs_ptr + token_idx * C + offs, mask=mask, other=0.0).to(tl.float32)

            gidx = tl.load(target_indices_ptr + token_idx * C + offs, mask=mask, other=0).to(tl.int64)

            if NO_DRAFT_PROBS:
                is_draft = (gidx == draft_token_id) & mask
                prob = tl.where(is_draft, 0.0, tprob)
            else:
                valid = (gidx >= 0) & (gidx < global_vocab_size) & mask
                dprob = tl.load(draft_probs_ptr + token_idx * global_vocab_size + gidx, mask=valid, other=0.0).to(
                    tl.float32
                )
                prob = tl.maximum(tprob - dprob, 0.0)

            qv = tl.load(q_ptr + req_idx * C + offs, mask=mask, other=1.0).to(tl.float32)

            bad_q = (qv <= 0) | tl.math.isinf(qv)
            score = tl.where(bad_q, float("-inf"), prob / qv)
            score = tl.where(mask, score, float("-inf"))

            block_best_score = tl.max(score, axis=0)
            block_best_idx = tl.argmax(score, axis=0).to(tl.int64)
            block_best_global_id = tl.load(target_indices_ptr + token_idx * C + (c_start + block_best_idx)).to(tl.int64)

            better = block_best_score > global_max_p
            global_max_p = tl.where(better, block_best_score, global_max_p)
            global_recovered_id = tl.where(better, block_best_global_id, global_recovered_id)

        tl.store(output_token_ids_ptr + token_idx, global_recovered_id)
    else:
        vocab_size = global_vocab_size
        n_loop = tl.cdiv(vocab_size, SUB_BLOCK)

        global_max_p = tl.full((), -float("inf"), tl.float32)
        global_recovered_id = tl.full((), -1, tl.int64)
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx).to(tl.int64)

        prefix_prob = 1.0
        if BLOCK_VERIFY:
            for prev_pos in range(pos):
                prev_token_idx = start_idx + prev_pos
                prev_draft_token_id = tl.load(draft_token_ids_ptr + prev_token_idx).to(tl.int64)
                prev_target_prob = tl.load(
                    target_probs_ptr + prev_token_idx * vocab_size + prev_draft_token_id
                ).to(tl.float32)
                if NO_DRAFT_PROBS:
                    prev_draft_prob = 1.0
                else:
                    prev_draft_prob = tl.load(
                        draft_probs_ptr + prev_token_idx * vocab_size + prev_draft_token_id
                    ).to(tl.float32)
                ratio = tl.where(prev_draft_prob > 0, prev_target_prob / prev_draft_prob, 0.0)
                prefix_prob = prefix_prob * tl.minimum(ratio, 1.0)

        for vi in range(n_loop):
            v_start = vi * SUB_BLOCK
            offs = v_start + tl.arange(0, SUB_BLOCK)
            mask = offs < vocab_size

            gidx = offs  # Global indices are just offset here

            tprob = tl.load(target_probs_ptr + token_idx * vocab_size + offs, mask=mask, other=0.0).to(tl.float32)

            if NO_DRAFT_PROBS:
                is_draft = (gidx == draft_token_id) & mask
                if BLOCK_VERIFY:
                    prob = tl.where(is_draft, 0.0, prefix_prob * tprob)
                else:
                    prob = tl.where(is_draft, 0.0, tprob)
            else:
                dprob = tl.load(draft_probs_ptr + token_idx * vocab_size + gidx, mask=mask, other=0.0).to(tl.float32)
                if BLOCK_VERIFY:
                    prob = tl.maximum(prefix_prob * tprob - dprob, 0.0)
                else:
                    prob = tl.maximum(tprob - dprob, 0.0)

            qv = tl.load(q_ptr + req_idx * vocab_size + offs, mask=mask, other=1.0).to(tl.float32)

            bad_q = (qv <= 0) | tl.math.isinf(qv)
            score = tl.where(bad_q, float("-inf"), prob / qv)
            score = tl.where(mask, score, float("-inf"))

            block_best_score = tl.max(score, axis=0)
            block_best_idx = tl.argmax(score, axis=0).to(tl.int64)
            block_best_global_id = v_start + block_best_idx

            better = block_best_score > global_max_p
            global_max_p = tl.where(better, block_best_score, global_max_p)
            global_recovered_id = tl.where(better, block_best_global_id.to(tl.int64), global_recovered_id)

        tl.store(output_token_ids_ptr + token_idx, global_recovered_id)

def rejection_greedy_sample_with_triton(
    output_token_ids,
    num_draft_tokens,
    cu_num_draft_tokens,
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
    is_greedy,
    max_spec_len,
    grid,
    block_size,
):
    vec_len = output_token_ids.shape[0]

    if min(num_draft_tokens) == 1 and max(num_draft_tokens) == 1 and is_greedy is None:
        rejection_greedy_sample_spec_len_1_triton[(grid,)](
            output_token_ids,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            vec_len,
            BLOCK_SIZE=block_size,
        )
    else:
        rejection_greedy_sample_triton[(grid,)](
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


def expand_triton(batch_size, expanded_x, x, cu_num_tokens, replace_from, replace_to, max_num_tokens):
    vec_len = batch_size
    grid, block_size = cal_grid_and_block_size(batch_size)

    expand_kernel[(grid,)](
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
    target_probs_ptr,  # [num_tokens, vocab_size] or [num_tokens, compressed_vocab_size] if COMPRESSED_MODE
    target_indices_ptr,  # [num_tokens, compressed_vocab_size] global vocab indices, only used if COMPRESSED_MODE
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,  # vocab_size or compressed_vocab_size if COMPRESSED_MODE
    global_vocab_size,  # global vocab size for draft_probs indexing (only used if COMPRESSED_MODE)
    vec_len,
    NO_DRAFT_PROBS: tl.constexpr,
    COMPRESSED_MODE: tl.constexpr,  # Whether using compressed logits
    BLOCK_SIZE: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr = 512,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0
    start_idxs = tl.where(offsets == 0, 0, tl.load(cu_num_draft_tokens_ptr + offsets - 1, not_greedy_mask))
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask)
    n_num_draft_tokens = end_idxs - start_idxs

    for req_i in range(BLOCK_SIZE):
        not_greedy = get_element(not_greedy_mask, (req_i,))
        if not_greedy:
            pi = 1.0
            uniform_prob = 1.0
            last_accepted_token_pos = -1
            start_idx = get_element(start_idxs, (req_i,))
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = get_element(n_num_draft_tokens, (req_i,))

            for pos in range(num_draft_tokens):
                token_idx = start_idx + pos
                draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

                if COMPRESSED_MODE:
                    target_prob = 0.0
                    found = False
                    
                    for v_offset in range(0, vocab_size, VOCAB_BLOCK_SIZE):
                        if not found:
                            vocab_offsets = v_offset + tl.arange(0, VOCAB_BLOCK_SIZE)
                            vocab_mask = vocab_offsets < vocab_size
                            
                            candidate_indices = tl.load(
                                target_indices_ptr + token_idx * vocab_size + vocab_offsets,
                                mask=vocab_mask,
                                other=-1
                            )
                            
                            match_mask = candidate_indices == draft_token_id
                            
                            candidate_probs = tl.load(
                                target_probs_ptr + token_idx * vocab_size + vocab_offsets,
                                mask=vocab_mask,
                                other=0.0
                            )
                            
                            current_match_prob = tl.sum(candidate_probs * match_mask, axis=0)
                            
                            if current_match_prob > 0.0:
                                target_prob = current_match_prob
                                found = True
                else:
                    target_prob = tl.load(target_probs_ptr + token_idx * vocab_size + draft_token_id)

                tmp_uniform_prob = tl.load(uniform_probs_ptr + token_idx)
                uniform_prob = uniform_prob * tmp_uniform_prob

                if NO_DRAFT_PROBS:
                    draft_prob = 1.0
                else:
                    vocab_for_draft = global_vocab_size if COMPRESSED_MODE else vocab_size
                    draft_prob = tl.load(draft_probs_ptr + token_idx * vocab_for_draft + draft_token_id)

                pi = min(pi * target_prob / draft_prob, 1.0)
                if draft_prob > 0 and pi >= uniform_prob:
                    last_accepted_token_pos = pos

            # Store accepted tokens
            if last_accepted_token_pos > -1:
                for pos in range(last_accepted_token_pos + 1):
                    token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)

            # Store recovered or bonus token
            if last_accepted_token_pos + 1 < num_draft_tokens:
                # Rejected - store recovered token
                recovered_token_id = tl.load(recovered_token_ids_ptr + start_idx + last_accepted_token_pos + 1)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + last_accepted_token_pos + 1,
                    recovered_token_id,
                )
            else:
                # All accepted - store bonus token
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens, bonus_token_id)
