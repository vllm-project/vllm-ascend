# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/utils.py

from vllm.triton_utils import tl, triton


@triton.jit
def prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,
    backup_next_token_ids_ptr,
    next_token_ids_ptr,
    valid_sampled_tokens_count_ptr,
    vocab_size,
    num_sampled_tokens_per_req,
    num_reqs,
    stride_sampled_token_ids,
    BLOCK_SIZE_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    token_offsets = tl.arange(0, BLOCK_SIZE_TOKENS)
    token_mask = token_offsets < num_sampled_tokens_per_req
    row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
    token_ids = tl.load(row_ptr + token_offsets, mask=token_mask, other=-1)
    is_valid = (token_ids != -1) & (token_ids < vocab_size) & token_mask
    valid_count = tl.sum(is_valid.to(tl.int32), axis=0)
    last_valid_index = tl.max(tl.where(is_valid, token_offsets, -1), axis=0)
    last_valid_token = tl.sum(tl.where(token_offsets == last_valid_index, token_ids, 0), axis=0)
    backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
    next_token = tl.where(valid_count > 0, last_valid_token, backup_token)

    tl.store(next_token_ids_ptr + req_idx, next_token)
    tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)


@triton.jit
def prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,
    num_reqs,  # tl.int32
):
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)
    if req_idx == 0:
        num_draft_tokens = cu_draft_curr
    else:
        cu_draft_prev = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        num_draft_tokens = cu_draft_curr - cu_draft_prev

    valid_count = tl.load(valid_sampled_tokens_count_ptr + req_idx)
    num_rejected = num_draft_tokens + 1 - valid_count
    num_rejected = tl.where(num_draft_tokens > 0, num_rejected, 0)

    q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + req_idx + 1) - 1
    index_to_sample = q_last_tok_idx - num_rejected
    tl.store(token_indices_to_sample_ptr + req_idx, index_to_sample)
    tl.store(num_rejected_tokens_gpu_ptr + req_idx, num_rejected)


@triton.jit
def copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid(
    # Inputs
    next_token_ids_ptr,  # [num_reqs]
    target_positions_ptr,  # [num_context]
    context_slot_mapping_ptr,  # [num_context]
    # Outputs
    out_input_ids_ptr,  # [num_query_total] (output)
    out_context_positions_ptr,  # [num_context] (output)
    out_query_positions_ptr,  # [num_query_total] (output)
    out_context_slot_mapping_ptr,  # [num_context] (output)
    out_query_slot_mapping_ptr,  # [num_query_total] (output)
    out_token_indices_ptr,  # [num_reqs * num_speculative_tokens] (output)
    # Block table
    block_table_ptr,  # [max_reqs, max_blocks]
    block_table_stride,  # stride of block_table dim 0 (in elements)
    # Metadata
    query_start_loc_ptr,  # [num_reqs + 1]
    seq_lens_ptr,  # [num_reqs]
    num_rejected_tokens_ptr,  # [num_reqs] or null (0) when not padded
    # Scalars
    parallel_drafting_token_id,  # tl.int32
    block_size,  # tl.int32
    num_query_per_req,  # tl.int32
    num_speculative_tokens,  # tl.int32
    total_input_tokens,  # tl.int32
    batch_size,  # tl.int32
    HAS_NUM_REJECTED: tl.constexpr = False,
    SAMPLE_FROM_ANCHOR: tl.constexpr = False,
):
    for req_idx in range(0, batch_size):
        ctx_start = tl.load(query_start_loc_ptr + req_idx)
        ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
        num_ctx = ctx_end - ctx_start

        for j in range(0, num_ctx):
            ctx_pos_idx = ctx_start + j
            pos = tl.load(target_positions_ptr + ctx_pos_idx)
            tl.store(out_context_positions_ptr + ctx_pos_idx, pos)

            slot = tl.load(context_slot_mapping_ptr + ctx_pos_idx)
            tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, slot)

        if HAS_NUM_REJECTED:
            num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
            valid_ctx_end = ctx_end - num_rejected
        else:
            num_rejected = 0
            valid_ctx_end = ctx_end

        seq_len = tl.load(seq_lens_ptr + req_idx)
        effective_seq_len = seq_len - num_rejected
        last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)

        for q_idx in range(0, num_query_per_req):
            query_pos = last_pos + 1 + q_idx
            query_out_idx = req_idx * num_query_per_req + q_idx

            tl.store(out_query_positions_ptr + query_out_idx, query_pos)

            query_cache_pos = effective_seq_len + q_idx
            block_num_q = query_cache_pos // block_size
            block_id_q = tl.load(block_table_ptr + req_idx * block_table_stride + block_num_q).to(tl.int64)
            slot_q = block_id_q * block_size + (query_cache_pos % block_size)
            tl.store(out_query_slot_mapping_ptr + query_out_idx, slot_q)

            if q_idx == 0:
                bonus_token = tl.load(next_token_ids_ptr + req_idx)
                tl.store(out_input_ids_ptr + query_out_idx, bonus_token)
            else:
                tl.store(out_input_ids_ptr + query_out_idx, parallel_drafting_token_id)

            if SAMPLE_FROM_ANCHOR:
                sample_out_idx = req_idx * num_speculative_tokens + q_idx
                tl.store(out_token_indices_ptr + sample_out_idx, query_out_idx)
            else:
                if q_idx > 0:
                    sample_out_idx = req_idx * num_speculative_tokens + (q_idx - 1)
                    tl.store(out_token_indices_ptr + sample_out_idx, query_out_idx)
