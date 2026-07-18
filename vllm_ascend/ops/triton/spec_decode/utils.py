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


@triton.jit(do_not_specialize=["num_reqs"])
def prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,
    num_reqs,  # tl.int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # Grid-Stride Loop:
    block_start_step = num_programs * BLOCK_SIZE

    for block_start in tl.range(pid * BLOCK_SIZE, num_reqs, block_start_step):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_reqs

        # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
        # cumulative sum (first entry is the first value, not zero).
        cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + offsets, mask=mask)

        prev_indices = offsets - 1
        has_prev = offsets > 0
        cu_draft_prev = tl.load(
            cu_num_draft_tokens_ptr + prev_indices,
            mask=mask & has_prev,
            other=0,
        )

        num_draft_tokens = tl.where(has_prev, cu_draft_curr - cu_draft_prev, cu_draft_curr)

        valid_count = tl.load(valid_sampled_tokens_count_ptr + offsets, mask=mask)
        num_rejected = num_draft_tokens + 1 - valid_count
        num_rejected = tl.where(num_draft_tokens > 0, num_rejected, 0)

        # query_start_loc[req_idx + 1] is the start position of the next request,
        # which is one past the last token of this request.
        q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + offsets + 1, mask=mask) - 1

        index_to_sample = q_last_tok_idx - num_rejected
        tl.store(token_indices_to_sample_ptr + offsets, index_to_sample, mask=mask)
        tl.store(num_rejected_tokens_gpu_ptr + offsets, num_rejected, mask=mask)


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


# TODO to delete
def copy_and_expand_dflash_inputs_py(
    # Inputs
    next_token_ids,  # [num_reqs]
    target_positions,  # [num_context]
    context_slot_mapping,  # [num_context]
    # Outputs
    out_input_ids,  # [num_query_total]
    out_context_positions,  # [num_context]
    out_query_positions,  # [num_query_total]
    out_context_slot_mapping,  # [num_context]
    out_query_slot_mapping,  # [num_query_total]
    out_token_indices,  # [num_reqs * num_speculative_tokens]
    # Block table
    block_table,  # [max_reqs, max_blocks]
    block_table_stride,  # stride of block_table dim 0 (unused on host; parity)
    # Metadata
    query_start_loc,  # [num_reqs + 1]
    seq_lens,  # [num_reqs]
    num_rejected_tokens,  # [num_reqs] | None
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_tokens,
    total_input_tokens,  # unused (kernel declares it); parity
    batch_size,
    HAS_NUM_REJECTED=False,
    SAMPLE_FROM_ANCHOR=False,
):
    """Host-side reference of ``copy_and_expand_dflash_inputs_kernel_single_grid``.

    Same interface and logic as the triton kernel (parameters aligned 1:1), for
    reuse from paths that run per kv-cache-group (e.g. DSpark, which has
    multiple groups) instead of launching the fused single-block-table kernel.
    ``block_table_stride`` and ``total_input_tokens`` are unused on host (the
    kernel needs them for pointer arithmetic / grid sizing) but kept for
    interface parity.
    """
    has_num_rejected = HAS_NUM_REJECTED
    for req_idx in range(batch_size):
        ctx_start = int(query_start_loc[req_idx].item())
        ctx_end = int(query_start_loc[req_idx + 1].item())

        out_context_positions[ctx_start:ctx_end] = target_positions[ctx_start:ctx_end]
        out_context_slot_mapping[ctx_start:ctx_end] = context_slot_mapping[ctx_start:ctx_end]

        if has_num_rejected:
            num_rejected = int(num_rejected_tokens[req_idx].item())
            valid_ctx_end = ctx_end - num_rejected
        else:
            num_rejected = 0
            valid_ctx_end = ctx_end

        seq_len = int(seq_lens[req_idx].item())
        effective_seq_len = seq_len - num_rejected
        last_pos = int(target_positions[valid_ctx_end - 1].item())

        for q_idx in range(num_query_per_req):
            query_out_idx = req_idx * num_query_per_req + q_idx

            out_query_positions[query_out_idx] = last_pos + 1 + q_idx

            query_cache_pos = effective_seq_len + q_idx
            block_num_q = query_cache_pos // block_size
            block_id_q = int(block_table[req_idx, block_num_q].item())
            out_query_slot_mapping[query_out_idx] = (
                block_id_q * block_size + (query_cache_pos % block_size)
            )

            if q_idx == 0:
                out_input_ids[query_out_idx] = next_token_ids[req_idx]
            else:
                out_input_ids[query_out_idx] = parallel_drafting_token_id

            # token_indices: map i-th sampled token -> its query position.
            # DFlash (SAMPLE_FROM_ANCHOR=False): bonus at q_idx=0 is NOT sampled,
            #   sample_out_idx = req*num_spec + (q_idx-1) (skips bonus).
            # DSpark (SAMPLE_FROM_ANCHOR=True): anchor at q_idx=0 IS sampled,
            #   sample_out_idx = req*num_spec + q_idx (includes anchor).
            if SAMPLE_FROM_ANCHOR or q_idx > 0:
                sample_out_idx = req_idx * num_speculative_tokens + (
                    q_idx if SAMPLE_FROM_ANCHOR else (q_idx - 1)
                )
                out_token_indices[sample_out_idx] = query_out_idx
