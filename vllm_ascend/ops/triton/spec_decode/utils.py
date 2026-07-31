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
def copy_and_expand_dflash_and_dspark_inputs_kernel(
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
    TILE_SIZE: tl.constexpr = 256,
):
    # Grid-stride kernel: launch grid is capped at the vector-core count by
    # the caller (grid = min(cdiv(total_work, TILE_SIZE), num_vectorcore)),
    # each program processes TILE_SIZE elements per iteration and strides by
    # num_programs * TILE_SIZE. TILE_SIZE is the Triton program tile width,
    # distinct from block_size (the KV-cache block size) above.
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    block_start_step = num_programs * TILE_SIZE

    # --- Part 1: context positions / slot_mapping copy ---
    # query_start_loc is a contiguous partition of [0, total_input_tokens),
    # so the per-request copy loops of the original kernel union into one
    # flat range that can be vectorized directly.
    block_start = pid * TILE_SIZE
    while block_start < total_input_tokens:
        offs = block_start + tl.arange(0, TILE_SIZE)
        mask = offs < total_input_tokens
        pos = tl.load(target_positions_ptr + offs, mask=mask)
        tl.store(out_context_positions_ptr + offs, pos, mask=mask)
        slot = tl.load(context_slot_mapping_ptr + offs, mask=mask)
        tl.store(out_context_slot_mapping_ptr + offs, slot, mask=mask)
        block_start += block_start_step

    # --- Part 2: query block expand ---
    # Flat offs covers [0, batch_size * num_query_per_req); req_idx / q_idx
    # are recovered from offs instead of iterating two serial loops.
    num_query_total = batch_size * num_query_per_req
    block_start = pid * TILE_SIZE
    while block_start < num_query_total:
        offs = block_start + tl.arange(0, TILE_SIZE)
        mask = offs < num_query_total

        req_idx = offs // num_query_per_req
        q_idx = offs % num_query_per_req

        ctx_end = tl.load(query_start_loc_ptr + req_idx + 1, mask=mask, other=0)
        if HAS_NUM_REJECTED:
            num_rejected = tl.load(num_rejected_tokens_ptr + req_idx, mask=mask, other=0)
        else:
            num_rejected = tl.zeros([TILE_SIZE], dtype=tl.int32)
        valid_ctx_end = ctx_end - num_rejected

        seq_len = tl.load(seq_lens_ptr + req_idx, mask=mask, other=0)
        effective_seq_len = seq_len - num_rejected
        last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1, mask=mask, other=0)

        # RoPE position id of the query token, derived from the last context
        # token's position. Written to out_query_positions for position embeddings.
        query_pos = last_pos + 1 + q_idx
        tl.store(out_query_positions_ptr + offs, query_pos, mask=mask)

        # Linear KV-cache token index used to look up the physical slot via the
        # block_table. This is kept separate from query_pos for multimodal
        # (e.g. M-RoPE) inputs: image/vision tokens can carry repeated or
        # non-contiguous position ids, so the position id != the linear token
        # index and the slot must be derived from the effective sequence length
        # rather than from query_pos. For text-only inputs the two values are
        # identical, so this only changes behaviour for multimodal inputs.
        query_kv_slot_pos = effective_seq_len + q_idx
        block_num_q = query_kv_slot_pos // block_size
        block_id_q = tl.load(block_table_ptr + req_idx * block_table_stride + block_num_q, mask=mask, other=0).to(
            tl.int64
        )
        slot_q = block_id_q * block_size + (query_kv_slot_pos % block_size)
        tl.store(out_query_slot_mapping_ptr + offs, slot_q, mask=mask)

        bonus = tl.load(next_token_ids_ptr + req_idx, mask=mask, other=0)
        in_id = tl.where(q_idx == 0, bonus, parallel_drafting_token_id)
        tl.store(out_input_ids_ptr + offs, in_id, mask=mask)

        if SAMPLE_FROM_ANCHOR:
            sample_out_idx = req_idx * num_speculative_tokens + q_idx
            tl.store(out_token_indices_ptr + sample_out_idx, offs, mask=mask)
        else:
            sample_mask = mask & (q_idx > 0)
            sample_out_idx = req_idx * num_speculative_tokens + (q_idx - 1)
            tl.store(out_token_indices_ptr + sample_out_idx, offs, mask=sample_mask)

        block_start += block_start_step
