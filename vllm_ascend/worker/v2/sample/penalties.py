# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/penalties.py.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

# Maximum program count accepted by the Ascend kernel launch API.
MAX_GRID_PROGRAMS = 65535

# Token-presence bits stored in each int32 prompt bitmap word.
BITS_PER_WORD = 32

# Prompt-presence flags and packing follow PR #15270:
# https://github.com/vllm-project/vllm-ascend/pull/15270
# Vocabulary words packed per program in _pack_prompt_bin_mask_kernel.
WORDS_PER_BLOCK = 128


@triton.jit
def _penalties_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    token_ids_ptr,
    expanded_local_pos_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    VOCAB_GRID_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0

    # NPU doesn't support chained 'or' operations like 'A or B or C'
    use_penalty = use_rep_penalty or use_freq_penalty
    use_penalty = use_penalty or use_pres_penalty
    if not use_penalty:
        # Early return to avoid loading logits.
        return

    vocab_program_idx = tl.program_id(1)
    for vocab_block_idx in tl.range(
        vocab_program_idx,
        NUM_VOCAB_BLOCKS,
        VOCAB_GRID_SIZE,
    ):
        block = vocab_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
        logits = logits.to(tl.float32)

        base_output_counts = tl.load(
            output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
            mask=mask,
            other=0,
        )

        # Accumulate draft token counts from previous positions directly into
        # output_bin_counts (preserves its native tensor layout, avoiding an
        # expensive shared-memory layout conversion after the loop).
        pos = tl.load(expanded_local_pos_ptr + token_idx)
        start_idx = token_idx - pos
        output_bin_counts = base_output_counts
        for prev_pos in tl.range(pos):
            prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
            token_match = block == prev_token
            output_bin_counts = output_bin_counts + token_match.to(tl.int32)
        output_bin_mask = output_bin_counts != 0

        # Apply repetition penalties.
        if use_rep_penalty:
            packed_block = vocab_block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
            packed_mask = tl.load(
                prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
                mask=packed_block < tl.cdiv(vocab_size, 32),
                other=0,
            )
            bit_masks = 1 << tl.arange(0, 32)
            bit_masks_expanded = bit_masks[None, :]
            packed_expanded = packed_mask[:, None]
            bits_matrix = (packed_expanded & bit_masks_expanded) != 0
            prompt_bin_mask = bits_matrix.reshape(BLOCK_SIZE)

            # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
            scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
            # If logits are positive, divide by penalty, otherwise multiply by penalty.
            logits *= tl.where(logits > 0, 1.0 / scale, scale)

        # Apply frequency penalties.
        logits -= freq_penalty * output_bin_counts
        # Apply presence penalties.
        logits -= pres_penalty * output_bin_mask
        # Store back to logits.
        tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 4096
    num_vocab_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    vocab_grid_size = min(num_vocab_blocks, MAX_GRID_PROGRAMS // num_tokens)
    _penalties_kernel[(num_tokens, vocab_grid_size)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        NUM_VOCAB_BLOCKS=num_vocab_blocks,
        VOCAB_GRID_SIZE=vocab_grid_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _bincount_kernel(
    expanded_idx_mapping_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    prefill_len_ptr,
    prompt_flag_ptr,
    prompt_flag_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    num_requests,
    token_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    work_items = num_requests * token_blocks
    quotient = work_items // programs
    remainder = work_items % programs
    first = pid * quotient + tl.minimum(pid, remainder)
    count = quotient + (pid < remainder)
    for work_item in range(first, first + count):
        token_idx = work_item // token_blocks
        block_idx = work_item - token_idx * token_blocks
        req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
        prefill_len = tl.load(prefill_len_ptr + req_state_idx)
        prompt_len = tl.load(prompt_len_ptr + req_state_idx)
        for start in range(block_idx * BLOCK_SIZE, tl.maximum(prompt_len, prefill_len), token_blocks * BLOCK_SIZE):
            block = start + tl.arange(0, BLOCK_SIZE)
            if start < prompt_len:
                mask = block < prompt_len
                prompt_tokens = tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask)
                tl.store(
                    prompt_flag_ptr + token_idx * prompt_flag_stride + prompt_tokens,
                    tl.full((BLOCK_SIZE,), 1, tl.int8),
                    mask=mask,
                )
            if start + BLOCK_SIZE >= prompt_len:
                mask = (block < prefill_len) & (block >= prompt_len)
                output_tokens = tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask)
                tl.atomic_add(
                    output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + output_tokens,
                    1,
                    mask=mask,
                )


@triton.jit
def _pack_prompt_bin_mask_kernel(
    expanded_idx_mapping_ptr,
    prompt_flag_ptr,
    prompt_flag_stride,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    num_words,
    vocab_size,
    WORDS_PER_BLOCK: tl.constexpr,
    BITS_PER_WORD: tl.constexpr,
):
    token_idx = tl.program_id(0)
    word_block_idx = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    words = word_block_idx * WORDS_PER_BLOCK + tl.arange(0, WORDS_PER_BLOCK)
    word_mask = words < num_words
    bits = tl.arange(0, BITS_PER_WORD)
    flat = words[:, None] * BITS_PER_WORD + bits[None, :]
    # The final word runs past vocab_size, so mask per element, not per word.
    elem_mask = word_mask[:, None] & (flat < vocab_size)
    flags = tl.load(prompt_flag_ptr + token_idx * prompt_flag_stride + flat, mask=elem_mask, other=0).to(tl.int32)
    # Each bit is contributed by exactly one lane, so the sum equals an OR.
    bit_masks = tl.full((BITS_PER_WORD,), 1, tl.uint32) << bits
    packed = tl.sum(tl.where(flags != 0, bit_masks[None, :], 0), axis=1).to(tl.int32)
    tl.store(prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + words, packed, mask=word_mask)


def bincount(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_prefill_len: int,
) -> None:
    num_tokens = expanded_idx_mapping.shape[0]
    if num_tokens == 0:
        return
    num_words = prompt_bin_mask.shape[1]
    vocab_size = output_bin_counts.shape[1]
    BLOCK_SIZE = 1024
    num_blocks = max(1, triton.cdiv(max_prefill_len, BLOCK_SIZE))
    init_device_properties_triton()
    statistics_grid = min(get_vectorcore_num(), num_tokens * num_blocks)
    # Scratch prompt-presence flags (0: absent, 1: present), one byte per
    # vocabulary entry per staged request. Packed into prompt_bin_mask below
    # and freed on return. num_tokens counts newly added penalty requests.
    prompt_flag = torch.zeros(
        num_tokens,
        num_words * BITS_PER_WORD,
        dtype=torch.int8,
        device=prompt_bin_mask.device,
    )
    output_bin_counts[expanded_idx_mapping] = 0
    _bincount_kernel[(statistics_grid,)](
        expanded_idx_mapping,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        prefill_len,
        prompt_flag,
        prompt_flag.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        num_tokens,
        num_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Writes every word of each row, so prompt_bin_mask needs no pre-zeroing.
    _pack_prompt_bin_mask_kernel[(num_tokens, triton.cdiv(num_words, WORDS_PER_BLOCK))](
        expanded_idx_mapping,
        prompt_flag,
        prompt_flag.stride(0),
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        num_words,
        vocab_size,
        WORDS_PER_BLOCK=WORDS_PER_BLOCK,
        BITS_PER_WORD=BITS_PER_WORD,
    )
