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

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

VOCAB_BLOCK_SIZE = 4096


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
    total_tiles,
    tiles_per_worker,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per vector core, each owning a contiguous run of the flattened
    # (token, vocab_block) space.
    start = tl.program_id(0) * tiles_per_worker
    end = tl.minimum(start + tiles_per_worker, total_tiles)
    if start >= end:
        return

    first_row = start // NUM_VOCAB_BLOCKS
    last_row = (end - 1) // NUM_VOCAB_BLOCKS

    for token_idx in tl.range(first_row, last_row + 1):
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
        if use_penalty:
            pos = tl.load(expanded_local_pos_ptr + token_idx)
            start_idx = token_idx - pos
            row_base = token_idx * NUM_VOCAB_BLOCKS
            lo = tl.maximum(start - row_base, 0)
            hi = tl.minimum(end - row_base, NUM_VOCAB_BLOCKS)

            for vocab_block_idx in tl.range(lo, hi):
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
    if num_tokens == 0:
        return
    num_vocab_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    num_workers = max(1, get_vectorcore_num())
    total_tiles = num_tokens * num_vocab_blocks
    tiles_per_worker = triton.cdiv(total_tiles, num_workers)
    _penalties_kernel[(num_workers,)](
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
        total_tiles,
        tiles_per_worker,
        NUM_VOCAB_BLOCKS=num_vocab_blocks,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
    )


@triton.jit
def _bincount_kernel(
    expanded_idx_mapping_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    prefill_len_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if block_idx * BLOCK_SIZE >= prefill_len:
        return

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if block_idx * BLOCK_SIZE < prompt_len:
        mask = block < prompt_len
        prompt_tokens = tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask)
        idx = prompt_tokens // 32

        bit_idx = prompt_tokens % 32
        bit = tl.full((BLOCK_SIZE,), 1, tl.int32) << bit_idx

        tl.atomic_or(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + idx,
            bit,
            mask=mask,
        )

    if (block_idx + 1) * BLOCK_SIZE >= prompt_len:
        mask = block < prefill_len
        mask &= block >= prompt_len
        output_tokens = tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask)
        tl.atomic_add(
            output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + output_tokens,
            1,
            mask=mask,
        )


def bincount(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_prefill_len: int,
) -> None:
    prompt_bin_mask[expanded_idx_mapping] = 0
    output_bin_counts[expanded_idx_mapping] = 0
    num_tokens = expanded_idx_mapping.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(max_prefill_len, BLOCK_SIZE)
    _bincount_kernel[(num_tokens, num_blocks)](
        expanded_idx_mapping,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
