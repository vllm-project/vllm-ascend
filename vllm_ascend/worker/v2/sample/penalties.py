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

from functools import lru_cache

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import (
    get_vectorcore_num,
    init_device_properties_triton,
)

_DEFAULT_PROGRAM_COUNT = 32


@lru_cache(maxsize=1)
def _detected_vectorcore_count() -> int:
    """Return the current device's VectorCore count."""
    try:
        vectorcore_count = int(get_vectorcore_num())
    except AssertionError:
        init_device_properties_triton()
        vectorcore_count = int(get_vectorcore_num())

    if vectorcore_count <= 0:
        raise RuntimeError(f"Invalid VectorCore count returned by the runtime: {vectorcore_count}")
    return vectorcore_count


@triton.jit(do_not_specialize=["num_tokens", "vocab_size"])
def _apply_penalties_vocab_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    num_tokens,
    vocab_size,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply prompt and committed-output penalties over the full vocabulary."""
    tl.static_assert(BLOCK_SIZE % 32 == 0)

    program_id = tl.program_id(0)
    row_start = program_id * ROWS_PER_PROGRAM
    row_end = tl.minimum(row_start + ROWS_PER_PROGRAM, num_tokens)

    vocab_offsets = tl.arange(0, BLOCK_SIZE)
    packed_offsets = tl.arange(0, BLOCK_SIZE // 32)
    bit_masks = tl.full((32,), 1, tl.int32) << tl.arange(0, 32)
    packed_vocab_size = (vocab_size + 31) // 32

    for token_idx in tl.range(row_start, row_end):
        req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
        rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
        freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
        pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

        use_rep_penalty = rep_penalty != 1.0
        use_freq_penalty = freq_penalty != 0.0
        use_pres_penalty = pres_penalty != 0.0
        use_penalty = use_rep_penalty or use_freq_penalty
        use_penalty = use_penalty or use_pres_penalty

        if use_penalty:
            logits_row_ptr = logits_ptr + token_idx * logits_stride
            output_counts_row_ptr = output_bin_counts_ptr + req_state_idx * output_bin_counts_stride
            prompt_mask_row_ptr = prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride
            inverse_rep_penalty = 1.0 / rep_penalty

            logits_tile_ptr = logits_row_ptr + vocab_offsets
            output_counts_tile_ptr = output_counts_row_ptr + vocab_offsets
            prompt_mask_tile_ptr = prompt_mask_row_ptr + packed_offsets

            for block_idx in tl.range(0, NUM_VOCAB_BLOCKS):
                block_start = block_idx * BLOCK_SIZE
                valid_mask = block_start + vocab_offsets < vocab_size

                logits = tl.load(
                    logits_tile_ptr,
                    mask=valid_mask,
                    other=0.0,
                ).to(tl.float32)
                output_counts = tl.load(
                    output_counts_tile_ptr,
                    mask=valid_mask,
                    other=0,
                ).to(tl.int32)
                output_seen = output_counts != 0

                if use_rep_penalty:
                    packed_start = block_idx * (BLOCK_SIZE // 32)
                    packed_mask = tl.load(
                        prompt_mask_tile_ptr,
                        mask=(packed_start + packed_offsets < packed_vocab_size),
                        other=0,
                    ).to(tl.int32)
                    prompt_seen = ((packed_mask[:, None] & bit_masks[None, :]) != 0).reshape(BLOCK_SIZE)
                    repetition_seen = prompt_seen | output_seen
                    repetition_result = tl.where(
                        logits > 0.0,
                        logits * inverse_rep_penalty,
                        logits * rep_penalty,
                    )
                    logits = tl.where(
                        repetition_seen,
                        repetition_result,
                        logits,
                    )

                if use_freq_penalty:
                    logits -= freq_penalty * output_counts.to(tl.float32)

                if use_pres_penalty:
                    logits -= pres_penalty * output_seen.to(tl.float32)

                tl.store(logits_tile_ptr, logits, mask=valid_mask)

                logits_tile_ptr += BLOCK_SIZE
                output_counts_tile_ptr += BLOCK_SIZE
                prompt_mask_tile_ptr += BLOCK_SIZE // 32


@triton.jit
def _apply_penalties_draft_kernel(
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
):
    """Apply corrections from earlier positions selected by local_pos."""
    token_idx = tl.program_id(0)
    local_pos = tl.load(expanded_local_pos_ptr + token_idx)

    # Match the reference implementation by ignoring positions before
    # the beginning of token_ids.
    local_pos = tl.minimum(local_pos, token_idx + 1)

    if local_pos <= 0:
        return

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0
    use_penalty = use_rep_penalty or use_freq_penalty
    use_penalty = use_penalty or use_pres_penalty
    if not use_penalty:
        return

    logits_row_ptr = logits_ptr + token_idx * logits_stride
    output_counts_row_ptr = output_bin_counts_ptr + req_state_idx * output_bin_counts_stride
    prompt_mask_row_ptr = prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride
    inverse_rep_penalty = 1.0 / rep_penalty
    draft_start_idx = token_idx - local_pos

    for candidate_pos in tl.range(local_pos):
        draft_token = tl.load(token_ids_ptr + draft_start_idx + candidate_pos + 1)

        draft_count = 0 * local_pos  # to ensure the type is tl.int32
        earlier_count = 0 * local_pos  # to ensure the type is tl.int32
        for scan_pos in tl.range(local_pos):
            scan_token = tl.load(token_ids_ptr + draft_start_idx + scan_pos + 1)
            is_same = scan_token == draft_token
            draft_count += is_same.to(tl.int32)
            earlier_count += (is_same & (scan_pos < candidate_pos)).to(tl.int32)

        # Only the first occurrence owns this logits element.
        if earlier_count == 0:
            base_output_count = tl.load(output_counts_row_ptr + draft_token)
            logit = tl.load(logits_row_ptr + draft_token).to(tl.float32)

            if use_rep_penalty:
                packed_word = tl.load(prompt_mask_row_ptr + draft_token // 32)
                prompt_seen = (packed_word & (1 << (draft_token % 32))) != 0
                repetition_already_applied = prompt_seen or (base_output_count != 0)
                if not repetition_already_applied:
                    logit = tl.where(
                        logit > 0.0,
                        logit * inverse_rep_penalty,
                        logit * rep_penalty,
                    )

            if use_freq_penalty:
                logit -= freq_penalty * draft_count.to(tl.float32)

            # Presence was already applied by the vocabulary kernel when the
            # token existed in committed output history.
            if use_pres_penalty:
                if base_output_count == 0:
                    logit -= pres_penalty

            tl.store(logits_row_ptr + draft_token, logit)


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
    """Apply repetition, frequency, and presence penalties in place."""
    num_tokens, vocab_size = logits.shape
    if num_tokens == 0 or vocab_size == 0:
        return

    program_count = min(max(_detected_vectorcore_count(), _DEFAULT_PROGRAM_COUNT), num_tokens)
    BLOCK_SIZE = 4096

    _apply_penalties_vocab_kernel[(program_count,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        num_tokens,
        vocab_size,
        NUM_VOCAB_BLOCKS=triton.cdiv(vocab_size, BLOCK_SIZE),
        ROWS_PER_PROGRAM=triton.cdiv(num_tokens, program_count),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    _apply_penalties_draft_kernel[(num_tokens,)](
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
        prompt_tokens = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + block,
            mask=mask,
        )
        packed_indices = prompt_tokens // 32
        bit_indices = prompt_tokens % 32
        bits = tl.full((BLOCK_SIZE,), 1, tl.int32) << bit_indices
        tl.atomic_or(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_indices,
            bits,
            mask=mask,
        )

    if (block_idx + 1) * BLOCK_SIZE >= prompt_len:
        mask = (block >= prompt_len) & (block < prefill_len)
        output_tokens = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + block,
            mask=mask,
        )
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
    if num_tokens == 0 or max_prefill_len <= 0:
        return

    BLOCK_SIZE = 1024
    _bincount_kernel[(num_tokens, triton.cdiv(max_prefill_len, BLOCK_SIZE))](
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
