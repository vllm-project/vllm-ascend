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


@triton.jit
def _penalties_kernel(
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    token_ids_ptr,
    expanded_local_pos_ptr,
    penalties_ptr,
    penalties_stride,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    INNER_BLOCK_SIZE: tl.constexpr,
    MAX_SPEC_LEN: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + token_idx)

    # first load penalties once
    rep_penalty = tl.load(penalties_ptr + req_state_idx * penalties_stride + 0)
    freq_penalty = tl.load(penalties_ptr + req_state_idx * penalties_stride + 1)
    pres_penalty = tl.load(penalties_ptr + req_state_idx * penalties_stride + 2)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0

    # NPU doesn't support chained 'or' operations like 'A or B or C'
    use_penalty = use_rep_penalty or use_freq_penalty
    use_penalty = use_penalty or use_pres_penalty

    if not use_penalty:
        # Early return to avoid loading logits.
        return

    bit_masks = tl.full((INNER_BLOCK_SIZE // 32, 32), 1, dtype=tl.int32) << tl.arange(0, 32)
    block_idx = tl.program_id(1)
    block_start = block_idx * BLOCK_SIZE

    pos = tl.load(expanded_local_pos_ptr + token_idx)
    start_idx = token_idx - pos

    inv_rep = 1.0 / rep_penalty

    for inner_offset in tl.static_range(0, BLOCK_SIZE, INNER_BLOCK_SIZE):
        inner_block_start = block_start + inner_offset
        inner_block = inner_block_start + tl.arange(0, INNER_BLOCK_SIZE)
        inner_mask = inner_block < vocab_size

        logits = tl.load(logits_ptr + token_idx * logits_stride + inner_block, mask=inner_mask, other=0.0)
        logits = logits.to(tl.float32)

        base_output_counts = tl.load(
            output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + inner_block,
            mask=inner_mask,
            other=0,
        )

        # Compute cumulative draft_counts from previous positions in this request
        total_counts = base_output_counts.to(tl.int32)
        for prev_pos in tl.static_range(MAX_SPEC_LEN):
            if prev_pos < pos:
                load_idx = start_idx + prev_pos + 1
                prev_token = tl.load(token_ids_ptr + load_idx)
                total_counts += inner_block == prev_token

        is_present = total_counts != 0

        # Apply repetition penalties.
        if use_rep_penalty:
            packed_inner_block_start = inner_block_start // 32
            packed_block = packed_inner_block_start + tl.arange(0, INNER_BLOCK_SIZE // 32)
            valid_packed_mask = packed_block < tl.cdiv(vocab_size, 32)

            packed_mask_val = tl.load(
                prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
                mask=valid_packed_mask,
                other=0,
            )
            prompt_mask = ((packed_mask_val[:, None] & bit_masks) != 0).reshape(INNER_BLOCK_SIZE)

            needs_scaling = prompt_mask | is_present

            base_factor = tl.where(logits > 0, inv_rep, rep_penalty)
            logits = tl.where(needs_scaling, logits * base_factor, logits)

        freq_term = freq_penalty * total_counts.to(tl.float32)
        pres_term = pres_penalty * is_present.to(tl.float32)

        logits = logits - freq_term - pres_term
        # Store back to logits.
        tl.store(logits_ptr + token_idx * logits_stride + inner_block, logits, mask=inner_mask)


def apply_penalties(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    num_speculative_tokens: int,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    INNER_BLOCK_SIZE = 4096
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)

    penalties = torch.stack(
        [repetition_penalty[:num_tokens], frequency_penalty[:num_tokens], presence_penalty[:num_tokens]], dim=1
    ).contiguous()
    penalties_stride = penalties.stride(0)

    _penalties_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        idx_mapping,
        token_ids,
        expanded_local_pos,
        penalties,
        penalties_stride,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        INNER_BLOCK_SIZE=INNER_BLOCK_SIZE,
        MAX_SPEC_LEN=num_speculative_tokens,
    )
