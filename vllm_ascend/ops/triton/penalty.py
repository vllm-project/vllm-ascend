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
# Triton-Ascend implementation of apply_penalties.
# Migrated from model_executor/layers/utils.apply_penalties.
# Reference: https://github.com/vllm-project/vllm-ascend/pull/6979

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.bincount import get_token_bin_counts_and_mask_triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def apply_all_penalties_kernel(
    logits_ptr,
    prompt_mask_ptr,
    output_mask_ptr,
    output_bin_counts_ptr,
    repetition_penalties_ptr,
    frequency_penalties_ptr,
    presence_penalties_ptr,
    num_seqs,
    vocab_size,
    stride_logits_seq,
    stride_logits_vocab,
    stride_prompt_mask_seq,
    stride_prompt_mask_vocab,
    stride_output_mask_seq,
    stride_output_mask_vocab,
    stride_bin_counts_seq,
    stride_bin_counts_vocab,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply repetition, frequency, and presence penalties to logits in place."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    seqs_per_program = (num_seqs + num_programs - 1) // num_programs

    start_seq = pid * seqs_per_program
    end_seq = tl.minimum(start_seq + seqs_per_program, num_seqs)

    for seq_idx in range(start_seq, end_seq):
        repetition_penalty = tl.load(repetition_penalties_ptr + seq_idx)
        frequency_penalty = tl.load(frequency_penalties_ptr + seq_idx)
        presence_penalty = tl.load(presence_penalties_ptr + seq_idx)

        for vocab_start in range(0, vocab_size, BLOCK_SIZE):
            vocab_offsets = vocab_start + tl.arange(0, BLOCK_SIZE)
            mask = vocab_offsets < vocab_size

            logits_offset = seq_idx * stride_logits_seq + vocab_offsets * stride_logits_vocab
            prompt_mask_offset = seq_idx * stride_prompt_mask_seq + vocab_offsets * stride_prompt_mask_vocab
            output_mask_offset = seq_idx * stride_output_mask_seq + vocab_offsets * stride_output_mask_vocab
            counts_offset = seq_idx * stride_bin_counts_seq + vocab_offsets * stride_bin_counts_vocab

            logits = tl.load(logits_ptr + logits_offset, mask=mask, other=0.0)
            prompt_mask_val = tl.load(prompt_mask_ptr + prompt_mask_offset, mask=mask, other=False)
            output_mask_val = tl.load(output_mask_ptr + output_mask_offset, mask=mask, other=False)
            output_bin_counts = tl.load(
                output_bin_counts_ptr + counts_offset,
                mask=mask,
                other=0,
            ).to(tl.float32)

            need_repetition_penalty = (prompt_mask_val | output_mask_val).to(tl.int1)
            penalty_factor = tl.where(need_repetition_penalty, repetition_penalty, 1.0)
            scaling = tl.where(
                (logits > 0.0).to(tl.int1),
                1.0 / penalty_factor,
                penalty_factor,
            )
            updated = logits * scaling

            updated -= frequency_penalty * output_bin_counts
            updated -= presence_penalty * output_mask_val.to(tl.float32)
            tl.store(logits_ptr + logits_offset, updated, mask=mask)


def apply_penalties_triton(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """Apply penalties to logits in place. Same interface as
    model_executor.layers.utils.apply_penalties.
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask_triton(prompt_tokens_tensor, vocab_size, num_seqs)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask_triton(
        output_tokens_tensor,
        vocab_size,
        num_seqs,
    )
    _apply_all_penalties_triton(
        logits,
        prompt_mask,
        output_mask,
        output_bin_counts,
        repetition_penalties,
        frequency_penalties,
        presence_penalties,
    )
    return logits


def _apply_all_penalties_triton(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    repetition_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    presence_penalties: torch.Tensor,
) -> None:
    """Apply all penalties given precomputed bin counts and masks."""
    num_seqs, vocab_size = logits.shape
    grid = (min(num_seqs, get_vectorcore_num()), 1, 1)

    apply_all_penalties_kernel[grid](
        logits,
        prompt_mask,
        output_mask,
        output_bin_counts,
        repetition_penalties,
        frequency_penalties,
        presence_penalties,
        num_seqs=num_seqs,
        vocab_size=vocab_size,
        stride_logits_seq=logits.stride(0),
        stride_logits_vocab=logits.stride(1),
        stride_prompt_mask_seq=prompt_mask.stride(0),
        stride_prompt_mask_vocab=prompt_mask.stride(1),
        stride_output_mask_seq=output_mask.stride(0),
        stride_output_mask_vocab=output_mask.stride(1),
        stride_bin_counts_seq=output_bin_counts.stride(0),
        stride_bin_counts_vocab=output_bin_counts.stride(1),
        BLOCK_SIZE=2048,
    )
