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
from vllm.v1.sample.metadata import SamplingMetadata


@triton.jit
def _penalties_and_temperature_kernel(
    logits_ptr,
    logits_stride,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    temperature_ptr,
    idx_mapping_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    rep_penalty = tl.load(repetition_penalty_ptr + batch_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + batch_idx)
    pres_penalty = tl.load(presence_penalty_ptr + batch_idx)
    temperature = tl.load(temperature_ptr + batch_idx)
    temperature = tl.where(temperature == 0.0, 1.0, temperature)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0
    # NOTE(Ronald1995): vllm original grammar `use_rep_penalty or
    # use_freq_penalty or use_pres_penalty`,
    # change it to `(use_rep_penalty or use_freq_penalty) or use_pres_penalty`,
    # because triton-ascend's compiler doesn't support chained boolean operator.
    use_penalty = (use_rep_penalty or use_freq_penalty) or use_pres_penalty
    use_temperature = temperature != 1.0
    if not (use_penalty or use_temperature):
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + batch_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)

    if use_penalty:
        req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
        output_bin_counts = tl.load(
            output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
            mask=mask,
        )
        # to use vector core, if use > 0 will use scalar to slow down performance
        output_bin_mask = output_bin_counts != 0

        # Apply repetition penalties.
        if use_rep_penalty:
            packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
            packed_mask = tl.load(
                prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
                mask=packed_block < tl.cdiv(vocab_size, 32),
            )
            # the compiler itself does not optimize right-shift operations, so we change the same func
            bit_masks = 1 << tl.arange(0, 32)
            bit_masks_expanded = bit_masks[None, :]

            packed_expanded = packed_mask[:, None]
            bits_matrix = (packed_expanded & bit_masks_expanded) != 0

            prompt_bin_mask = bits_matrix.reshape(BLOCK_SIZE)

            prompt_bin_mask = prompt_bin_mask.to(tl.int1)
            prompt_bin_mask = prompt_bin_mask.reshape(BLOCK_SIZE)

            # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
            scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
            # If logits are positive, divide by penalty, otherwise multiply by penalty.
            logits *= tl.where(logits > 0, 1.0 / scale, scale)

        # Apply frequency penalties.
        logits -= freq_penalty * output_bin_counts
        # Apply presence penalties.
        logits -= pres_penalty * output_bin_mask

    # Apply temperature.
    logits = logits / temperature

    # Store back to logits.
    tl.store(logits_ptr + batch_idx * logits_stride + block, logits, mask=mask)


def apply_penalties_and_temperature(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> None:
    """Override the function because there are some bugs
    when _penalties_and_temperature_kernel runs on npu, we need to make some fixes.
    you could read NOTE(Ronald1995) comments to understand.
    """
    num_reqs, vocab_size = logits.shape
    # NOTE(Ronald1995): change BLOCK_SIZE from 8192 to 4096 in case UB overflow
    # in triton-ascend.
    BLOCK_SIZE = 4096
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    # TODO(Ronald1995): Optimize the performance of the kernel in npu.
    _penalties_and_temperature_kernel[(num_reqs, num_blocks)](
        logits,
        logits.stride(0),
        sampling_metadata.repetition_penalty,
        sampling_metadata.frequency_penalty,
        sampling_metadata.presence_penalty,
        sampling_metadata.temperature,
        sampling_metadata.idx_mapping,
        sampling_metadata.prompt_bin_mask,
        sampling_metadata.prompt_bin_mask.stride(0),
        sampling_metadata.output_bin_counts,
        sampling_metadata.output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
