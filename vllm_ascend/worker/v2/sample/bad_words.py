# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/bad_words.py.
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

MAX_BAD_WORDS_TOTAL_TOKENS = 1024  # Max total tokens for all bad words per request
MAX_NUM_BAD_WORDS = 128  # Max number of bad words per request


@triton.jit(do_not_specialize=["num_tokens", "max_num_bad_words"])
def _bad_words_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    bad_word_token_ids_ptr,
    bad_word_token_ids_stride,
    bad_word_offsets_ptr,
    bad_word_offsets_stride,
    num_bad_words_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    num_tokens,
    max_num_bad_words,
    MAX_PREFIX_LEN: tl.constexpr,
):
    """
    Optimized bad words filtering kernel for Ascend NPU.

    Key optimizations:
    - Optimized memory access patterns
    - Reduced redundant calculations
    - Enhanced data locality
    - Minimized conditional branches
    - Improved load balancing
    """
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    # Calculate tokens per core for better load balancing
    tokens_per_core = (num_tokens + num_cores - 1) // num_cores
    start_token = pid * tokens_per_core
    end_token = min(start_token + tokens_per_core, num_tokens)

    # Process each token assigned to this core
    for token_idx in range(start_token, end_token):
        # Load request state index
        req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
        num_bad_words = tl.load(num_bad_words_ptr + req_state_idx)

        # Only process if there are bad words for this request
        if num_bad_words > 0:
            # Load position information
            pos = tl.load(expanded_local_pos_ptr + token_idx)
            cur_req_first_pos = token_idx - pos

            # Load length information
            prompt_len = tl.load(prompt_len_ptr + req_state_idx)
            total_len = tl.load(total_len_ptr + req_state_idx)
            output_len = total_len - prompt_len
            effective_len = output_len + pos

            # Precompute base addresses
            bd_offsets_base = bad_word_offsets_ptr + req_state_idx * bad_word_offsets_stride
            bd_tokens_base = bad_word_token_ids_ptr + req_state_idx * bad_word_token_ids_stride
            output_base = all_token_ids_ptr + req_state_idx * all_token_ids_stride + prompt_len

            # Process each bad word for this token
            for bw_idx in range(max_num_bad_words):
                if bw_idx < num_bad_words:
                    # Load bad word range
                    start = tl.load(bd_offsets_base + bw_idx)
                    end = tl.load(bd_offsets_base + bw_idx + 1)
                    bad_word_len = end - start
                    prefix_len = bad_word_len - 1

                    # Check prefix length validity
                    if prefix_len <= effective_len:
                        # Load last token
                        last_token = tl.load(bd_tokens_base + end - 1)

                        # Match checking with early termination
                        match = 1
                        j = 0
                        while j < prefix_len and match:
                            # Load expected token
                            expected = tl.load(bd_tokens_base + start + j)

                            # Calculate actual position and load actual token
                            actual_pos = effective_len - prefix_len + j
                            if actual_pos >= output_len:
                                spec_offset = actual_pos - output_len
                                actual = tl.load(input_ids_ptr + cur_req_first_pos + spec_offset)
                            else:
                                actual = tl.load(output_base + actual_pos)

                            # Check for mismatch
                            if expected != actual:
                                match = 0
                            j += 1

                        # Store result if match found
                        if match:
                            tl.store(logits_ptr + token_idx * logits_stride + last_token, -float("inf"))


def apply_bad_words(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    bad_word_token_ids: torch.Tensor,
    bad_word_offsets: torch.Tensor,
    num_bad_words: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    max_num_bad_words: int,
) -> None:
    """
    Apply bad words filtering to logits.

    Args:
        logits: [num_tokens, vocab_size] - Model output logits
        expanded_idx_mapping: [num_tokens] - Token to request mapping
        bad_word_token_ids: [max_num_reqs, MAX_BAD_WORDS_TOTAL_TOKENS] - Bad word token IDs
        bad_word_offsets: [max_num_reqs, MAX_NUM_BAD_WORDS + 1] - Bad word offsets
        num_bad_words: [max_num_reqs] - Number of bad words per request
        all_token_ids: [max_num_reqs, max_seq_len] - All token IDs
        prompt_len: [max_num_reqs] - Prompt length
        total_len: [max_num_reqs] - Total length
        input_ids: [num_tokens] - Input IDs
        expanded_local_pos: [num_tokens] - Expanded local position
        max_num_bad_words: Maximum number of bad words to check
    """
    num_tokens = logits.shape[0]

    core_num = get_vectorcore_num()

    MAX_PREFIX_LEN = 32

    _bad_words_kernel[(core_num,)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        bad_word_token_ids,
        bad_word_token_ids.stride(0),
        bad_word_offsets,
        bad_word_offsets.stride(0),
        num_bad_words,
        all_token_ids,
        all_token_ids.stride(0),
        prompt_len,
        total_len,
        input_ids,
        expanded_local_pos,
        num_tokens,
        max_num_bad_words,
        MAX_PREFIX_LEN,
    )
