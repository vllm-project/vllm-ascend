# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/min_p.py.
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

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["num_tokens"])
def _min_p_kernel(
    in_logits_ptr,
    out_logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    min_p_ptr,
    vocab_size,
    num_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    core_num = tl.num_programs(0)

    tokens_per_block = (num_tokens + core_num - 1) // core_num
    start_token = pid * tokens_per_block
    end_token = tl.minimum(start_token + tokens_per_block, num_tokens)

    for token_idx in tl.range(start_token, end_token):
        req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
        min_p = tl.load(min_p_ptr + req_state_idx).to(tl.float32)
        if min_p != 0.0:
            max_val = float("-inf")
            for i in range(0, vocab_size, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < vocab_size
                logits = tl.load(
                    in_logits_ptr + token_idx * logits_stride + block,
                    mask=mask,
                    other=float("-inf"),
                )
                max_val = tl.max(tl.maximum(logits, max_val))
            max_val = max_val.to(tl.float32)  # type: ignore

            threshold = max_val + tl.log(min_p)
            for i in range(0, vocab_size, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < vocab_size
                logits = tl.load(
                    in_logits_ptr + token_idx * logits_stride + block,
                    mask=mask,
                    other=float("-inf"),
                )
                logits = tl.where(logits < threshold, float("-inf"), logits)
                tl.store(out_logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_min_p(logits: torch.Tensor, expanded_idx_mapping: torch.Tensor, min_p: torch.Tensor) -> None:

    num_tokens, vocab_size = logits.shape

    vec_core = get_vectorcore_num()
    core_nums = min(num_tokens, vec_core)

    BLOCK_SIZE = min(triton.next_power_of_2(vocab_size), 8192)

    _min_p_kernel[(core_nums,)](
        logits,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        min_p,
        vocab_size,
        num_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
        multibuffer=False,
    )
