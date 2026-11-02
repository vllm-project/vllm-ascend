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


from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_logsumexp as _compute_global_lse,
)


@triton.jit
def _entropy_partials_kernel(
    logits_ptr,
    logits_stride,
    max_ptr,
    sumexp_ptr,
    stats_stride,
    entropy_ptr,
    mapping_ptr,
    local_pos_ptr,
    temp_ptr,
    vocab_size,
    vocab_num_blocks,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    EPSILON: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    req = tl.load(mapping_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req)
    local_pos = tl.load(local_pos_ptr + row)
    entropy = 0.0
    if temp != 0.0 and local_pos < num_speculative_steps:
        lse = _compute_global_lse(
            max_ptr,
            stats_stride,
            sumexp_ptr,
            stats_stride,
            row,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS,
        )
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(logits_ptr + row * logits_stride + offsets, offsets < vocab_size, other=float("-inf"))
        probs = tl.exp(logits.to(tl.float32) - lse)
        entropy = tl.sum(-probs * tl.log(probs + EPSILON), 0)
    tl.store(entropy_ptr + row * stats_stride + block, entropy)


@triton.jit
def _entropy_verify(
    entropy_partials_ptr,
    target_local_sumexp_stride,
    logit_idx,
    vocab_num_blocks,
    target_log_prob,
    draft_log_prob,
    u,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    POSTERIOR_THRESHOLD: tl.constexpr,
    POSTERIOR_ALPHA: tl.constexpr,
):
    threshold = POSTERIOR_THRESHOLD
    if POSTERIOR_ALPHA > 0:
        blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
        entropy = tl.sum(
            tl.load(
                entropy_partials_ptr + logit_idx * target_local_sumexp_stride + blocks,
                mask=blocks < vocab_num_blocks,
                other=0.0,
            ),
            0,
        )
        threshold = tl.minimum(tl.exp(-entropy * POSTERIOR_ALPHA), POSTERIOR_THRESHOLD)
    target_prob = tl.exp(target_log_prob)
    draft_prob = tl.exp(draft_log_prob)
    valid_prob = draft_prob > 0.0
    return valid_prob & (target_prob / tl.where(valid_prob, draft_prob, 1.0) >= threshold * u)
