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
# NPU-compatible structured output bitmask kernel.
#
# Upstream cannot be used directly on Ascend NPU: `BLOCK_SIZE=8192` overflows
# UB, while a smaller `BLOCK_SIZE` makes the grid unstable. We therefore keep
# `BLOCK_SIZE=8192` and split each block with `BLOCK_SIZE_SUB=1024`.
#
#

from vllm.triton_utils import tl, triton


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
# Ascend NPU bitmask kernel (BLOCK_SIZE_SUB tiling)
# TODO: Optimize the kernel performance with NPU profiling data.
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    BLOCK_SIZE_SUB: tl.constexpr = 1024
    bitmask_idx = tl.program_id(0)
    block_id = tl.program_id(1)
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

    # Sub-block tiling loop: process BLOCK_SIZE_SUB tokens per iteration
    for sub_offset in tl.range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
        global_token_offset = block_id * BLOCK_SIZE + sub_offset
        bitmask_word_start = global_token_offset // 32
        bitmask_offset = bitmask_word_start + tl.arange(0, BLOCK_SIZE_SUB // 32)
        packed_bitmask = tl.load(
            bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
            mask=bitmask_offset < bitmask_stride,
            other=0,
        )
        bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
        bitmask = bitmask.reshape(BLOCK_SIZE_SUB)

        # Apply: set blocked positions to -inf
        block_offset = global_token_offset + tl.arange(0, BLOCK_SIZE_SUB)
        tl.store(
            logits_ptr + logits_idx * logits_stride + block_offset,
            -float("inf"),
            mask=bitmask & (block_offset < vocab_size),
        )
