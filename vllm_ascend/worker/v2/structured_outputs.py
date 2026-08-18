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

from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import (
    get_vectorcore_num,
    init_device_properties_triton,
)


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
# Ascend NPU bitmask kernel (BLOCK_SIZE_SUB tiling)
# TODO: Optimize the kernel performance with NPU profiling data.
@triton.jit
def _apply_grammar_bitmask_kernel_impl(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    total_tasks,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    BLOCK_SIZE_SUB: tl.constexpr = 8192
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Split the original (bitmask row, vocab block) work evenly across the
    # programs in the Ascend launch. This changes scheduling only; each
    # logical task still owns the same BLOCK_SIZE-wide output range.
    tasks_per_program = total_tasks // num_programs
    remainder = total_tasks % num_programs
    task_count = tasks_per_program + tl.where(pid < remainder, 1, 0)
    task_start = pid * tasks_per_program + tl.minimum(pid, remainder)

    # Test a packed bit directly instead of applying a per-lane variable shift.
    bit_mask = tl.full((32,), 1, tl.int32) << tl.arange(0, 32)

    for local_task in tl.range(0, task_count):
        task_id = task_start + local_task
        bitmask_idx = task_id // NUM_VOCAB_BLOCKS
        block_id = task_id - bitmask_idx * NUM_VOCAB_BLOCKS
        logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

        # Keep the existing 1024-token sub-block tiling to preserve the
        # current Ascend UB workaround.
        for sub_offset in tl.range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            global_token_offset = block_id * BLOCK_SIZE + sub_offset
            bitmask_word_start = global_token_offset // 32
            bitmask_offset = bitmask_word_start + tl.arange(0, BLOCK_SIZE_SUB // 32)
            packed_bitmask = tl.load(
                bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
                mask=bitmask_offset < bitmask_stride,
                other=0,
            )
            bitmask = (packed_bitmask[:, None] & bit_mask[None, :]) == 0
            bitmask = bitmask.reshape(BLOCK_SIZE_SUB)

            # Apply: set blocked positions to -inf.
            block_offset = global_token_offset + tl.arange(0, BLOCK_SIZE_SUB)
            tl.store(
                logits_ptr + logits_idx * logits_stride + block_offset,
                -float("inf"),
                mask=bitmask & (block_offset < vocab_size),
            )


class _ApplyGrammarBitmaskKernelLauncher:
    """Adapt the upstream logical grid to an Ascend-specific launch grid.

    vLLM calls `_apply_grammar_bitmask_kernel[grid](...)`, where `grid` is
    `(num_masks, num_vocab_blocks)`. Keep that call ABI unchanged and use the
    two dimensions only to recover the total number of logical tasks.
    """

    def __getitem__(self, grid):
        num_masks, num_vocab_blocks = grid
        total_tasks = num_masks * num_vocab_blocks

        # vLLM-Ascend stores the detected VectorCore count in its Triton device
        # properties. The initializer is guarded, so repeated calls do not
        # re-query the device once the properties have been populated.
        init_device_properties_triton()
        num_programs = min(get_vectorcore_num(), total_tasks)
        ascend_grid = (num_programs,)

        def launch(
            logits_ptr,
            logits_stride,
            logits_indices_ptr,
            bitmask_ptr,
            bitmask_stride,
            vocab_size,
            BLOCK_SIZE,
        ):
            return _apply_grammar_bitmask_kernel_impl[ascend_grid](
                logits_ptr,
                logits_stride,
                logits_indices_ptr,
                bitmask_ptr,
                bitmask_stride,
                vocab_size,
                total_tasks,
                NUM_VOCAB_BLOCKS=num_vocab_blocks,
                BLOCK_SIZE=BLOCK_SIZE,
                multibuffer=False,
            )

        return launch


# Keep the symbol patched by patch_triton.py compatible with the upstream
# `_apply_grammar_bitmask_kernel[grid](...)` launch syntax.
_apply_grammar_bitmask_kernel = _ApplyGrammarBitmaskKernelLauncher()
