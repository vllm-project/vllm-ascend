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

from vllm_ascend.ops.triton.triton_utils import (
    get_vectorcore_num,
    init_device_properties_triton,
)


@triton.jit
def _apply_grammar_bitmask_kernel_impl(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    cu_num_logits_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    total_tasks,
    MASK_STRIDE: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    tasks_per_program = total_tasks // NUM_PROGRAMS
    remainder = total_tasks % NUM_PROGRAMS
    task_start = pid * tasks_per_program + tl.minimum(pid, remainder)
    task_end = task_start + tasks_per_program + tl.where(pid < remainder, 1, 0)

    bit_mask = tl.full((32,), 1, tl.int32) << tl.arange(0, 32)

    for task_id in tl.range(task_start, task_end):
        bitmask_idx = task_id // NUM_VOCAB_BLOCKS
        block_id = task_id - bitmask_idx * NUM_VOCAB_BLOCKS
        # The mapping is keyed by (request, position) rather than the absolute
        # logit index: adaptive verification finalizes per-request logit
        # offsets on device, so the kernel resolves the logits row from the
        # GPU cu_num_logits and skips positions trimmed by the draft budget.
        mapping_idx = tl.load(logits_indices_ptr + bitmask_idx)
        req_idx = mapping_idx // MASK_STRIDE
        position_idx = mapping_idx - req_idx * MASK_STRIDE
        logits_start = tl.load(cu_num_logits_ptr + req_idx)
        num_req_logits = tl.load(cu_num_logits_ptr + req_idx + 1) - logits_start
        logits_idx = logits_start + position_idx
        position_is_active = position_idx < num_req_logits

        bitmask_offset = block_id * (BLOCK_SIZE // 32) + tl.arange(0, BLOCK_SIZE // 32)
        packed_bitmask = tl.load(
            bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
            mask=bitmask_offset < bitmask_stride,
            other=0,
        )
        blocked = (packed_bitmask[:, None] & bit_mask[None, :]) == 0
        blocked = blocked.reshape(BLOCK_SIZE)

        block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(
            logits_ptr + logits_idx * logits_stride + block_offset,
            -float("inf"),
            mask=position_is_active & blocked & (block_offset < vocab_size),
        )


class _ApplyGrammarBitmaskKernelLauncher:
    """Map the upstream logical grid to the Ascend VectorCore launch grid."""

    def __getitem__(self, grid):
        num_masks, num_vocab_blocks = grid
        total_tasks = num_masks * num_vocab_blocks

        init_device_properties_triton()
        num_programs = min(get_vectorcore_num(), total_tasks)

        def launch(
            logits_ptr,
            logits_stride,
            logits_indices_ptr,
            cu_num_logits_ptr,
            bitmask_ptr,
            bitmask_stride,
            vocab_size,
            MASK_STRIDE,
            BLOCK_SIZE,
        ):
            # Disable MULTIBUFFER to better utilize the UB buffer, which gives
            # a significant performance improvement on A2/A3. It can be enabled
            # again on A5.
            return _apply_grammar_bitmask_kernel_impl[(num_programs,)](
                logits_ptr,
                logits_stride,
                logits_indices_ptr,
                cu_num_logits_ptr,
                bitmask_ptr,
                bitmask_stride,
                vocab_size,
                total_tasks,
                MASK_STRIDE=MASK_STRIDE,
                NUM_PROGRAMS=num_programs,
                NUM_VOCAB_BLOCKS=num_vocab_blocks,
                BLOCK_SIZE=BLOCK_SIZE,
                multibuffer=False,
            )

        return launch


_apply_grammar_bitmask_kernel = _ApplyGrammarBitmaskKernelLauncher()
