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
    logits_stride: tl.constexpr,
    logits_indices_ptr,
    cu_num_logits_ptr,
    bitmask_ptr,
    bitmask_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    num_masks,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    MASK_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply masks with mask-major scheduling to reuse request metadata."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    masks_per_program = tl.cdiv(num_masks, num_programs)
    start_mask = pid * masks_per_program
    end_mask = tl.minimum(start_mask + masks_per_program, num_masks)
    bit_mask = tl.full((32,), 1, tl.int32) << tl.arange(0, 32)

    for bitmask_idx in tl.range(start_mask, end_mask):
        # Resolve the request/position mapping once for all vocabulary blocks.
        mapping_idx = tl.load(logits_indices_ptr + bitmask_idx)
        req_idx = mapping_idx // MASK_STRIDE
        position_idx = mapping_idx % MASK_STRIDE
        logits_idx = tl.load(cu_num_logits_ptr + req_idx)
        num_req_logits = tl.load(cu_num_logits_ptr + req_idx + 1) - logits_idx
        logits_idx += position_idx
        if position_idx < num_req_logits:
            for block_id in tl.range(0, NUM_VOCAB_BLOCKS):
                bitmask_offset = block_id * (BLOCK_SIZE // 32) + tl.arange(0, BLOCK_SIZE // 32)
                valid_words = bitmask_offset < tl.cdiv(vocab_size, 32)
                packed_bitmask = tl.load(
                    bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
                    mask=valid_words & (bitmask_offset < bitmask_stride),
                    other=0,
                )
                any_blocked = tl.sum((valid_words & (packed_bitmask != -1)).to(tl.int32), 0) > 0
                if any_blocked:
                    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    ptrs = logits_ptr + logits_idx * logits_stride + block_offset
                    valid = block_offset < vocab_size
                    any_allowed = tl.sum((valid_words & (packed_bitmask != 0)).to(tl.int32), 0) > 0
                    if any_allowed:
                        blocked = ((packed_bitmask[:, None] & bit_mask[None, :]) == 0).reshape(BLOCK_SIZE)
                        values = tl.load(ptrs, mask=valid, other=0)
                        negative_inf = tl.full((BLOCK_SIZE,), -float("inf"), values.dtype)
                        tl.store(ptrs, tl.where(blocked, negative_inf, values), mask=valid)
                    else:
                        tl.store(ptrs, -float("inf"), mask=valid)


class _ApplyGrammarBitmaskKernelLauncher:
    """Map the upstream logical grid to the Ascend VectorCore launch grid."""

    def __getitem__(self, grid):
        num_masks, num_vocab_blocks = grid

        init_device_properties_triton()
        num_programs = min(get_vectorcore_num(), num_masks)

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
            if num_masks == 0:
                return None
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
                num_masks,
                NUM_VOCAB_BLOCKS=num_vocab_blocks,
                MASK_STRIDE=MASK_STRIDE,
                BLOCK_SIZE=BLOCK_SIZE,
                multibuffer=False,
            )

        return launch


_apply_grammar_bitmask_kernel = _ApplyGrammarBitmaskKernelLauncher()
