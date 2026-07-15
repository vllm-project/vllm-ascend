# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_kernel(
    num_tokens,
    max_num_tokens,
    num_reqs,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32
    block_table_stride,
    block_size,
    slot_mapping_ptr,  # [max_num_tokens], int32
    KV_CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_KV_BLOCK: tl.constexpr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx >= num_reqs:
        # Pad remaining slots for CUDA graph compatibility. Use one program per
        # BLOCK_SIZE tile instead of making a single program sweep the tail.
        pad_block_idx = req_idx - num_reqs
        offsets = num_tokens + pad_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(slot_mapping_ptr + offsets, PAD_ID, mask=offsets < max_num_tokens)
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1)
    row_offset = req_idx * block_table_stride
    block_table_offsets = tl.arange(0, BLOCK_TABLE_WINDOW_SIZE)
    virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE

    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        positions = tl.load(positions_ptr + offsets, mask=mask, other=0).to(tl.int32)

        virtual_block_indices = positions // virtual_block_size
        virtual_block_offsets = positions - virtual_block_size * virtual_block_indices

        if TOTAL_CP_WORLD_SIZE == 1:
            is_local = mask
            local_block_offsets = virtual_block_offsets
        else:
            interleave_chunks = virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
            rank_in_chunk = interleave_chunks - TOTAL_CP_WORLD_SIZE * (interleave_chunks // TOTAL_CP_WORLD_SIZE)
            is_local = rank_in_chunk == TOTAL_CP_RANK
            rounds = virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            remainder_base = virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
            remainder = virtual_block_offsets - CP_KV_CACHE_INTERLEAVE_SIZE * remainder_base
            local_block_offsets = rounds * CP_KV_CACHE_INTERLEAVE_SIZE + remainder

        local_block_indices = local_block_offsets // block_size
        block_indices = virtual_block_indices * BLOCKS_PER_KV_BLOCK + local_block_indices

        # Non-contiguous block_table loads degrade to scalar on Ascend. Positions
        # are grouped by request, so a token tile only spans a small block window.
        valid_block_indices = tl.where(mask, block_indices, 2147483647)
        block_idx_base = tl.min(valid_block_indices, axis=0)
        block_table_window_offsets = block_idx_base + block_table_offsets
        block_table_window = tl.load(
            block_table_ptr + row_offset + block_table_window_offsets,
            mask=block_table_window_offsets < block_table_stride,
            other=0,
        ).to(tl.float32)
        relative_block_indices = tl.where(mask & is_local, block_indices - block_idx_base, 0)
        block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)

        slot_offsets = local_block_offsets - block_size * local_block_indices
        slot_ids = block_numbers * block_size + slot_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)
