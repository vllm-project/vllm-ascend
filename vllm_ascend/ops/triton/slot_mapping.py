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
# Fused multi-group slot mapping Triton kernel.
# Merges N separate _compute_slot_mapping_kernel launches into a single
# 2D-grid launch, reducing kernel launch overhead on Ascend NPU (910B).
#
# The per-group computation is aligned with the Ascend-optimised single-group
# kernel in ``ops/triton/compute_slot_mapping.py`` (PR #13048):
#   - ``pos`` cast to int32 to reduce scalar-arithmetic overhead;
#   - ``TOTAL_CP_WORLD_SIZE == 1`` dedicated fast path (no CP interleave);
#   - windowed block-table load + ``tl.gather`` to fix non-contiguous access.
#
# Design:
#   - 2D grid: (num_reqs + 1, num_groups)
#       axis 0 → req_idx  (including one padding row)
#       axis 1 → group_idx
#   - Each program handles exactly one (req, group) pair.
#   - All groups run concurrently in the same launch (no strided loop).
#   - Per-group parameter arrays are pre-built once during initialisation
#     (see MultiGroupBlockTable._build_fused_params) and reused every step.

from __future__ import annotations

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def compute_slot_mapping_fused_kernel(
    # ---- scalar inputs (same for every program) -----------------------
    num_tokens,  # int: actual number of tokens in the batch
    max_num_tokens,  # int: max buffer size for padding
    # ---- tensor inputs (shared across all groups / requests) ----------
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    # ---- per-group parameter arrays [num_groups] (pre-built, cached) --
    group_block_table_ptrs,  # [num_groups], int64  (raw data_ptr values)
    group_block_table_strides,  # [num_groups], int32
    group_block_sizes,  # [num_groups], int32
    group_slot_mapping_ptrs,  # [num_groups], int64  (raw data_ptr values)
    group_kv_cache_block_sizes,  # [num_groups], int32
    group_blocks_per_kv,  # [num_groups], int32
    # ---- compile-time constants ---------------------------------------
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TILE_BLOCK_SIZE: tl.constexpr,  # positions tile size (1024)
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,  # window for block-table load
):
    """
    2D-grid fused slot-mapping kernel.
    Grid: ``(num_reqs + 1, num_groups)``
    Each program (req_idx, group_idx):
      1. Loads the per-group parameters for ``group_idx``.
      2. If ``req_idx`` is the last row → pads ``slot_mapping[group]``.
         Otherwise → computes the standard slot-mapping for that request.
    The normal-path logic mirrors ``_compute_slot_mapping_kernel`` in
    ``ops/triton/compute_slot_mapping.py`` (PR #13048), with ``block_size``,
    ``kv_cache_block_size`` and ``blocks_per_kv_block`` loaded dynamically
    (per-group) instead of being ``tl.constexpr``.
    """
    # ---- resolve (req, group) from the 2-D grid -----------------------
    req_idx = tl.program_id(axis=0)
    group_idx = tl.program_id(axis=1)
    num_reqs_plus_one = tl.num_programs(axis=0)

    # ---- load per-group parameters ------------------------------------
    block_table_ptr = tl.load(group_block_table_ptrs + group_idx).to(tl.pointer_type(tl.int32))
    block_table_stride = tl.load(group_block_table_strides + group_idx)
    block_size = tl.load(group_block_sizes + group_idx)
    slot_mapping_ptr = tl.load(group_slot_mapping_ptrs + group_idx).to(tl.pointer_type(tl.int32))
    kv_cache_block_size = tl.load(group_kv_cache_block_sizes + group_idx)
    blocks_per_kv_block = tl.load(group_blocks_per_kv + group_idx)

    # ---- padding row --------------------------------------------------
    if req_idx == num_reqs_plus_one - 1:
        for p in range(num_tokens, max_num_tokens, TILE_BLOCK_SIZE):
            pad_offs = p + tl.arange(0, TILE_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + pad_offs, PAD_ID, mask=pad_offs < max_num_tokens)
        return

    # ---- normal request -----------------------------------------------
    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    row_offset = req_idx * block_table_stride
    block_table_offsets = tl.arange(0, BLOCK_TABLE_WINDOW_SIZE)

    # Sentinel used to mask out-of-range positions before the min reduction.
    INT32_MAX = 2147483647

    for i in range(start_idx, end_idx, TILE_BLOCK_SIZE):
        offsets = i + tl.arange(0, TILE_BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0).to(tl.int32)

        # Compute block indices / slot offsets.
        if TOTAL_CP_WORLD_SIZE == 1:
            # Fast path: no CP interleave arithmetic.
            block_indices = pos // block_size
            slot_offsets = pos - block_indices * block_size
        else:
            virtual_block_size = kv_cache_block_size * TOTAL_CP_WORLD_SIZE
            virtual_block_indices = pos // virtual_block_size
            virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
            is_local = (virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
            local_block_offsets = (
                virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            ) * CP_KV_CACHE_INTERLEAVE_SIZE + (virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE)
            block_indices = virtual_block_indices * blocks_per_kv_block + local_block_offsets // block_size
            slot_offsets = local_block_offsets % block_size

        # Windowed block-table load (fixes non-contiguous access):
        # load [block_idx_base, block_idx_base + WINDOW) once, then gather.
        valid_block_indices = tl.where(mask, block_indices, INT32_MAX)
        block_idx_base = tl.min(valid_block_indices, axis=0)
        block_table_window_offsets = block_idx_base + block_table_offsets
        block_table_window = tl.load(
            block_table_ptr + row_offset + block_table_window_offsets,
            mask=block_table_window_offsets < block_table_stride,
            other=0,
        ).to(tl.float32)

        if TOTAL_CP_WORLD_SIZE == 1:
            relative_block_indices = tl.where(mask, block_indices - block_idx_base, 0)
        else:
            relative_block_indices = tl.where(mask & is_local, block_indices - block_idx_base, 0)
        block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)

        slot_ids = block_numbers * block_size + slot_offsets
        if TOTAL_CP_WORLD_SIZE != 1:
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)
