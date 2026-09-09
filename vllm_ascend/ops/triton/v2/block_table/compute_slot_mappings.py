# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.block_table import _load_ptr


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,
    query_start_loc,
    pos,
    block_table_ptrs,
    block_table_strides,
    block_sizes,
    slot_mappings_ptr,
    slot_mappings_stride,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    BLOCK_TABLE_WINDOW_SIZE: tl.constexpr,
):
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)

    # An empty request can use -1 as its idx_mapping sentinel. Do not resolve
    # its block-table row: there are no slot IDs to produce for this program.
    if start_idx < end_idx:
        block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
        block_table_stride = tl.load(block_table_strides + group_id)
        block_size = tl.load(block_sizes + group_id)
        req_state_idx = tl.load(idx_mapping + batch_idx)

        lane_offsets = tl.arange(0, TRITON_BLOCK_SIZE)
        block_table_offsets = tl.arange(0, BLOCK_TABLE_WINDOW_SIZE)
        for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
            offset = i + lane_offsets
            valid = offset < end_idx
            positions = tl.load(pos + offset, mask=valid, other=0).to(tl.int32)
            block_indices = positions // (block_size * CP_SIZE)
            # block_offset = positions % (block_size * CP_SIZE). Replacing the
            # remainder with multiply/subtract avoids scalar fallback on Ascend.
            block_offsets = positions - (block_size * CP_SIZE) * block_indices

            # Stage only the contiguous portion of this request row used by
            # the current token tile. The window bound is selected at launch
            # from the smallest group block size.
            valid_block_indices = tl.where(valid, block_indices, 2147483647)
            block_idx_base = tl.min(valid_block_indices, axis=0)
            window_offsets = block_idx_base + block_table_offsets
            block_table_window = tl.load(
                block_table_ptr + req_state_idx * block_table_stride + window_offsets,
                mask=window_offsets < block_table_stride,
                other=0,
            ).to(tl.float32)
            relative_block_indices = tl.where(valid, block_indices - block_idx_base, 0)
            block_numbers = tl.gather(block_table_window, relative_block_indices, 0).to(tl.int32)

            if CP_SIZE == 1:
                slot_ids = block_numbers * block_size + block_offsets
            else:
                is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
                rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
                remainder = block_offsets % CP_INTERLEAVE
                local_offsets = rounds * CP_INTERLEAVE + remainder
                slot_ids = block_numbers * block_size + local_offsets
                slot_ids = tl.where(is_local, slot_ids, PAD_ID)

            tl.store(slot_mapping_ptr + offset, slot_ids, mask=valid)
