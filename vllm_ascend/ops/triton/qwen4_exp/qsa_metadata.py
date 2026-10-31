# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build QSA metadata in persistent owner buffers on Ascend.

The upstream owner supplies the persistent buffers. This producer preserves its
four returned views and optional work-buffer writes without CUDA PDL, temporary
device tensors or a CPU read of rejection-corrected sequence lengths.
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata

_TOKEN_BLOCK_SIZE = 128
_WORK_BLOCK_SIZE = 256


@triton.jit
def _build_qsa_metadata_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    common_slot_mapping_ptr,
    block_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    visible_blocks_ptr,
    slot_mapping_ptr,
    k_work_metadata_ptr,
    query_start_stride: tl.constexpr,
    seq_lens_stride: tl.constexpr,
    common_slot_stride: tl.constexpr,
    block_table_stride_0: tl.constexpr,
    block_table_stride_1: tl.constexpr,
    token_to_req_stride: tl.constexpr,
    logical_positions_stride: tl.constexpr,
    visible_blocks_stride: tl.constexpr,
    slot_mapping_stride: tl.constexpr,
    work_stride_0: tl.constexpr,
    work_stride_1: tl.constexpr,
    num_reqs: tl.constexpr,
    num_mapped_tokens: tl.constexpr,
    num_tokens: tl.constexpr,
    slot_capacity: tl.constexpr,
    max_num_work: tl.constexpr,
    storage_block_size: tl.constexpr,
    compress_ratio: tl.constexpr,
    circular_buffer_size: tl.constexpr,
    num_block_table_rows: tl.constexpr,
    num_block_table_columns: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    WORK_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid * TOKEN_BLOCK_SIZE + tl.arange(0, TOKEN_BLOCK_SIZE)
    mapped = (token_idx < num_mapped_tokens) & (token_idx < num_tokens)
    request_idx = tl.zeros((TOKEN_BLOCK_SIZE,), tl.int32)
    query_start = tl.zeros((TOKEN_BLOCK_SIZE,), tl.int64)
    query_end = tl.zeros((TOKEN_BLOCK_SIZE,), tl.int64)
    seq_len = tl.zeros((TOKEN_BLOCK_SIZE,), tl.int64)
    if k_work_metadata_ptr is not None and max_num_work > 0:
        work = pid * WORK_BLOCK_SIZE + tl.arange(0, WORK_BLOCK_SIZE)
        work_in_bounds = work < max_num_work
        work_request = tl.full((WORK_BLOCK_SIZE,), -1, tl.int32)
        work_in_request = tl.full((WORK_BLOCK_SIZE,), -1, tl.int64)
        work_prefix = tl.full((), 0, tl.int64)

    # A rolled request scan bounds compiler IR for larger batches. Every CTA
    # derives the same work prefix, so work tiles need no inter-CTA barrier.
    for request in tl.range(0, num_reqs, loop_unroll_factor=1):
        request_start = tl.load(query_start_loc_ptr + request * query_start_stride).to(tl.int64)
        request_end = tl.load(query_start_loc_ptr + (request + 1) * query_start_stride).to(tl.int64)
        request_seq_len = tl.load(seq_lens_ptr + request * seq_lens_stride).to(tl.int64)
        token_in_request = mapped & (token_idx >= request_start) & (token_idx < request_end)
        request_idx = tl.where(token_in_request, request, request_idx)
        query_start = tl.where(token_in_request, request_start, query_start)
        query_end = tl.where(token_in_request, request_end, query_end)
        seq_len = tl.where(token_in_request, request_seq_len, seq_len)
        if k_work_metadata_ptr is not None and max_num_work > 0:
            request_query_len = request_end - request_start
            chunk_start = request_seq_len - request_query_len
            # Torch uses floor division, including graph dummy requests whose
            # sequence length can be zero while their query span is nonempty.
            chunk_group = tl.where(
                chunk_start >= 0,
                chunk_start // compress_ratio,
                -((-chunk_start + compress_ratio - 1) // compress_ratio),
            )
            num_groups = request_seq_len // compress_ratio - chunk_group
            work_count = tl.where(request_query_len > 0, tl.maximum(num_groups, 1), 0)
            owns_work = work_in_bounds & (work >= work_prefix) & (work < work_prefix + work_count)
            work_request = tl.where(owns_work, request, work_request)
            work_in_request = tl.where(owns_work, work - work_prefix, work_in_request)
            work_prefix += work_count

    logical_position = seq_len - (query_end - query_start) + token_idx - query_start
    logical_position = tl.where(mapped, logical_position, -1)
    if num_tokens > 0:
        token_mask = token_idx < num_tokens
        safe_token = tl.minimum(token_idx, num_tokens - 1).to(tl.int64)
        tl.store(token_to_req_ptr + safe_token * token_to_req_stride, request_idx, mask=token_mask)
        tl.store(logical_positions_ptr + safe_token * logical_positions_stride, logical_position, mask=token_mask)
        visible_blocks = tl.maximum(
            0, tl.minimum(tl.maximum(logical_position + 1, 0) // compress_ratio, seq_len // compress_ratio)
        )
        tl.store(visible_blocks_ptr + safe_token * visible_blocks_stride, visible_blocks, mask=token_mask)

    if circular_buffer_size > 0 or compress_ratio != 1:
        slot = tl.full((TOKEN_BLOCK_SIZE,), -1, tl.int64)
        # Zero-size tables can have null GM pointers. Eliminate their loads at
        # compile time; for nonempty tables keep even masked addresses in bounds.
        if num_tokens > 0 and (num_block_table_rows > 0 and num_block_table_columns > 0):
            valid = mapped & (logical_position >= 0) & (request_idx < num_block_table_rows)
            if circular_buffer_size > 0:
                valid &= token_idx + circular_buffer_size >= query_end
                safe_request = tl.where(valid, request_idx, 0).to(tl.int64)
                if num_reqs == 1:
                    # Avoid the old Ascend compiler's scalar-address/vector-mask
                    # broadcast limitation. The table is known to be nonempty.
                    physical_block = tl.load(block_table_ptr).to(tl.int64)
                else:
                    physical_block = tl.load(
                        block_table_ptr + safe_request * block_table_stride_0, mask=valid, other=-1
                    ).to(tl.int64)
                valid &= physical_block > 0
                slot = physical_block * circular_buffer_size + tl.maximum(logical_position, 0) % circular_buffer_size
            else:
                compressed_position = tl.maximum(logical_position, 0) // compress_ratio
                logical_block = compressed_position // storage_block_size
                valid &= ((logical_position + 1) % compress_ratio == 0) & (logical_block < num_block_table_columns)
                safe_request = tl.where(valid, request_idx, 0).to(tl.int64)
                safe_block = tl.where(valid, logical_block, 0).to(tl.int64)
                physical_block = tl.load(
                    block_table_ptr + safe_request * block_table_stride_0 + safe_block * block_table_stride_1,
                    mask=valid,
                    other=-1,
                ).to(tl.int64)
                safe_token = tl.where(mapped, token_idx, 0).to(tl.int64)
                common_slot = tl.load(common_slot_mapping_ptr + safe_token * common_slot_stride, mask=mapped, other=-1)
                valid &= (physical_block >= 0) & (common_slot >= 0)
                slot = physical_block * storage_block_size + compressed_position % storage_block_size
            slot = tl.where(valid, slot, -1)
        if slot_capacity > 0:
            # circular/compressed Torch helpers clear the entire supplied
            # persistent slot buffer, not only the returned num_tokens view.
            safe_slot = tl.minimum(token_idx, slot_capacity - 1).to(tl.int64)
            tl.store(slot_mapping_ptr + safe_slot * slot_mapping_stride, slot, mask=token_idx < slot_capacity)

    if k_work_metadata_ptr is not None and max_num_work > 0:
        safe_work = tl.minimum(work, max_num_work - 1).to(tl.int64)
        tl.store(k_work_metadata_ptr + safe_work * work_stride_0, work_request, mask=work_in_bounds)
        tl.store(
            k_work_metadata_ptr + safe_work * work_stride_0 + work_stride_1,
            work_in_request,
            mask=work_in_bounds,
        )


def build_qsa_metadata_triton(
    common_attn_metadata: CommonAttentionMetadata,
    token_to_req_buffer: torch.Tensor,
    logical_positions_buffer: torch.Tensor,
    visible_blocks_buffer: torch.Tensor,
    slot_mapping_buffer: torch.Tensor,
    *,
    storage_block_size: int,
    compress_ratio: int,
    circular_buffer_size: int = 0,
    k_work_metadata_buffer: torch.Tensor | None = None,
    request_capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write owner buffers; raw ratio-one slots retain the common-buffer alias."""
    del request_capacity
    num_tokens = common_attn_metadata.num_actual_tokens
    num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
    num_reqs = common_attn_metadata.query_start_loc.shape[0] - 1
    token_to_req = token_to_req_buffer[:num_tokens]
    logical_positions = logical_positions_buffer[:num_tokens]
    visible_blocks = visible_blocks_buffer[:num_tokens]
    uses_side_slots = circular_buffer_size > 0 or compress_ratio != 1
    slot_capacity = slot_mapping_buffer.numel() if uses_side_slots else 0
    slot_mapping = (
        slot_mapping_buffer[:num_tokens] if uses_side_slots else common_attn_metadata.slot_mapping[:num_tokens]
    )
    max_num_work = k_work_metadata_buffer.shape[0] if k_work_metadata_buffer is not None else 0
    if num_tokens == 0 and slot_capacity == 0 and max_num_work == 0:
        return token_to_req, logical_positions, visible_blocks, slot_mapping

    block_table = common_attn_metadata.block_table_tensor
    grid = (max(cdiv(max(num_tokens, slot_capacity), _TOKEN_BLOCK_SIZE), cdiv(max_num_work, _WORK_BLOCK_SIZE), 1),)
    _build_qsa_metadata_kernel[grid](
        common_attn_metadata.query_start_loc,
        common_attn_metadata.seq_lens,
        common_attn_metadata.slot_mapping,
        block_table,
        token_to_req_buffer,
        logical_positions_buffer,
        visible_blocks_buffer,
        slot_mapping_buffer,
        k_work_metadata_buffer,
        common_attn_metadata.query_start_loc.stride(0),
        common_attn_metadata.seq_lens.stride(0),
        common_attn_metadata.slot_mapping.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        token_to_req_buffer.stride(0),
        logical_positions_buffer.stride(0),
        visible_blocks_buffer.stride(0),
        slot_mapping_buffer.stride(0),
        k_work_metadata_buffer.stride(0) if k_work_metadata_buffer is not None else 0,
        k_work_metadata_buffer.stride(1) if k_work_metadata_buffer is not None else 0,
        num_reqs,
        num_mapped_tokens,
        num_tokens,
        slot_capacity,
        max_num_work,
        storage_block_size,
        compress_ratio,
        circular_buffer_size,
        block_table.shape[0],
        block_table.shape[1],
        TOKEN_BLOCK_SIZE=_TOKEN_BLOCK_SIZE,
        WORK_BLOCK_SIZE=_WORK_BLOCK_SIZE,
        num_warps=4,
    )
    return token_to_req, logical_positions, visible_blocks, slot_mapping
