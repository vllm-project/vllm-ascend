# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in Ascend QSA metadata producer for the upstream cache owner.

The upstream cache owner supplies persistent output buffers and the builder
ABI. This kernel avoids CUDA PDL and vector gathers, and eliminates block-table
loads at compile time when the table has zero columns.
"""

import math

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import CommonAttentionMetadata


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
    block_table_stride_0: tl.constexpr,
    block_table_stride_1: tl.constexpr,
    num_reqs: tl.constexpr,
    num_mapped_tokens: tl.constexpr,
    num_tokens: tl.constexpr,
    max_num_work: tl.constexpr,
    num_search_steps: tl.constexpr,
    work_search_steps: tl.constexpr,
    storage_block_size: tl.constexpr,
    compress_ratio: tl.constexpr,
    circular_buffer_size: tl.constexpr,
    num_block_table_columns: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    REQUEST_SCAN_SIZE: tl.constexpr,
    WORK_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid * TOKEN_BLOCK_SIZE + tl.arange(0, TOKEN_BLOCK_SIZE)
    store_mask = token_idx < num_tokens
    mapped = token_idx < num_mapped_tokens
    request_idx = tl.zeros((TOKEN_BLOCK_SIZE,), tl.int32)
    query_start = tl.zeros((TOKEN_BLOCK_SIZE,), tl.int32)
    query_end = tl.zeros((TOKEN_BLOCK_SIZE,), tl.int32)
    seq_len = tl.zeros((TOKEN_BLOCK_SIZE,), tl.int32)
    # Pointer presence is static: the raw-ring/ratio-one kernel contains no
    # work-metadata pointer expression. No CTA-dependent branch is needed.
    if k_work_metadata_ptr is not None:
        work = pid * WORK_BLOCK_SIZE + tl.arange(0, WORK_BLOCK_SIZE)
        work_in_bounds = work < max_num_work
        work_request = tl.full((WORK_BLOCK_SIZE,), -1, tl.int32)
        work_in_request = tl.full((WORK_BLOCK_SIZE,), -1, tl.int32)
        work_prefix = tl.full((), 0, tl.int32)

    # Keep hot small batches static. Larger request batches use a rolled loop
    # to bound compiler IR size; both paths read actual device request metadata.
    # Empty requests own neither tokens nor work items.
    MAX_STATIC_REQUESTS: tl.constexpr = 8
    if num_reqs <= MAX_STATIC_REQUESTS:
        for request in tl.static_range(0, num_reqs):
            request_start = tl.load(query_start_loc_ptr + request)
            request_end = tl.load(query_start_loc_ptr + request + 1)
            request_seq_len = tl.load(seq_lens_ptr + request)
            token_in_request = mapped & (token_idx >= request_start) & (token_idx < request_end)
            request_idx = tl.where(token_in_request, request, request_idx)
            query_start = tl.where(token_in_request, request_start, query_start)
            query_end = tl.where(token_in_request, request_end, query_end)
            seq_len = tl.where(token_in_request, request_seq_len, seq_len)
            if k_work_metadata_ptr is not None:
                request_query_len = request_end - request_start
                chunk_start = request_seq_len - request_query_len
                num_groups = request_seq_len // compress_ratio - chunk_start // compress_ratio
                # Item zero also commits a nonempty request's current raw-K suffix.
                work_count = tl.where(request_query_len > 0, tl.maximum(num_groups, 1), 0)
                owns_work = work_in_bounds & (work >= work_prefix) & (work < work_prefix + work_count)
                work_request = tl.where(owns_work, request, work_request)
                work_in_request = tl.where(owns_work, work - work_prefix, work_in_request)
                work_prefix += work_count
    else:
        for request in tl.range(0, num_reqs, loop_unroll_factor=1):
            request_start = tl.load(query_start_loc_ptr + request)
            request_end = tl.load(query_start_loc_ptr + request + 1)
            request_seq_len = tl.load(seq_lens_ptr + request)
            token_in_request = mapped & (token_idx >= request_start) & (token_idx < request_end)
            request_idx = tl.where(token_in_request, request, request_idx)
            query_start = tl.where(token_in_request, request_start, query_start)
            query_end = tl.where(token_in_request, request_end, query_end)
            seq_len = tl.where(token_in_request, request_seq_len, seq_len)
            if k_work_metadata_ptr is not None:
                request_query_len = request_end - request_start
                chunk_start = request_seq_len - request_query_len
                num_groups = request_seq_len // compress_ratio - chunk_start // compress_ratio
                # Item zero also commits a nonempty request's current raw-K suffix.
                work_count = tl.where(request_query_len > 0, tl.maximum(num_groups, 1), 0)
                owns_work = work_in_bounds & (work >= work_prefix) & (work < work_prefix + work_count)
                work_request = tl.where(owns_work, request, work_request)
                work_in_request = tl.where(owns_work, work - work_prefix, work_in_request)
                work_prefix += work_count
    logical_position = seq_len - (query_end - query_start) + token_idx - query_start
    logical_position = tl.where(mapped, logical_position, -1)
    tl.store(
        token_to_req_ptr + token_idx,
        tl.where(mapped, request_idx, 0),
        mask=store_mask,
    )
    tl.store(
        logical_positions_ptr + token_idx,
        logical_position,
        mask=store_mask,
    )
    visible_blocks = tl.maximum(
        0,
        tl.minimum(
            (logical_position + 1) // compress_ratio,
            seq_len // compress_ratio,
        ),
    ).to(tl.int32)
    tl.store(visible_blocks_ptr + token_idx, visible_blocks, mask=store_mask)

    # A zero-column tensor may carry a null device pointer. Masked tl.load
    # is insufficient on this NPU compiler: eliminate all block-table loads
    # at compile time for the empty-table case.
    if num_block_table_columns == 0:
        if (circular_buffer_size > 0) or (compress_ratio != 1):
            tl.store(slot_mapping_ptr + token_idx, -1, mask=store_mask)
    else:
        # circular_buffer_size is constexpr, so each builder instance compiles out
        # the other QSA cache owner's slot-mapping rule.
        if circular_buffer_size > 0:
            valid = (
                mapped
                & (logical_position >= 0)
                & (token_idx + circular_buffer_size >= query_end)
                & (num_block_table_columns > 0)
            )
            if num_reqs == 1:
                # One request always reads table[0, 0]. Load it as a scalar:
                # a splat address with a vector mask fails in this compiler.
                physical_block = tl.load(block_table_ptr)
            else:
                physical_block = tl.load(
                    block_table_ptr + request_idx * block_table_stride_0,
                    mask=valid,
                    other=-1,
                )
            # Dummy and padding requests own no ring and sit on the null block.
            valid &= physical_block > 0
            slot = physical_block.to(tl.int64) * circular_buffer_size + (logical_position % circular_buffer_size)
        elif compress_ratio != 1:
            compressed_position = tl.maximum(logical_position, 0) // compress_ratio
            logical_block = compressed_position // storage_block_size
            valid = (
                mapped
                & (logical_position >= 0)
                & ((logical_position + 1) % compress_ratio == 0)
                & (logical_block < num_block_table_columns)
            )
            physical_block = tl.load(
                block_table_ptr + request_idx * block_table_stride_0 + logical_block * block_table_stride_1,
                mask=valid,
                other=-1,
            )
            valid &= physical_block >= 0
            valid &= tl.load(common_slot_mapping_ptr + token_idx, mask=mapped, other=-1) >= 0
            slot = physical_block.to(tl.int64) * storage_block_size + (compressed_position % storage_block_size)
        if (circular_buffer_size > 0) or (compress_ratio != 1):
            tl.store(
                slot_mapping_ptr + token_idx,
                tl.where(valid, slot, -1),
                mask=store_mask,
            )
    # All CTAs execute the same small request scan. Only live work lanes store;
    # there is no runtime has_work_tile branch or vector gather/reduction.
    if k_work_metadata_ptr is not None:
        tl.store(
            k_work_metadata_ptr + work * 2,
            work_request,
            mask=work_in_bounds,
        )
        tl.store(
            k_work_metadata_ptr + work * 2 + 1,
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
    """Build QSA side-cache and optional pre-indexer work metadata."""
    num_tokens = common_attn_metadata.num_actual_tokens
    num_mapped_tokens = int(common_attn_metadata.query_start_loc_cpu[-1])
    token_to_req = token_to_req_buffer[:num_tokens]
    logical_positions = logical_positions_buffer[:num_tokens]
    visible_blocks = visible_blocks_buffer[:num_tokens]
    slot_mapping = slot_mapping_buffer[:num_tokens]
    num_reqs = common_attn_metadata.query_start_loc.shape[0] - 1
    assert num_reqs > 0

    if k_work_metadata_buffer is not None:
        if request_capacity is None:
            request_capacity = num_reqs
        assert request_capacity >= num_reqs
        # Pad for tl.arange while keeping the scan width stable across live batches.
        request_scan_size = 1 << int(math.ceil(math.log2(request_capacity)))
        max_num_work = k_work_metadata_buffer.shape[0]
    else:
        request_scan_size = 1
        max_num_work = 0

    if num_tokens == 0 and k_work_metadata_buffer is None:
        return token_to_req, logical_positions, visible_blocks, slot_mapping

    block_table = common_attn_metadata.block_table_tensor
    num_search_steps = int(math.ceil(math.log2(num_reqs)))
    work_search_steps = int(math.ceil(math.log2(num_reqs)))
    # The same grid covers token tiles and, for the compressed cache, work tiles.
    num_token_blocks = cdiv(num_tokens, 128)
    num_work_blocks = cdiv(max_num_work, 256) if k_work_metadata_buffer is not None else 0
    _build_qsa_metadata_kernel[(max(num_token_blocks, num_work_blocks, 1),)](
        common_attn_metadata.query_start_loc,
        common_attn_metadata.seq_lens,
        common_attn_metadata.slot_mapping,
        block_table,
        token_to_req,
        logical_positions,
        visible_blocks,
        slot_mapping,
        k_work_metadata_buffer,
        block_table.stride(0),
        block_table.stride(1),
        num_reqs,
        num_mapped_tokens,
        num_tokens,
        max_num_work,
        num_search_steps,
        work_search_steps,
        storage_block_size,
        compress_ratio,
        circular_buffer_size,
        block_table.shape[1],
        TOKEN_BLOCK_SIZE=128,
        REQUEST_SCAN_SIZE=request_scan_size,
        WORK_BLOCK_SIZE=256,
        num_warps=4,
    )
    if circular_buffer_size == 0 and compress_ratio == 1:
        slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]
    return token_to_req, logical_positions, visible_blocks, slot_mapping
