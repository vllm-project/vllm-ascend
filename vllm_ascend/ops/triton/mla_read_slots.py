# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded device-side address construction for the strided MLA cache reader."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

MAX_FAST_SEQUENCES = 8
MAX_SAFE_TOKENS = (2**63 - 1) // MAX_FAST_SEQUENCES


def _validate_slot_metadata(block_table, seq_lens, seq_offset, num_tokens, block_size):
    """Check request endpoints using the same indexing operation as the fallback.

    Each request traverses a contiguous range of logical pages, so checking its
    first and last page catches out-of-range columns without a per-token index.
    Empty requests use column zero and never examine their offset. Invalid
    lengths select an out-of-range column, preserving an indexing failure.
    All checks stay on device and participate in graph capture.
    """
    lengths = seq_lens.to(torch.int64)
    offsets = seq_offset[: lengths.numel()].to(torch.int64) if seq_offset is not None else torch.zeros_like(lengths)
    active = lengths > 0
    first = torch.where(active, offsets // block_size, 0)
    last_position = offsets + torch.clamp(lengths - 1, min=0)
    last = torch.where(active, last_position // block_size, 0)
    bad_lengths = torch.any((lengths < 0) | (lengths > num_tokens)) | (lengths.sum() != num_tokens)
    position_overflow = torch.any(active & (last_position < offsets))
    first = torch.where(bad_lengths | position_overflow, block_table.shape[1], first)
    columns = torch.stack((first, last), dim=1)
    requests = torch.arange(lengths.numel(), device=lengths.device, dtype=torch.int64).unsqueeze(1)
    return block_table[requests.expand_as(columns), columns]


@triton.jit(do_not_specialize=["num_tokens"])
def _mla_read_slots_kernel(
    table,
    lengths,
    offsets,
    endpoints,
    output,
    num_tokens,
    table_stride0: tl.constexpr,
    table_stride1: tl.constexpr,
    length_stride: tl.constexpr,
    offset_stride: tl.constexpr,
    endpoint_stride: tl.constexpr,
    endpoint_column_stride: tl.constexpr,
    table_columns: tl.constexpr,
    block_size: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    HAS_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(num_tokens, BLOCK), tl.num_programs(0)):
        token = tile * BLOCK + tl.arange(0, BLOCK)
        result = tl.full((BLOCK,), -1, tl.int64)
        start = tl.full((), 0, tl.int64)
        for request in tl.static_range(0, NUM_SEQUENCES):
            length = tl.load(lengths + request * length_stride).to(tl.int64)
            belongs = (token >= start) & (token < start + length) & (token < num_tokens)
            position = token.to(tl.int64) - start
            if HAS_OFFSET:
                position += tl.load(offsets + request * offset_stride).to(tl.int64)
            # Use safe addresses even when masked indirect loads are lowered
            # unconditionally on Ascend. Keep Python floor/remainder semantics.
            position = tl.where(belongs, position, 0)
            column = position // block_size
            remainder = position % block_size
            negative = remainder < 0
            column -= negative.to(tl.int64)
            remainder += tl.where(negative, block_size, 0)
            column = tl.where(column < 0, column + table_columns, column)
            valid_column = (column >= 0) & (column < table_columns)
            safe_column = tl.where(valid_column, column, 0)
            physical = tl.load(table + request * table_stride0 + safe_column * table_stride1).to(tl.int64)
            first_physical = tl.load(endpoints + request * endpoint_stride).to(tl.int64)
            last_physical = tl.load(endpoints + request * endpoint_stride + endpoint_column_stride).to(tl.int64)
            physical = tl.where(belongs & (token == start), first_physical, physical)
            physical = tl.where(belongs & (token == start + length - 1), last_physical, physical)
            slot = physical * block_size + remainder
            result = tl.where(belongs & valid_column, slot, result)
            start += length
        # The allocation is padded to a whole tile; invalid tail entries remain -1.
        tl.store(output + token, result)


def _build_mla_read_slots_original(block_table, seq_lens, seq_offset, num_tokens, block_size, endpoints):
    """Return exact slots for valid metadata, or None to retain the original path.

    Fast scope is 1..8 requests with integer device metadata. As in the original
    reader, lengths must sum to num_tokens and block indices must be valid.
    """
    count = seq_lens.numel()
    if (
        not 1 <= count <= 8
        or block_table.shape[1] == 0
        or seq_lens.dtype not in (torch.int32, torch.int64)
        or block_table.dtype not in (torch.int32, torch.int64)
        or (seq_offset is not None and seq_offset.dtype not in (torch.int32, torch.int64))
    ):
        return None
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.int64, device=seq_lens.device)
    block = 256
    padded = triton.cdiv(num_tokens, block) * block
    output = torch.empty(padded, dtype=torch.int64, device=seq_lens.device)
    init_device_properties_triton()
    grid = (min(triton.cdiv(num_tokens, block), get_vectorcore_num()),)
    _mla_read_slots_kernel[grid](
        block_table,
        seq_lens,
        seq_offset if seq_offset is not None else seq_lens,
        endpoints,
        output,
        num_tokens,
        block_table.stride(0),
        block_table.stride(1),
        seq_lens.stride(0),
        seq_offset.stride(0) if seq_offset is not None else 0,
        endpoints.stride(0),
        endpoints.stride(1),
        block_table.shape[1],
        block_size,
        count,
        seq_offset is not None,
        block,
    )
    return output[:num_tokens]


@triton.jit(do_not_specialize=["num_tokens", "logical_pages"])
def _mla_read_slots_page_kernel(
    table,
    lengths,
    offsets,
    endpoints,
    output,
    num_tokens,
    logical_pages,
    table_stride0: tl.constexpr,
    table_stride1: tl.constexpr,
    length_stride: tl.constexpr,
    offset_stride: tl.constexpr,
    endpoint_stride: tl.constexpr,
    endpoint_column_stride: tl.constexpr,
    TABLE_COLUMNS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    HAS_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK).to(tl.int64)
    for page_iter in range(tl.program_id(0), logical_pages, tl.num_programs(0)):
        page_id = page_iter.to(tl.int64)
        prior_tokens = tl.full((), 0, tl.int64)
        prior_pages = tl.full((), 0, tl.int64)
        for request in tl.static_range(0, NUM_SEQUENCES):
            length = tl.load(lengths + request * length_stride).to(tl.int64)
            offset = tl.full((), 0, tl.int64)
            if HAS_OFFSET:
                offset = tl.load(offsets + request * offset_stride).to(tl.int64)
            remainder = offset % PAGE_SIZE
            remainder = tl.where(remainder < 0, remainder + PAGE_SIZE, remainder)
            first_page = (offset - remainder) // PAGE_SIZE
            pages = tl.where(length > 0, (length + remainder + PAGE_SIZE - 1) // PAGE_SIZE, 0)
            page_in_request = page_id - prior_pages
            belongs = (page_in_request >= 0) & (page_in_request < pages)
            column = first_page + page_in_request
            column = tl.where(column < 0, column + TABLE_COLUMNS, column)
            valid_column = (column >= 0) & (column < TABLE_COLUMNS)
            safe_column = tl.where(belongs & valid_column, column, 0)
            physical = tl.load(table + request * table_stride0 + safe_column * table_stride1).to(tl.int64)
            first_physical = tl.load(endpoints + request * endpoint_stride).to(tl.int64)
            last_physical = tl.load(endpoints + request * endpoint_stride + endpoint_column_stride).to(tl.int64)
            physical = tl.where(belongs & (page_in_request == 0), first_physical, physical)
            physical = tl.where(belongs & (page_in_request == pages - 1), last_physical, physical)
            sequence_index = page_in_request * PAGE_SIZE + lanes - remainder
            output_index = prior_tokens + sequence_index
            valid = (
                belongs
                & (lanes < PAGE_SIZE)
                & (sequence_index >= 0)
                & (sequence_index < length)
                & (output_index >= 0)
                & (output_index < num_tokens)
            )
            safe_output_index = tl.where(valid, output_index, 0)
            slot = tl.where(valid_column, physical * PAGE_SIZE + lanes, -1)
            tl.store(output + safe_output_index, slot, mask=valid)
            prior_tokens += length
            prior_pages += pages


def build_mla_read_slots(block_table, seq_lens, seq_offset, num_tokens, block_size):
    """Build slots on device for up to eight sequences, or return ``None``.

    One or two sequences with block sizes up to 512 use a page-based kernel.
    Other supported shapes use the token-based kernel. Metadata must describe
    valid table indices and nonnegative lengths summing to ``num_tokens``, as
    required by the existing reader. Unsupported dtypes or shapes fall back.
    """
    count = seq_lens.numel()
    if (
        seq_lens.ndim != 1
        or block_table.ndim != 2
        or not 1 <= count <= MAX_FAST_SEQUENCES
        or block_table.shape[0] < count
        or block_table.shape[1] == 0
        or block_size <= 0
        or not 0 <= num_tokens <= MAX_SAFE_TOKENS
        or seq_lens.device.type != "npu"
        or block_table.device != seq_lens.device
        or (seq_offset is not None and (seq_offset.ndim != 1 or seq_offset.numel() < count))
        or (seq_offset is not None and seq_offset.device != seq_lens.device)
        or seq_lens.dtype not in (torch.int32, torch.int64)
        or block_table.dtype not in (torch.int32, torch.int64)
        or (seq_offset is not None and seq_offset.dtype not in (torch.int32, torch.int64))
    ):
        return None
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.int64, device=seq_lens.device)
    endpoints = _validate_slot_metadata(block_table, seq_lens, seq_offset, num_tokens, block_size)
    if count not in (1, 2) or block_size > 512:
        return _build_mla_read_slots_original(block_table, seq_lens, seq_offset, num_tokens, block_size, endpoints)
    padded = triton.cdiv(num_tokens, 256) * 256
    output = torch.full((padded,), -1, dtype=torch.int64, device=seq_lens.device)
    block = max(128, triton.next_power_of_2(block_size))
    logical_pages = triton.cdiv(num_tokens, block_size) + 2 * count
    init_device_properties_triton()
    grid = (min(logical_pages, get_vectorcore_num()),)
    _mla_read_slots_page_kernel[grid](
        block_table,
        seq_lens,
        seq_offset if seq_offset is not None else seq_lens,
        endpoints,
        output,
        num_tokens,
        logical_pages,
        block_table.stride(0),
        block_table.stride(1),
        seq_lens.stride(0),
        seq_offset.stride(0) if seq_offset is not None else 0,
        endpoints.stride(0),
        endpoints.stride(1),
        block_table.shape[1],
        block_size,
        count,
        seq_offset is not None,
        block,
    )
    return output[:num_tokens]
