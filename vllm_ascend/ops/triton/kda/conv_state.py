# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare cache indices and stage unsupported KDA convolution state layouts."""

import torch
from vllm.triton_utils import tl, triton

_CONV_STATE_COPY_BLOCK_SIZE = 256
_CONV_STATE_COPY_MAX_PROGRAMS = 65535
_CONV_INDEX_BLOCK_SIZE = 128


@triton.jit
def _prepare_conv_indices_rows(
    indices, starts, output, index_stride, num_slots, REQUESTS: tl.constexpr, BLOCK: tl.constexpr
):
    request = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = request < REQUESTS
    slot = tl.load(indices + request * index_stride, mask=in_range, other=-1).to(tl.int64)
    start = tl.load(starts + request, mask=in_range, other=0)
    end = tl.load(starts + request + 1, mask=in_range, other=0)
    active = (slot >= 0) & (slot < num_slots) & (end > start)
    tl.store(output + request, tl.where(active, slot, num_slots), mask=in_range)


@triton.jit
def _prepare_conv_indices_kernel(
    indices,
    starts,
    output,
    index_stride,
    second_indices,
    second_starts,
    second_output,
    second_index_stride,
    num_slots,
    REQUESTS: tl.constexpr,
    SECOND_REQUESTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.program_id(1) == 0:
        _prepare_conv_indices_rows(indices, starts, output, index_stride, num_slots, REQUESTS, BLOCK)
    else:
        _prepare_conv_indices_rows(
            second_indices, second_starts, second_output, second_index_stride, num_slots, SECOND_REQUESTS, BLOCK
        )


def prepare_conv_indices(
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_slots: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select the first index column and map inactive rows to a nonnegative null slot."""
    requests = cache_indices.shape[0]
    if out is not None and requests > out.numel():
        raise ValueError("Convolution index count exceeds output buffer capacity")
    indices = torch.empty(requests, dtype=torch.int32, device=cache_indices.device) if out is None else out[:requests]
    if requests:
        _prepare_conv_indices_kernel[(triton.cdiv(requests, _CONV_INDEX_BLOCK_SIZE),)](
            cache_indices,
            query_start_loc,
            indices,
            cache_indices.stride(0),
            cache_indices,
            query_start_loc,
            indices,
            cache_indices.stride(0),
            num_slots,
            requests,
            0,
            BLOCK=_CONV_INDEX_BLOCK_SIZE,
        )
    return indices


def prepare_conv_indices_pair(
    first_indices: torch.Tensor,
    first_starts: torch.Tensor,
    first_out: torch.Tensor,
    second_indices: torch.Tensor,
    second_starts: torch.Tensor,
    second_out: torch.Tensor,
    num_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare non-spec and spec request indices with one metadata-stage launch."""
    first_requests, second_requests = first_indices.shape[0], second_indices.shape[0]
    if first_requests > first_out.numel() or second_requests > second_out.numel():
        raise ValueError("Convolution index count exceeds output buffer capacity")
    first, second = first_out[:first_requests], second_out[:second_requests]
    if first_requests or second_requests:
        _prepare_conv_indices_kernel[(triton.cdiv(max(first_requests, second_requests), _CONV_INDEX_BLOCK_SIZE), 2)](
            first_indices,
            first_starts,
            first,
            first_indices.stride(0),
            second_indices,
            second_starts,
            second,
            second_indices.stride(0),
            num_slots,
            first_requests,
            second_requests,
            BLOCK=_CONV_INDEX_BLOCK_SIZE,
        )
    return first, second


@triton.jit
def _copy_conv_state_kernel(
    cache,
    packed,
    cache_indices,
    starts,
    packed_indices,
    cache_stride,
    index_stride,
    num_slots,
    REQUESTS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    DIM: tl.constexpr,
    STATE_STRIDE: tl.constexpr,
    DIM_STRIDE: tl.constexpr,
    WRITE_BACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel_tiles = tl.cdiv(DIM, BLOCK)
    request_tiles = STATE_LEN * channel_tiles
    # Bound the launch grid, including wide TP shards and large batches.
    for tile in range(tl.program_id(0), REQUESTS * request_tiles, tl.num_programs(0)):
        request = tile // request_tiles
        row_tile = tile % request_tiles
        state_row = row_tile // channel_tiles
        channels = row_tile % channel_tiles * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(cache_indices + request * index_stride).to(tl.int64)
        active = (slot >= 0) & (slot < num_slots) & (tl.load(starts + request + 1) > tl.load(starts + request))
        in_range = channels < DIM
        safe_slot = tl.where(active, slot, 0)
        # Keep the potentially large page address scalar. Within-page offsets
        # fit int32; broadcasting the page address makes every lane use int64.
        cache_row = cache + safe_slot * cache_stride + state_row * STATE_STRIDE
        cache_offsets = channels * DIM_STRIDE
        packed_offsets = request * STATE_LEN * DIM + state_row * DIM + channels
        if WRITE_BACK:
            values = tl.load(packed + packed_offsets, mask=in_range, other=0)
            tl.store(cache_row + cache_offsets, values, mask=active & in_range)
        else:
            values = tl.load(cache_row + cache_offsets, mask=active & in_range, other=0)
            tl.store(packed + packed_offsets, values, mask=in_range)
            if row_tile == 0:
                tl.store(packed_indices + request, tl.where(active, request, -1))


def copy_conv_state(
    cache: torch.Tensor,
    packed: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    packed_indices: torch.Tensor,
    *,
    write_back: bool,
) -> None:
    """Pack active convolution states or restore them to the original cache view."""
    requests = cache_indices.shape[0]
    if requests == 0:
        return
    state_len, dim = cache.shape[1:]
    grid = (min(requests * state_len * triton.cdiv(dim, _CONV_STATE_COPY_BLOCK_SIZE), _CONV_STATE_COPY_MAX_PROGRAMS),)
    _copy_conv_state_kernel[grid](
        cache,
        packed,
        cache_indices,
        query_start_loc,
        packed_indices,
        cache.stride(0),
        cache_indices.stride(0),
        cache.shape[0],
        requests,
        state_len,
        dim,
        cache.stride(1),
        cache.stride(2),
        WRITE_BACK=write_back,
        BLOCK=_CONV_STATE_COPY_BLOCK_SIZE,
    )
