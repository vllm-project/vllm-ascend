# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack and restore strided convolution states for KDA."""

import torch
from vllm.triton_utils import tl, triton

_CONV_STATE_COPY_BLOCK_SIZE = 256
_CONV_STATE_COPY_MAX_PROGRAMS = 65535


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
