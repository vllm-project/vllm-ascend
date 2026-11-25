# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write the candidate checkpoint pages produced by a MegaRecurrentKda verification."""

import torch
from vllm.triton_utils import tl, triton

# Use one complete key row per value row. The deployed 128x128 state then
# has eight tiles per head, enough parallel work for a 12-head TP8 shard.
_VALUE_TILE = 16


@triton.jit
def _store_mega_kda_candidate_pages_kernel(
    cache_ptr,
    initial_state_ptr,
    correction_ptr,
    key_ptr,
    decay_ptr,
    destinations_ptr,
    lengths_ptr,
    active_ptr,
    cache_page_stride,
    destination_request_stride,
    destination_step_stride,
    lengths_stride,
    active_stride,
    num_pages,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    VALUE_TILES: tl.constexpr,
    VALUE_TILE: tl.constexpr,
    KEY_TILE: tl.constexpr,
):
    tile = tl.program_id(0)
    tile_v = tile % VALUE_TILES
    request_head = tile // VALUE_TILES
    request = request_head // HEADS
    head = request_head % HEADS
    active = tl.load(active_ptr + request * active_stride)
    if active:
        length = tl.load(lengths_ptr + request * lengths_stride)
        values = tile_v * VALUE_TILE + tl.arange(0, VALUE_TILE)
        keys = tl.arange(0, KEY_TILE)
        mask = (values[:, None] < DIM) & (keys[None, :] < DIM)
        head_offsets = (head * DIM + values[:, None]) * DIM + keys[None, :]
        state = tl.load(
            initial_state_ptr + request * HEADS * DIM * DIM + head_offsets, mask, 0
        )
        factors = request_head * WINDOW * DIM
        # Retain the state tile between candidate positions. Missing page
        # destinations suppress stores but still contribute to later prefixes.
        for step in range(tl.minimum(length, WINDOW)):
            factor = factors + step * DIM
            correction = tl.load(correction_ptr + factor + values, values < DIM, 0)
            key = tl.load(key_ptr + factor + keys, keys < DIM, 0).to(tl.float32)
            decay = tl.load(decay_ptr + factor + keys, keys < DIM, 0)
            state = state * decay[None, :] + correction[:, None] * key[None, :]
            page = tl.load(
                destinations_ptr
                + request * destination_request_stride
                + step * destination_step_stride
            ).to(tl.int64)
            if (page > 0) & (page < num_pages):
                tl.store(
                    cache_ptr + page * cache_page_stride + head_offsets, state, mask
                )


def store_candidate_pages(
    state_cache: torch.Tensor,
    initial_state: torch.Tensor,
    correction: torch.Tensor,
    key: torch.Tensor,
    decay: torch.Tensor,
    destinations: torch.Tensor,
    lengths: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Reconstruct each valid prefix and write it to a physical cache page.

    The FP32 checkpoint is a packed [B, H, D, D] buffer in [value, key]
    order. The packed correction, key and decay each have [B, H, W, D]
    layout. Page rows are contiguous, with a possibly padded page stride.
    Each live request owns its destination pages.
    """
    batch, heads, window, dim = correction.shape
    if batch == 0:
        return
    if state_cache.device.type != "npu":
        state = initial_state
        for step in range(window):
            state = state * decay[:, :, step].unsqueeze(-2) + (
                correction[:, :, step].unsqueeze(-1)
                * key[:, :, step].float().unsqueeze(-2)
            )
            pages = destinations[:, step]
            valid = (
                active & (step < lengths) & (pages > 0) & (pages < state_cache.size(0))
            )
            state_cache.index_copy_(0, pages[valid].long(), state[valid])
        return

    value_tile = min(_VALUE_TILE, triton.next_power_of_2(dim))
    value_tiles = triton.cdiv(dim, value_tile)
    _store_mega_kda_candidate_pages_kernel[(batch * heads * value_tiles,)](
        state_cache,
        initial_state,
        correction,
        key,
        decay,
        destinations,
        lengths,
        active,
        state_cache.stride(0),
        destinations.stride(0),
        destinations.stride(1),
        lengths.stride(0),
        active.stride(0),
        state_cache.size(0),
        HEADS=heads,
        DIM=dim,
        WINDOW=window,
        VALUE_TILES=value_tiles,
        VALUE_TILE=value_tile,
        KEY_TILE=triton.next_power_of_2(dim),
        enable_fp_fusion=False,
    )
