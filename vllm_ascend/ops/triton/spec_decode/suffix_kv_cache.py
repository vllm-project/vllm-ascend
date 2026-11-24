# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Stride-aware BF16 cache writes for supported suffix FIA layouts.

Caller contract: unique valid destination slots, or -1 for padding. The kernel
clamps all address arithmetic and masks invalid slots for device-memory safety.
K/V input strides are honored; slots must be dense. Writes are in place.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _scatter_layout_kernel(
    key,
    value,
    key_cache,
    value_cache,
    slots,
    KEY_TOKEN_STRIDE: tl.constexpr,
    KEY_HEAD_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    VALUE_TOKEN_STRIDE: tl.constexpr,
    VALUE_HEAD_STRIDE: tl.constexpr,
    VALUE_DIM_STRIDE: tl.constexpr,
    KEY_BLOCK_STRIDE: tl.constexpr,
    KEY_CACHE_HEAD_STRIDE: tl.constexpr,
    KEY_CACHE_TOKEN_STRIDE: tl.constexpr,
    KEY_CACHE_DIM_STRIDE: tl.constexpr,
    VALUE_BLOCK_STRIDE: tl.constexpr,
    VALUE_CACHE_HEAD_STRIDE: tl.constexpr,
    VALUE_CACHE_TOKEN_STRIDE: tl.constexpr,
    VALUE_CACHE_DIM_STRIDE: tl.constexpr,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + token).to(tl.int64)
    valid_slot = (slot >= 0) & (slot < num_blocks * BLOCK_SIZE)
    safe_slot = tl.minimum(tl.maximum(slot, 0), num_blocks * BLOCK_SIZE - 1)
    block = safe_slot // BLOCK_SIZE
    position = safe_slot % BLOCK_SIZE
    head = tl.arange(0, HEADS)[:, None]
    dim = tl.arange(0, DIM)[None, :]
    # Keep wide slot arithmetic scalar; local head/dimension offsets stay
    # small affine int32 tensors rather than flattened int64 gather offsets.
    key_row = key + token * KEY_TOKEN_STRIDE
    value_row = value + token * VALUE_TOKEN_STRIDE
    keys = tl.load(key_row + head * KEY_HEAD_STRIDE + dim * KEY_DIM_STRIDE, mask=valid_slot, other=0)
    values = tl.load(value_row + head * VALUE_HEAD_STRIDE + dim * VALUE_DIM_STRIDE, mask=valid_slot, other=0)
    key_slot = key_cache + block * KEY_BLOCK_STRIDE + position * KEY_CACHE_TOKEN_STRIDE
    value_slot = value_cache + block * VALUE_BLOCK_STRIDE + position * VALUE_CACHE_TOKEN_STRIDE
    tl.store(key_slot + head * KEY_CACHE_HEAD_STRIDE + dim * KEY_CACHE_DIM_STRIDE, keys, mask=valid_slot)
    tl.store(value_slot + head * VALUE_CACHE_HEAD_STRIDE + dim * VALUE_CACHE_DIM_STRIDE, values, mask=valid_slot)


def scatter(key, value, key_cache, value_cache, slots, *, head_major):
    if key.ndim != 3 or key.shape != value.shape or key_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        raise ValueError("scatter requires matching [T,H,D] inputs and 4D cache pairs")
    if (
        slots.ndim != 1
        or slots.stride(0) != 1
        or slots.numel() != key.shape[0]
        or slots.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("invalid slot mapping geometry")
    if any(t.dtype != torch.bfloat16 or t.device != key.device for t in (key, value, key_cache, value_cache)):
        raise ValueError("this bounded scatter requires same-device BF16 tensors")
    if key.device.type != "npu" or slots.device != key.device:
        raise ValueError("scatter inputs must be on one NPU")
    head_axis, token_axis = (1, 2) if head_major else (2, 1)
    heads, dim = key.shape[1:]
    if heads <= 0 or dim <= 0 or key_cache.shape[token_axis] <= 0:
        raise ValueError("scatter dimensions must be positive")
    if heads & (heads - 1) or dim & (dim - 1):
        raise ValueError("suffix scatter requires power-of-two head geometry")
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        raise ValueError("layout comparison requires dense non-overlapping cache storage")
    if key_cache.shape[head_axis] != heads or key_cache.shape[3] != dim or key_cache.shape[0] <= 0:
        raise ValueError("cache head geometry mismatch")
    if key.shape[0] == 0:
        return
    _scatter_layout_kernel[(key.shape[0],)](
        key,
        value,
        key_cache,
        value_cache,
        slots,
        KEY_TOKEN_STRIDE=key.stride(0),
        KEY_HEAD_STRIDE=key.stride(1),
        KEY_DIM_STRIDE=key.stride(2),
        VALUE_TOKEN_STRIDE=value.stride(0),
        VALUE_HEAD_STRIDE=value.stride(1),
        VALUE_DIM_STRIDE=value.stride(2),
        KEY_BLOCK_STRIDE=key_cache.stride(0),
        KEY_CACHE_HEAD_STRIDE=key_cache.stride(head_axis),
        KEY_CACHE_TOKEN_STRIDE=key_cache.stride(token_axis),
        KEY_CACHE_DIM_STRIDE=key_cache.stride(3),
        VALUE_BLOCK_STRIDE=value_cache.stride(0),
        VALUE_CACHE_HEAD_STRIDE=value_cache.stride(head_axis),
        VALUE_CACHE_TOKEN_STRIDE=value_cache.stride(token_axis),
        VALUE_CACHE_DIM_STRIDE=value_cache.stride(3),
        num_blocks=key_cache.shape[0],
        BLOCK_SIZE=key_cache.shape[token_axis],
        HEADS=heads,
        DIM=dim,
        num_warps=4,
    )
