# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exact integer HiSparse invalidation on AI Core, following upstream semantics."""

import torch
import triton
import triton.language as tl


@triton.jit
def _invalidate_hot_slots(
    dgi,
    sorted_slots,
    state_indices,
    STATES: tl.constexpr,
    HOT: tl.constexpr,
    SLOTS: tl.constexpr,
    ITERATIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    state = tl.load(state_indices + tl.program_id(0))
    slot = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (state >= 0) & (state < STATES) & (slot < HOT)
    value = tl.load(dgi + state * HOT + slot, mask=valid, other=0)
    first = tl.load(sorted_slots)
    last = tl.load(sorted_slots + SLOTS - 1)
    valid = valid & (value >= first) & (value <= last)
    low = tl.full((BLOCK,), 0, tl.int32)
    high = tl.full((BLOCK,), SLOTS, tl.int32)
    for _ in tl.static_range(ITERATIONS):
        active = valid & (low < high)
        middle = low + (high - low) // 2
        key = tl.load(sorted_slots + middle, mask=active, other=0)
        advance = active & (key < value)
        high = tl.where(active & ~advance, middle, high)
        low = tl.where(advance, middle + 1, low)
    key = tl.load(sorted_slots + low, mask=valid & (low < SLOTS), other=0)
    tl.store(dgi + state * HOT + slot, -1, mask=valid & (low < SLOTS) & (key == value))


def try_invalidate(owner, dgi, sorted_slots, state_indices) -> bool:
    """Use current group-local values; retain upstream fallback for other layouts."""
    if not getattr(owner, "use_fused_index_ops", False):
        return False
    if not (
        dgi.device.type == "npu"
        and dgi.ndim == 2
        and dgi.dtype == torch.int32
        and sorted_slots.ndim == state_indices.ndim == 1
        and sorted_slots.dtype == torch.int32
        and state_indices.dtype in (torch.int32, torch.int64)
        and sorted_slots.device == state_indices.device == dgi.device
        and all(t.is_contiguous() for t in (dgi, sorted_slots, state_indices))
    ):
        return False
    slots = sorted_slots.numel()
    active = state_indices.numel()
    if slots and active and dgi.shape[1]:
        _invalidate_hot_slots[(active, triton.cdiv(dgi.shape[1], 256))](
            dgi,
            sorted_slots,
            state_indices,
            dgi.shape[0],
            dgi.shape[1],
            slots,
            max(1, slots.bit_length()),
            256,
        )
    return True
