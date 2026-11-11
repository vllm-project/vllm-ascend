# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _compute_slot_mapping_2d_kernel(
    active_slots_ptr,  # [num_tokens], int32 or int64
    slot2d_ptr,  # [2 * num_tokens], int32 (row-major flat view)
    flat_ptr,  # [num_tokens], int64
    num_tokens,
    storage_block_size: tl.constexpr,
    ratio: tl.constexpr,
    BLOCK_LOG2: tl.constexpr,
    RATIO_LOG2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_tokens
    # slot dtype follows the input pointer dtype (int32 or int64).
    slot = tl.load(active_slots_ptr + offs, mask=mask, other=-1)
    if ratio > 1:
        # Power-of-two ratio (asserted by the wrapper; DS V4.1 uses 1 or 2).
        # Never emit an int32 division: CANN 9.2.0 stacks lower it through
        # fp32, which rounds dividends >= 2**24 (222217721 -> 222217728) and
        # produced off-by-one page indices. Bit ops are exact.
        valid_comp = (slot >= 0) & (((slot + 1) & (ratio - 1)) == 0)
        slot = tl.where(valid_comp, slot >> RATIO_LOG2, -1)
    valid = slot >= 0
    # storage_block_size is a power of two (asserted by the wrapper), so page
    # and in-page offset are exact shifts/masks. No clamping is needed: padded
    # lanes are already -1 and ``where(valid, ...)`` masks invalid lanes below.
    page = slot >> BLOCK_LOG2
    row = slot & (storage_block_size - 1)
    tl.store(
        slot2d_ptr + offs * 2 + 0,
        tl.where(valid, page, -1).to(tl.int32),
        mask=mask,
    )
    tl.store(
        slot2d_ptr + offs * 2 + 1,
        tl.where(valid, row, -1).to(tl.int32),
        mask=mask,
    )
    tl.store(
        flat_ptr + offs,
        tl.where(valid, slot, -1).to(tl.int64),
        mask=mask,
    )


def _pow2_log2(value: int) -> int:
    """log2 of a positive power of two."""
    assert value > 0 and (value & (value - 1)) == 0, f"{value} is not a power of two"
    return value.bit_length() - 1


def compute_slot_mapping_2d(
    active_slots,
    storage_block_size: int,
    ratio: int,
    slot2d,
    flat,
) -> None:
    """Fuse the per-group slot-mapping 2D/flat computation into one kernel.

    This is equivalent to the eager chain in ``dsa_v41`` build():

        valid = active_slots >= 0            (plus optional compression)
        slot2d[:, 0] = where(valid, max(slot, 0) // block, -1)
        slot2d[:, 1] = where(valid, max(slot, 0) %  block, -1)
        flat         = where(valid, slot, -1)

    ``ratio > 1`` applies the compressed-slot mapping (complete groups only).
    The compressed ratio==2 masking path (valid_end / positions / ring state)
    is kept in the eager implementation and is not routed through this kernel.

    ``storage_block_size`` and ``ratio`` must be powers of two (both always
    hold for DeepSeek V4.1: vLLM block sizes are powers of two and
    ``tokens_per_state`` is 1 or 2); the wrapper asserts this and the kernel
    uses exact shifts/masks only. No int32 division is ever emitted because
    CANN 9.2.0 stacks lower it through fp32, which rounds dividends >= 2**24
    (222217721 -> 222217728) and produces off-by-one page indices.

    ``active_slots``: 1D contiguous device tensor of original physical slots
    (int32/int64).
    ``slot2d``: [num_tokens, 2] int32 persistent buffer, written in place.
    ``flat``: [num_tokens] int64 contiguous persistent buffer, written in place.
    The kernel walks ``active_slots`` and ``flat`` with unit stride, so both
    must be contiguous; non-contiguous views are rejected instead of silently
    producing a wrong mapping. ``slot2d`` is flattened with ``view(-1)``, which
    already requires a contiguous layout.
    """
    num_tokens = active_slots.shape[0]
    if num_tokens == 0:
        return
    assert active_slots.is_contiguous(), "active_slots must be contiguous"
    assert flat.is_contiguous(), "flat must be contiguous"
    assert storage_block_size >= 1, "storage_block_size must be positive"
    assert ratio >= 1, "ratio must be positive"
    block = 1024
    grid = (triton.cdiv(num_tokens, block),)
    _compute_slot_mapping_2d_kernel[grid](
        active_slots,
        slot2d.view(-1),
        flat,
        num_tokens,
        storage_block_size=storage_block_size,
        ratio=ratio,
        BLOCK_LOG2=_pow2_log2(storage_block_size),
        RATIO_LOG2=_pow2_log2(ratio),
        BLOCK=block,
    )
