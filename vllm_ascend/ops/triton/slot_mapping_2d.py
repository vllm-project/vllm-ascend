# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _compute_slot_mapping_2d_kernel(
    active_slots_ptr,  # [num_tokens], int32 or int64
    slot2d_ptr,  # [2 * num_tokens], int32 (row-major flat view)
    flat_ptr,  # [num_tokens], int64
    num_tokens,
    storage_block_size,
    ratio,
    COMPRESSED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_tokens
    # slot dtype follows the input pointer dtype (int32 or int64).
    slot = tl.load(active_slots_ptr + offs, mask=mask, other=-1)
    if COMPRESSED:
        valid_comp = (slot >= 0) & ((slot + 1) % ratio == 0)
        slot = tl.where(valid_comp, slot // ratio, -1)
    valid = slot >= 0
    physical = tl.maximum(slot, 0)
    block = storage_block_size
    tl.store(
        slot2d_ptr + offs * 2 + 0,
        tl.where(valid, physical // block, -1).to(tl.int32),
        mask=mask,
    )
    tl.store(
        slot2d_ptr + offs * 2 + 1,
        tl.where(valid, physical % block, -1).to(tl.int32),
        mask=mask,
    )
    tl.store(
        flat_ptr + offs,
        tl.where(valid, slot, -1).to(tl.int64),
        mask=mask,
    )


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

    ``active_slots``: 1D device tensor of original physical slots (int32/int64).
    ``slot2d``: [num_tokens, 2] int32 persistent buffer, written in place.
    ``flat``: [num_tokens] int64 persistent buffer, written in place.
    """
    num_tokens = active_slots.shape[0]
    if num_tokens == 0:
        return
    block = 1024
    grid = (triton.cdiv(num_tokens, block),)
    _compute_slot_mapping_2d_kernel[grid](
        active_slots,
        slot2d.view(-1),
        flat,
        num_tokens,
        storage_block_size,
        ratio,
        COMPRESSED=ratio > 1,
        BLOCK=block,
    )
