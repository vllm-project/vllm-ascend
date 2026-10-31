# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build packed cache slot metadata in one NPU launch."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_actual_reqs",
        "num_actual_tokens",
        "skip_update",
    ]
)
def _build_packed_cache_slot_mapping_kernel(
    slots_ptr,
    positions_ptr,
    query_start_loc_ptr,
    coordinates_ptr,
    flat_slots_ptr,
    num_tokens,
    num_actual_reqs,
    num_actual_tokens,
    skip_update,
    slot_stride,
    position_stride,
    query_start_stride,
    PAGE_SIZE: tl.constexpr,
    LOG2_PAGE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tokens < num_tokens
    # The whole chain is integer address bookkeeping that fits in int32. AIV has
    # no integer divider, so 64-bit div/mod expands into long fixup sequences;
    # keeping the natural slot dtype avoids them entirely.
    slots = tl.load(slots_ptr + tokens * slot_stride, mask=mask, other=-1)
    request_end = tl.load(query_start_loc_ptr + num_actual_reqs * query_start_stride)
    valid_end = tl.minimum(request_end, num_actual_tokens)
    # Full graphs keep a fixed token carrier whose padded tail may still hold
    # slot IDs from an earlier replay.  Never let those rows write any packed cache
    # plane, including the uncompressed SWA/C1 planes.
    valid = mask & (tokens < valid_end) & (slots >= 0)

    if COMPRESS_RATIO == 2:
        positions = tl.load(
            positions_ptr + tokens * position_stride,
            mask=mask,
            other=0,
        )
        # (s + 1) % 2 == 0 and p % 2 == 1 reduce to low-bit tests.
        valid &= ((slots & 1) == 1) & (skip_update == 0) & ((positions & 1) == 1)
        physical = slots >> 1
    else:
        physical = slots

    # Clamp before shifting, so the right shift and the mask stay non-negative.
    safe_physical = tl.maximum(physical, 0)
    if LOG2_PAGE >= 0:
        pages = safe_physical >> LOG2_PAGE
        rows = safe_physical & (PAGE_SIZE - 1)
    else:
        # Non power-of-two page sizes keep the generic (slower) form.
        pages = safe_physical // PAGE_SIZE
        rows = safe_physical % PAGE_SIZE
    flat_slots = tl.where(valid, physical, -1)
    pages = tl.where(valid, pages, -1)
    rows = tl.where(valid, rows, -1)
    tl.store(flat_slots_ptr + tokens, flat_slots, mask=mask)
    tl.store(coordinates_ptr + tokens * 2, pages, mask=mask)
    tl.store(coordinates_ptr + tokens * 2 + 1, rows, mask=mask)


def supports_packed_slot_mapping(compress_ratio: int, has_positions: bool) -> bool:
    """Return whether the fused kernel can serve this cache group.

    The kernel only implements the ratio-1 and ratio-2 cache planes. Ratio 1
    ignores token positions entirely, while ratio 2 reads them for the
    completion test, so a ratio-2 group without positions has to stay on the
    eager chain.
    """
    if compress_ratio == 1:
        return True
    return compress_ratio == 2 and has_positions


def build_packed_cache_slot_mapping(
    slots: torch.Tensor,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_tokens: int,
    num_actual_reqs: int,
    num_actual_tokens: int,
    page_size: int,
    compress_ratio: int,
    *,
    skip_update: bool = False,
    coordinates_output: torch.Tensor,
    flat_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish coordinate and flat packed cache addresses together.

    This replaces the eager GE/Add/Remainder/And/Div/Where/Copy chain used by
    every KV cache group. Ratio-2 completion and padded-row masking follow the
    existing metadata builder contract exactly.

    ``positions`` is dereferenced only when ``compress_ratio == 2``; the ratio-1
    branch compiles the load away and reads nothing but ``positions.stride(0)``
    on the host. A ratio-1 caller without positions may therefore pass ``slots``
    itself as a stride-only placeholder.
    """
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    if num_tokens:
        # A decode step carries only a few hundred tokens, so a 256-row block
        # collapses the launch onto a single AIV core. Size the block for
        # occupancy instead: ~6-12 programs at decode, wider tiles at prefill.
        block_size = 16 if num_tokens <= 512 else 64
        page_is_pow2 = page_size > 0 and (page_size & (page_size - 1)) == 0
        log2_page = page_size.bit_length() - 1 if page_is_pow2 else -1
        _build_packed_cache_slot_mapping_kernel[(triton.cdiv(num_tokens, block_size),)](
            slots,
            positions,
            query_start_loc,
            coordinates_output,
            flat_output,
            num_tokens,
            num_actual_reqs,
            num_actual_tokens,
            int(skip_update),
            slots.stride(0),
            positions.stride(0),
            query_start_loc.stride(0),
            PAGE_SIZE=page_size,
            LOG2_PAGE=log2_page,
            COMPRESS_RATIO=compress_ratio,
            BLOCK_SIZE=block_size,
        )
    return coordinates_output, flat_output
