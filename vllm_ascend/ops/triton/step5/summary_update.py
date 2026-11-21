# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend CSA region summaries, compressed once when a region closes.

Adapted from the upstream Triton implementation in
``vllm.models.step5.nvidia.ops.summary_update``: the ring addressing is the
upstream deterministic layout (one ring per request inside its KV block, slot
= block_table[req] * rows_per_block + position % capacity), so the update
kernels only store open-region rows and summarize closed regions. The FP8 e4m3
summary store (``fp32 -> bfloat16 -> e4m3`` double rounding) is bit-identical
on triton-ascend, verified against the torch reference.
"""

import torch
from vllm.triton_utils import tl, triton

# Fixed Step5 DSA summary layout, shared by the kernels below (passed in as
# constexpr) and their wrappers. The proxy head dim is 256 lanes; one 1 KiB
# ring row packs [key | z] as 2 * PROXY_DIM bf16 elements.
PROXY_DIM = 256
RING_ROW_ELEMENTS = 2 * PROXY_DIM
# Tokens per summarized region (region_size in the Step5 config).
REGION_SIZE = 8
# Metadata row layout: [region, first_row, REGION_SIZE member row ids].
METADATA_WIDTH = 2 + REGION_SIZE
# Online-softmax denominator floor from the CUDA reference; keeps the
# summary division finite when every weight underflows to zero.
DENOMINATOR_FLOOR = 1e-20
# Largest top_k the indexer region-score operator supports (csrc README).
MAX_TOPK = 512


@triton.jit
def _append_current_region_kernel(
    ids_ptr,
    positions_ptr,
    wide_ptr,
    counts_ptr,
    tokens,
    region_size,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Pack ``BLOCK_T`` rows of one group, streaming ``TOPK`` in chunks."""
    group = tl.program_id(0)
    tile = tl.program_id(1)
    row = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = row < tokens
    rows = group * tokens + row

    live = tl.zeros((BLOCK_T,), dtype=tl.int32)
    for k0 in range(0, TOPK, BLOCK_K):
        col = k0 + tl.arange(0, BLOCK_K)
        mask = row_mask[:, None] & (col < TOPK)[None, :]
        ids = tl.load(
            ids_ptr + rows[:, None] * TOPK + col[None, :],
            mask=mask,
            other=-1,
        )
        live += tl.sum((ids != -1).to(tl.int32), axis=1)
        tl.store(
            wide_ptr + rows[:, None] * (TOPK + 1) + col[None, :],
            ids,
            mask=mask,
        )

    pad = tl.full((BLOCK_T,), -1, dtype=tl.int32)
    tl.store(wide_ptr + rows * (TOPK + 1) + TOPK, pad, mask=row_mask)

    positions = tl.load(positions_ptr + row, mask=row_mask, other=0)
    current = (positions // region_size).to(tl.int32)
    slot = tl.minimum(live, TOPK)
    tl.store(wide_ptr + rows * (TOPK + 1) + slot, current, mask=row_mask)

    tl.store(counts_ptr + rows, live + 1, mask=row_mask)


def append_current_region(
    ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    groups: int,
    region_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append each row's current region into a ``topk + 1`` wide id tensor.

    ``ids`` is ``[G*T, topk]`` ascending logical ids with ``-1`` tail padding;
    the selection pool is strictly below ``floor(position / region_size)``,
    so the appended current region is larger than every selected id and the
    row stays ascending. It lands at slot ``count`` (the extra ``topk``-th
    column when the selection is full), matching the Triton pack kernels'
    current-region append. Counts become ``live + 1``.

    The whole pack (count + copy + append) runs as a single fused Triton
    kernel.

    Returns ``(idx [G, T, topk + 1], counts [G, T])`` ready for the operator.
    """
    rows, topk = ids.shape
    tokens = rows // groups
    wide = torch.empty((groups, tokens, topk + 1), device=ids.device, dtype=ids.dtype)
    counts = torch.empty((groups, tokens), device=ids.device, dtype=torch.int32)
    block_k = triton.next_power_of_2(min(topk, MAX_TOPK))
    # Cap the per-program tile at 64 rows / ~8K elements so a single program
    # never spans an unreasonable share of the id tensor.
    block_t = max(1, min(64, 8192 // block_k))
    grid = (groups, triton.cdiv(tokens, block_t))
    _append_current_region_kernel[grid](
        ids,
        positions,
        wide,
        counts,
        tokens,
        region_size,
        TOPK=topk,
        BLOCK_K=block_k,
        BLOCK_T=block_t,
    )
    return wide, counts


# TODO: The kernels below will be removed once triton-ascend is compatible with upstream implementations
@triton.jit
def _maximum(x, y):
    return tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _accumulate(numerator, denominator, maximum, key, z):
    updated_max = _maximum(maximum, tl.reduce(z, 0, _maximum))
    scale = tl.where(denominator > 0, tl.exp(maximum - updated_max), 0)
    weight = tl.exp(z - updated_max)
    return (
        numerator * scale + weight * key,
        denominator * scale + weight,
        updated_max,
    )


@triton.jit
def _store_summary(
    mean_ptr,
    slot,
    dim,
    numerator,
    denominator,
    PROXY_DIM: tl.constexpr,
    REGION_SIZE: tl.constexpr,
    DENOMINATOR_FLOOR: tl.constexpr,
    MEAN_BLOCK_STRIDE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    """Write the summary of the region holding KV ``slot``; summary blocks
    start ``MEAN_BLOCK_STRIDE`` bytes apart and hold ``BLOCK_TOKENS`` tokens."""
    summary = tl.div_rn(numerator, _maximum(denominator, DENOMINATOR_FLOOR)).to(tl.bfloat16)
    summary_fp8 = summary.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    # Slots and ring bases arrive as int64 (vLLM slot_mapping and the ring
    # metadata are int64 end to end); the cast only guards against a future
    # int32 buffer, whose late-block byte offsets could exceed 2 GiB.
    address = slot.to(tl.int64) // BLOCK_TOKENS * MEAN_BLOCK_STRIDE + slot % BLOCK_TOKENS // REGION_SIZE * PROXY_DIM
    tl.store(mean_ptr + address + dim, summary_fp8)


@triton.jit
def _replay_ring(
    ring_ptr,
    base,
    first,
    last,
    dim,
    PROXY_DIM: tl.constexpr,
    RING_ROW_ELEMENTS: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
):
    """Accumulate the ring rows at positions ``[first, last)`` in order."""
    numerator = tl.zeros((PROXY_DIM,), tl.float32)
    denominator = tl.zeros((PROXY_DIM,), tl.float32)
    maximum = tl.max(tl.full((PROXY_DIM,), -float("inf"), tl.float32), 0)
    for position in range(first, last):
        row = ring_ptr + (base + position % RING_CAPACITY) * RING_ROW_ELEMENTS
        key = tl.load(row + dim).to(tl.float32)
        z = tl.load(row + PROXY_DIM + dim).to(tl.float32)
        numerator, denominator, maximum = _accumulate(numerator, denominator, maximum, key, z)
        # Serializes the replay loop: the vectorizer cannot prove safety for
        # cross-iteration vectorization when the trip count is a runtime
        # difference (real prefill replays up to a full ring capacity).
        tl.debug_barrier()
    return numerator, denominator, maximum


@triton.jit
def _update_decode(
    key_ptr,
    z_ptr,
    slots_ptr,
    positions_ptr,
    ring_base_ptr,
    mean_ptr,
    ring_ptr,
    KEY_ROW_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK_STRIDE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    REGION_SIZE: tl.constexpr,
    RING_ROW_ELEMENTS: tl.constexpr,
    DENOMINATOR_FLOOR: tl.constexpr,
):
    """Store an open-region row, or summarize the region this row closes.

    The close folds the region's other ``REGION_SIZE - 1`` ring rows plus
    this row as flat masked tiles; loops carrying 256-wide tensors with
    scalar guards inside defeat the Ascend vectorizer (see _update_prefill).
    """
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row)
    base = tl.load(ring_base_ptr + row)
    if (slot >= 0) & (base >= 0):
        position = tl.load(positions_ptr + row)
        dim = tl.arange(0, PROXY_DIM)
        key = tl.load(key_ptr + row * KEY_ROW_STRIDE + dim).to(tl.float32)
        z = tl.load(z_ptr + row * Z_ROW_STRIDE + dim).to(tl.float32)
        if position % REGION_SIZE == REGION_SIZE - 1:
            offset = tl.arange(0, REGION_SIZE)
            carry = offset.to(tl.int64) < REGION_SIZE - 1
            ring_off = (
                base.to(tl.int64) + (position - (REGION_SIZE - 1) + offset.to(tl.int64)) % RING_CAPACITY
            ) * RING_ROW_ELEMENTS
            rkey = tl.load(ring_ptr + ring_off[:, None] + dim[None, :], mask=carry[:, None], other=0.0).to(tl.float32)
            rz = tl.load(
                ring_ptr + ring_off[:, None] + PROXY_DIM + dim[None, :],
                mask=carry[:, None],
                other=-float("inf"),
            ).to(tl.float32)
            maximum = tl.maximum(tl.max(tl.where(carry[:, None], rz, -float("inf"))), tl.max(z))
            rweight = tl.exp(rz - maximum)
            cweight = tl.exp(z - maximum)
            numerator = tl.sum(rweight * rkey, 0) + cweight * key
            denominator = tl.sum(rweight, 0) + cweight
            _store_summary(
                mean_ptr,
                slot,
                dim,
                numerator,
                denominator,
                PROXY_DIM,
                REGION_SIZE,
                DENOMINATOR_FLOOR,
                MEAN_BLOCK_STRIDE,
                BLOCK_TOKENS,
            )
        else:
            ring_row = ring_ptr + (base.to(tl.int64) + position % RING_CAPACITY) * RING_ROW_ELEMENTS
            tl.store(ring_row + dim, key.to(tl.bfloat16))
            tl.store(ring_row + PROXY_DIM + dim, z.to(tl.bfloat16))


def summary_layout(mean: torch.Tensor) -> tuple[int, int]:
    """(bytes between blocks, tokens per block) of a ``[B, R, 256]`` cache."""
    assert mean.stride(1) == PROXY_DIM and mean.stride(2) == 1
    return mean.stride(0), mean.shape[1] * REGION_SIZE


def update_decode(key, z, slots, positions, ring_base, mean, ring, capacity):
    # num_warps=4 with fp fusion disabled is the launch configuration the
    # bit-exactness against the torch reference was verified with.
    if slots.numel():
        _update_decode[(slots.numel(),)](
            key,
            z,
            slots,
            positions,
            ring_base,
            mean,
            ring,
            key.stride(0),
            z.stride(0),
            capacity,
            *summary_layout(mean),
            PROXY_DIM,
            REGION_SIZE,
            RING_ROW_ELEMENTS,
            DENOMINATOR_FLOOR,
            num_warps=4,
            enable_fp_fusion=False,
        )


@triton.jit
def _update_verify(
    key_ptr,
    z_ptr,
    slots_ptr,
    query_start_ptr,
    seq_lens_ptr,
    ring_base_ptr,
    mean_ptr,
    ring_ptr,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    Z_DIM_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK_STRIDE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    REGION_SIZE: tl.constexpr,
    RING_ROW_ELEMENTS: tl.constexpr,
    DENOMINATOR_FLOOR: tl.constexpr,
):
    # One program per request: its rows update the same open region in order.
    request = tl.program_id(0)
    start = tl.load(query_start_ptr + request)
    end = tl.load(query_start_ptr + request + 1)
    if start < end:
        base = tl.load(ring_base_ptr + start)
        seq = tl.load(seq_lens_ptr + request)
        dim = tl.arange(0, PROXY_DIM)
        for row in range(start, end):
            slot = tl.load(slots_ptr + row)
            if (slot >= 0) & (base >= 0):
                position = seq - (end - row)
                ring_row = ring_ptr + (base + position % RING_CAPACITY) * RING_ROW_ELEMENTS
                key = tl.load(key_ptr + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE)
                z = tl.load(z_ptr + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE)
                tl.store(ring_row + dim, key.to(tl.bfloat16))
                tl.store(ring_row + PROXY_DIM + dim, z.to(tl.bfloat16))
                if position % REGION_SIZE == REGION_SIZE - 1:
                    tl.debug_barrier()
                    numerator, denominator, _ = _replay_ring(
                        ring_ptr,
                        base,
                        position - (REGION_SIZE - 1),
                        position + 1,
                        dim,
                        PROXY_DIM,
                        RING_ROW_ELEMENTS,
                        RING_CAPACITY,
                    )
                    _store_summary(
                        mean_ptr,
                        slot,
                        dim,
                        numerator,
                        denominator,
                        PROXY_DIM,
                        REGION_SIZE,
                        DENOMINATOR_FLOOR,
                        MEAN_BLOCK_STRIDE,
                        BLOCK_TOKENS,
                    )
            tl.debug_barrier()


def update_verify(key, z, slots, query_start_loc, seq_lens, ring_base, mean, ring, capacity):
    """Uniform MTP verify update; graph capturable, no host metadata."""
    requests = seq_lens.shape[0]
    if requests:
        _update_verify[(requests,)](
            key,
            z,
            slots,
            query_start_loc,
            seq_lens,
            ring_base,
            mean,
            ring,
            key.stride(0),
            key.stride(2),
            z.stride(0),
            z.stride(2),
            capacity,
            *summary_layout(mean),
            PROXY_DIM,
            REGION_SIZE,
            RING_ROW_ELEMENTS,
            DENOMINATOR_FLOOR,
            num_warps=4,
            enable_fp_fusion=False,
        )


@triton.jit
def _build_prefill_groups(
    slots_ptr,
    positions_ptr,
    owners_ptr,
    query_start_ptr,
    metadata_ptr,
    POSITION_STRIDE: tl.constexpr,
    REGION_SIZE: tl.constexpr,
    METADATA_WIDTH: tl.constexpr,
):
    # One program per token row. A group is the run of one request's rows in
    # an eight-token region; its first row writes the metadata, others -1.
    row = tl.program_id(0)
    output = metadata_ptr + row.to(tl.int64) * METADATA_WIDTH
    owner = tl.load(owners_ptr + row)
    start = tl.load(query_start_ptr + owner)
    end = tl.load(query_start_ptr + owner + 1)
    position = tl.load(positions_ptr + row * POSITION_STRIDE)
    head = (row == start) | (position % REGION_SIZE == 0)
    offset = tl.arange(0, REGION_SIZE)
    step = offset - position % REGION_SIZE
    rows = row + step
    in_group = head & (step >= 0) & (rows < end)
    slot = tl.load(slots_ptr + rows, in_group, other=-1)
    valid = in_group & (slot >= 0)
    first = tl.min(tl.where(valid, offset, REGION_SIZE), 0)
    region = tl.max(tl.where(valid, slot // REGION_SIZE, -1), 0)
    tl.store(output, tl.where(first < REGION_SIZE, region, -1))
    tl.store(output + 1, first)
    tl.store(output + 2 + offset, tl.where(valid, rows, -1).to(tl.int64))


def build_prefill_groups(slots, positions, owners, query_start_loc) -> torch.Tensor:
    """Summary update metadata on device: one synchronous launch, no host
    round trip (the Ascend builder computes the same grouping on device
    instead of staging pinned host plans asynchronously)."""
    rows = slots.shape[0]
    metadata = torch.empty((rows, METADATA_WIDTH), dtype=torch.int64, device=slots.device)
    if rows:
        _build_prefill_groups[(rows,)](
            slots,
            positions,
            owners,
            query_start_loc,
            metadata,
            positions.stride(0),
            REGION_SIZE,
            METADATA_WIDTH,
        )
    return metadata


@triton.jit
def _update_prefill(
    key_ptr,
    z_ptr,
    metadata_ptr,
    positions_ptr,
    ring_base_ptr,
    mean_ptr,
    ring_ptr,
    POSITION_STRIDE: tl.constexpr,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    Z_DIM_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK_STRIDE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    REGION_SIZE: tl.constexpr,
    RING_ROW_ELEMENTS: tl.constexpr,
    METADATA_WIDTH: tl.constexpr,
    DENOMINATOR_FLOOR: tl.constexpr,
):
    metadata = metadata_ptr + tl.program_id(0) * METADATA_WIDTH
    # Only a group holding its region's eighth row closes the region.
    if (tl.load(metadata) >= 0) & (tl.load(metadata + METADATA_WIDTH - 1) >= 0):
        region = tl.load(metadata)
        # Rows before ``first`` were stored by an earlier step (in the
        # ring); rows of this chunk follow. Both folds are flat (8, 256)
        # tiles with masks -- loops that carry 256-wide tensors and hold
        # scalar guards inside defeat the Ascend vectorizer, and the
        # reference NPU kernels fold the same math as masked tiles.
        first = tl.load(metadata + 1)
        head = tl.load(metadata + 2 + first)
        position = tl.load(positions_ptr + head * POSITION_STRIDE)
        base = tl.load(ring_base_ptr + head)
        dim = tl.arange(0, PROXY_DIM)
        offset = tl.arange(0, REGION_SIZE)
        carry = offset.to(tl.int64) < first
        ring_row = (base.to(tl.int64) + (position - first + offset.to(tl.int64)) % RING_CAPACITY) * RING_ROW_ELEMENTS
        rkey = tl.load(ring_ptr + ring_row[:, None] + dim[None, :], mask=carry[:, None], other=0.0).to(tl.float32)
        rz = tl.load(
            ring_ptr + ring_row[:, None] + PROXY_DIM + dim[None, :], mask=carry[:, None], other=-float("inf")
        ).to(tl.float32)
        rows = tl.load(metadata + 2 + offset)
        valid = rows >= 0
        safe = tl.where(valid, rows, 0)
        key = tl.load(
            key_ptr + safe[:, None] * KEY_ROW_STRIDE + dim[None, :] * KEY_DIM_STRIDE, mask=valid[:, None], other=0.0
        ).to(tl.float32)
        z = tl.load(
            z_ptr + safe[:, None] * Z_ROW_STRIDE + dim[None, :] * Z_DIM_STRIDE, mask=valid[:, None], other=-float("inf")
        ).to(tl.float32)
        maximum = tl.maximum(
            tl.max(tl.where(carry[:, None], rz, -float("inf"))),
            tl.max(tl.where(valid[:, None], z, -float("inf"))),
        )
        # An invalid row loads z = -inf, so its weight is exp(-inf) = 0;
        # the region always closes here, so ``maximum`` is finite.
        rweight = tl.exp(rz - maximum)
        cweight = tl.exp(z - maximum)
        numerator = tl.sum(rweight * rkey, 0) + tl.sum(cweight * key, 0)
        denominator = tl.sum(rweight, 0) + tl.sum(cweight, 0)
        _store_summary(
            mean_ptr,
            region * REGION_SIZE,
            dim,
            numerator,
            denominator,
            PROXY_DIM,
            REGION_SIZE,
            DENOMINATOR_FLOOR,
            MEAN_BLOCK_STRIDE,
            BLOCK_TOKENS,
        )


@triton.jit
def _write_ring(
    key_ptr,
    z_ptr,
    ring_slots_ptr,
    ring_ptr,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    Z_DIM_STRIDE: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    RING_ROW_ELEMENTS: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(ring_slots_ptr + row)
    if slot >= 0:
        dim = tl.arange(0, PROXY_DIM)
        key = tl.load(key_ptr + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE)
        z = tl.load(z_ptr + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE)
        tl.store(ring_ptr + slot * RING_ROW_ELEMENTS + dim, key.to(tl.bfloat16))
        tl.store(ring_ptr + slot * RING_ROW_ELEMENTS + PROXY_DIM + dim, z.to(tl.bfloat16))


def update_prefill(key, z, metadata, positions, ring_base, ring_slots, mean, ring, capacity):
    """Summarize the regions this step closes, then store open-region rows.

    Every ring read of the step precedes its writes, so a long chunk cannot
    overwrite open-region rows it still has to read.
    """
    rows = metadata.shape[0]
    if not rows:
        return
    strides = (key.stride(0), key.stride(2), z.stride(0), z.stride(2))
    _update_prefill[(rows,)](
        key,
        z,
        metadata,
        positions,
        ring_base,
        mean,
        ring,
        positions.stride(0),
        *strides,
        capacity,
        *summary_layout(mean),
        PROXY_DIM,
        REGION_SIZE,
        RING_ROW_ELEMENTS,
        METADATA_WIDTH,
        DENOMINATOR_FLOOR,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _write_ring[(ring_slots.shape[0],)](key, z, ring_slots, ring, *strides, PROXY_DIM, RING_ROW_ELEMENTS, num_warps=4)
