# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend CSA region summaries, compressed once when a region closes.

Adapted from the CUDA implementation in
``vllm.models.step5.nvidia.ops.summary_update``: the ring addressing is the
upstream deterministic layout (one ring per request inside its KV block, slot
= block_table[req] * rows_per_block + position % capacity), so the update
kernels only store open-region rows and summarize closed regions. The CUDA
``libdevice`` exponential is replaced with ``tl.exp``; the FP8 e4m3 summary
store (``fp32 -> bfloat16 -> e4m3`` double rounding) is bit-identical on
triton-ascend, verified against the torch reference.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _maximum(x, y):
    return tl.maximum(x, y)


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
    Mean,
    slot,
    dim,
    numerator,
    denominator,
    MEAN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Write the summary of the region holding KV ``slot``; blocks of ``Mean``
    are ``MEAN_BLOCK`` bytes apart."""
    summary = tl.div_rn(numerator, _maximum(denominator, 1e-20)).to(tl.bfloat16)
    summary_fp8 = summary.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    # Slot buffers are int32 on Ascend; the byte offset can exceed 2 GiB.
    address = (
        slot.to(tl.int64) // BLOCK_SIZE * MEAN_BLOCK
        + slot % BLOCK_SIZE // 8 * 256
    )
    tl.store(Mean + address + dim, summary_fp8)


@triton.jit
def _replay_ring(Ring, base, first, last, dim, RING_CAPACITY: tl.constexpr):
    """Accumulate the ring rows at positions ``[first, last)`` in order."""
    numerator = tl.zeros((256,), tl.float32)
    denominator = tl.zeros((256,), tl.float32)
    maximum = tl.max(tl.full((256,), -float("inf"), tl.float32), 0)
    for position in range(first, last):
        row = Ring + (base + position % RING_CAPACITY) * 512
        key = tl.load(row + dim).to(tl.float32)
        z = tl.load(row + 256 + dim).to(tl.float32)
        numerator, denominator, maximum = _accumulate(
            numerator, denominator, maximum, key, z
        )
        # Serializes the replay loop: the vectorizer cannot prove safety for
        # cross-iteration vectorization when the trip count is a runtime
        # difference (real prefill replays up to a full ring capacity).
        tl.debug_barrier()
    return numerator, denominator, maximum


@triton.jit
def _update_decode(
    Key,
    Z,
    Slots,
    Positions,
    RingBase,
    Mean,
    Ring,
    KEY_STRIDE: tl.constexpr,
    Z_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Store an open-region row, or summarize the region this row closes.

    The close folds the region's 7 earlier ring rows plus this row as flat
    masked tiles; loops carrying 256-wide tensors with scalar guards inside
    defeat the Ascend vectorizer (see _update_prefill).
    """
    row = tl.program_id(0)
    slot = tl.load(Slots + row)
    base = tl.load(RingBase + row)
    if (slot >= 0) & (base >= 0):
        position = tl.load(Positions + row)
        dim = tl.arange(0, 256)
        key = tl.load(Key + row * KEY_STRIDE + dim).to(tl.float32)
        z = tl.load(Z + row * Z_STRIDE + dim).to(tl.float32)
        if position % 8 == 7:
            offset = tl.arange(0, 8)
            carry = offset.to(tl.int64) < 7
            ring_off = (
                base.to(tl.int64)
                + (position - 7 + offset.to(tl.int64)) % RING_CAPACITY
            ) * 512
            rkey = tl.load(
                Ring + ring_off[:, None] + dim[None, :], mask=carry[:, None], other=0.0
            ).to(tl.float32)
            rz = tl.load(
                Ring + ring_off[:, None] + 256 + dim[None, :],
                mask=carry[:, None],
                other=-float("inf"),
            ).to(tl.float32)
            maximum = tl.maximum(tl.max(tl.where(carry[:, None], rz, -float("inf"))), tl.max(z))
            rweight = tl.exp(rz - maximum)
            cweight = tl.exp(z - maximum)
            numerator = tl.sum(rweight * rkey, 0) + cweight * key
            denominator = tl.sum(rweight, 0) + cweight
            _store_summary(
                Mean, slot, dim, numerator, denominator, MEAN_BLOCK, BLOCK_SIZE
            )
        else:
            ring_row = Ring + (base.to(tl.int64) + position % RING_CAPACITY) * 512
            tl.store(ring_row + dim, key.to(tl.bfloat16))
            tl.store(ring_row + 256 + dim, z.to(tl.bfloat16))


def summary_layout(mean: torch.Tensor) -> tuple[int, int]:
    """(bytes between blocks, tokens per block) of a ``[B, R, 256]` cache."""
    assert mean.stride(1) == 256 and mean.stride(2) == 1
    return mean.stride(0), mean.shape[1] * 8


def update_decode(key, z, slots, positions, ring_base, mean, ring, capacity):
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
            num_warps=4,
            enable_fp_fusion=False,
        )


@triton.jit
def _update_verify(
    Key,
    Z,
    Slots,
    QueryStart,
    SeqLens,
    RingBase,
    Mean,
    Ring,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    Z_DIM_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per request: its rows update the same open region in order.
    request = tl.program_id(0)
    start = tl.load(QueryStart + request)
    end = tl.load(QueryStart + request + 1)
    if start < end:
        base = tl.load(RingBase + start)
        seq = tl.load(SeqLens + request)
        dim = tl.arange(0, 256)
        for row in range(start, end):
            slot = tl.load(Slots + row)
            if (slot >= 0) & (base >= 0):
                position = seq - (end - row)
                ring_row = Ring + (base + position % RING_CAPACITY) * 512
                key = tl.load(Key + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE)
                z = tl.load(Z + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE)
                tl.store(ring_row + dim, key.to(tl.bfloat16))
                tl.store(ring_row + 256 + dim, z.to(tl.bfloat16))
                if position % 8 == 7:
                    numerator, denominator, _ = _replay_ring(
                        Ring, base, position - 7, position + 1, dim, RING_CAPACITY
                    )
                    _store_summary(
                        Mean, slot, dim, numerator, denominator, MEAN_BLOCK, BLOCK_SIZE
                    )
            tl.debug_barrier()


def update_verify(
    key, z, slots, query_start_loc, seq_lens, ring_base, mean, ring, capacity
):
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
            num_warps=4,
            enable_fp_fusion=False,
        )


@triton.jit
def _build_prefill_groups(
    Slots,
    Positions,
    Owners,
    QueryStart,
    Metadata,
    POSITION_STRIDE: tl.constexpr,
):
    # One program per token row. A group is the run of one request's rows in
    # an eight-token region; its first row writes the metadata, others -1.
    row = tl.program_id(0)
    output = Metadata + row.to(tl.int64) * 10
    owner = tl.load(Owners + row)
    start = tl.load(QueryStart + owner)
    end = tl.load(QueryStart + owner + 1)
    position = tl.load(Positions + row * POSITION_STRIDE)
    head = (row == start) | (position % 8 == 0)
    offset = tl.arange(0, 8)
    step = offset - position % 8
    rows = row + step
    in_group = head & (step >= 0) & (rows < end)
    slot = tl.load(Slots + rows, in_group, other=-1)
    valid = in_group & (slot >= 0)
    first = tl.min(tl.where(valid, offset, 8), 0)
    region = tl.max(tl.where(valid, slot // 8, -1), 0)
    tl.store(output, tl.where(first < 8, region, -1))
    tl.store(output + 1, first)
    tl.store(output + 2 + offset, tl.where(valid, rows, -1).to(tl.int64))


def build_prefill_groups(slots, positions, owners, query_start_loc) -> torch.Tensor:
    """Summary update metadata on device: one synchronous launch, no host
    round trip (the Ascend builder computes the same grouping on device
    instead of staging pinned host plans asynchronously)."""
    rows = slots.shape[0]
    metadata = torch.empty((rows, 10), dtype=torch.int64, device=slots.device)
    if rows:
        _build_prefill_groups[(rows,)](
            slots,
            positions,
            owners,
            query_start_loc,
            metadata,
            positions.stride(0),
        )
    return metadata


@triton.jit
def _update_prefill(
    Key,
    Z,
    Metadata,
    Positions,
    RingBase,
    Mean,
    Ring,
    POSITION_STRIDE: tl.constexpr,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    Z_DIM_STRIDE: tl.constexpr,
    RING_CAPACITY: tl.constexpr,
    MEAN_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    metadata = Metadata + tl.program_id(0) * 10
    # Only a group holding its region's eighth row closes the region.
    if (tl.load(metadata) >= 0) & (tl.load(metadata + 9) >= 0):
        region = tl.load(metadata)
        # Rows before ``first`` were stored by an earlier step (in the
        # ring); rows of this chunk follow. Both folds are flat (8, 256)
        # tiles with masks -- loops that carry 256-wide tensors and hold
        # scalar guards inside defeat the Ascend vectorizer, and the
        # reference NPU kernels fold the same math as masked tiles.
        first = tl.load(metadata + 1)
        head = tl.load(metadata + 2 + first)
        position = tl.load(Positions + head * POSITION_STRIDE)
        base = tl.load(RingBase + head)
        dim = tl.arange(0, 256)
        offset = tl.arange(0, 8)
        carry = offset.to(tl.int64) < first
        ring_row = (base.to(tl.int64) + (position - first + offset.to(tl.int64)) % RING_CAPACITY) * 512
        rkey = tl.load(Ring + ring_row[:, None] + dim[None, :], mask=carry[:, None], other=0.0).to(tl.float32)
        rz = tl.load(Ring + ring_row[:, None] + 256 + dim[None, :], mask=carry[:, None], other=-float("inf")).to(tl.float32)
        rows = tl.load(metadata + 2 + offset)
        valid = rows >= 0
        safe = tl.where(valid, rows, 0)
        key = tl.load(Key + safe[:, None] * KEY_ROW_STRIDE + dim[None, :] * KEY_DIM_STRIDE, mask=valid[:, None], other=0.0).to(tl.float32)
        z = tl.load(Z + safe[:, None] * Z_ROW_STRIDE + dim[None, :] * Z_DIM_STRIDE, mask=valid[:, None], other=-float("inf")).to(tl.float32)
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
            Mean, region * 8, dim, numerator, denominator, MEAN_BLOCK, BLOCK_SIZE
        )


@triton.jit
def _write_ring(
    Key,
    Z,
    RingSlots,
    Ring,
    KEY_ROW_STRIDE: tl.constexpr,
    KEY_DIM_STRIDE: tl.constexpr,
    Z_ROW_STRIDE: tl.constexpr,
    Z_DIM_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(RingSlots + row)
    if slot >= 0:
        dim = tl.arange(0, 256)
        key = tl.load(Key + row * KEY_ROW_STRIDE + dim * KEY_DIM_STRIDE)
        z = tl.load(Z + row * Z_ROW_STRIDE + dim * Z_DIM_STRIDE)
        tl.store(Ring + slot * 512 + dim, key.to(tl.bfloat16))
        tl.store(Ring + slot * 512 + 256 + dim, z.to(tl.bfloat16))


def update_prefill(
    key, z, metadata, positions, ring_base, ring_slots, mean, ring, capacity
):
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
        num_warps=4,
        enable_fp_fusion=False,
    )
    _write_ring[(ring_slots.shape[0],)](key, z, ring_slots, ring, *strides, num_warps=4)
