# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mapped-Host refill, invalidation and packed KV layout operations.

Residency planning is provided by the native AscendC resolver.
"""

from dataclasses import dataclass
from typing import Any

import torch
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import LaunchSpec, TritonWarmupTensor
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.ascend_jit_warmup import AscendTritonJitKernel

_INVALID_INDEX = tl.constexpr(-1)
# Hot table entries hold nonnegative signed int32 Host slots.
_HOST_SLOT_BITS = tl.constexpr(torch.iinfo(torch.int32).bits - 1)
_MAX_HOST_SLOT = tl.constexpr(torch.iinfo(torch.int32).max)
# The largest signed int64 sorts after every valid (state, slot) key.
_INVALID_SORTED_SLOT_KEY = tl.constexpr(torch.iinfo(torch.int64).max)
_PREFILL_REPRESENTATIVE_TILE = 256
_MAX_PREFILL_REPRESENTATIVE_ROWS = torch.iinfo(torch.int32).max - _PREFILL_REPRESENTATIVE_TILE


@triton.jit(
    do_not_specialize=["num_rows"],
    do_not_specialize_on_alignment=["index_ptr", "source_ptr", "representatives_ptr"],
)
def _scatter_min_representatives_kernel(index_ptr, source_ptr, representatives_ptr, num_rows, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = row < num_rows
    destination = tl.load(index_ptr + row, valid, other=0)
    position = tl.load(source_ptr + row, valid, other=0)
    tl.atomic_min(representatives_ptr + destination, position, mask=valid)


class TritonScatterMinRepresentativesKernel(AscendTritonJitKernel["TritonScatterMinRepresentativesKernel.CompileKey"]):
    kernel = _scatter_min_representatives_kernel
    # The launch spec supplies all arguments explicitly: avoid inspect.signature
    # binding on each small prefill reduction.
    bind_launch_inputs = False

    @dataclass(frozen=True)
    class CompileKey:
        block_size: int

    def dispatch(self, *, BLOCK_SIZE: int) -> CompileKey:
        return self.CompileKey(block_size=BLOCK_SIZE)

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        # All pointers are int64; runtime length and pointer alignment do not
        # specialize. The tile is the only configurable compile-time argument.
        return self._trace_dispatch(self.dispatch)(BLOCK_SIZE=_PREFILL_REPRESENTATIVE_TILE)

    def warmup_inputs(self, key: CompileKey) -> dict[str, Any]:
        return dict(
            index=TritonWarmupTensor(torch.int64, shape=(key.block_size,)),
            source=TritonWarmupTensor(torch.int64, shape=(key.block_size,)),
            representatives=TritonWarmupTensor(torch.int64, shape=(key.block_size,)),
            num_rows=key.block_size,
            BLOCK_SIZE=key.block_size,
        )

    @kernel_launcher
    def __call__(self, index, source, representatives, num_rows: int, BLOCK_SIZE: int) -> LaunchSpec:
        return (triton.cdiv(num_rows, BLOCK_SIZE),), dict(
            index_ptr=index,
            source_ptr=source,
            representatives_ptr=representatives,
            num_rows=num_rows,
            BLOCK_SIZE=BLOCK_SIZE,
        )


# One process-local launcher is shared by startup registration and runtime.
# It retains compilation metadata only, never request tensors or outputs.
_TRITON_SCATTER_MIN_REPRESENTATIVES = TritonScatterMinRepresentativesKernel()


def scatter_min_representatives(
    representatives: torch.Tensor, index: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """In-place int64 amin on contiguous, nonoverlapping vectors on one NPU.

    The caller supplies same-NPU vectors and valid compact IDs
    in ``[0, representatives.numel())``.
    Existing output values participate in the min (``include_self=True``).
    No input contents are read on the host; empty input launches no kernel.
    """
    tensors = (representatives, index, source)
    if any(t.ndim != 1 or t.dtype != torch.int64 or not t.is_contiguous() for t in tensors):
        raise ValueError("HiSparse representative selection requires contiguous int64 vectors")
    if index.shape != source.shape:
        raise ValueError("HiSparse representative indices and source must have matching shapes")
    num_rows = index.numel()
    # Keep both Triton's runtime scalar type and the row offset arithmetic int32.
    if num_rows > _MAX_PREFILL_REPRESENTATIVE_ROWS:
        raise ValueError("HiSparse representative input exceeds the int32 row limit")
    if num_rows:
        if not representatives.numel():
            raise ValueError("HiSparse representative output must be nonempty for nonempty input")
        _TRITON_SCATTER_MIN_REPRESENTATIVES(index, source, representatives, num_rows, _PREFILL_REPRESENTATIVE_TILE)
    return representatives


@triton.jit
def gather_rows_kernel(
    host_address,
    hot,
    source,
    destination,
    counts,
    mask_ptr,
    attention,
    state,
    HOST_ROWS: tl.constexpr,
    HOT_ROWS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    HOT_BLOCK: tl.constexpr,
    HOT_STRIDE_BYTES: tl.constexpr,
    TOP_K: tl.constexpr,
    ATTENTION_BLOCK: tl.constexpr,
    HAS_ATTENTION: tl.constexpr,
    HAS_STATE: tl.constexpr,
    BLOCK: tl.constexpr,
    PARALLEL_COLUMNS: tl.constexpr = 0,
):
    row = tl.program_id(1) if PARALLEL_COLUMNS > 0 else tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    padding = tl.load(state + row) < 0 if HAS_STATE else False
    if PARALLEL_COLUMNS > 0:
        tl.static_assert(not HAS_ATTENTION)
        count = tl.minimum(TOP_K, tl.maximum(tl.load(counts + row), 0))
        # Empty compact plans do not inspect stale source/destination entries.
        # Keep enough independent programs for small misses and retain the
        # original contiguous byte-copy shape instead of a wide 2-D payload.
        if ~padding:
            for col in range(tl.program_id(0), count, PARALLEL_COLUMNS):
                index = row * TOP_K + col
                src = tl.load(source + index)
                dst = tl.load(destination + index)
                valid = (src >= 0) & (dst >= 0) & (dst < HOT_ROWS)
                if valid:
                    host = host_address.to(tl.pointer_type(tl.uint8))
                    destination_offset = dst.to(tl.int64) // HOT_BLOCK * HOT_STRIDE_BYTES + dst % HOT_BLOCK * ROW_BYTES
                    for start in range(0, ROW_BYTES, BLOCK):
                        payload = tl.load(
                            host + src.to(tl.int64) * ROW_BYTES + start + offset,
                            mask=(src < HOST_ROWS) & (start + offset < ROW_BYTES),
                            other=0,
                        )
                        tl.store(hot + destination_offset + start + offset, payload, mask=start + offset < ROW_BYTES)
    else:
        col = tl.program_id(1)
        index = row * TOP_K + col
        dst = tl.load(destination + index)
        if HAS_ATTENTION:
            value = tl.where(
                (dst >= 0) & ~padding, dst // HOT_BLOCK * ATTENTION_BLOCK + dst % HOT_BLOCK, _INVALID_INDEX
            )
            tl.store(attention + index, value)
        copy = tl.load(mask_ptr + index) != 0
        if copy & ~padding:
            src = tl.load(source + index)
            valid = (src >= 0) & (dst >= 0) & (dst < HOT_ROWS)
            if valid:
                host = host_address.to(tl.pointer_type(tl.uint8))
                destination_offset = dst.to(tl.int64) // HOT_BLOCK * HOT_STRIDE_BYTES + dst % HOT_BLOCK * ROW_BYTES
                for start in range(0, ROW_BYTES, BLOCK):
                    payload = tl.load(
                        host + src.to(tl.int64) * ROW_BYTES + start + offset,
                        mask=(src < HOST_ROWS) & (start + offset < ROW_BYTES),
                        other=0,
                    )
                    tl.store(hot + destination_offset + start + offset, payload, mask=start + offset < ROW_BYTES)


@triton.jit
def invalidate_kernel(
    tables,
    states,
    requests,
    written,
    REQUESTS: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    REGION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    dgi = tl.load(tables + tl.program_id(2)).to(tl.pointer_type(tl.int32))
    token = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(requests + token)
    state = tl.load(states + request, mask=(request >= 0) & (request < REQUESTS), other=_INVALID_INDEX)
    value = tl.load(written + token)
    valid = (state >= 0) & (state < STATE_ROWS) & (value >= 0) & (offset < REGION)
    ptr = dgi + state.to(tl.int64) * REGION + offset
    cached = tl.load(ptr, mask=valid, other=_INVALID_INDEX)
    tl.store(ptr, _INVALID_INDEX, mask=valid & (cached == value))


@triton.jit(do_not_specialize=["ROWS"])
def invalidate_written_slots_batched(
    tables,
    states,
    requests,
    written,
    ROWS,
    REQUESTS: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    REGION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    dgi = tl.load(tables + tl.program_id(1)).to(tl.pointer_type(tl.int32))
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # One program owns this tile across all states. Process writes in order
    # so a contiguous store preserves other tags without cross-program races.
    for token in range(ROWS):
        request = tl.load(requests + token)
        state = tl.load(states + request, mask=(request >= 0) & (request < REQUESTS), other=_INVALID_INDEX)
        value = tl.load(written + token)
        valid = (state >= 0) & (state < STATE_ROWS) & (value >= 0) & (value <= _MAX_HOST_SLOT) & (offset < REGION)
        ptr = dgi + state.to(tl.int64) * REGION + offset
        cached = tl.load(ptr, mask=valid, other=_INVALID_INDEX)
        # Tags are INT32; reject out-of-range writes before narrowing INT64.
        tl.store(ptr, tl.where(cached == value.to(tl.int32), _INVALID_INDEX, cached), mask=valid)


@triton.jit(do_not_specialize=["ROWS", "REQUESTS", "STATES"])
def pack_written_slot_keys(
    states,
    requests,
    written,
    keys,
    ROWS,
    REQUESTS,
    STATES,
    TILE: tl.constexpr,
):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    request = tl.load(requests + row, mask=row < ROWS, other=_INVALID_INDEX)
    slot = tl.load(written + row, mask=row < ROWS, other=_INVALID_INDEX)
    valid = (request >= 0) & (request < REQUESTS) & (slot >= 0) & (slot <= _MAX_HOST_SLOT)
    state = tl.load(states + request, mask=valid, other=_INVALID_INDEX)
    valid = valid & (state >= 0) & (state < STATES)
    # Each key identifies a Host slot within one request state. Slots above
    # int32 cannot match a Hot entry and share the padding sentinel.
    key = (state.to(tl.int64) << _HOST_SLOT_BITS) | slot.to(tl.int64)
    tl.store(keys + row, tl.where(valid, key, _INVALID_SORTED_SLOT_KEY), mask=row < ROWS)


@triton.jit(do_not_specialize=["ROWS"])
def invalidate_written_slots_sorted(
    tables,
    keys,
    ROWS,
    HOT: tl.constexpr,
    TILE: tl.constexpr,
    STEPS: tl.constexpr,
    UNROLL_SEARCH: tl.constexpr = True,
):
    table = tl.load(tables + tl.program_id(0)).to(tl.pointer_type(tl.int32))
    state = tl.program_id(1)
    offset = tl.program_id(2) * TILE + tl.arange(0, TILE)
    pointer = table + state.to(tl.int64) * HOT + offset
    cached = tl.load(pointer, mask=offset < HOT, other=_INVALID_INDEX)
    valid = (offset < HOT) & (cached >= 0)
    if tl.sum(valid.to(tl.int32), 0) > 0:
        query = (state.to(tl.int64) << _HOST_SLOT_BITS) | cached.to(tl.int64)
        low = tl.full((TILE,), 0, tl.int32)
        high = tl.full((TILE,), ROWS, tl.int32)
        # Keep low/high loop-carried on arch22: fully unrolling this masked
        # binary search fails Triton-Ascend's TritonToUnstructure pass.
        # Ascend 950 retains the existing unrolled search.
        for _ in tl.range(STEPS, loop_unroll_factor=STEPS if UNROLL_SEARCH else 1):
            active = valid & (low < high)
            mid = low + (high - low) // 2
            value = tl.load(keys + mid, mask=active, other=_INVALID_SORTED_SLOT_KEY)
            advance = active & (value < query)
            low = tl.where(advance, mid + 1, low)
            high = tl.where(active & ~advance, mid, high)
        inside = valid & (low < ROWS)
        key = tl.load(keys + low, mask=inside, other=_INVALID_SORTED_SLOT_KEY)
        tl.store(pointer, _INVALID_INDEX, mask=inside & (key == query))


@triton.jit
def pack_attention_rows_kernel(
    cache,
    indices,
    latent,
    rope,
    K: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    LATENT: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    i = tl.arange(0, BLOCK)
    physical = tl.load(indices + row * INDEX_STRIDE + col)
    valid = (physical >= 0) & (physical < CACHE_ROWS)
    base = physical.to(tl.int64) // CACHE_BLOCK * CACHE_STRIDE + physical % CACHE_BLOCK * WIDTH
    value = tl.load(cache + base + i, mask=valid & (i < WIDTH), other=0)
    tl.store(latent + (row * K + col).to(tl.int64) * LATENT + i, value, mask=i < LATENT)
    tl.store(
        rope + (row * K + col).to(tl.int64) * (WIDTH - LATENT) + i - LATENT, value, mask=(i >= LATENT) & (i < WIDTH)
    )


@triton.jit
def scatter_packed_kv_kernel(
    latent,
    rope,
    cache,
    slots,
    valid_rows,
    CACHE_ROWS: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    LATENT: tl.constexpr,
    LATENT_STRIDE: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    HAS_VALID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    slot = tl.load(slots + row)
    valid = (slot >= 0) & (slot < CACHE_ROWS)
    if HAS_VALID:
        valid = valid & (row < tl.load(valid_rows))
    a = tl.load(latent + row.to(tl.int64) * LATENT_STRIDE + i, mask=valid & (i < LATENT), other=0)
    b = tl.load(rope + row.to(tl.int64) * ROPE_STRIDE + i - LATENT, mask=valid & (i >= LATENT) & (i < WIDTH), other=0)
    base = slot.to(tl.int64) // CACHE_BLOCK * CACHE_STRIDE + slot % CACHE_BLOCK * WIDTH
    tl.store(cache + base + i, tl.where(i < LATENT, a, b), mask=valid & (i < WIDTH))


@triton.jit
def gather_resident_rows_kernel(
    source,
    destination,
    row_ids,
    ROWS,
    SOURCE_ROWS,
    SOURCE_BLOCK: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    DEST_BLOCK: tl.constexpr,
    DEST_STRIDE: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    row = (tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
    src = tl.load(row_ids + row, mask=row < ROWS, other=_INVALID_INDEX).to(tl.int64)
    src = tl.maximum(src, 0)
    src_offset = src // SOURCE_BLOCK * SOURCE_STRIDE + src % SOURCE_BLOCK * ROW_BYTES
    dst_offset = row // DEST_BLOCK * DEST_STRIDE + row % DEST_BLOCK * ROW_BYTES
    byte = tl.program_id(1) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    valid = (row < ROWS) & (src < SOURCE_ROWS)
    mask = valid[:, None] & (byte[None, :] < ROW_BYTES)
    payload = tl.load(source + src_offset[:, None] + byte[None, :], mask=mask, other=0)
    tl.store(destination + dst_offset[:, None] + byte[None, :], payload, mask=mask)


@triton.jit
def _invalidate_recycled_groups(
    tables,
    blocks,
    state_indices,
    STATES: tl.constexpr,
    HOT: tl.constexpr,
    BLOCK_COUNT: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    ITERATIONS: tl.constexpr,
    TILE: tl.constexpr,
):
    table = tl.load(tables + tl.program_id(0)).to(tl.pointer_type(tl.int32))
    state = tl.load(state_indices + tl.program_id(1))
    slot = tl.program_id(2) * TILE + tl.arange(0, TILE)
    valid = (state >= 0) & (state < STATES) & (slot < HOT)
    value = tl.load(table + state * HOT + slot, mask=valid, other=_INVALID_INDEX)
    valid = valid & (value >= 0)
    page = tl.maximum(value, 0) // PAGE_ROWS
    first = tl.load(blocks)
    last = tl.load(blocks + BLOCK_COUNT - 1)
    valid = valid & (page >= first) & (page <= last)
    low = tl.full((TILE,), 0, tl.int32)
    high = tl.full((TILE,), BLOCK_COUNT, tl.int32)
    for _ in tl.static_range(ITERATIONS):
        active = valid & (low < high)
        middle = low + (high - low) // 2
        key = tl.load(blocks + middle, mask=active, other=0)
        advance = active & (key < page)
        high = tl.where(active & ~advance, middle, high)
        low = tl.where(advance, middle + 1, low)
    key = tl.load(blocks + low, mask=valid & (low < BLOCK_COUNT), other=0)
    tl.store(table + state * HOT + slot, _INVALID_INDEX, mask=valid & (low < BLOCK_COUNT) & (key == page))
