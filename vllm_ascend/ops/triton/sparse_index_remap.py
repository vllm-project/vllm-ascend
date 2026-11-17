#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Triton-Ascend implementation of AscendSFADCPImpl._remap_sparse_indices.

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.triton.triton_utils import _resolve_triton_ascend_op, get_element

_sort = _resolve_triton_ascend_op("sort") if HAS_TRITON else None


@triton.jit
def _exact_div_nonnegative(value, DIVISOR: tl.constexpr, SHIFT: tl.constexpr):
    WIDTH: tl.constexpr = value.dtype.primitive_bitwidth
    if DIVISOR >= 1 << (WIDTH - 1):
        quotient = tl.full(value.shape, 0, value.dtype)
    elif DIVISOR == 1 << SHIFT:
        quotient = value >> SHIFT
    else:
        # The high word of value * ceil(2**WIDTH / d) overestimates the
        # quotient by at most one. Half-word products avoid uint64 mulhi.
        HALF: tl.constexpr = WIDTH // 2
        MASK: tl.constexpr = (1 << HALF) - 1
        MAGIC: tl.constexpr = ((1 << WIDTH) + DIVISOR - 1) // DIVISOR
        low = value & MASK
        high = value >> HALF
        product = low * (MAGIC & MASK)
        # Mask the extracted words: A3 may lower an unsigned shift as a
        # signed shift when the product's most significant bit is set.
        middle = high * (MAGIC & MASK) + ((product >> HALF) & MASK)
        carry = low * (MAGIC >> HALF) + (middle & MASK)
        quotient = high * (MAGIC >> HALF) + ((middle >> HALF) & MASK) + ((carry >> HALF) & MASK)
        # -d < q*d - value < d, so the signed difference is exact. Avoid
        # unsigned comparison, which A3 lowers through an unsupported cast.
        difference = (quotient * DIVISOR - value).to(tl.int64 if WIDTH == 64 else tl.int32)
        quotient -= (difference > 0).to(value.dtype)
    return quotient


@triton.jit
def _remap_values(
    idx,
    DCP_SIZE: tl.constexpr,
    RANK,
    INTERLEAVE_SIZE: tl.constexpr,
    DCP_BITS: tl.constexpr,
    INTERLEAVE_BITS: tl.constexpr,
):
    WIDTH: tl.constexpr = idx.dtype.primitive_bitwidth
    if DCP_SIZE == 1 << DCP_BITS and INTERLEAVE_SIZE == 1 << INTERLEAVE_BITS and DCP_BITS + INTERLEAVE_BITS < WIDTH:
        owner = (idx >> INTERLEAVE_BITS) & (DCP_SIZE - 1)
        local = ((idx >> (INTERLEAVE_BITS + DCP_BITS)) << INTERLEAVE_BITS) + (idx & (INTERLEAVE_SIZE - 1))
    else:
        # Negative sentinels do not participate in the unsigned arithmetic.
        value = tl.maximum(idx, 0).to(tl.uint64 if idx.dtype == tl.int64 else tl.uint32)
        block = _exact_div_nonnegative(value, INTERLEAVE_SIZE, INTERLEAVE_BITS)
        cycle = _exact_div_nonnegative(block, DCP_SIZE, DCP_BITS)
        owner = block - cycle * DCP_SIZE
        local = (cycle * INTERLEAVE_SIZE + value - block * INTERLEAVE_SIZE).to(idx.dtype)
    return local, (idx >= 0) & (owner == RANK)


@triton.jit
def _remap_sorted_values(
    idx,
    offsets,
    K: tl.constexpr,
    DCP_BITS: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    INTERLEAVE_SIZE: tl.constexpr,
    COUNT: tl.constexpr = False,
):
    local, valid = _remap_values(idx, DCP_SIZE, RANK, INTERLEAVE_SIZE, DCP_BITS, INTERLEAVE_BITS)
    valid &= offsets < K
    # Unique FP32 position keys preserve top-k order, including duplicate IDs.
    # Only positions are converted numerically; integer payloads stay exact.
    order = _sort(tl.where(valid, offsets.to(tl.float32), float("inf")), descending=False)
    lookup = tl.minimum(order, BLOCK - 1).to(tl.int32)
    selected = tl.gather(local, lookup, 0)
    count = tl.sum(valid.to(tl.int32), 0) if COUNT else 0
    return tl.where(order < float("inf"), selected, -1), count


@triton.jit
def _store_remap_sort(
    result, count, output, write_pos, chunk, K: tl.constexpr, BLOCK: tl.constexpr, NUM_CHUNKS: tl.constexpr
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    if NUM_CHUNKS == 1:
        tl.store(output + row * K + offsets, result, offsets < K)
    else:
        # Previous tiles write at most chunk * BLOCK entries. Initializing
        # this tile cannot overwrite their already-compacted prefix.
        input_offsets = chunk * BLOCK + offsets
        tl.store(output + row * K + input_offsets, -1, input_offsets < K)
        tl.store(output + row * K + write_pos + offsets, result, offsets < count)
    return write_pos + count


@triton.jit
def _remap_sort(
    indices,
    output,
    K: tl.constexpr,
    DCP_BITS: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    INTERLEAVE_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    idx = tl.load(indices + row * K + offsets, offsets < K, -1)
    result, _ = _remap_sorted_values(idx, offsets, K, DCP_BITS, RANK, INTERLEAVE_BITS, BLOCK, DCP_SIZE, INTERLEAVE_SIZE)
    tl.store(output + row * K + offsets, result, offsets < K)


@triton.jit
def _remap_sort_tiles(
    indices,
    index_words,
    output,
    K: tl.constexpr,
    DCP_BITS: tl.constexpr,
    RANK: tl.constexpr,
    INTERLEAVE_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    INTERLEAVE_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NARROW_INT64: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    write_pos = 0
    for chunk in range(NUM_CHUNKS):
        input_offsets = chunk * BLOCK + offsets
        if NARROW_INT64:
            # Inspect the two INT32 words without scalarized INT64 arithmetic.
            # Negative inputs are invalid regardless of their remaining bits.
            words = index_words + 2 * (row * K + input_offsets)
            low = tl.load(words, input_offsets < K, 0).to(tl.uint32)
            high = tl.load(words + 1, input_offsets < K, -1).to(tl.uint32)
            negative = (high >> 31) != 0
            wide = ~negative & ((high != 0) | ((low >> 31) != 0))
            if tl.sum(wide.to(tl.int32), 0) == 0:
                narrow_idx = tl.where(negative, -1, low.to(tl.int32))
                narrow_result, narrow_count = _remap_sorted_values(
                    narrow_idx,
                    offsets,
                    K,
                    DCP_BITS,
                    RANK,
                    INTERLEAVE_BITS,
                    BLOCK,
                    DCP_SIZE,
                    INTERLEAVE_SIZE,
                    NUM_CHUNKS > 1,
                )
                write_pos = _store_remap_sort(
                    narrow_result, narrow_count, output, write_pos, chunk, K, BLOCK, NUM_CHUNKS
                )
            else:
                idx = tl.load(indices + row * K + input_offsets, input_offsets < K, -1)
                result, count = _remap_sorted_values(
                    idx,
                    offsets,
                    K,
                    DCP_BITS,
                    RANK,
                    INTERLEAVE_BITS,
                    BLOCK,
                    DCP_SIZE,
                    INTERLEAVE_SIZE,
                    NUM_CHUNKS > 1,
                )
                write_pos = _store_remap_sort(result, count, output, write_pos, chunk, K, BLOCK, NUM_CHUNKS)
        else:
            idx = tl.load(indices + row * K + input_offsets, input_offsets < K, -1)
            result, count = _remap_sorted_values(
                idx,
                offsets,
                K,
                DCP_BITS,
                RANK,
                INTERLEAVE_BITS,
                BLOCK,
                DCP_SIZE,
                INTERLEAVE_SIZE,
                NUM_CHUNKS > 1,
            )
            write_pos = _store_remap_sort(result, count, output, write_pos, chunk, K, BLOCK, NUM_CHUNKS)


@triton.jit(do_not_specialize=["topk_count", "dcp_rank"])
def remap_sparse_indices_fused_kernel(
    indices_ptr,  # [rows, topk_count] int32, replicated-view topk indices
    chunk_out_ptr,  # [rows * num_chunks, BLOCK] int32 (output)
    chunk_count_ptr,  # [rows, num_chunks] int32 (output)
    out_ptr,  # [rows, topk_count] int32 (output): tail pre-filled with -1 here
    topk_count,
    dcp_rank,
    DCP_BITS: tl.constexpr,
    INTERLEAVE_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    INTERLEAVE_SIZE: tl.constexpr,
):
    """Fused remap + per-chunk compaction + output tail pre-fill.

    One program per (chunk, row). The vectorized remap computes the
    DCP-local index and the owner-validity mask for this chunk; then the
    chunk's valid entries are compacted to the front of its ``chunk_out``
    slot in order via scalar ``get_element`` operations, the chunk's valid
    count is written to ``chunk_count``, and the chunk's region of ``out``
    is pre-filled with -1 (the gather kernel later overwrites the compacted
    front).
    """
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < topk_count

    idx = tl.load(indices_ptr + row * topk_count + offsets, mask=in_bounds, other=-1)
    remapped, valid = _remap_values(idx, DCP_SIZE, dcp_rank, INTERLEAVE_SIZE, DCP_BITS, INTERLEAVE_BITS)
    valid &= in_bounds

    valid_i32 = valid.to(tl.int32)
    chunk_out = chunk_out_ptr + (row * NUM_CHUNKS + chunk) * BLOCK
    # Pre-fill this chunk's output region with -1 first; the serial stores
    # below overwrite the compacted front in program order.
    tl.store(out_ptr + row * topk_count + offsets, -1, mask=in_bounds)
    cnt = 0
    for i in range(BLOCK):
        if get_element(valid_i32, (i,)) == 1:
            tl.store(chunk_out + cnt, get_element(remapped, (i,)))
            cnt += 1
    tl.store(chunk_count_ptr + row * NUM_CHUNKS + chunk, cnt)


@triton.jit(do_not_specialize=["topk_count"])
def remap_sparse_indices_compact_gather_kernel(
    chunk_out_ptr,  # [rows * num_chunks, BLOCK] int32
    chunk_count_ptr,  # [rows, num_chunks] int32
    out_ptr,  # [rows, topk_count] int32 (output): tail pre-filled with -1
    topk_count,
    BLOCK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """Gather chunks to the row front in chunk order.

    One program per row. The per-chunk copy is vectorized (masked load +
    masked contiguous store); serial iteration count is num_chunks only.
    """
    row = tl.program_id(0)
    write_pos = 0
    for chunk in range(NUM_CHUNKS):
        cnt = tl.load(chunk_count_ptr + row * NUM_CHUNKS + chunk)
        offs = tl.arange(0, BLOCK)
        m = offs < cnt
        x = tl.load(chunk_out_ptr + (row * NUM_CHUNKS + chunk) * BLOCK + offs, mask=m, other=0)
        tl.store(out_ptr + row * topk_count + write_pos + offs, x, mask=m)
        write_pos += cnt


def remap_sparse_indices_triton(
    topk_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
) -> torch.Tensor:
    """Triton replacement for the torch implementation in
    ``AscendSFADCPImpl._remap_sparse_indices``.

    Args:
        topk_indices: [..., topk_count] int32 or int64 replicated-view topk indices.
        dcp_size: Number of DCP ranks.
        dcp_rank: This rank's index in the DCP group.
        interleave_size: KV cache interleave size of the replicated view.

    Returns:
        Tensor with the same shape and dtype as ``topk_indices``: indices
        owned by this rank remapped to DCP-local KV positions and compacted
        to the front of the last dim in top-k order, padded with -1.

    Design notes (verified on triton-ascend 3.2.0 / CANN 9.1):
    - Do not use a ``tl.cumsum`` prefix-sum compaction: on a2 it produces
      nondeterministic garbage values, and on a3 it traps the vector core
      under repeated launches (single-shot results are correct). ``tl.cumsum``
      alone carries a large per-program fixed cost that makes row-parallel
      layouts very slow.
    - Do not fill the output tail inside the gather kernel after its copy
      loop with a scalar-comparison mask: it writes wrong values. The tail
      is pre-filled in the fused kernel with a plain ``in_bounds`` mask
      instead, so this path has only two device ops (fused + gather).
    """
    if topk_indices.numel() == 0:
        return topk_indices
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("DCP sparse indices must be INT32 or INT64")
    if dcp_size <= 0 or interleave_size <= 0 or not 0 <= dcp_rank < dcp_size:
        raise ValueError("DCP size/interleave must be positive and rank must belong to the group")
    orig_shape = topk_indices.shape
    if not topk_indices.is_contiguous():
        topk_indices = topk_indices.contiguous()
    indices = topk_indices
    topk_count = indices.shape[-1]
    rows = indices.numel() // topk_count
    # The torch implementation operates per-row on the last dim, so arbitrary
    # leading dims (e.g. [dcp_size, 1, topk_count] from the DCP all_gather) can
    # be flattened to a 2D view and restored afterwards.
    indices = indices.view(rows, topk_count)

    # INT64 K4096 exceeds 192 KiB UB on A3. Keep one dtype/shape policy
    # across devices; larger rows use bounded chunk buffers below.
    sort_limit = 4096 if indices.dtype == torch.int32 else 2048
    if topk_count <= sort_limit:
        out = torch.empty_like(indices)
        power_of_two_layout = not (dcp_size & (dcp_size - 1) or interleave_size & (interleave_size - 1))
        # A3 cannot reliably lower the INT64 width-selection loop for a
        # single tile. Small rows use the direct row kernel for every layout.
        if topk_count <= 1024 or ((power_of_two_layout or dcp_size == 1) and topk_count <= 2048):
            _remap_sort[(rows,)](
                indices,
                out,
                topk_count,
                dcp_size.bit_length() - 1,
                dcp_rank,
                interleave_size.bit_length() - 1,
                next_power_of_2(topk_count),
                dcp_size,
                interleave_size,
            )
            return out.view(orig_shape)
        narrow_int64 = (
            indices.dtype == torch.int64
            and dcp_size < 1 << 31
            and interleave_size < 1 << 31
            and not power_of_two_layout
        )
        # A3 needs 208 KiB for INT32 K4096 and 216 KiB for the INT64 K2048
        # width-selection branches. Bound tiles to its 192 KiB UB on all devices.
        block = min(2048 if indices.dtype == torch.int32 else 1024, next_power_of_2(topk_count))
        num_chunks = triton.cdiv(topk_count, block)
        _remap_sort_tiles[(rows,)](
            indices,
            indices.view(torch.int32) if narrow_int64 else indices,
            out,
            topk_count,
            dcp_size.bit_length() - 1,
            dcp_rank,
            interleave_size.bit_length() - 1,
            block,
            dcp_size,
            interleave_size,
            num_chunks,
            narrow_int64,
        )
        out = out.view(orig_shape)
        return out

    block = min(128, next_power_of_2(topk_count))
    num_chunks = triton.cdiv(topk_count, block)
    chunk_out = torch.empty((rows * num_chunks, block), dtype=indices.dtype, device=indices.device)
    chunk_count = torch.empty((rows, num_chunks), dtype=torch.int32, device=indices.device)
    out = torch.empty_like(indices)

    remap_sparse_indices_fused_kernel[(num_chunks, rows)](
        indices,
        chunk_out,
        chunk_count,
        out,
        topk_count,
        dcp_rank,
        DCP_BITS=dcp_size.bit_length() - 1,
        INTERLEAVE_BITS=interleave_size.bit_length() - 1,
        BLOCK=block,
        NUM_CHUNKS=num_chunks,
        DCP_SIZE=dcp_size,
        INTERLEAVE_SIZE=interleave_size,
        multibuffer=False,
    )
    remap_sparse_indices_compact_gather_kernel[(rows,)](
        chunk_out,
        chunk_count,
        out,
        topk_count,
        BLOCK=block,
        NUM_CHUNKS=num_chunks,
        multibuffer=False,
    )
    out = out.view(orig_shape)
    return out
