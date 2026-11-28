# SPDX-License-Identifier: Apache-2.0
"""Batched DCP packing and row-weighted combine kernels."""

from vllm.triton_utils import tl, triton

from vllm_ascend.device.hardware_profile import DCP_HEAD_SCATTER_MULTI_ROW, DCP_HEAD_SCATTER_SINGLE_ROW
from vllm_ascend.ops.triton.triton_utils import get_element

_SINGLE_ROW = tl.constexpr(DCP_HEAD_SCATTER_SINGLE_ROW)
_MULTI_ROW = tl.constexpr(DCP_HEAD_SCATTER_MULTI_ROW)


@triton.jit
def _decode_lse_significand(hi, mid, lo):
    BYTE_RADIX: tl.constexpr = 256.0
    HIGH_BYTE_SCALE: tl.constexpr = BYTE_RADIX * BYTE_RADIX
    return hi * HIGH_BYTE_SCALE + mid * BYTE_RADIX + lo


@triton.jit
def _encode_lse_fields(lse):
    """Encode the existing four-field wire format without transcendental ops."""
    bits = lse.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7FFFFF
    finite = exponent.to(tl.float32) < 255.0
    # Wire exponent bias is 128. Subnormals share code 2 with the smallest
    # normal, but omit its implicit significand bit. Zero keeps code 128.
    significand = fraction | tl.where(exponent == 0, 0, 0x800000)
    significand = tl.where(finite, significand, 0)
    code = tl.maximum(exponent.to(tl.float32), 1.0) + 1.0
    code = tl.where((bits >> 31) == 0, code, -code)
    code = tl.where((bits & 0x7FFFFFFF) == 0, 128.0, code)
    code = tl.where(finite, code, 0.0)
    return (
        code,
        ((significand >> 16) & 255).to(tl.float32),
        ((significand >> 8) & 255).to(tl.float32),
        (significand & 255).to(tl.float32),
    )


@triton.jit
def _pack_dcp_output_lse_batched_kernel(
    output_ptr,
    lse_ptr,
    sum_ptr,
    send_ptr,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    lse_stride_t,
    lse_stride_h,
    sum_stride_t,
    sum_stride_h,
    send_stride_rank,
    send_stride_scatter,
    send_stride_replicated,
    send_stride_d,
    local_scatter_size,
    head_dim,
    num_heads,
    total_rows,
    SCATTER_TOKENS: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    FROM_STATS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    OUTPUT_HEAD_MAJOR: tl.constexpr = False,
    output_mask_ptr=None,
    HAS_OUTPUT_MASK: tl.constexpr = False,
):
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d_offsets = tl.arange(0, BLOCK_D)[None, :]

    if OUTPUT_HEAD_MAJOR:
        num_tokens = total_rows // num_heads
        token_tiles = tl.cdiv(num_tokens, BLOCK_ROWS)
        tiles = token_tiles * num_heads
    else:
        tiles = tl.cdiv(total_rows, BLOCK_ROWS)
    for tile in range(program_idx, tiles, num_programs):
        if OUTPUT_HEAD_MAJOR:
            # Keep each tile inside one head so both copies have affine strides.
            head_idx = (tile // token_tiles).to(tl.int64)
            token_idx = (tile % token_tiles) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
            row_mask = token_idx < num_tokens
            token_idx = token_idx.to(tl.int64)
        else:
            linear_idx = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
            row_mask = linear_idx < total_rows
            token_idx = (linear_idx // num_heads).to(tl.int64)
            head_idx = (linear_idx % num_heads).to(tl.int64)

        if OUTPUT_HEAD_MAJOR:
            send_base = (head_idx * num_tokens + token_idx) * send_stride_replicated
        else:
            if SCATTER_TOKENS:
                rank_idx = token_idx // local_scatter_size
                scatter_idx = token_idx % local_scatter_size
                replicated_idx = head_idx
            else:
                rank_idx = head_idx // local_scatter_size
                scatter_idx = head_idx % local_scatter_size
                replicated_idx = token_idx
            send_base = (
                rank_idx * send_stride_rank
                + scatter_idx * send_stride_scatter
                + replicated_idx * send_stride_replicated
            )
        output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
        d_mask = (d_offsets < head_dim) & row_mask
        output = tl.load(output_ptr + output_offsets, mask=d_mask, other=0.0)
        if HAS_OUTPUT_MASK:
            invalid = tl.load(output_mask_ptr + token_idx, mask=row_mask, other=True).to(tl.int1)
            output = tl.where(invalid, 0.0, output)
        tl.store(send_ptr + send_base + d_offsets * send_stride_d, output, mask=d_mask)

        lse = tl.load(lse_ptr + token_idx * lse_stride_t + head_idx * lse_stride_h, mask=row_mask, other=0.0).to(
            tl.float32
        )
        if FROM_STATS:
            total = tl.load(sum_ptr + token_idx * sum_stride_t + head_idx * sum_stride_h, mask=row_mask, other=1.0).to(
                tl.float32
            )
            lse = lse + tl.log(total)
        if HAS_OUTPUT_MASK:
            lse = tl.where(invalid, -float("inf"), lse)
        if LSE_PACK_DIM == 1:
            tl.store(send_ptr + send_base + head_dim * send_stride_d, lse.to(send_ptr.dtype.element_ty), mask=row_mask)
        else:
            exponent_code, significand_hi, significand_mid, significand_lo = _encode_lse_fields(lse)
            tl.store(
                send_ptr + send_base + head_dim * send_stride_d,
                exponent_code.to(send_ptr.dtype.element_ty),
                mask=row_mask,
            )
            tl.store(
                send_ptr + send_base + (head_dim + 1) * send_stride_d,
                significand_hi.to(send_ptr.dtype.element_ty),
                mask=row_mask,
            )
            tl.store(
                send_ptr + send_base + (head_dim + 2) * send_stride_d,
                significand_mid.to(send_ptr.dtype.element_ty),
                mask=row_mask,
            )
            tl.store(
                send_ptr + send_base + (head_dim + 3) * send_stride_d,
                significand_lo.to(send_ptr.dtype.element_ty),
                mask=row_mask,
            )


@triton.jit
def _accumulate_dcp_output(ptr, mask, valid, weight, numerator):
    values = tl.load(ptr, mask, 0.0).to(tl.float32)
    return numerator + tl.where(valid, values, 0.0) * weight


@triton.jit
def _fused_dcp_lse_combine_head_scatter_kernel(
    recv_ptr,
    output_ptr,
    local_output_ptr,
    local_lse_ptr,
    local_output_stride_t,
    local_output_stride_h,
    local_output_stride_d,
    local_lse_stride_t,
    local_lse_stride_h,
    recv_stride_rank,
    recv_stride_scatter,
    recv_stride_replicated,
    recv_stride_d,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    head_dim,
    num_heads,
    total_rows,
    DCP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MODE: tl.constexpr,
    HAS_LOCAL: tl.constexpr = True,
    local_output_mask_ptr=None,
    HAS_LOCAL_OUTPUT_MASK: tl.constexpr = False,
):
    # A3 avoids scalarized multirow loads. A5 streams eight rows per tile for higher memory throughput.
    BLOCK_ROWS: tl.constexpr = 1 if MODE == _SINGLE_ROW else 8
    BLOCK_RANKS: tl.constexpr = triton.next_power_of_2(DCP_SIZE + HAS_LOCAL)
    ranks = tl.arange(0, BLOCK_RANKS)
    rank_offsets = tl.minimum(ranks, DCP_SIZE - 1) * recv_stride_rank
    features = tl.arange(0, BLOCK_D)
    program = tl.program_id(0)
    programs = tl.num_programs(0)
    tile_count = tl.cdiv(total_rows, BLOCK_ROWS)
    quotient = tile_count // programs
    remainder = tile_count % programs
    first = program * quotient + tl.minimum(program, remainder)
    stop = first + quotient + (program < remainder).to(tl.int32)
    for tile in range(first, stop):
        if MODE == _SINGLE_ROW:
            rows = tile
        else:
            rows = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        tokens = (rows // num_heads).to(tl.int64)
        heads = (rows % num_heads).to(tl.int64)
        bases = heads * recv_stride_scatter + tokens * recv_stride_replicated
        if MODE == _SINGLE_ROW:
            addresses = rank_offsets + heads * recv_stride_scatter + tokens * recv_stride_replicated
            packed = tl.load(
                recv_ptr + addresses[:, None] + (head_dim + tl.arange(0, 4)[None, :]) * recv_stride_d,
                (ranks < DCP_SIZE)[:, None],
                0.0,
            ).to(tl.float32)
            even, odd = tl.split(tl.reshape(packed, (BLOCK_RANKS, 2, 2)))
            code, significand_mid = tl.split(even)
            significand_hi, significand_lo = tl.split(odd)
            rank_ids = ranks
            feature_ids = features
            mask = features < head_dim
            feature_bases = bases
            numerator = tl.zeros((BLOCK_D,), tl.float32)
        else:
            rank_ids = ranks[None, :]
            feature_ids = features[None, :]
            row_mask = rows < total_rows
            mask = row_mask[:, None] & (feature_ids < head_dim)
            feature_bases = bases[:, None]
            addresses = feature_bases + rank_offsets[None, :] + head_dim * recv_stride_d
            field_mask = row_mask[:, None] & (rank_ids < DCP_SIZE)
            code = tl.load(recv_ptr + addresses, field_mask, 0.0).to(tl.float32)
            significand_hi = tl.load(recv_ptr + addresses + recv_stride_d, field_mask, 0.0).to(tl.float32)
            significand_mid = tl.load(recv_ptr + addresses + 2 * recv_stride_d, field_mask, 0.0).to(tl.float32)
            significand_lo = tl.load(recv_ptr + addresses + 3 * recv_stride_d, field_mask, 0.0).to(tl.float32)
            numerator = tl.zeros((BLOCK_ROWS, BLOCK_D), tl.float32)

        significand = _decode_lse_significand(significand_hi, significand_mid, significand_lo)
        bits = (tl.maximum(tl.abs(code).to(tl.int32) - 2, 0) << 23) + significand.to(tl.int32)
        magnitude = tl.where(significand != 0.0, bits, 0).to(tl.float32, bitcast=True)
        lses = tl.where(code < 0.0, -magnitude, magnitude)
        valid = (rank_ids < DCP_SIZE) & (code != 0.0) & (lses == lses) & (tl.abs(lses) != float("inf"))
        if HAS_LOCAL:
            lse_ptr = local_lse_ptr + tokens * local_lse_stride_t + heads * local_lse_stride_h
            if MODE == _SINGLE_ROW:
                local_lse = tl.load(lse_ptr).to(tl.float32)
            else:
                local_lse = tl.load(lse_ptr, row_mask, 0.0).to(tl.float32)[:, None]
            local_valid = (local_lse == local_lse) & (tl.abs(local_lse) != float("inf"))
            if HAS_LOCAL_OUTPUT_MASK:
                if MODE == _SINGLE_ROW:
                    invalid = tl.load(local_output_mask_ptr + tokens).to(tl.int1)
                else:
                    invalid = tl.load(local_output_mask_ptr + tokens, row_mask, other=True).to(tl.int1)[:, None]
                local_valid = local_valid & ~invalid
            lses = tl.where(rank_ids == DCP_SIZE, local_lse, lses)
            valid = tl.where(rank_ids == DCP_SIZE, local_valid, valid)
        maximum = tl.max(tl.where(valid, lses, -float("inf")), -1)
        maximum = tl.where(maximum != -float("inf"), maximum, 0.0)
        if MODE == _MULTI_ROW:
            maximum = maximum[:, None]
        weights = tl.where(valid, tl.exp(lses - maximum), 0.0)
        denominator = tl.sum(weights, -1)
        if MODE == _SINGLE_ROW:
            for rank in tl.static_range(DCP_SIZE):
                weight = get_element(weights, (rank,))
                finite = get_element(valid, (rank,))
                offsets = rank * recv_stride_rank + feature_bases + feature_ids * recv_stride_d
                numerator = _accumulate_dcp_output(recv_ptr + offsets, mask, finite, weight, numerator)
        else:
            for rank in tl.range(DCP_SIZE, loop_unroll_factor=1):
                index = tl.full((BLOCK_ROWS, 1), rank, tl.int32)
                weight = tl.gather(weights, index, axis=1)
                finite = tl.gather(valid.to(tl.int32), index, axis=1) != 0
                offsets = rank * recv_stride_rank + feature_bases + feature_ids * recv_stride_d
                numerator = _accumulate_dcp_output(recv_ptr + offsets, mask, finite, weight, numerator)
        if HAS_LOCAL:
            local_bases = tokens * local_output_stride_t + heads * local_output_stride_h
            if MODE == _SINGLE_ROW:
                weight = get_element(weights, (DCP_SIZE,))
            else:
                weight = tl.gather(weights, tl.full((BLOCK_ROWS, 1), DCP_SIZE, tl.int32), axis=1)
                local_bases = local_bases[:, None]
            offsets = local_bases + feature_ids * local_output_stride_d
            numerator = _accumulate_dcp_output(local_output_ptr + offsets, mask, local_valid, weight, numerator)
        scale = tl.where(denominator > 0.0, denominator, 1.0)
        output_bases = tokens * output_stride_t + heads * output_stride_h
        if MODE == _MULTI_ROW:
            scale = scale[:, None]
            output_bases = output_bases[:, None]
        tl.store(output_ptr + output_bases + feature_ids * output_stride_d, numerator / scale, mask)
