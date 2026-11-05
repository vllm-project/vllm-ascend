# SPDX-License-Identifier: Apache-2.0
"""Batch independent DCP rows while keeping DCP rank accumulation sequential.

Eight D=512 rows use 16 KiB for the FP32 accumulator. The rank dimension
is streamed, so it does not multiply the accumulator size.
"""

from vllm.triton_utils import tl, triton

MAX_UNROLLED_COMBINE_ELEMENTS = 36 * 1024


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
def _encode_dcp_lse_kernel(
    maxima,
    sums,
    encoded,
    MAX_T: tl.constexpr,
    MAX_H: tl.constexpr,
    SUM_T: tl.constexpr,
    SUM_H: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    FROM_STATS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    blocks = tl.cdiv(ROWS, BLOCK)
    per_core = tl.cdiv(blocks, tl.num_programs(0))
    begin = tl.program_id(0) * per_core
    for tile in range(begin, tl.minimum(begin + per_core, blocks)):
        row = tile * BLOCK + tl.arange(0, BLOCK)
        token = row // HEADS
        head = row - token * HEADS
        lse = tl.load(maxima + token * MAX_T + head * MAX_H, row < ROWS, 0).to(tl.float32)
        if FROM_STATS:
            total = tl.load(sums + token * SUM_T + head * SUM_H, row < ROWS, 1).to(tl.float32)
            lse += tl.log(total)
        code, hi, mid, lo = _encode_lse_fields(lse)
        values = tl.reshape(tl.join(tl.join(code, mid), tl.join(hi, lo)), (4 * BLOCK,))
        offsets = tile * (4 * BLOCK) + tl.arange(0, 4 * BLOCK)
        tl.store(encoded + offsets, values, offsets < 4 * ROWS)


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
        tl.store(send_ptr + send_base + d_offsets * send_stride_d, output, mask=d_mask)

        lse = tl.load(lse_ptr + token_idx * lse_stride_t + head_idx * lse_stride_h, mask=row_mask, other=0.0).to(
            tl.float32
        )
        if FROM_STATS:
            total = tl.load(sum_ptr + token_idx * sum_stride_t + head_idx * sum_stride_h, mask=row_mask, other=1.0).to(
                tl.float32
            )
            lse = lse + tl.log(total)
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
def _fused_dcp_lse_combine_batched_kernel(
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
    BLOCK_RANKS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HAS_LOCAL: tl.constexpr = True,
    COALESCED_LSE: tl.constexpr = False,
    RANK_UNROLL: tl.constexpr = 1,
):
    # The caller selects BF16, scatter-head, no-return-LSE cases.
    # Each received row therefore ends with four packed LSE values.
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d_offsets = tl.arange(0, BLOCK_D)[None, :]
    ranks = tl.arange(0, BLOCK_RANKS)[:, None]

    for row_start in range(program_idx * BLOCK_ROWS, total_rows, num_programs * BLOCK_ROWS):
        linear_idx = row_start + tl.arange(0, BLOCK_ROWS)[:, None]
        row_mask = linear_idx < total_rows
        token_idx = (linear_idx // num_heads).to(tl.int64)
        head_idx = (linear_idx % num_heads).to(tl.int64)

        if COALESCED_LSE:
            # Coalesced four-slot loads use the upstream streaming-rank
            # formulation; only one output accumulator stays live.
            lse_max = tl.full((BLOCK_ROWS, 1), -float("inf"), tl.float32)
            if HAS_LOCAL:
                local_lse = tl.load(
                    local_lse_ptr + token_idx * local_lse_stride_t + head_idx * local_lse_stride_h,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                local_valid = tl.abs(local_lse) < float("inf")
                lse_max = tl.where(local_valid, local_lse, -float("inf"))
            for rank_idx in tl.range(0, DCP_SIZE, loop_unroll_factor=RANK_UNROLL):
                recv_base = (
                    rank_idx * recv_stride_rank + head_idx * recv_stride_scatter + token_idx * recv_stride_replicated
                )
                packed = tl.load(
                    recv_ptr + recv_base + (head_dim + tl.arange(0, 4)[None, :]) * recv_stride_d,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                even, odd = tl.split(tl.reshape(packed, (BLOCK_ROWS, 2, 2)))
                code, mid = tl.split(even)
                hi, lo = tl.split(odd)
                exponent_code = code[:, None]
                significand_hi = hi[:, None]
                significand_mid = mid[:, None]
                significand_lo = lo[:, None]
                packed_valid = exponent_code != 0.0
                sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
                exponent_magnitude = tl.where(exponent_code < 0.0, -exponent_code, exponent_code)
                significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
                # Reconstruct IEEE FP32 bits directly, including subnormals.
                magnitude = exponent_magnitude.to(tl.int32)
                lse_bits = (tl.maximum(magnitude - 2, 0) << 23) + significand.to(tl.int32)
                lse_bits = tl.where(significand != 0.0, lse_bits, 0)
                lse = sign * lse_bits.to(tl.float32, bitcast=True)
                valid_lse = packed_valid & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
                lse_max = tl.maximum(lse_max, tl.where(valid_lse, lse, -float("inf")))
            any_valid_lse = lse_max != -float("inf")
            safe_lse_max = tl.where(any_valid_lse, lse_max, 0.0)
            weight_sum = tl.zeros((BLOCK_ROWS, 1), tl.float32)
            merged = tl.zeros([BLOCK_ROWS, BLOCK_D], dtype=tl.float32)
            d_mask = (d_offsets < head_dim) & row_mask
            for rank_idx in tl.range(0, DCP_SIZE, loop_unroll_factor=RANK_UNROLL):
                recv_base = (
                    rank_idx * recv_stride_rank + head_idx * recv_stride_scatter + token_idx * recv_stride_replicated
                )
                packed = tl.load(
                    recv_ptr + recv_base + (head_dim + tl.arange(0, 4)[None, :]) * recv_stride_d,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                even, odd = tl.split(tl.reshape(packed, (BLOCK_ROWS, 2, 2)))
                code, mid = tl.split(even)
                hi, lo = tl.split(odd)
                exponent_code = code[:, None]
                significand_hi = hi[:, None]
                significand_mid = mid[:, None]
                significand_lo = lo[:, None]
                packed_valid = exponent_code != 0.0
                sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
                exponent_magnitude = tl.where(exponent_code < 0.0, -exponent_code, exponent_code)
                significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
                # Reconstruct IEEE FP32 bits directly, including subnormals.
                magnitude = exponent_magnitude.to(tl.int32)
                lse_bits = (tl.maximum(magnitude - 2, 0) << 23) + significand.to(tl.int32)
                lse_bits = tl.where(significand != 0.0, lse_bits, 0)
                lse = sign * lse_bits.to(tl.float32, bitcast=True)
                valid_lse = packed_valid & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
                weight = tl.where(valid_lse, tl.exp(lse - safe_lse_max), 0.0)
                partial_output = tl.load(recv_ptr + recv_base + d_offsets * recv_stride_d, mask=d_mask, other=0.0).to(
                    tl.float32
                )
                partial_output = tl.where(valid_lse, partial_output, 0.0)
                merged += partial_output * weight
                weight_sum += weight
            if HAS_LOCAL:
                local_weight = tl.where(local_valid, tl.exp(local_lse - safe_lse_max), 0.0)
                local_offsets = (
                    token_idx * local_output_stride_t
                    + head_idx * local_output_stride_h
                    + d_offsets * local_output_stride_d
                )
                local_output = tl.load(local_output_ptr + local_offsets, mask=d_mask, other=0.0).to(tl.float32)
                merged += tl.where(local_valid, local_output, 0.0) * local_weight
                weight_sum += local_weight
            denominator = tl.where(weight_sum > 0.0, weight_sum, 1.0)
            merged /= denominator
            output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
            tl.store(output_ptr + output_offsets, merged, mask=d_mask)
        else:
            # Decode each rank's four BF16 LSE fields once. Only the small
            # [BLOCK_RANKS, BLOCK_ROWS] LSE tile stays live across the output loop.
            recv_lse_base = (
                ranks * recv_stride_rank
                + tl.trans(head_idx) * recv_stride_scatter
                + tl.trans(token_idx) * recv_stride_replicated
                + head_dim * recv_stride_d
            )
            lse_mask = (ranks < DCP_SIZE) & tl.trans(row_mask)
            exponent_code = tl.load(recv_ptr + recv_lse_base, lse_mask, 0.0).to(tl.float32)
            significand_hi = tl.load(recv_ptr + recv_lse_base + recv_stride_d, lse_mask, 0.0).to(tl.float32)
            significand_mid = tl.load(recv_ptr + recv_lse_base + 2 * recv_stride_d, lse_mask, 0.0).to(tl.float32)
            significand_lo = tl.load(recv_ptr + recv_lse_base + 3 * recv_stride_d, lse_mask, 0.0).to(tl.float32)
            sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
            exponent = tl.abs(exponent_code) - 128.0
            significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
            decoded_lses = sign * significand * tl.exp2(exponent - 23.0)
            valid_lses = lse_mask & (exponent_code != 0.0) & (tl.abs(decoded_lses) < float("inf"))
            rank_lses = tl.where(valid_lses, decoded_lses, -float("inf"))
            lse_max = tl.max(rank_lses, 0)[:, None]
            if HAS_LOCAL:
                local_lse = tl.load(
                    local_lse_ptr + token_idx * local_lse_stride_t + head_idx * local_lse_stride_h,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                local_valid = (local_lse == local_lse) & (local_lse != float("inf")) & (local_lse != -float("inf"))
                lse_max = tl.maximum(lse_max, tl.where(local_valid, local_lse, -float("inf")))

            any_valid_lse = lse_max != -float("inf")
            safe_lse_max = tl.where(any_valid_lse, lse_max, 0.0)
            # Normalize the small rank/row tile before applying it to D features.
            # This avoids dividing the much wider output accumulator afterwards.
            rank_weights = tl.exp(rank_lses - tl.trans(safe_lse_max))
            weight_sum = tl.sum(rank_weights, 0)[:, None]
            if HAS_LOCAL:
                local_weight = tl.where(local_valid, tl.exp(local_lse - safe_lse_max), 0.0)
                weight_sum += local_weight
            denominator = tl.where(weight_sum > 0.0, weight_sum, 1.0)
            rank_weights /= tl.trans(denominator)
            if HAS_LOCAL:
                local_weight /= denominator
            merged = tl.zeros([BLOCK_ROWS, BLOCK_D], dtype=tl.float32)
            d_mask = (d_offsets < head_dim) & row_mask
            for rank_idx in tl.range(0, DCP_SIZE, loop_unroll_factor=DCP_SIZE if HAS_LOCAL else 1):
                recv_base = (
                    rank_idx * recv_stride_rank + head_idx * recv_stride_scatter + token_idx * recv_stride_replicated
                )
                rank_index = tl.full((1, BLOCK_ROWS), rank_idx, tl.int32)
                lse = tl.trans(tl.gather(rank_lses, rank_index, 0))
                valid_lse = tl.abs(lse) < float("inf")
                weight = tl.trans(tl.gather(rank_weights, rank_index, 0))
                partial_output = tl.load(
                    recv_ptr + recv_base + d_offsets * recv_stride_d,
                    mask=d_mask,
                    other=0.0,
                ).to(tl.float32)
                # Select before multiplication: multiplying a zero weight by a
                # NaN from an invalid rank would otherwise contaminate the result.
                partial_output = tl.where(valid_lse, partial_output, 0.0)
                merged += partial_output * weight

            if HAS_LOCAL:
                local_offsets = (
                    token_idx * local_output_stride_t
                    + head_idx * local_output_stride_h
                    + d_offsets * local_output_stride_d
                )
                local_output = tl.load(local_output_ptr + local_offsets, mask=d_mask, other=0.0).to(tl.float32)
                merged += tl.where(local_valid, local_output, 0.0) * local_weight
            output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
            tl.store(output_ptr + output_offsets, merged, mask=d_mask)
