# SPDX-License-Identifier: Apache-2.0
"""Batch independent DCP rows while keeping DCP rank accumulation sequential.

The combine kernel's eight D=512 rows use 16 KiB for the FP32 accumulator.
Its rank dimension is streamed, so it does not multiply the accumulator size.
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
    num_tokens,
    num_heads,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FROM_STATS: tl.constexpr,
):
    """Pack BF16 head-scatter rows with one two-token, four-head tile."""
    # Four wire fields carry FP32 LSE; cap feature tiles to bound strided UB use.
    LSE_FIELDS: tl.constexpr = 4
    MAX_FEATURE_TILE: tl.constexpr = 512
    TILE_T: tl.constexpr = 2
    TILE_H: tl.constexpr = 4
    # Bound the feature-strided address and copy working set for wide rows.
    TILE_D: tl.constexpr = min(BLOCK_D, MAX_FEATURE_TILE)
    token_tiles = tl.cdiv(num_tokens, TILE_T)
    head_tiles = tl.cdiv(num_heads, TILE_H)
    token_offsets = tl.arange(0, TILE_T)
    head_offsets = tl.arange(0, TILE_H)
    feature_offsets = tl.arange(0, TILE_D)
    program = tl.program_id(0)
    programs = tl.num_programs(0)
    tiles = token_tiles * head_tiles
    quotient = tiles // programs
    remainder = tiles % programs
    first = program * quotient + tl.minimum(program, remainder)
    count = quotient + (program < remainder).to(tl.int32)
    token_start = first % token_tiles * TILE_T
    head_start = first // token_tiles * TILE_H
    # Consecutive intervals cover each tile once; carry replaces per-tile div/rem.
    for _ in range(count):
        token = (token_start + token_offsets).to(tl.int64)
        head = (head_start + head_offsets).to(tl.int64)
        row_mask = (token[:, None] < num_tokens) & (head[None, :] < num_heads)
        # The caller allocates contiguous [rank, local_head, token, D + 4].
        send_base = (head[None, :] * num_tokens + token[:, None]) * (HEAD_DIM + LSE_FIELDS)
        for feature_start in range(0, HEAD_DIM, TILE_D):
            feature = feature_start + feature_offsets
            output_offsets = (
                token[:, None, None] * output_stride_t
                + head[None, :, None] * output_stride_h
                + feature[None, None, :] * output_stride_d
            )
            mask = row_mask[:, :, None] & (feature[None, None, :] < HEAD_DIM)
            output = tl.load(output_ptr + output_offsets, mask, 0.0)
            tl.store(
                send_ptr + send_base[:, :, None] + feature[None, None, :],
                output,
                mask,
            )
        lse = tl.load(lse_ptr + token[:, None] * lse_stride_t + head[None, :] * lse_stride_h, row_mask, 0.0).to(
            tl.float32
        )
        if FROM_STATS:
            total = tl.load(sum_ptr + token[:, None] * sum_stride_t + head[None, :] * sum_stride_h, row_mask, 1.0).to(
                tl.float32
            )
            lse = lse + tl.log(total)
        exponent_code, significand_hi, significand_mid, significand_lo = _encode_lse_fields(lse)
        tl.store(send_ptr + send_base + HEAD_DIM, exponent_code, row_mask)
        tl.store(send_ptr + send_base + HEAD_DIM + 1, significand_hi, row_mask)
        tl.store(send_ptr + send_base + HEAD_DIM + 2, significand_mid, row_mask)
        tl.store(send_ptr + send_base + HEAD_DIM + 3, significand_lo, row_mask)
        token_start += TILE_T
        carry = token_start >= num_tokens
        token_start = tl.where(carry, 0, token_start)
        head_start += tl.where(carry, TILE_H, 0)


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
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HAS_LOCAL: tl.constexpr = True,
    RANK_UNROLL: tl.constexpr = 1,
):
    # The caller selects BF16, scatter-head, no-return-LSE cases.
    # Each received row therefore ends with four packed LSE values.
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d_offsets = tl.arange(0, BLOCK_D)[None, :]

    for row_start in range(program_idx * BLOCK_ROWS, total_rows, num_programs * BLOCK_ROWS):
        linear_idx = row_start + tl.arange(0, BLOCK_ROWS)[:, None]
        row_mask = linear_idx < total_rows
        token_idx = (linear_idx // num_heads).to(tl.int64)
        head_idx = (linear_idx % num_heads).to(tl.int64)

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
                token_idx * local_output_stride_t + head_idx * local_output_stride_h + d_offsets * local_output_stride_d
            )
            local_output = tl.load(local_output_ptr + local_offsets, mask=d_mask, other=0.0).to(tl.float32)
            merged += tl.where(local_valid, local_output, 0.0) * local_weight
            weight_sum += local_weight
        denominator = tl.where(weight_sum > 0.0, weight_sum, 1.0)
        merged /= denominator
        output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
        tl.store(output_ptr + output_offsets, merged, mask=d_mask)
