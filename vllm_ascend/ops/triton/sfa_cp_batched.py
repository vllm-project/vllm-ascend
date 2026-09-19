# SPDX-License-Identifier: Apache-2.0
"""Batch independent SFA rows while keeping DCP rank accumulation sequential.

Eight D=512 rows use 16 KiB for the FP32 accumulator. The rank dimension
is streamed, so it does not multiply the accumulator size.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def _pack_sfa_dcp_output_lse_batched_kernel(
    output_ptr,
    lse_ptr,
    send_ptr,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    lse_stride_t,
    lse_stride_h,
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
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d_offsets = tl.arange(0, BLOCK_D)[None, :]

    for row_start in range(program_idx * BLOCK_ROWS, total_rows, num_programs * BLOCK_ROWS):
        linear_idx = row_start + tl.arange(0, BLOCK_ROWS)[:, None]
        row_mask = linear_idx < total_rows
        token_idx = (linear_idx // num_heads).to(tl.int64)
        head_idx = (linear_idx % num_heads).to(tl.int64)

        if SCATTER_TOKENS:
            rank_idx = token_idx // local_scatter_size
            scatter_idx = token_idx % local_scatter_size
            replicated_idx = head_idx
        else:
            rank_idx = head_idx // local_scatter_size
            scatter_idx = head_idx % local_scatter_size
            replicated_idx = token_idx

        send_base = (
            rank_idx * send_stride_rank + scatter_idx * send_stride_scatter + replicated_idx * send_stride_replicated
        )
        output_offsets = token_idx * output_stride_t + head_idx * output_stride_h + d_offsets * output_stride_d
        d_mask = (d_offsets < head_dim) & row_mask
        output = tl.load(output_ptr + output_offsets, mask=d_mask, other=0.0)
        tl.store(send_ptr + send_base + d_offsets * send_stride_d, output, mask=d_mask)

        lse = tl.load(lse_ptr + token_idx * lse_stride_t + head_idx * lse_stride_h, mask=row_mask, other=0.0).to(
            tl.float32
        )
        if LSE_PACK_DIM == 1:
            tl.store(send_ptr + send_base + head_dim * send_stride_d, lse.to(send_ptr.dtype.element_ty), mask=row_mask)
        else:
            # Store a finite FP32 LSE as a signed exponent code plus three
            # base-256 significand digits. Every stored value is an integer in
            # [-255, 255], so FP16 and BF16 preserve it exactly.
            finite_lse = (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            abs_lse = tl.abs(lse)
            nonzero_lse = abs_lse != 0.0
            safe_abs_lse = tl.where(finite_lse & nonzero_lse, abs_lse, 1.0)
            lse_exponent = tl.floor(tl.log2(safe_abs_lse))
            lse_exponent = tl.maximum(-126.0, tl.minimum(lse_exponent, 127.0))
            lse_exponent = tl.where(nonzero_lse, lse_exponent, 0.0)
            significand = tl.where(
                finite_lse & nonzero_lse,
                abs_lse * tl.exp2(23.0 - lse_exponent),
                0.0,
            )
            significand_hi = tl.floor(significand / 65536.0)
            significand_remainder = significand - significand_hi * 65536.0
            significand_mid = tl.floor(significand_remainder / 256.0)
            significand_lo = significand_remainder - significand_mid * 256.0
            exponent_code = lse_exponent + 128.0
            exponent_code = tl.where(lse < 0.0, -exponent_code, exponent_code)
            exponent_code = tl.where(finite_lse, exponent_code, 0.0)
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
def _fused_sfa_dcp_lse_combine_batched_kernel(
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
    SCATTER_TOKENS: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    RETURN_LSE: tl.constexpr = False,
    HAS_LOCAL: tl.constexpr = False,
):
    program_idx = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d_offsets = tl.arange(0, BLOCK_D)[None, :]

    for row_start in range(program_idx * BLOCK_ROWS, total_rows, num_programs * BLOCK_ROWS):
        linear_idx = row_start + tl.arange(0, BLOCK_ROWS)[:, None]
        row_mask = linear_idx < total_rows
        token_idx = (linear_idx // num_heads).to(tl.int64)
        head_idx = (linear_idx % num_heads).to(tl.int64)

        if SCATTER_TOKENS:
            scatter_idx = token_idx
            replicated_idx = head_idx
        else:
            scatter_idx = head_idx
            replicated_idx = token_idx

        # Keep LSE state per row and one [BLOCK_ROWS, BLOCK_D] accumulator.
        # Stream ranks to avoid a [DCP_SIZE, BLOCK_ROWS, BLOCK_D] live buffer.
        lse_max = -float("inf")
        if HAS_LOCAL:
            local_lse = tl.load(
                local_lse_ptr + token_idx * local_lse_stride_t + head_idx * local_lse_stride_h, mask=row_mask, other=0.0
            ).to(tl.float32)
            local_valid = (local_lse == local_lse) & (local_lse != float("inf")) & (local_lse != -float("inf"))
            lse_max = tl.where(local_valid, local_lse, -float("inf"))
        for rank_idx in tl.static_range(DCP_SIZE):
            recv_base = (
                rank_idx * recv_stride_rank
                + scatter_idx * recv_stride_scatter
                + replicated_idx * recv_stride_replicated
            )
            if LSE_PACK_DIM == 1:
                lse = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d, mask=row_mask, other=0.0).to(tl.float32)
                valid_lse = (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            else:
                exponent_code = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d, mask=row_mask, other=0.0).to(
                    tl.float32
                )
                significand_hi = tl.load(
                    recv_ptr + recv_base + (head_dim + 1) * recv_stride_d, mask=row_mask, other=0.0
                ).to(tl.float32)
                significand_mid = tl.load(
                    recv_ptr + recv_base + (head_dim + 2) * recv_stride_d, mask=row_mask, other=0.0
                ).to(tl.float32)
                significand_lo = tl.load(
                    recv_ptr + recv_base + (head_dim + 3) * recv_stride_d, mask=row_mask, other=0.0
                ).to(tl.float32)
                packed_valid = exponent_code != 0.0
                sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
                exponent_magnitude = tl.where(exponent_code < 0.0, -exponent_code, exponent_code)
                safe_exponent = exponent_magnitude - 128.0
                significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
                lse = sign * significand * tl.exp2(safe_exponent - 23.0)
                valid_lse = packed_valid & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            lse_max = tl.maximum(lse_max, tl.where(valid_lse, lse, -float("inf")))

        any_valid_lse = lse_max != -float("inf")
        safe_lse_max = tl.where(any_valid_lse, lse_max, 0.0)
        weight_sum = 0.0
        merged = tl.zeros([BLOCK_ROWS, BLOCK_D], dtype=tl.float32)
        d_mask = (d_offsets < head_dim) & row_mask
        for rank_idx in tl.static_range(DCP_SIZE):
            recv_base = (
                rank_idx * recv_stride_rank
                + scatter_idx * recv_stride_scatter
                + replicated_idx * recv_stride_replicated
            )
            if LSE_PACK_DIM == 1:
                lse = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d, mask=row_mask, other=0.0).to(tl.float32)
                valid_lse = (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            else:
                exponent_code = tl.load(recv_ptr + recv_base + head_dim * recv_stride_d, mask=row_mask, other=0.0).to(
                    tl.float32
                )
                significand_hi = tl.load(
                    recv_ptr + recv_base + (head_dim + 1) * recv_stride_d, mask=row_mask, other=0.0
                ).to(tl.float32)
                significand_mid = tl.load(
                    recv_ptr + recv_base + (head_dim + 2) * recv_stride_d, mask=row_mask, other=0.0
                ).to(tl.float32)
                significand_lo = tl.load(
                    recv_ptr + recv_base + (head_dim + 3) * recv_stride_d, mask=row_mask, other=0.0
                ).to(tl.float32)
                packed_valid = exponent_code != 0.0
                sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
                exponent_magnitude = tl.where(exponent_code < 0.0, -exponent_code, exponent_code)
                safe_exponent = exponent_magnitude - 128.0
                significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
                lse = sign * significand * tl.exp2(safe_exponent - 23.0)
                valid_lse = packed_valid & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
            weight = tl.where(valid_lse, tl.exp(lse - safe_lse_max), 0.0)
            partial_output = tl.load(
                recv_ptr + recv_base + d_offsets * recv_stride_d,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            # Select before multiplication: multiplying a zero weight by a
            # NaN from an invalid rank would otherwise contaminate the result.
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
        if RETURN_LSE:
            merged_lse = tl.where(any_valid_lse, safe_lse_max + tl.log(denominator), -float("inf"))
            lse_offset = token_idx * output_stride_t + head_idx * output_stride_h + head_dim * output_stride_d
            tl.store(output_ptr + lse_offset, merged_lse, mask=row_mask)
