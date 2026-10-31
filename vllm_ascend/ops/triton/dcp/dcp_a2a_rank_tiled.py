# SPDX-License-Identifier: Apache-2.0

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["NUM_HEADS", "NUM_TOKENS"])
def _fused_dcp_lse_combine_rank_tiled_kernel(
    recv_ptr,
    output_ptr,
    local_output_ptr,
    local_lse_ptr,
    local_stride_t,
    local_stride_h,
    local_stride_d,
    local_lse_stride_t,
    local_lse_stride_h,
    NUM_HEADS,
    NUM_TOKENS,
    DCP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Reduce ranks together while retaining the four-lane LSE wire format.

    A feature tile bounds temporary storage. Weights and accumulation remain
    FP32; the parallel reduction can round differently from the scalar FMA loop.
    """
    ranks = tl.arange(0, BLOCK_R)
    features = tl.arange(0, BLOCK_D)
    tiles_per_head = tl.cdiv(HEAD_DIM, BLOCK_D)
    rank_mask = ranks < DCP_SIZE
    for job in range(tl.program_id(0), NUM_TOKENS * NUM_HEADS * tiles_per_head, tl.num_programs(0)):
        row = job // tiles_per_head
        tile = job % tiles_per_head
        token = row // NUM_HEADS
        head = row % NUM_HEADS
        # Receive layout: [source ranks, local heads, tokens, D + four LSE lanes].
        base = ((ranks * NUM_HEADS + head) * NUM_TOKENS + token) * (HEAD_DIM + 4)
        exponent_code = tl.load(recv_ptr + base + HEAD_DIM, mask=rank_mask, other=0.0).to(tl.float32)
        significand_hi = tl.load(recv_ptr + base + HEAD_DIM + 1, mask=rank_mask, other=0.0).to(tl.float32)
        significand_mid = tl.load(recv_ptr + base + HEAD_DIM + 2, mask=rank_mask, other=0.0).to(tl.float32)
        significand_lo = tl.load(recv_ptr + base + HEAD_DIM + 3, mask=rank_mask, other=0.0).to(tl.float32)
        sign = tl.where(exponent_code < 0.0, -1.0, 1.0)
        exponent_magnitude = tl.where(exponent_code < 0.0, -exponent_code, exponent_code)
        significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
        lse = sign * significand * tl.exp2(exponent_magnitude - 128.0 - 23.0)
        valid = rank_mask & (exponent_code != 0.0) & (lse == lse) & (lse != float("inf")) & (lse != -float("inf"))
        local_lse = tl.load(local_lse_ptr + token * local_lse_stride_t + head * local_lse_stride_h).to(tl.float32)
        local_valid = (local_lse == local_lse) & (local_lse != float("inf")) & (local_lse != -float("inf"))
        maximum = tl.maximum(
            tl.max(tl.where(valid, lse, -float("inf")), 0),
            tl.where(local_valid, local_lse, -float("inf")),
        )
        safe_maximum = tl.where(maximum != -float("inf"), maximum, 0.0)
        weights = tl.where(valid, tl.exp(lse - safe_maximum), 0.0)
        local_weight = tl.where(local_valid, tl.exp(local_lse - safe_maximum), 0.0)
        feature_offsets = tile * BLOCK_D + features
        feature_mask = feature_offsets < HEAD_DIM
        partial = tl.load(
            recv_ptr + base[:, None] + feature_offsets[None, :],
            mask=rank_mask[:, None] & feature_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        partial = tl.where(valid[:, None], partial, 0.0)
        merged = tl.sum(partial * weights[:, None], axis=0)
        local_values = tl.load(
            local_output_ptr + token * local_stride_t + head * local_stride_h + feature_offsets * local_stride_d,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        merged += tl.where(local_valid, local_values, 0.0) * local_weight
        denominator = tl.sum(weights, 0) + local_weight
        merged /= tl.where(denominator > 0.0, denominator, 1.0)
        tl.store(output_ptr + row * HEAD_DIM + feature_offsets, merged, mask=feature_mask)
