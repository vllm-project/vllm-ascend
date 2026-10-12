# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram lookup kernels; model code owns buffers and launch scheduling."""

from vllm.triton_utils import tl, triton


@triton.jit
def _decode_e8m0(scale):
    # E8M0 has no sign or mantissa. 0 is 2^-127 (not FP32 zero),
    # and 255 is NaN (not FP32 infinity).
    # Ascend byte loads can sign-extend values >= 128 during widening.
    exponent = scale.to(tl.int32) & 0xFF
    bits = exponent << 23
    bits = tl.where(exponent == 0, 0x00400000, bits)
    bits = tl.where(exponent == 255, 0x7FC00000, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _engram_int8_gather_dequant_kernel(
    weight_ptr,
    scale_ptr,
    ids_ptr,
    output_ptr,
    rows,
    vocab_start,
    vocab_end,
    ids_stride_t,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
    QUANTIZED: tl.constexpr,
    MXFP8: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return
    offsets = tl.arange(0, WIDTH)
    # Row `row` is (token, local head); ids is [tokens, n_hash_cols] and this
    # shard owns heads [HEAD_START, HEAD_START + LOCAL_HEADS). Local heads are
    # written contiguously and the rest of PAD_HEADS stays untouched, so a
    # narrower shard never writes into the padding other ranks read.
    token = row // LOCAL_HEADS
    local = row % LOCAL_HEADS
    source_row = tl.load(ids_ptr + token * ids_stride_t + HEAD_START + local).to(tl.int64)
    # Same last line of defence as the host-uva kernel, and the same contract
    # as upstream's lookup: ids are global and only this shard's vocab range is
    # owned; anything else reads row 0 and is masked back to zero.
    owned = (source_row >= vocab_start) & (source_row < vocab_end)
    local_row = tl.where(owned, source_row - vocab_start, 0)
    codes = tl.load(weight_ptr + local_row * WIDTH + offsets).to(tl.float32)
    if QUANTIZED:
        scales = tl.load(scale_ptr + local_row * (WIDTH // GROUP) + offsets // GROUP)
        if MXFP8:
            scales = _decode_e8m0(scales)
        codes = codes * scales
    result = codes.to(tl.bfloat16)
    tl.store(
        output_ptr + (token * PAD_HEADS + local) * WIDTH + offsets,
        tl.where(owned, result, tl.zeros_like(result)),
    )


@triton.jit
def _engram_host_uva_gather_dequant_kernel(
    codes_ptrs,
    scales_ptrs,
    ids,
    output,
    rows,
    vocab_start,
    vocab_end,
    ids_stride_t,
    CHUNK: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
    QUANTIZED: tl.constexpr,
    MXFP8: tl.constexpr,
):
    row = tl.program_id(0)
    if row < rows:
        token = row // LOCAL_HEADS
        head_local = row % LOCAL_HEADS
        index = tl.load(ids + token * ids_stride_t + HEAD_START + head_local).to(tl.int64)
        owned = (index >= vocab_start) & (index < vocab_end)
        local_row = tl.where(owned, index - vocab_start, 0)
        chunk = local_row // CHUNK
        local = local_row % CHUNK
        if MXFP8:
            codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.float8e4nv))
        elif QUANTIZED:
            codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.int8))
        else:
            codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.bfloat16))
        col = tl.arange(0, WIDTH)
        value = tl.load(codes + local * WIDTH + col).to(tl.float32)
        if QUANTIZED:
            if MXFP8:
                scales = tl.load(scales_ptrs + chunk).to(tl.pointer_type(tl.uint8))
            else:
                scales = tl.load(scales_ptrs + chunk).to(tl.pointer_type(tl.float32))
            if MXFP8:
                # Read and decode each group scale once, then broadcast to
                # its values instead of gathering WIDTH repeated bytes.
                groups = tl.arange(0, WIDTH // GROUP)
                scale = tl.load(scales + local * (WIDTH // GROUP) + groups)
                scale = _decode_e8m0(scale)
                value = tl.reshape(value, (WIDTH // GROUP, GROUP)) * scale[:, None]
                value = tl.reshape(value, (WIDTH,))
            else:
                scale = tl.load(scales + local * (WIDTH // GROUP) + col // GROUP)
                value = value * scale
        result = value.to(tl.bfloat16)
        tl.store(
            output + (token * PAD_HEADS + head_local) * WIDTH + col,
            tl.where(owned, result, tl.zeros_like(result)),
        )
