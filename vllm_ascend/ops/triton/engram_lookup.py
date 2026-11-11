# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram lookup kernels; model code owns buffers and launch scheduling."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

_SCALE_GROUP = 32


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
def _dequantize_packed_engram_kernel(
    packed, scale_ptr, output, ROW_BYTES: tl.constexpr, E8M0: tl.constexpr, MXFP8: tl.constexpr
):
    row = tl.program_id(0)
    columns = tl.arange(0, 256)
    if MXFP8:
        codes = tl.load((packed + row * ROW_BYTES + columns).to(tl.pointer_type(tl.float8e4nv))).to(tl.float32)
    else:
        codes = tl.load(packed + row * ROW_BYTES + columns).to(tl.int8).to(tl.float32)
    if E8M0:
        scales = _decode_e8m0(tl.load(scale_ptr + row * ROW_BYTES + 256 + columns // 32))
    else:
        scales = tl.load(scale_ptr + row * (ROW_BYTES // 4) + 64 + columns // 32)
    tl.store(output + row * 256 + columns, (codes * scales).to(tl.bfloat16))


def dequantize_packed_engram(packed, output, rows, row_bytes, scale_dtype, code_dtype="I8"):
    if code_dtype not in ("I8", "INT8", "F8_E4M3"):
        raise ValueError("unsupported packed Engram code dtype")
    if scale_dtype not in ("F8_E8M0", "F32") or row_bytes != (264 if scale_dtype == "F8_E8M0" else 288):
        raise ValueError("unsupported packed Engram row format")
    if not output.is_contiguous() or output.numel() != rows * 256:
        raise ValueError("packed Engram requires contiguous output rows")
    init_device_properties_triton()
    scales = packed if scale_dtype == "F8_E8M0" else packed.view(torch.float32)
    _dequantize_packed_engram_kernel[(rows,)](
        packed,
        scales,
        output,
        ROW_BYTES=row_bytes,
        E8M0=scale_dtype == "F8_E8M0",
        MXFP8=code_dtype == "F8_E4M3",
        num_warps=4,
    )


@triton.jit
def _engram_offload_encode_kernel(
    ids,
    owners,
    encoded,
    count,
    size,
    row_offset,
    ids_stride,
    vocab_start,
    vocab_end,
    ROWS_PER_OWNER: tl.constexpr,
    ROWS_PER_SLOT: tl.constexpr,
    TABLE_OFFSET: tl.constexpr,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    index = row + row_offset
    source = tl.load(
        ids + (index // LOCAL_HEADS).to(tl.int64) * ids_stride + HEAD_START + index % LOCAL_HEADS,
        row < size,
        other=-1,
    ).to(tl.int64)
    valid = (source >= vocab_start) & (source < vocab_end)
    safe_source = tl.where(valid, source, 0)
    owner = tl.load(owners + safe_source // ROWS_PER_OWNER)
    zero_owner = tl.load(owners)
    entry = owner * ROWS_PER_SLOT + TABLE_OFFSET + safe_source % ROWS_PER_OWNER
    tl.store(encoded + row, tl.where(valid, entry, zero_owner * ROWS_PER_SLOT), row < size)
    if tl.program_id(0) == 0:
        tl.store(count, size)


@triton.jit
def _engram_offload_decode_kernel(
    codes,
    scales,
    ids,
    output,
    row_offset,
    ids_stride,
    vocab_start,
    vocab_end,
    CODE_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
    QUANTIZED: tl.constexpr,
    MXFP8: tl.constexpr,
):
    row = tl.program_id(0)
    index = row + row_offset
    token = (index // LOCAL_HEADS).to(tl.int64)
    head = index % LOCAL_HEADS
    source = tl.load(ids + token * ids_stride + HEAD_START + head).to(tl.int64)
    valid = (source >= vocab_start) & (source < vocab_end)
    col = tl.arange(0, WIDTH)
    value = tl.load(codes + row * CODE_STRIDE + col).to(tl.float32)
    if QUANTIZED:
        scale = tl.load(scales + row * SCALE_STRIDE + col // GROUP)
        if MXFP8:
            scale = _decode_e8m0(scale)
        value = value * scale
    tl.store(output + (token * PAD_HEADS + head) * WIDTH + col, tl.where(valid, value, 0).to(tl.bfloat16))


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


@triton.jit
def _mapped_engram_gather_dequant_kernel(
    bases_ptr,
    ids_ptr,
    output_ptr,
    ids_stride,
    vocab_start,
    vocab_end,
    ROWS_PER_OWNER: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    WEIGHT_BYTES: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
    QUANTIZED: tl.constexpr,
    MXFP8: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // LOCAL_HEADS
    head = row % LOCAL_HEADS
    source = tl.load(ids_ptr + token.to(tl.int64) * ids_stride + HEAD_START + head).to(tl.int64)
    valid = (source >= vocab_start) & (source < vocab_end)
    source = tl.where(valid, source, 0).to(tl.uint32)
    owner = source // ROWS_PER_OWNER
    local_row = source % ROWS_PER_OWNER
    address = tl.load(bases_ptr + owner).to(tl.uint64) + local_row.to(tl.uint64) * ROW_BYTES
    if MXFP8:
        codes = address.to(tl.pointer_type(tl.uint8))
    elif QUANTIZED:
        codes = address.to(tl.pointer_type(tl.int8))
    else:
        codes = address.to(tl.pointer_type(tl.bfloat16))
    col = tl.arange(0, WIDTH)
    if MXFP8:
        # Decode E4M3FN from the packed byte mapping without requiring native
        # FP8 loads/casts, preserving subnormals, signed zero and NaN encodings.
        bits = tl.load(codes + col).to(tl.int32) & 0xFF
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 7
        normal = ((bits & 0x80) << 24) | ((exponent + 120) << 23) | (mantissa << 20)
        subnormal = mantissa.to(tl.float32) * (2.0**-9) * tl.where(bits & 0x80, -1.0, 1.0)
        value = tl.where(exponent == 0, subnormal, normal.to(tl.float32, bitcast=True))
        value = tl.where((bits & 0x7F) == 0x7F, float("nan"), value)
    else:
        value = tl.load(codes + col).to(tl.float32)
    if QUANTIZED:
        if MXFP8:
            scales = (address + WEIGHT_BYTES).to(tl.pointer_type(tl.uint8))
            groups = tl.arange(0, WIDTH // GROUP)
            scale = _decode_e8m0(tl.load(scales + groups))
            value = tl.reshape(value, (WIDTH // GROUP, GROUP)) * scale[:, None]
            value = tl.reshape(value, (WIDTH,))
        else:
            scales = (address + WEIGHT_BYTES).to(tl.pointer_type(tl.float32))
            value = value * tl.load(scales + col // GROUP)
    tl.store(
        output_ptr + (token.to(tl.int64) * PAD_HEADS + head) * WIDTH + col,
        tl.where(valid, value, 0).to(tl.bfloat16),
    )


def mapped_engram_gather_dequant(
    bases, indices, out, rows_per_owner, row_bytes, weight_bytes, head_start, local_heads, vocab_start, vocab_end, dtype
):
    """Read registered MemFabric rank slots into out, leaving padded heads untouched.

    bases is an int64 NPU vector of table starts in owner order. IDs have contiguous
    columns; out is contiguous BF16. Each row stores
    contiguous codes followed by group-32 scales (INT8/FP32 or MXFP8/E8M0), or
    unquantized BF16 values. The pool must outlive this call and graph replays.
    """
    rows = indices.shape[0] * local_heads
    if rows == 0:
        return
    init_device_properties_triton()
    _mapped_engram_gather_dequant_kernel[(rows,)](
        bases,
        indices,
        out,
        indices.stride(0),
        vocab_start,
        vocab_end,
        ROWS_PER_OWNER=rows_per_owner,
        ROW_BYTES=row_bytes,
        WEIGHT_BYTES=weight_bytes,
        WIDTH=out.shape[-1],
        GROUP=_SCALE_GROUP,
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        PAD_HEADS=out.shape[1],
        QUANTIZED=dtype != torch.bfloat16,
        MXFP8=dtype == torch.float8_e4m3fn,
        num_warps=4,
    )
