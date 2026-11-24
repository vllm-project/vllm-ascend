# SPDX-License-Identifier: Apache-2.0
"""Gather and dequantize INT8/MXFP8 PLE rows from registered host memory."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

PLE_GROUP_SIZE = 32


@triton.jit(do_not_specialize=["shard_start", "shard_end"])
def _lookup_quantized(
    code_chunk_ptrs,
    scales,
    embedding_ids,
    output,
    shard_start,
    shard_end,
    EMBEDDING_WIDTH: tl.constexpr,
    CHUNK_ROWS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_MXFP8: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    index = tl.load(embedding_ids + row).to(tl.int64)
    owned = (index >= shard_start) & (index < shard_end)
    col = tl.arange(0, BLOCK_SIZE)
    # INT8 has one shard decision per program. Branch before host gathers so
    # non-owned rows only write zeros, and owned rows use column-only masks.
    # MXFP8 uses the same ownership branch to avoid non-owned host gathers.
    if not IS_MXFP8:
        if owned:
            local = index - shard_start
            base = tl.load(code_chunk_ptrs + local // CHUNK_ROWS).to(tl.pointer_type(tl.int8))
            bits = tl.load(base + (local % CHUNK_ROWS) * EMBEDDING_WIDTH + col, mask=col < EMBEDDING_WIDTH, other=0)
            group = tl.arange(0, BLOCK_SIZE // GROUP_SIZE)
            scale_offsets = local * (EMBEDDING_WIDTH // GROUP_SIZE) + group
            scale = tl.load(scales + scale_offsets, mask=group < EMBEDDING_WIDTH // GROUP_SIZE, other=0)
            dequantized = (bits.to(tl.float32).reshape(BLOCK_SIZE // GROUP_SIZE, GROUP_SIZE) * scale[:, None]).reshape(
                BLOCK_SIZE
            )
            tl.store(output + row * EMBEDDING_WIDTH + col, dequantized, mask=col < EMBEDDING_WIDTH)
        else:
            tl.store(output + row * EMBEDDING_WIDTH + col, 0, mask=col < EMBEDDING_WIDTH)
    else:
        if owned:
            local = index - shard_start
            base = tl.load(code_chunk_ptrs + local // CHUNK_ROWS).to(tl.pointer_type(tl.uint8))
            bits = tl.load(base + (local % CHUNK_ROWS) * EMBEDDING_WIDTH + col, mask=col < EMBEDDING_WIDTH, other=0)
            # Load each scale once, then broadcast over its 32 code elements.
            group = tl.arange(0, BLOCK_SIZE // GROUP_SIZE)
            scale_offsets = local * (EMBEDDING_WIDTH // GROUP_SIZE) + group
            scale_mask = group < EMBEDDING_WIDTH // GROUP_SIZE
            value = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            E8M0_EXPONENT_BIAS: tl.constexpr = 127
            E8M0_NAN_ENCODING: tl.constexpr = 255
            FP32_MANTISSA_BITS: tl.constexpr = 23
            E8M0_MIN_SCALE: tl.constexpr = 5.877471754111438e-39  # 2**-127
            exponent = tl.load(scales + scale_offsets, mask=scale_mask, other=E8M0_EXPONENT_BIAS).to(tl.int32)
            # E8M0 byte 0 denotes 2**-127; byte 255 denotes NaN.
            scale = (exponent << FP32_MANTISSA_BITS).to(tl.float32, bitcast=True)
            scale = tl.where(exponent == 0, E8M0_MIN_SCALE, scale)
            scale = tl.where(exponent == E8M0_NAN_ENCODING, float("nan"), scale)
            dequantized = (value.reshape(BLOCK_SIZE // GROUP_SIZE, GROUP_SIZE) * scale[:, None]).reshape(BLOCK_SIZE)
            tl.store(output + row * EMBEDDING_WIDTH + col, dequantized, mask=col < EMBEDDING_WIDTH)
        else:
            tl.store(output + row * EMBEDDING_WIDTH + col, 0, mask=col < EMBEDDING_WIDTH)


def lookup_quantized(code_chunk_ptrs, scales, embedding_ids, output, shard_start, shard_end, chunk_rows, *, is_mxfp8):
    """Write selected group-32 host rows into a separate contiguous BF16 output.

    Codes use a registered host pointer table. Scales reside in a dense NPU shard.
    Only metadata is checked here: inspecting pointer values or ID contents
    would synchronize the NPU and break graph capture. See docs/ple_lookup_quantized.md.
    """
    if not isinstance(is_mxfp8, bool):
        raise ValueError("is_mxfp8 must be a bool")
    if embedding_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("PLE IDs must have int32 or int64 dtype")
    if output.ndim != embedding_ids.ndim + 1 or tuple(output.shape[:-1]) != tuple(embedding_ids.shape):
        raise ValueError("PLE output shape must be (*embedding_ids.shape, width)")
    if output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise ValueError("PLE output must be contiguous BF16")
    if output.device.type != "npu" or embedding_ids.device != output.device:
        raise ValueError("PLE IDs and output must be on the same NPU")
    width = output.shape[-1]
    if width <= 0 or width % PLE_GROUP_SIZE:
        raise ValueError("Quantized PLE embedding width must be positive and divisible by 32")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in (shard_start, shard_end, chunk_rows)
    ):
        raise ValueError("PLE shard bounds and chunk must be integers")
    if shard_start < 0 or shard_end <= shard_start or chunk_rows <= 0:
        raise ValueError("PLE requires 0 <= start < end and a positive chunk")
    chunks = triton.cdiv(shard_end - shard_start, chunk_rows)
    expected_dtype = torch.uint8 if is_mxfp8 else torch.float32
    if (
        scales.ndim != 2
        or scales.dtype != expected_dtype
        or scales.device != output.device
        or not scales.is_contiguous()
        or scales.shape[0] < shard_end - shard_start
        or scales.shape[1] != width // PLE_GROUP_SIZE
    ):
        raise ValueError("PLE resident scales must cover the local shard with one scale per group")
    if code_chunk_ptrs.dtype != torch.int64 or code_chunk_ptrs.ndim != 1 or not code_chunk_ptrs.is_contiguous():
        raise ValueError("PLE pointer tables must be contiguous 1-D int64 tensors")
    if code_chunk_ptrs.device != output.device or code_chunk_ptrs.numel() < chunks:
        raise ValueError("PLE pointer tables must cover the shard and be on the output NPU")
    if embedding_ids.numel() == 0:
        return
    # Preserve the existing host-lookup setup: lazily cache core counts and UB size.
    # These properties do not select the fixed tile size in this kernel.
    init_device_properties_triton()
    _lookup_quantized[(embedding_ids.numel(),)](
        code_chunk_ptrs,
        scales,
        embedding_ids.contiguous(),
        output,
        shard_start,
        shard_end,
        width,
        chunk_rows,
        PLE_GROUP_SIZE,
        triton.next_power_of_2(width),
        is_mxfp8,
    )
