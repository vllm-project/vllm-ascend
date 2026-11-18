# SPDX-License-Identifier: Apache-2.0
"""Gather BF16 or group-scaled INT8/MXFP8 PLE rows from registered host memory."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@triton.jit
def _lookup(
    ptrs,
    ids,
    out,
    start,
    end,
    WIDTH: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    index = tl.load(ids + row).to(tl.int64)
    owned = (index >= start) & (index < end)
    local = tl.where(owned, index - start, 0)
    base = tl.load(ptrs + local // CHUNK).to(tl.pointer_type(tl.bfloat16))
    col = tl.arange(0, BLOCK)
    value = tl.load(base + (local % CHUNK) * WIDTH + col, mask=owned & (col < WIDTH), other=0)
    tl.store(out + row * WIDTH + col, value, mask=col < WIDTH)


def lookup(ptrs, ids, output, start, end, chunk):
    if ids.numel() == 0:
        return
    init_device_properties_triton()
    ids = ids.contiguous()
    width = output.shape[-1]
    _lookup[(ids.numel(),)](ptrs, ids, output, start, end, width, chunk, triton.next_power_of_2(width))


PLE_GROUP_SIZE = 32


@triton.jit(do_not_specialize=["start", "end"])
def _lookup_quantized(
    ptrs,
    scale_ptrs,
    ids,
    out,
    start,
    end,
    WIDTH: tl.constexpr,
    CHUNK: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
    MXFP8: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    index = tl.load(ids + row).to(tl.int64)
    owned = (index >= start) & (index < end)
    col = tl.arange(0, BLOCK)
    # INT8 has one shard decision per program. Branch before host gathers so
    # non-owned rows only write zeros, and owned rows use column-only masks.
    # MXFP8 uses the same ownership branch to avoid non-owned host gathers.
    if not MXFP8:
        if owned:
            local = index - start
            base = tl.load(ptrs + local // CHUNK).to(tl.pointer_type(tl.int8))
            scales = tl.load(scale_ptrs + local // CHUNK).to(tl.pointer_type(tl.float32))
            bits = tl.load(base + (local % CHUNK) * WIDTH + col, mask=col < WIDTH, other=0)
            group = tl.arange(0, BLOCK // GROUP)
            scale_offsets = (local % CHUNK) * (WIDTH // GROUP) + group
            scale = tl.load(scales + scale_offsets, mask=group < WIDTH // GROUP, other=0)
            dequantized = (bits.to(tl.float32).reshape(BLOCK // GROUP, GROUP) * scale[:, None]).reshape(BLOCK)
            tl.store(out + row * WIDTH + col, dequantized, mask=col < WIDTH)
        else:
            tl.store(out + row * WIDTH + col, 0, mask=col < WIDTH)
    else:
        if owned:
            local = index - start
            base = tl.load(ptrs + local // CHUNK).to(tl.pointer_type(tl.uint8))
            scales = tl.load(scale_ptrs + local // CHUNK).to(tl.pointer_type(tl.uint8))
            bits = tl.load(base + (local % CHUNK) * WIDTH + col, mask=col < WIDTH, other=0)
            # Load each host scale once, then broadcast over its 32 code elements.
            group = tl.arange(0, BLOCK // GROUP)
            scale_offsets = (local % CHUNK) * (WIDTH // GROUP) + group
            scale_mask = group < WIDTH // GROUP
            value = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            exponent = tl.load(scales + scale_offsets, mask=scale_mask, other=127).to(tl.int32)
            # E8M0 byte 0 denotes 2**-127; byte 255 denotes NaN.
            scale = (exponent << 23).to(tl.float32, bitcast=True)
            scale = tl.where(exponent == 0, 5.877471754111438e-39, scale)
            scale = tl.where(exponent == 255, float("nan"), scale)
            dequantized = (value.reshape(BLOCK // GROUP, GROUP) * scale[:, None]).reshape(BLOCK)
            tl.store(out + row * WIDTH + col, dequantized, mask=col < WIDTH)
        else:
            tl.store(out + row * WIDTH + col, 0, mask=col < WIDTH)


def lookup_quantized(ptrs, scale_ptrs, ids, output, start, end, chunk, *, mxfp8):
    """Write selected group-32 host rows into a separate contiguous BF16 output.

    Pointer tables address live CANN-registered storage, owned by the caller.
    Only metadata is checked here: inspecting pointer values or ID contents
    would synchronize the NPU and break graph capture. See docs/ple_lookup.md.
    """
    if not isinstance(mxfp8, bool):
        raise ValueError("mxfp8 must be a bool")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("PLE IDs must have int32 or int64 dtype")
    if output.ndim != ids.ndim + 1 or tuple(output.shape[:-1]) != tuple(ids.shape):
        raise ValueError("PLE output shape must be (*ids.shape, width)")
    if output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise ValueError("PLE output must be contiguous BF16")
    if output.device.type != "npu" or ids.device != output.device:
        raise ValueError("PLE IDs and output must be on the same NPU")
    width = output.shape[-1]
    if width <= 0 or width % PLE_GROUP_SIZE:
        raise ValueError("Quantized PLE embedding width must be positive and divisible by 32")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (start, end, chunk)):
        raise ValueError("PLE shard bounds and chunk must be integers")
    if start < 0 or end <= start or chunk <= 0:
        raise ValueError("PLE requires 0 <= start < end and a positive chunk")
    chunks = triton.cdiv(end - start, chunk)
    for pointers in (ptrs, scale_ptrs):
        if pointers.dtype != torch.int64 or pointers.ndim != 1 or not pointers.is_contiguous():
            raise ValueError("PLE pointer tables must be contiguous 1-D int64 tensors")
        if pointers.device != output.device or pointers.numel() < chunks:
            raise ValueError("PLE pointer tables must cover the shard and be on the output NPU")
    if ids.numel() == 0:
        return
    init_device_properties_triton()
    _lookup_quantized[(ids.numel(),)](
        ptrs,
        scale_ptrs,
        ids.contiguous(),
        output,
        start,
        end,
        width,
        chunk,
        PLE_GROUP_SIZE,
        triton.next_power_of_2(width),
        mxfp8,
    )
