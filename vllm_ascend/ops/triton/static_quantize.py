# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ND QINT8 quantization with independent core partition and UB tiles.

Read the original stored reciprocal/offset, never recompute the reciprocal.
Per-channel parameters retain their actual dtype, address and stride. The
existing FP32 scalar interface remains supported.
"""

import configparser
import platform
from functools import cache
from pathlib import Path
from typing import NamedTuple

import torch
import torch_npu
import triton.language.extra.cann.extension as al
from vllm.triton_utils import tl, triton

from vllm_ascend import envs
from vllm_ascend.ops.triton.static_quantize_tiling import DMA_ALIGNMENT_BYTES, _quantize_tiling


class QuantizeResources(NamedTuple):
    vector_cores: int
    ub_bytes: int
    cache_line_bytes: int


@cache
def get_device_resources(device_index: int) -> QuantizeResources | None:
    """Read the current device's CANN tiling geometry once per device."""
    if not envs.ASCEND_HOME_PATH:
        return None
    root = Path(envs.ASCEND_HOME_PATH)
    device_name = torch.npu.get_device_properties(device_index).name
    relative = Path("data/platform_config") / (device_name + ".ini")
    config = configparser.ConfigParser()
    # CANN supports both toolkit layouts. Prefer the generic path if present.
    for path in (root / relative, root / (platform.machine() + "-linux") / relative):
        if config.read(path):
            break
    else:
        return None
    arch = config.getint("version", "NpuArch", fallback=0)
    ub_bytes = config.getint("AICoreSpec", "ub_size", fallback=0)
    cores = triton.runtime.driver.active.utils.get_device_properties(device_index).get("num_vectorcore", 0)
    if min(cores, ub_bytes, arch) <= 0:
        return None
    return QuantizeResources(cores, ub_bytes, 512 if arch in (2201, 3003, 3113) else 256)


@triton.jit
def _quantize_row(x, reciprocal, offset):
    """One CANN Compute row; preserve the established FP32 rounding contract."""
    product = x.to(tl.float32) * reciprocal
    affine = product + offset
    # Preserve NaNs through rounding. The final saturating device cast has
    # CANN's NaN-to-zero behavior; gate it with all BF16/FP16 bit patterns.
    bounded = tl.minimum(
        tl.maximum(affine, -128.0, propagate_nan=tl.PropagateNan.ALL),
        127.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    RNE_MAGIC: tl.constexpr = 12582912.0
    rounded = (bounded + RNE_MAGIC) - RNE_MAGIC
    # These finite, bounded integral FP32 values are exactly representable in FP16.
    # CANN uses an INT32 carrier between RINT and FP16, but this backend
    # lowers that carrier through INT32 -> FP32 -> FP16. Keep the already
    # rounded value instead of adding a redundant integer round trip.
    return rounded.to(tl.float16)


@triton.jit
def _quantize_layout_copy_kernel(
    source,
    output,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    DENSE_STRIDES: tl.constexpr,
    AXIS: tl.constexpr,
    OUTER: tl.constexpr,
    INNER_BLOCK: tl.constexpr,
    OUTER_BLOCK: tl.constexpr,
    FULL_TILE: tl.constexpr,
):
    """Bit-copy an ND view, reading along its physically contiguous axis."""
    inner_start = tl.program_id(0).to(tl.int64) * INNER_BLOCK
    outer_start = tl.program_id(1).to(tl.int64) * OUTER_BLOCK
    inner = tl.arange(0, INNER_BLOCK)
    outer = tl.arange(0, OUTER_BLOCK)
    source_base = tl.full((), 0, tl.int64)
    output_base = tl.full((), 0, tl.int64)
    remaining = tl.program_id(2).to(tl.int64)
    for axis in tl.static_range(len(SHAPE) - 2, -1, -1):
        if axis != AXIS:
            coordinate = remaining % SHAPE.value[axis]
            remaining = remaining // SHAPE.value[axis]
            source_base += coordinate * STRIDES.value[axis]
            output_base += coordinate * DENSE_STRIDES.value[axis]
    source_origin = source + source_base + outer_start * STRIDES.value[-1] + inner_start
    output_origin = output + output_base + inner_start * DENSE_STRIDES.value[AXIS] + outer_start
    # Keep the GM tile origin in INT64. Local offsets fit INT32 unless the
    # actual stride requires a wider address; avoid vector INT64 conversion.
    MAX_INT32_OFFSET: tl.constexpr = (1 << 31) - 1
    if (OUTER_BLOCK - 1) * STRIDES.value[-1] + INNER_BLOCK > MAX_INT32_OFFSET:
        source_outer = outer.to(tl.int64)
    else:
        source_outer = outer
    if (INNER_BLOCK - 1) * DENSE_STRIDES.value[AXIS] + OUTER_BLOCK > MAX_INT32_OFFSET:
        output_inner = inner.to(tl.int64)
    else:
        output_inner = inner
    source_tile = source_origin + source_outer[:, None] * STRIDES.value[-1] + inner[None, :]
    output_tile = output_origin + output_inner[:, None] * DENSE_STRIDES.value[AXIS] + outer[None, :]
    if FULL_TILE:
        value = tl.load(source_tile)
        tl.store(output_tile, tl.trans(value))
    else:
        valid_inner = inner < SHAPE.value[AXIS] - inner_start
        valid_outer = outer < OUTER - outer_start
        value = tl.load(source_tile, valid_outer[:, None] & valid_inner[None, :], other=0)
        # Rebuild the rectangular mask in output order. Transposing a boolean
        # mask triggered a read/modify/write lowering on the installed backend.
        tl.store(output_tile, tl.trans(value), valid_inner[:, None] & valid_outer[None, :])


def _quantize_contiguous_copy(x, axis, resources):
    """Normalize costly transposed views with Triton; include this call in timing."""
    width = x.shape[axis]
    outer = x.shape[-1]
    batches = x.numel() // width // outer
    inner_block = min(triton.next_power_of_2(width), resources.cache_line_bytes // x.element_size())
    # Two payload carriers, two INT64 address vectors, and the predicate.
    capacity = (resources.ub_bytes - 2048) // (2 * x.element_size() + 2 * 8 + 1)
    per_core = triton.cdiv(x.numel(), resources.vector_cores)
    outer_block = min(
        triton.next_power_of_2(outer),
        triton.next_power_of_2(triton.cdiv(per_core, inner_block)),
        1 << (max(1, capacity // inner_block).bit_length() - 1),
    )
    alignment = DMA_ALIGNMENT_BYTES // x.element_size()
    while (
        2
        * triton.cdiv(inner_block, alignment)
        * alignment
        * triton.cdiv(outer_block, alignment)
        * alignment
        * x.element_size()
        + (inner_block + outer_block) * (2 * 8 + 1)
        > resources.ub_bytes - 2048
    ):
        # Transpose carriers pad BOTH axes to a DMA block, including narrow
        # rows. Logical element count alone misses this physical allocation.
        if outer_block > 1:
            outer_block //= 2
        elif inner_block > 1:
            inner_block //= 2
        else:
            raise RuntimeError("UB cannot hold one DMA-aligned transpose tile")
    result = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    raw_dtype = torch.int16 if x.element_size() == 2 else torch.int32
    _quantize_layout_copy_kernel[(triton.cdiv(width, inner_block), triton.cdiv(outer, outer_block), batches)](
        x.view(raw_dtype),
        result.view(raw_dtype),
        SHAPE=tuple(x.shape),
        STRIDES=tuple(x.stride()),
        DENSE_STRIDES=tuple(result.stride()),
        AXIS=axis,
        OUTER=outer,
        INNER_BLOCK=inner_block,
        OUTER_BLOCK=outer_block,
        FULL_TILE=width % inner_block == 0 and outer % outer_block == 0,
        multibuffer=False,
    )
    return result


@triton.jit
def _load_quantize_columns(
    ptr,
    start,
    end,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    DMA_BYTES: tl.constexpr,
    ELEMENT_BYTES: tl.constexpr,
    FULL: tl.constexpr,
    SPAN: tl.constexpr,
):
    """Coalesce short-stride reads before selecting the logical columns.

    The A3 strided DMA lowering otherwise reserves a 32-byte block for each
    element. Read the physical span when it is smaller than that staging
    space; selection stays in UB. No address outside the valid row is read.
    """
    lanes = tl.arange(0, BLOCK)
    if STRIDE > 1 and SPAN * ELEMENT_BYTES < BLOCK * DMA_BYTES:
        physical = tl.arange(0, SPAN)
        if FULL:
            span_size = (BLOCK - 1) * STRIDE + 1
        else:
            span_size = (end - start - 1) * STRIDE + 1
        values = tl.load(ptr + start * STRIDE + physical, physical < span_size, other=0)
        return tl.gather(values, lanes * STRIDE, axis=0)
    else:
        if FULL:
            return tl.load(ptr + (start + lanes) * STRIDE)
        else:
            return tl.load(ptr + (start + lanes) * STRIDE, start + lanes < end, other=0)


@triton.jit
def _static_quantize_kernel(
    x_ptr,
    reciprocal_ptr,
    offset_ptr,
    output_ptr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
    ROW_COUNT: tl.constexpr,
    COLUMN_COUNT: tl.constexpr,
    PER_TENSOR: tl.constexpr,
    RECIPROCAL_STRIDE: tl.constexpr,
    OFFSET_STRIDE: tl.constexpr,
    INPUT_SPAN: tl.constexpr,
    RECIPROCAL_SPAN: tl.constexpr,
    OFFSET_SPAN: tl.constexpr,
    ROW_FACTOR: tl.constexpr,
    COLUMN_FACTOR: tl.constexpr,
    COLUMN_CORES: tl.constexpr,
    BASE_N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    PARAMETER_BLOCK: tl.constexpr,
    DMA_BYTES: tl.constexpr,
    INPUT_BYTES: tl.constexpr,
    PARAMETER_BYTES: tl.constexpr,
):
    core = tl.program_id(0).to(tl.int64)
    row_start = (core // COLUMN_CORES) * ROW_FACTOR
    column_start = (core % COLUMN_CORES) * COLUMN_FACTOR
    row_end = tl.minimum(row_start + ROW_FACTOR, ROW_COUNT)
    column_end = tl.minimum(column_start + COLUMN_FACTOR, COLUMN_COUNT)
    # Like CANN's full-length / tail loops, prove the full tile from the
    # partition dimensions. Avoid dynamic padding and tail bookkeeping when
    # every column tile is complete; arbitrary tails retain masked accesses.
    FULL_COLUMNS: tl.constexpr = COLUMN_COUNT % COLUMN_FACTOR == 0 and COLUMN_FACTOR % BLOCK_COLS == 0
    if PER_TENSOR:
        if reciprocal_ptr.dtype.element_ty == tl.float32:
            # Original PerTensorFP32 uses GM GetValue. Retain Tensor updates.
            reciprocal = tl.load(reciprocal_ptr)
            offset = tl.load(offset_ptr)
        else:
            # PerTensorFP16/BF16 stages a DMA block, then reads its first value.
            # Masked lanes never read beyond a one-element parameter Tensor.
            lanes = tl.arange(0, PARAMETER_BLOCK)
            raw_reciprocal = tl.load(reciprocal_ptr + lanes, lanes == 0, other=0)
            raw_offset = tl.load(offset_ptr + lanes, lanes == 0, other=0)
            reciprocal = al.get_element(raw_reciprocal, (0,)).to(tl.float32)
            offset = al.get_element(raw_offset, (0,)).to(tl.float32)
    for column_tile in range(0, column_end - column_start, BLOCK_COLS):
        columns = column_start + column_tile + tl.arange(0, BLOCK_COLS)
        if FULL_COLUMNS:
            column_mask = tl.full((BLOCK_COLS,), True, tl.int1)
        else:
            column_mask = columns < column_end
        if not PER_TENSOR:
            # Scalar tile origin plus consecutive lanes gives a contiguous
            # slice for stride=1. Strided parameter views keep their semantics.
            reciprocal = _load_quantize_columns(
                reciprocal_ptr,
                column_start + column_tile,
                column_end,
                RECIPROCAL_STRIDE,
                BLOCK_COLS,
                DMA_BYTES,
                PARAMETER_BYTES,
                FULL_COLUMNS,
                RECIPROCAL_SPAN,
            ).to(tl.float32)
            offset = _load_quantize_columns(
                offset_ptr,
                column_start + column_tile,
                column_end,
                OFFSET_STRIDE,
                BLOCK_COLS,
                DMA_BYTES,
                PARAMETER_BYTES,
                FULL_COLUMNS,
                OFFSET_SPAN,
            ).to(tl.float32)
        for row_batch in range(row_start, row_end, BASE_N):
            if BASE_N == 1:
                index = row_batch * COLUMN_COUNT + columns
                if CONTIGUOUS:
                    input_index = index
                elif not PER_TENSOR:
                    # Decode the leading dimensions once per logical row;
                    # the column slice retains its affine physical stride.
                    row_offset = tl.full((), 0, tl.int64)
                    remaining = row_batch
                    for axis in tl.static_range(len(SHAPE) - 2, -1, -1):
                        row_offset += (remaining % SHAPE.value[axis]) * STRIDES.value[axis]
                        remaining = remaining // SHAPE.value[axis]
                else:
                    input_index = tl.full((BLOCK_COLS,), 0, tl.int64)
                    remaining = index
                    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
                        input_index += (remaining % SHAPE.value[axis]) * STRIDES.value[axis]
                        remaining = remaining // SHAPE.value[axis]
                if not CONTIGUOUS and not PER_TENSOR:
                    x = _load_quantize_columns(
                        x_ptr + row_offset,
                        column_start + column_tile,
                        column_end,
                        STRIDES.value[-1],
                        BLOCK_COLS,
                        DMA_BYTES,
                        INPUT_BYTES,
                        FULL_COLUMNS,
                        INPUT_SPAN,
                    )
                else:
                    x = tl.load(x_ptr + input_index, mask=column_mask, other=0)
                half = _quantize_row(x, reciprocal, offset)
                tl.store(output_ptr + index, al.cast(half, tl.int8, overflow_mode="saturate"), mask=column_mask)
            else:
                rows = row_batch + tl.arange(0, BLOCK_ROWS)
                batch_end = tl.minimum(row_batch + BASE_N, row_end)
                mask = (rows[:, None] < batch_end) & column_mask[None, :]
                index = rows[:, None] * COLUMN_COUNT + columns[None, :]
                if CONTIGUOUS:
                    batch_x = tl.load(x_ptr + index, mask=mask, other=0)
                # A3 insert_slice on INT8 converts the whole batch twice per
                # row. FP16 preserves every rounded INT8 value exactly and
                # permits one final narrowing cast after all row insertions.
                batch_y = tl.full((BLOCK_ROWS, BLOCK_COLS), 0, tl.float16)
                for row in range(batch_end - row_batch):
                    if CONTIGUOUS:
                        row_x = al.extract_slice(batch_x, (row, 0), (1, BLOCK_COLS), (1, 1))
                        x = tl.reshape(row_x, (BLOCK_COLS,))
                    else:
                        # A3's strided 2D load lowering creates a dynamic
                        # staging slice that fails static buffer-size analysis.
                        # Read each physical row directly; retain the same
                        # parameter reuse and batched contiguous output.
                        row_offset = tl.full((), 0, tl.int64)
                        remaining = row_batch + row
                        for axis in tl.static_range(len(SHAPE) - 2, -1, -1):
                            row_offset += (remaining % SHAPE.value[axis]) * STRIDES.value[axis]
                            remaining = remaining // SHAPE.value[axis]
                        x = _load_quantize_columns(
                            x_ptr + row_offset,
                            column_start + column_tile,
                            column_end,
                            STRIDES.value[-1],
                            BLOCK_COLS,
                            DMA_BYTES,
                            INPUT_BYTES,
                            FULL_COLUMNS,
                            INPUT_SPAN,
                        )
                    half = _quantize_row(x, reciprocal, offset)
                    batch_y = al.insert_slice(
                        batch_y, tl.reshape(half, (1, BLOCK_COLS)), (row, 0), (1, BLOCK_COLS), (1, 1)
                    )
                flat_y = tl.reshape(batch_y, (BLOCK_ROWS * BLOCK_COLS,))
                result = al.cast(flat_y, tl.int8, overflow_mode="saturate")
                tl.store(output_ptr + index, tl.reshape(result, (BLOCK_ROWS, BLOCK_COLS)), mask=mask)


def quantize(x: torch.Tensor, reciprocal: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Quantize original scalar/channel parameters; return contiguous INT8.

    Accept rank 1..8 strided ND BF16/FP16/FP32 inputs. Parameters are matching
    ND scalars (FP32 or input dtype) or 1D channel vectors in the input dtype.
    Both parameter vectors use their actual strides and remain Tensor inputs
    during graph replay. No host value reads, reciprocal recomputation, or
    persistent parameter copies are performed here.
    """
    parameter_count = reciprocal.numel()
    per_tensor = parameter_count == 1
    if not (
        x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.layout == torch.strided
        and 1 <= x.ndim <= 8
        and x.device == reciprocal.device == offset.device
        and reciprocal.layout == offset.layout == torch.strided
        and reciprocal.shape == offset.shape
        and reciprocal.dtype == offset.dtype
        and reciprocal.ndim in (0, 1)
        and (
            (per_tensor and reciprocal.dtype in (x.dtype, torch.float32))
            or (not per_tensor and parameter_count == x.shape[-1] and reciprocal.dtype == x.dtype)
        )
        and all(torch_npu.get_npu_format(tensor) == 2 for tensor in (x, reciprocal, offset))
    ):
        return torch_npu.npu_quantize(x, reciprocal, offset, torch.qint8, -1, False)
    size = x.numel()
    resources = get_device_resources(x.device.index) if size else None
    if size and resources is None:
        return torch_npu.npu_quantize(x, reciprocal, offset, torch.qint8, -1, False)
    output = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    if resources is not None:
        contiguous = x.is_contiguous()
        if not contiguous:
            axis = next((i for i in range(x.ndim) if x.shape[i] > 1 and x.stride(i) == 1), None)
            if (
                axis is not None
                and axis != x.ndim - 1
                and (per_tensor or x.stride(-1) * x.element_size() >= DMA_ALIGNMENT_BYTES)
            ):
                x = _quantize_contiguous_copy(x, axis, resources)
                contiguous = True
        reciprocal_stride = reciprocal.stride()[0] if reciprocal.ndim else 0
        offset_stride = offset.stride()[0] if offset.ndim else 0
        width = size if per_tensor else x.shape[-1]
        rows = 1 if per_tensor else size // width
        plan = _quantize_tiling(
            rows,
            width,
            x.element_size(),
            reciprocal.element_size(),
            contiguous,
            reciprocal_stride == 1 and offset_stride == 1,
            per_tensor,
            resources.vector_cores,
            resources.ub_bytes,
            resources.cache_line_bytes,
        )
        cores = plan.pop("NUM_CORES")
        _static_quantize_kernel[(cores,)](
            x,
            reciprocal,
            offset,
            output,
            SHAPE=tuple(x.shape),
            STRIDES=tuple(x.stride()),
            CONTIGUOUS=contiguous,
            ROW_COUNT=rows,
            COLUMN_COUNT=width,
            PER_TENSOR=per_tensor,
            RECIPROCAL_STRIDE=reciprocal_stride,
            OFFSET_STRIDE=offset_stride,
            # Worker patches replace next_power_of_2 with an int-only helper.
            # Derive the same spans on the host, before entering Triton JIT.
            INPUT_SPAN=triton.next_power_of_2((plan["BLOCK_COLS"] - 1) * x.stride(-1) + 1),
            RECIPROCAL_SPAN=triton.next_power_of_2((plan["BLOCK_COLS"] - 1) * reciprocal_stride + 1),
            OFFSET_SPAN=triton.next_power_of_2((plan["BLOCK_COLS"] - 1) * offset_stride + 1),
            PARAMETER_BLOCK=DMA_ALIGNMENT_BYTES // reciprocal.element_size(),
            DMA_BYTES=DMA_ALIGNMENT_BYTES,
            **plan,
            INPUT_BYTES=x.element_size(),
            PARAMETER_BYTES=reciprocal.element_size(),
            enable_fp_fusion=False,
            multibuffer=False,
            enable_ubuf_saving=True,
        )
    return output
