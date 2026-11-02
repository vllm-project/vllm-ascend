# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Copy current C8 resident rows directly into owned Prefill staging."""

import torch
import triton
import triton.language as tl


@triton.jit
def _resident_rows_to_staging(
    source,
    destination,
    indices,
    rows: tl.constexpr,
    source_block_stride: tl.constexpr,
    source_row_stride: tl.constexpr,
    destination_stride: tl.constexpr,
    index_stride: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    byte = tl.arange(0, BLOCK_BYTES)
    # The production interleaved allocation exceeds a 32-bit byte address.
    physical = tl.load(indices + row * index_stride, row < rows, other=0).to(tl.int64)
    physical = tl.maximum(physical, 0)
    offset = (physical // 128) * source_block_stride + (physical % 128) * source_row_stride
    values = tl.load(source + offset[:, None] + byte[None, :], (row[:, None] < rows) & (byte[None, :] < 656), other=0)
    tl.store(
        destination + row[:, None].to(tl.int64) * destination_stride + byte[None, :],
        values,
        (row[:, None] < rows) & (byte[None, :] < 656),
    )


def try_gather_resident_rows(source: torch.Tensor, destination: torch.Tensor, indices: torch.Tensor) -> bool:
    """Use the A5 packed-C8 contract; unsupported layouts retain the base path.

    Each call reads current indices on the caller's current stream. Negative
    indices select resident row zero, exactly as the base path does, before
    the existing Host-miss gather overwrites those destination rows. Positive
    indices must address resident rows, as required by the staging plan.
    No address, group metadata, output buffer or completion event is cached.
    """
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    if get_ascend_device_type() != AscendDeviceType.A5:
        return False
    if not (source.device.type == "npu" and source.device == destination.device == indices.device):
        return False
    if not (source.dtype == destination.dtype and source.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)):
        return False
    if not (
        source.ndim == 3
        and source.shape[0] > 0
        and source.shape[1:] == (128, 656)
        and destination.ndim == 2
        and destination.shape[1] == 656
        and indices.ndim == 1
        and indices.numel() == destination.shape[0]
        and indices.dtype in (torch.int32, torch.int64)
        and source.stride(2) == destination.stride(1) == 1
        and source.stride(1) >= 656
        and source.stride(0) >= 128 * source.stride(1)
        and destination.stride(0) >= 656
        and indices.stride(0) > 0
    ):
        return False
    if source.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr():
        return False
    if indices.numel():
        _resident_rows_to_staging[(triton.cdiv(indices.numel(), 8),)](
            source.view(torch.uint8),
            destination.view(torch.uint8),
            indices,
            indices.numel(),
            source.stride(0),
            source.stride(1),
            destination.stride(0),
            indices.stride(0),
            BLOCK_ROWS=8,
            BLOCK_BYTES=1024,
            num_warps=4,
        )
    return True
