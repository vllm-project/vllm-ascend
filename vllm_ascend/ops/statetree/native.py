# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU execution of fused StateTree GEMM and direct-layout reconstruction."""

from __future__ import annotations

import torch

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.statetree.codec import PackedWeight
from vllm_ascend.ops.statetree.compiler import build_kernel
from vllm_ascend.ops.statetree.config import FUSED_BATCH_TILES, select_batch_tile


def prepare(packed: PackedWeight) -> None:
    """Compile every serving batch variant before graph capture."""
    for batch_tile in FUSED_BATCH_TILES:
        build_kernel(batch_tile, packed.n_tile, packed.k_tile, packed.escape_slots)


def _validate_npu_weight(packed: PackedWeight) -> None:
    packed.validate()
    if packed.device.type != "npu":
        raise ValueError("StateTree packed weights must be on an NPU")
    if not is_950():
        raise RuntimeError("StateTree currently targets Ascend 950 / dav-3510 only")


def linear(
    x: torch.Tensor,
    packed: PackedWeight,
    batch_tile: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fused packed-weight decode and BF16 GEMM with a row-dependent tile."""
    _validate_npu_weight(packed)
    if x.dtype != torch.bfloat16 or x.device != packed.device or x.ndim < 1 or x.shape[-1] != packed.k:
        raise ValueError("input must be BF16, have last dimension K, and share the packed-weight device")
    if not x.is_contiguous():
        raise ValueError("StateTree input must be contiguous")
    rows = x.numel() // packed.k
    batch_tile = batch_tile or select_batch_tile(rows)
    output_shape = (*x.shape[:-1], packed.n)
    if out is None:
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    elif out.shape != output_shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
        raise ValueError("StateTree output buffer has incompatible shape, dtype, device, or stride")
    if rows == 0:
        return out
    kernel = build_kernel(batch_tile, packed.n_tile, packed.k_tile, packed.escape_slots)
    with torch.npu.device(x.device):
        status = kernel.launch(
            torch.npu.current_stream().npu_stream,
            packed.payload.data_ptr(),
            x.data_ptr(),
            out.data_ptr(),
            rows,
            packed.n,
            packed.k,
        )
    if status != 0:
        raise RuntimeError(f"StateTree fused kernel launch failed with status {status}")
    return out


def decode(packed: PackedWeight) -> torch.Tensor:
    """Decode directly to row-major BF16 without a full-matrix layout copy."""
    _validate_npu_weight(packed)
    kernel = build_kernel(16, packed.n_tile, packed.k_tile, packed.escape_slots)
    dense = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
    n_tiles = (packed.n + packed.n_tile - 1) // packed.n_tile
    units = n_tiles * packed.k // packed.k_tile * 2
    with torch.npu.device(packed.device):
        status = kernel.decode(
            torch.npu.current_stream().npu_stream,
            packed.payload.data_ptr(),
            dense.data_ptr(),
            units,
            packed.n,
            packed.k,
        )
    if status != 0:
        raise RuntimeError(f"StateTree decode kernel launch failed with status {status}")
    return dense
