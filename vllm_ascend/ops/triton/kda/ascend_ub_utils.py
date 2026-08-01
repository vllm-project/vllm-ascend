# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""Minimal Ascend launch helpers used by the FLA fused norm-gate kernel.

The block-size calculations are the single-dimension and row-tile cases from
FLA's ``ascend_ub_manager.py``. Keeping the focused subset here avoids a
runtime dependency on the private ``fla.modules.backends`` package.
"""

import warnings
from functools import cache

import torch
from vllm.triton_utils import triton

ASCEND_MAX_GRID_DIM = 65535
_FALLBACK_UB_CAPACITY_BITS = 65536 * 8


@cache
def _get_ub_capacity_bits() -> int:
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                from tbe.common.platform import (  # type: ignore[import-not-found]
                    get_soc_spec,
                    set_current_compile_soc_info,
                )

                device = torch.npu
                set_current_compile_soc_info(device.get_device_name(device.current_device()))
                ub_size_bytes = get_soc_spec("UB_SIZE")
                if ub_size_bytes is not None and ub_size_bytes > 0:
                    return int(ub_size_bytes) * 8
        except Exception as error:
            warnings.warn(
                f"Failed to detect Ascend UB capacity; using the conservative 64-KiB fallback: {error}",
                stacklevel=3,
            )
            return _FALLBACK_UB_CAPACITY_BITS

    return _FALLBACK_UB_CAPACITY_BITS


def get_multiprocessor_count(device_index: int = 0) -> int:
    """Return the number of NPU vector cores used to cap the launch grid."""
    try:
        properties = triton.runtime.driver.active.utils.get_device_properties(device_index)
        return properties["multiprocessor_count"]
    except Exception:
        try:
            target = triton.runtime.driver.active.get_current_target()
            if target.backend == "npu":
                properties = triton.runtime.driver.active.utils.get_device_properties(device_index)
                return properties["num_vectorcore"]
        except Exception:
            pass
        return 1


def _largest_power_of_two_at_most(value: int) -> int:
    value = max(1, value)
    return triton.next_power_of_2(value + 1) // 2


def _ub_safe_block(
    desired: int,
    fixed_size: int,
    memory_multiplier: float,
    safety_margin: float,
) -> int:
    safe_capacity_bits = int(_get_ub_capacity_bits() * safety_margin)
    bytes_per_element = 4
    max_block = int(safe_capacity_bits // (memory_multiplier * max(1, fixed_size) * bytes_per_element * 8))
    return min(desired, _largest_power_of_two_at_most(max_block))


def compute_ub_block_size(
    dim_size: int,
    memory_multiplier: float,
    *,
    safety_margin: float = 0.9,
    fallback: int = 2048,
    min_block: int = 1,
    max_block: int | None = None,
    desired: int | None = None,
) -> int:
    """Compute the FLA UB-safe block size for one tilable dimension."""
    if desired is None:
        desired = triton.next_power_of_2(dim_size)
    try:
        block = _ub_safe_block(desired, 1, memory_multiplier, safety_margin)
    except Exception:
        block = min(desired, fallback)
    block = max(min_block, block)
    if max_block is not None:
        block = min(block, max_block)
    return block


def compute_row_tile_block_size(
    row_dim: int,
    fixed_dim: int,
    memory_multiplier: float,
    *,
    tiling_row: bool = True,
    safety_margin: float = 0.85,
    fallback: int = 16,
    min_block: int = 1,
    max_block: int | None = None,
) -> int:
    """Compute the FLA UB-safe tile along one axis of a two-dimensional tile."""
    tiled_dim = row_dim if tiling_row else fixed_dim
    fixed_size = fixed_dim if tiling_row else row_dim
    desired = triton.next_power_of_2(tiled_dim)
    try:
        block = _ub_safe_block(desired, fixed_size, memory_multiplier, safety_margin)
    except Exception:
        block = min(desired, fallback)
    block = max(min_block, block)
    if max_block is not None:
        block = min(block, max_block)
    return block
