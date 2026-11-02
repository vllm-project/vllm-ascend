# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Map and stably pack C8 selections with optional captured-forward group reuse."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _map_compact_prefix1417(
    source,
    output,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    token = tl.load(source + row.to(tl.int64) * ROW_STRIDE + column * COL_STRIDE, mask=column < WIDTH, other=-1)
    valid = (column < WIDTH) & (token >= 0) & (token < CAPACITY)
    count = tl.sum(valid.to(tl.int32), 0)
    safe = tl.maximum(token, 0).to(tl.int64)
    mapped = (safe // BLOCK * PAGE_STRIDE + safe % BLOCK).to(tl.int32)
    base = output + row.to(tl.int64) * WIDTH
    # A valid prefix already has the exact stable-compaction order. Test the
    # current device values on every replay; do not infer it from old counts.
    prefix_valid = tl.sum(((column < count) & ~valid).to(tl.int32), 0) == 0
    if prefix_valid:
        tl.store(base + column, tl.where(valid, mapped, -1), mask=column < WIDTH)
    else:
        # Arbitrary holes retain the original stable scan, including duplicates.
        destination = tl.cumsum(valid.to(tl.int32), 0) - 1
        tl.store(base + destination, mapped, mask=valid)
        tl.store(base + column, -1, mask=(column < WIDTH) & (column >= count))


def map_compact_c8_indices(
    indices: torch.Tensor, block: int, stride_rows: int, capacity: int, *, _scope=None
) -> torch.Tensor:
    """Preserve original map+stable-compaction values, including all duplicates.

    The default path owns fresh output. The private scope accepts only the
    current group resolver output during one capture; its producer kernel reads
    current values on every replay. Eager and unsupported inputs stay fresh.
    """
    if indices.ndim != 2 or indices.dtype != torch.int32 or block <= 0 or stride_rows < block:
        raise ValueError("C8 mapping requires int32 [queries,top_k] and valid page geometry")
    if capacity <= 0 or capacity % block or (capacity // block - 1) * stride_rows + block > 2**31 - 1:
        raise ValueError("C8 physical row capacity is invalid")
    width = indices.shape[1]
    if indices.device.type != "npu" or not 0 < width <= 4096:
        from vllm_ascend.ops.hisparse_c8 import compact_c8_indices, map_c8_physical_indices

        return compact_c8_indices(map_c8_physical_indices(indices, block, stride_rows, capacity))
    binding = None
    if _scope is not None and _scope.context() is not None and torch.npu.is_current_stream_capturing():
        binding = _scope.binding(indices, block, stride_rows, capacity)
        if binding is not None:
            if _scope.result is not None:
                _scope.hits += 1
                return _scope.result
            # Only current-group resolver output enters this path. The physical
            # domain is derived from its full paged backing, independent of the
            # compatible layer alias's unused tail. Kernel values update on
            # every graph replay; eager/mixed inputs keep the original path.
            capacity = _scope.physical_bound
    result = torch.empty(indices.shape, device=indices.device, dtype=torch.int32)
    if indices.shape[0]:
        _map_compact_prefix1417[(indices.shape[0],)](
            indices,
            result,
            indices.stride(0),
            indices.stride(1),
            width,
            block,
            stride_rows,
            capacity,
            triton.next_power_of_2(width),
        )
    if binding is not None:
        _scope.key = binding
        _scope.result = result
    return result
