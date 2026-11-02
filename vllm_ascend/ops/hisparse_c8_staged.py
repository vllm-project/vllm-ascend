# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Fresh logical-to-staging mapping and stable C8 index packing in one call."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _staged_map_compact(
    logical,
    tables,
    requests,
    output,
    LOGICAL_ROW: tl.constexpr,
    LOGICAL_COL: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    selected = tl.load(logical + row.to(tl.int64) * LOGICAL_ROW + column * LOGICAL_COL, column < WIDTH, other=-1).to(
        tl.int64
    )
    request = tl.load(requests + row.to(tl.int64) * REQUEST_STRIDE).to(tl.int64)
    request = tl.where(request < 0, request + TABLE_ROWS, request)
    block = tl.maximum(selected, 0) // BLOCK_SIZE
    # Positive logical indices and request rows are bounded by the existing
    # staging-plan contract. Mask invalid blocks defensively before loading.
    readable = (column < WIDTH) & (selected >= 0) & (block < TABLE_COLS) & (request >= 0) & (request < TABLE_ROWS)
    page = tl.load(tables + request * TABLE_ROW + block * TABLE_COL, readable, other=-1)
    # The old Torch chain stores physical rows as int32 before C8 range checks.
    physical = (page.to(tl.int64) * BLOCK_SIZE + tl.maximum(selected, 0) % BLOCK_SIZE).to(tl.int32)
    valid = readable & (physical >= 0) & (physical < CAPACITY)
    position = tl.cumsum(valid.to(tl.int32), 0) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    base = output + row.to(tl.int64) * WIDTH
    tl.store(base + position, physical, valid)
    tl.store(base + column, -1, (column < WIDTH) & (column >= count))


def prepare_staged_c8_indices(logical, tables, requests, cache, block_size):
    """Return a newly owned, stably packed selection or retain the old path.

    Inputs belong to the current Prefill staging plan. Request rows must index
    tables, and nonnegative logical indices must address its block columns,
    as required by the original Torch gather. Values, group-local addresses,
    table strides and request order are consumed afresh on every invocation.
    Only contiguous C8 staging has an identity row layout after translation.
    """
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    if get_ascend_device_type() != AscendDeviceType.A5:
        return None
    if not (logical.device.type == "npu" and logical.device == tables.device == requests.device == cache.device):
        return None
    if not (
        logical.ndim == 2
        and logical.dtype == torch.int32
        and 0 < logical.shape[1] <= 4096
        and tables.ndim == 2
        and tables.shape[0] > 0
        and tables.shape[1] > 0
        and tables.dtype in (torch.int32, torch.int64)
        and requests.ndim == 1
        and requests.numel() == logical.shape[0]
        and requests.dtype in (torch.int32, torch.int64)
        and cache.dtype == torch.float8_e4m3fn
        and cache.ndim == 3
        and cache.shape[0] > 0
        and cache.shape[1:] == (block_size, 656)
        and cache.stride() == (block_size * 656, 656, 1)
        and block_size > 0
        and cache.shape[0] * block_size <= 2**31 - 1
    ):
        return None
    result = torch.empty(logical.shape, dtype=torch.int32, device=logical.device)
    if logical.shape[0]:
        _staged_map_compact[(logical.shape[0],)](
            logical,
            tables,
            requests,
            result,
            logical.stride(0),
            logical.stride(1),
            tables.stride(0),
            tables.stride(1),
            tables.shape[1],
            tables.shape[0],
            requests.stride(0),
            block_size,
            cache.shape[0] * block_size,
            logical.shape[1],
            triton.next_power_of_2(logical.shape[1]),
        )
    return result
