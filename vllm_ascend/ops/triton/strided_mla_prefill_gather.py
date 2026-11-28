# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gather ragged MLA history directly into separate TND output buffers."""

import torch
import torch_npu
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit
def _gather(
    K,
    R,
    TABLE,
    CU,
    STARTS,
    OUT_K,
    OUT_R,
    batches,
    table_row_stride,
    max_pages,
    k_page_stride,
    r_page_stride,
    K_ROW: tl.constexpr,
    R_ROW: tl.constexpr,
    PAGE: tl.constexpr,
    ROWS: tl.constexpr,
    K_DIM: tl.constexpr,
    R_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    offsets = tl.arange(0, ROWS)
    k_dim = tl.arange(0, K_DIM)
    r_dim = tl.arange(0, R_DIM)
    prior_tiles = 0
    for request in range(batches):
        dst_start = tl.load(CU + request)
        dst_stop = tl.load(CU + request + 1)
        start = tl.load(STARTS + request)
        left_pad = start % ROWS
        active_rows = dst_stop - dst_start
        active_tiles = tl.where(active_rows > 0, tl.cdiv(active_rows + left_pad, ROWS), 0)
        first_tile = (pid + programs - prior_tiles % programs) % programs
        source_start = start - left_pad
        for tile in range(first_tile, active_tiles, programs):
            source = source_start + tile * ROWS
            logical = source // PAGE
            physical = tl.load(TABLE + request * table_row_stride + logical)
            valid_page = (physical >= 0) & (physical < max_pages)
            safe_page = tl.where(valid_page, physical, 0).to(tl.int64)
            k_base = K + safe_page * k_page_stride + (source % PAGE) * K_ROW
            r_base = R + safe_page * r_page_stride + (source % PAGE) * R_ROW
            payload_k = tl.load(k_base + offsets[:, None] * K_ROW + k_dim[None, :])
            payload_r = tl.load(r_base + offsets[:, None] * R_ROW + r_dim[None, :])
            payload_k = tl.where(valid_page, payload_k, 0)
            payload_r = tl.where(valid_page, payload_r, 0)
            relative = tile * ROWS + offsets - left_pad
            valid_rows = (relative >= 0) & (relative < active_rows)
            dst = tl.where(valid_rows, dst_start + relative, 0)
            tl.store(OUT_K + dst[:, None] * K_DIM + k_dim[None, :], payload_k, valid_rows[:, None])
            tl.store(OUT_R + dst[:, None] * R_DIM + r_dim[None, :], payload_r, valid_rows[:, None])
        prior_tiles += active_tiles


def try_gather_strided_prefill(key_cache, rope_cache, page_table, cumulative, starts, key, rope) -> bool:
    """Write separate TND outputs using the existing device chunk boundaries."""
    batches = starts.numel()
    if (
        not HAS_TRITON
        or key_cache.ndim != 4
        or key_cache.shape[1:] != (128, 1, 512)
        or rope_cache.shape != (key_cache.size(0), 128, 1, 64)
        or (key_cache.is_contiguous() and rope_cache.is_contiguous())
        or key_cache.dtype not in (torch.bfloat16, torch.float16)
        or rope_cache.dtype != key_cache.dtype
        or key_cache.stride(-1) != 1
        or rope_cache.stride(-1) != 1
        or page_table.ndim != 2
        or page_table.size(0) < batches
        or page_table.stride(1) != 1
        or page_table.dtype != torch.int32
        or cumulative.shape != (batches + 1,)
        or cumulative.dtype != torch.int32
        or not cumulative.is_contiguous()
        or starts.shape != (batches,)
        or starts.dtype != torch.int32
        or not starts.is_contiguous()
        or key.ndim != 3
        or key.shape != (key.size(0), 1, 512)
        or rope.shape != (key.size(0), 1, 64)
        or key.dtype != key_cache.dtype
        or rope.dtype != key_cache.dtype
        or not key.is_contiguous()
        or not rope.is_contiguous()
        or key_cache.device.type != "npu"
        or any(x.device != key_cache.device for x in (rope_cache, page_table, cumulative, starts, key, rope))
        or any(torch_npu.get_npu_format(x) not in (0, 2) for x in (key_cache, rope_cache))
    ):
        return False
    if not key.numel():
        return True
    if not batches or not page_table.size(1):
        return False
    init_device_properties_triton()
    cores = get_vectorcore_num()
    _gather[(cores,)](
        key_cache,
        rope_cache,
        page_table,
        cumulative,
        starts,
        key,
        rope,
        batches,
        page_table.stride(0),
        key_cache.size(0),
        key_cache.stride(0),
        rope_cache.stride(0),
        key_cache.stride(1),
        rope_cache.stride(1),
        key_cache.size(1),
        8,
        key_cache.size(-1),
        rope_cache.size(-1),
    )
    return True
