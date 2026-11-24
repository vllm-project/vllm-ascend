# SPDX-License-Identifier: Apache-2.0
"""Copy one contiguous lane range from each logical row of the HY4 cache."""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.device.device_config import is_950

_PAGE_ROWS = 64
_CACHE_ROW_BYTES = 1000
_WORD_BYTES = 4


def supports_paged_cache_copy(cache, table, block_size):
    return (
        HAS_TRITON
        and cache.device.type == "npu"
        and is_950()
        and cache.dtype == torch.uint8
        and cache.ndim == 2
        and cache.is_contiguous()
        and cache.size(-1) == _CACHE_ROW_BYTES
        and cache.storage_offset() % _WORD_BYTES == 0
        and block_size == _PAGE_ROWS
        and table.dtype in (torch.int32, torch.int64)
        and table.ndim == 1
        and table.is_contiguous()
        and table.device == cache.device
    )


@triton.jit(do_not_specialize=["rows"])
def _copy_paged_cache_lanes(
    cache,
    table,
    output,
    rows,
    PAGE_ROWS: tl.constexpr,
    CACHE_WORDS: tl.constexpr,
    FIRST_WORD: tl.constexpr,
    WORDS: tl.constexpr,
    TILE_WORDS: tl.constexpr,
):
    logical_page = tl.program_id(0)
    physical_page = tl.load(table + logical_page).to(tl.int64)
    in_page = tl.arange(0, PAGE_ROWS)
    logical_row = logical_page * PAGE_ROWS + in_page
    physical_row = physical_page * PAGE_ROWS + in_page
    word = tl.arange(0, TILE_WORDS)
    mask = (logical_row[:, None] < rows) & (word[None, :] < WORDS)
    data = tl.load(cache + physical_row[:, None] * CACHE_WORDS + FIRST_WORD + word[None, :], mask, other=0)
    tl.store(output + logical_row[:, None] * WORDS + word[None, :], data, mask)


def copy_paged_cache_lanes(cache, table, rows, first_byte, row_bytes):
    output = torch.empty((rows, row_bytes), device=cache.device, dtype=torch.uint8)
    words = row_bytes // _WORD_BYTES
    _copy_paged_cache_lanes[(triton.cdiv(rows, _PAGE_ROWS),)](
        cache.view(torch.int32),
        table,
        output.view(torch.int32),
        rows,
        _PAGE_ROWS,
        _CACHE_ROW_BYTES // _WORD_BYTES,
        first_byte // _WORD_BYTES,
        words,
        triton.next_power_of_2(words),
        num_warps=1,
    )
    return output
