# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unfold causal short-query paged attention rows into single-query TND FIA rows."""

from collections.abc import Sequence
from typing import Any

import torch

_MAX_INDEX_CACHE_ENTRIES = 256


def is_all_single_query(query_ends: Sequence[int], num_tokens: int) -> bool:
    """Return True iff query_ends partitions [0, num_tokens) into unit-width queries."""
    return num_tokens > 0 and len(query_ends) == num_tokens and list(query_ends) == list(range(1, num_tokens + 1))


def single_query_layout(
    query_ends: Sequence[int],
    kv_lengths: Sequence[int],
    max_expand_width: int,
) -> tuple[list[int], list[int], list[int]]:
    """Expand causal paged rows with 1 < query width <= max_expand_width into single-query entries."""
    new_query_ends: list[int] = []
    new_kv_lengths: list[int] = []
    page_table_rows: list[int] = []
    start = 0
    for row, (end, kv_length) in enumerate(zip(query_ends, kv_lengths, strict=True)):
        width = end - start
        if 1 < width <= max_expand_width:
            assert kv_length >= width
            new_query_ends.extend(range(start + 1, end + 1))
            new_kv_lengths.extend(range(kv_length - width + 1, kv_length + 1))
            page_table_rows.extend([row] * width)
        else:
            new_query_ends.append(end)
            new_kv_lengths.append(kv_length)
            page_table_rows.append(row)
        start = end
    return new_query_ends, new_kv_lengths, page_table_rows


def _select_block_table_rows(
    block_table: torch.Tensor,
    rows: list[int],
    device_index_cache: dict[Any, torch.Tensor] | None = None,
) -> torch.Tensor:
    if rows == list(range(len(rows))) and len(rows) <= block_table.shape[0]:
        return block_table[: len(rows)]
    if not isinstance(device_index_cache, dict):
        device_index_cache = None
    device = block_table.device
    cache_key = (tuple(rows), device)
    if device_index_cache is not None and cache_key in device_index_cache:
        indices = device_index_cache[cache_key]
    else:
        indices = torch.tensor(rows, dtype=torch.long)
        if device.type != "cpu":
            indices = indices.pin_memory().to(device, non_blocking=True)
        if device_index_cache is not None:
            if len(device_index_cache) >= _MAX_INDEX_CACHE_ENTRIES:
                device_index_cache.pop(next(iter(device_index_cache)))
            device_index_cache[cache_key] = indices
    return block_table.index_select(0, indices)


def prepare_fia_layout(
    metadata: Any,
    query_ends: list[int],
    kv_lengths: list[int],
    block_table: torch.Tensor | None,
    device_index_cache: dict[Any, torch.Tensor] | None = None,
) -> tuple[list[int], list[int], torch.Tensor | None]:
    """Return unfolded TND FIA metadata for eager and piecewise paged attention."""
    raw_expand_width = getattr(metadata, "fia_max_expand_width", 0)
    max_expand_width = (
        raw_expand_width if isinstance(raw_expand_width, int) and not isinstance(raw_expand_width, bool) else 0
    )
    if (
        block_table is None
        or max_expand_width <= 1
        or not query_ends
        or not isinstance(getattr(metadata, "num_actual_tokens", None), int)
        or query_ends[-1] > metadata.num_actual_tokens
    ):
        return query_ends, kv_lengths, block_table
    cache_key = (
        tuple(query_ends),
        tuple(kv_lengths),
        block_table.data_ptr(),
        int(block_table.shape[0]),
        max_expand_width,
    )
    cached = getattr(metadata, "fia_single_query", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1], cached[2], cached[3]
    ends, lengths, rows = single_query_layout(query_ends, kv_lengths, max_expand_width)
    if len(rows) == len(query_ends):
        result = (query_ends, kv_lengths, block_table)
    else:
        result = (ends, lengths, _select_block_table_rows(block_table, rows, device_index_cache))
    metadata.fia_single_query = (cache_key, *result)
    return result


def prepare_graph_fia_layout(
    query_ends: Sequence[int],
    kv_lengths: Sequence[int],
    block_table: torch.Tensor,
    num_actual_tokens: int,
    num_input_tokens: int | None = None,
    device_index_cache: dict[Any, torch.Tensor] | None = None,
) -> tuple[list[int], list[int], torch.Tensor]:
    """Unfold uniform, varlen, or short-extend decode-graph batches into single-query TND FIA rows."""
    target_tokens = num_input_tokens if num_input_tokens is not None else (query_ends[-1] if query_ends else 0)
    lengths: list[int] = []
    rows: list[int] = []
    start = 0
    for row, (end, kv_length) in enumerate(zip(query_ends, kv_lengths, strict=True)):
        width = end - start
        if width == 0:
            continue
        assert width > 0
        if end <= num_actual_tokens:
            assert kv_length >= width
            lengths.extend(range(kv_length - width + 1, kv_length + 1))
            rows.extend([row] * width)
        elif start < num_actual_tokens:
            live_width = num_actual_tokens - start
            assert kv_length >= live_width
            lengths.extend(range(kv_length - live_width + 1, kv_length + 1))
            rows.extend([row] * live_width)
            pad_width = end - num_actual_tokens
            lengths.extend([1] * pad_width)
            rows.extend([row] * pad_width)
        else:
            # Padded query rows do not read a live request or write KV cache.
            lengths.extend([1] * width)
            rows.extend([row] * width)
        start = end
    assert start >= num_actual_tokens
    if target_tokens > start:
        pad_tail = target_tokens - start
        lengths.extend([1] * pad_tail)
        rows.extend([0] * pad_tail)
        start = target_tokens
    table = _select_block_table_rows(block_table, rows, device_index_cache)
    return list(range(1, start + 1)), lengths, table
