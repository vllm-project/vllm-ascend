# SPDX-License-Identifier: Apache-2.0
"""A device-side plan for distributing live KV pages over A3 Cube cores."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _balance_pages(
    LENGTHS,
    SCHEDULE,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
    CORES: tl.constexpr,
    QUERY: tl.constexpr,
    CAUSAL: tl.constexpr,
    LANES: tl.constexpr,
):
    batch = tl.arange(0, LANES)
    length = tl.load(LENGTHS + batch, batch < BATCH, other=0)
    length = tl.where((length > 0) & (length <= WIDTH * 128), length, 0)
    pages = tl.cdiv(length, 128)
    units = pages
    if CAUSAL and QUERY > 1:
        tail = length % 128
        join_tail = (pages > 1) & (tail > 0) & (tail < QUERY)
        units -= join_tail.to(tl.int32)

    # Compare extra work in the largest unsplit stream against a two-page
    # reduction budget. One unit then covers a complete physical request.
    largest = tl.max(units, axis=0)
    average = tl.cdiv(tl.sum(units, axis=0), CORES)
    avoid_reduction = (BATCH <= CORES) & (largest <= average + 2)
    units = tl.where(avoid_reduction & (units > 0), 1, units)
    end = tl.cumsum(units, axis=0)
    first = end - units
    total = tl.sum(units, axis=0)
    active = tl.minimum(total, CORES)
    divisor = tl.maximum(total, 1)
    # Invert the floor division that assigns a contiguous interval to each
    # core. A request crossing a boundary contributes one partial per core.
    first_core = ((first.to(tl.int64) + 1) * active - 1) // divisor
    last_core = (end.to(tl.int64) * active - 1) // divisor
    first_core = tl.where(units > 0, first_core, 0).to(tl.int32)
    splits = tl.where(units > 0, last_core - first_core + 1, 0).to(tl.int32)

    field = tl.arange(0, 4)
    needs_finish = (splits != 1) | ((length < QUERY) if CAUSAL else (length <= 0))
    any_finish = tl.sum(((batch < BATCH) & needs_finish).to(tl.int32), axis=0) > 0
    header = tl.where(field == 0, total, tl.where(field == 1, active, tl.where(field == 2, any_finish.to(tl.int32), 1)))
    tl.store(SCHEDULE + field, header)
    values = tl.where(field[None, :] == 0, first[:, None], units[:, None])
    values = tl.where(field[None, :] == 2, first_core[:, None], values)
    values = tl.where(field[None, :] == 3, splits[:, None], values)
    tl.store(SCHEDULE + (batch[:, None] + 1) * 4 + field[None, :], values, batch[:, None] < BATCH)


@triton.jit
def _legacy_page_ranges(
    LENGTHS,
    SCHEDULE,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
    CORES: tl.constexpr,
    QUERY: tl.constexpr,
    CAUSAL: tl.constexpr,
    LANES: tl.constexpr,
):
    batch = tl.arange(0, LANES)
    length = tl.load(LENGTHS + batch, batch < BATCH, other=0)
    length = tl.where((length > 0) & (length <= WIDTH * 128), length, 0)
    pages = tl.cdiv(length, 128)
    units = pages
    if CAUSAL and QUERY > 1:
        tail = length % 128
        join_tail = (pages > 1) & (tail > 0) & (tail < QUERY)
        units -= join_tail.to(tl.int32)

    end = tl.cumsum(units, axis=0)
    first = end - units
    total = tl.sum(units, axis=0)
    active = tl.minimum(total, CORES)
    divisor = tl.maximum(total, 1)
    # Invert the floor division that assigns a contiguous interval to each
    # core. A request crossing a boundary contributes one partial per core.
    first_core = ((first.to(tl.int64) + 1) * active - 1) // divisor
    last_core = (end.to(tl.int64) * active - 1) // divisor
    first_core = tl.where(units > 0, first_core, 0).to(tl.int32)
    splits = tl.where(units > 0, last_core - first_core + 1, 0).to(tl.int32)

    field = tl.arange(0, 4)
    header = tl.where(field == 0, total, tl.where(field == 1, active, 0))
    tl.store(SCHEDULE + field, header)
    values = tl.where(field[None, :] == 0, first[:, None], units[:, None])
    values = tl.where(field[None, :] == 2, first_core[:, None], values)
    values = tl.where(field[None, :] == 3, splits[:, None], values)
    tl.store(SCHEDULE + (batch[:, None] + 1) * 4 + field[None, :], values, batch[:, None] < BATCH)


def schedule_pages(lengths, page_table_width, cores, *, query_width=1, causal=True, out=None, direct_output=False):
    """Build the A3 page schedule from persistent int32 device lengths.

    The default layout is the original [total_units, active_cores, 0, 0] header
    and one [first_unit, unit_count, first_core, split_count] record per request.
    An eligible MTP tail shares the final unit with the previous page.

    With direct_output enabled, the decode kernel can give a short request
    one owner and finish it in NTD layout. Header word 2 requests the final
    merge or padding pass; word 3 enables this direct-output contract.
    """
    batch = lengths.numel()
    if (
        lengths.ndim != 1
        or lengths.dtype != torch.int32
        or lengths.device.type != "npu"
        or not lengths.is_contiguous()
        or not 0 < batch <= 256
    ):
        raise ValueError("expected 1 to 256 contiguous int32 device lengths")
    if not 0 < page_table_width <= (2**31 - 1) // max(batch, 128) or not 0 < cores <= 64 or query_width not in (1, 4):
        raise ValueError("invalid page schedule geometry")
    size = (batch + 1, 4)
    if out is None:
        out = lengths.new_empty(size)
    if out.shape != size or out.dtype != lengths.dtype or out.device != lengths.device or not out.is_contiguous():
        raise ValueError("invalid page schedule output buffer")
    if not direct_output:
        _legacy_page_ranges[(1,)](
            lengths,
            out,
            batch,
            page_table_width,
            cores,
            query_width,
            bool(causal),
            triton.next_power_of_2(batch),
            num_stages=1,
        )
        return out
    _balance_pages[(1,)](
        lengths,
        out,
        batch,
        page_table_width,
        cores,
        query_width,
        bool(causal),
        triton.next_power_of_2(batch),
        num_stages=1,
    )
    return out
