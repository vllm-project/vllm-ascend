# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 Flash prefill ops indexer."""

from functools import lru_cache

import torch
from vllm.models.hy_v4_flash.reference import FP8_MAX, INDEX_DIM, sparse_candidates
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.device.device_config import is_950

from .cache import copy_paged_cache_lanes, supports_paged_cache_copy


@triton.jit(do_not_specialize=["First", "Past", "Current", "Rows", "Destination", "HistoryIndex"])
def _assemble_index_pool_groups(
    Keys,
    CacheKeys,
    CachePositions,
    BlockTable,
    Raw,
    GroupMap,
    HistoryPositions,
    First,
    Past,
    Current,
    Rows,
    Destination,
    HistoryIndex,
    PAGE_ROWS: tl.constexpr,
    KEY_ROW: tl.constexpr,
    POSITION_ROW: tl.constexpr,
    KEY_START: tl.constexpr,
    POSITION_START: tl.constexpr,
    AXES: tl.constexpr,
    WITH_HISTORY: tl.constexpr,
    TILE: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tile * TILE + tl.arange(0, TILE)
    channel = tl.arange(0, 128)
    if WITH_HISTORY and tile == 0:
        is_history = row < Past
        token = First + row
        page = tl.load(BlockTable + token // PAGE_ROWS, is_history, other=0).to(tl.int64)
        slot = page * PAGE_ROWS + token % PAGE_ROWS
        history = tl.load(
            CacheKeys + slot[:, None] * KEY_ROW + KEY_START + channel[None, :],
            is_history[:, None],
            other=0,
        )
        current = tl.load(
            Keys + (Current + row[:, None] - Past) * 128 + channel[None, :],
            ((row < Rows) & ~is_history)[:, None],
            other=0,
        )
        key = tl.where(is_history[:, None], history, current)
        first_page = tl.load(BlockTable + First // PAGE_ROWS).to(tl.int64)
        first_slot = first_page * PAGE_ROWS + First % PAGE_ROWS
        axis = tl.arange(0, 1 if AXES == 1 else 4)
        coordinate = tl.load(
            CachePositions + first_slot * POSITION_ROW + POSITION_START + axis,
            axis < AXES,
            other=0,
        )
        tl.store(HistoryPositions + HistoryIndex * AXES + axis, coordinate, axis < AXES)
    else:
        key = tl.load(
            Keys + (Current + row[:, None] - Past) * 128 + channel[None, :],
            row[:, None] < Rows,
            other=0,
        )
    tl.store(Raw + (Destination + row[:, None]) * 128 + channel[None, :], key.to(tl.float32), row[:, None] < Rows)

    group = tile * (TILE // 4) + tl.arange(0, TILE // 4)
    first_row = group * 4
    # Negative entries address the compact history instead of the current batch.
    source = tl.where(first_row < Past, -1 - HistoryIndex, Current + first_row - Past)
    tl.store(GroupMap + Destination // 4 + group, source, first_row < Rows)


def assemble_index_pool_groups(cache, block_table, block_size, key_offset, keys, positions_ndim, segments):
    axes = 1 if positions_ndim == 1 else 3
    total_rows = sum(rows for _, _, _, _, rows in segments)
    history_count = sum(past > 0 for _, _, past, _, _ in segments)
    raw = torch.empty((total_rows, 128), dtype=torch.float32, device=keys.device)
    group_map = torch.empty(total_rows // 4, dtype=torch.int32, device=keys.device)
    history = torch.empty((history_count, axes), dtype=torch.int64, device=keys.device)
    key_cache = cache.view(torch.bfloat16)
    position_cache = cache.view(torch.int64)
    tile = 32
    destination = history_index = 0
    for request, first_token, past, first_new, rows in segments:
        _assemble_index_pool_groups[(triton.cdiv(rows, tile),)](
            keys,
            key_cache,
            position_cache,
            block_table[request],
            raw,
            group_map,
            history,
            first_token,
            past,
            first_new,
            rows,
            destination,
            history_index,
            block_size,
            key_cache.stride(0),
            position_cache.stride(0),
            key_offset // 2,
            (key_offset + 256) // 8,
            axes,
            past > 0,
            tile,
        )
        destination += rows
        history_index += past > 0
    return raw, group_map, history[:, 0] if axes == 1 else history.T


def _pool_key_addresses(cache, block_table, groups, live, block_size, ratio):
    lanes_per_row = INDEX_DIM // ratio
    lane = torch.arange(INDEX_DIM, device=cache.device)
    token = (groups * ratio).masked_fill(~live, 0)
    block = block_table.index_select(0, token // block_size).long()
    slot = block * block_size + token % block_size
    row_width = cache.shape[-1]
    row = slot[:, None] + lane[None, :] // lanes_per_row
    column = row_width - lanes_per_row + lane % lanes_per_row
    return row * row_width + column


def read_prefill_pool(cache, block_table, groups, block_size, ratio):
    """Read full IndexPool groups from the BF16 lanes at the end of each cache row."""
    lanes_per_row = INDEX_DIM // ratio
    groups_per_page = block_size // ratio
    if (
        ratio == 4
        and 0 < groups <= block_table.numel() * groups_per_page
        and supports_paged_cache_copy(cache, block_table, block_size)
    ):
        row_bytes = lanes_per_row * torch.bfloat16.itemsize
        packed = copy_paged_cache_lanes(cache, block_table, groups * ratio, cache.size(-1) - row_bytes, row_bytes)
        return packed.view(torch.bfloat16).view(groups, INDEX_DIM)
    cache = cache.view(torch.bfloat16)
    pages_to_read = (groups + groups_per_page - 1) // groups_per_page
    page_lanes = cache.view(-1, block_size, cache.shape[-1])[..., -lanes_per_row:]
    pages = page_lanes.index_select(0, block_table[:pages_to_read])
    return pages.reshape(-1, INDEX_DIM)[:groups]


def gather_pool_keys(cache, block_table, groups, live, block_size, ratio):
    """Gather the IndexPool lanes that are spread across each group's rows."""
    cache = cache.view(torch.bfloat16)
    addresses = _pool_key_addresses(cache, block_table, groups, live, block_size, ratio)
    addresses = addresses.masked_fill(~live[:, None], 0)
    values = cache.flatten().index_select(0, addresses.flatten()).view(-1, INDEX_DIM)
    return values.masked_fill(~live[:, None], 0)


def copy_pool_keys(cache, block_table, keys, first_group, block_size, ratio, *, write, end_group=None):
    if not keys.shape[0]:
        return
    if (
        write
        and end_group is None
        and ratio == 4
        and isinstance(first_group, int)
        and first_group >= 0
        and (first_group + keys.shape[0]) * ratio <= block_table.numel() * block_size
        and keys.ndim == 2
        and keys.shape[1] == INDEX_DIM
        and keys.is_contiguous()
        and keys.dtype in (torch.bfloat16, torch.float32)
        and keys.device == cache.device
        and supports_paged_cache_copy(cache, block_table, block_size)
    ):
        from ..indexer import copy_pool_keys as write_paged_pool

        write_paged_pool(cache, block_table, keys, first_group, block_size, ratio, write=True)
        return
    groups = torch.arange(keys.shape[0], device=cache.device) + first_group
    live = groups >= 0 if end_group is None else (groups >= 0) & (groups < end_group)
    if write:
        cache = cache.view(torch.bfloat16)
        addresses = _pool_key_addresses(cache, block_table, groups, live, block_size, ratio)
        # Offset zero is outside the IndexPool lanes, even on a dummy page.
        addresses = addresses.masked_fill(~live[:, None], 0)
        flat = cache.flatten()
        values = torch.where(live[:, None], keys.bfloat16(), flat[:1])
        flat.index_copy_(0, addresses.flatten(), values.flatten())
    else:
        keys.copy_(gather_pool_keys(cache, block_table, groups, live, block_size, ratio))


_MIN_AMAX = 1e-4
_MIN_NORMAL = 2.0**-6
_FP32_FRACTION_BITS = 23
_FP32_EXPONENT_BIAS = 127
_E4M3_FRACTION_BITS = 3


def encode_indexpool_fp8(vectors: torch.Tensor) -> torch.Tensor:
    """Return FP32 Hadamard rows rounded to the original E4M3 bins."""
    if not vectors.numel():
        return vectors.float()
    width = vectors.shape[-1]
    values = vectors.float()
    stride = 1
    while stride < width:
        left, right = values.reshape(-1, 2, stride).unbind(1)
        values = torch.stack((left + right, left - right), dim=1).reshape(vectors.shape)
        stride *= 2
    values = values * width**-0.5
    magnitude = values.abs()
    maxima = magnitude.amax(dim=-1, keepdim=True)
    maxima = maxima.masked_fill(~(magnitude < torch.inf).all(dim=-1, keepdim=True), torch.nan)
    scale = torch.exp2(torch.ceil(torch.log2(maxima.clamp_min(_MIN_AMAX) / FP8_MAX)))
    magnitude = (values / scale).abs().clamp_max(FP8_MAX)
    bits = magnitude.clamp_min(_MIN_NORMAL).contiguous().view(torch.int32)
    exponent = (bits >> _FP32_FRACTION_BITS) - _FP32_EXPONENT_BIAS - _E4M3_FRACTION_BITS
    unit = ((exponent + _FP32_EXPONENT_BIAS) << _FP32_FRACTION_BITS).view(torch.float32)
    inverse_unit = ((_FP32_EXPONENT_BIAS - exponent) << _FP32_FRACTION_BITS).view(torch.float32)
    sign = (values > 0).float() - (values < 0).float()
    result = (magnitude * inverse_unit).round() * unit * sign * scale
    return result.masked_fill(scale.isnan(), torch.nan)


@triton.jit(do_not_specialize=["rows"])
def _round_prefill_fp8_rows(
    transformed,
    output,
    rows,
    TILE_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    NORMALIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_AMAX: tl.constexpr,
):
    row = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    lane = tl.arange(0, WIDTH)
    offset = row[:, None] * WIDTH + lane[None, :]
    data = tl.load(transformed + offset, row[:, None] < rows, other=0) * NORMALIZE
    magnitude = tl.abs(data)
    valid = tl.sum((~(magnitude < float("inf"))).to(tl.int32), axis=1) == 0
    amax = tl.maximum(tl.max(magnitude, axis=1), MIN_AMAX)
    exponent = tl.ceil(tl.log2(tl.div_rn(amax, FP8_MAX))).to(tl.int32)
    scale = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
    inverse_scale = ((127 - exponent) << 23).to(tl.float32, bitcast=True)

    scaled = tl.minimum(magnitude * inverse_scale[:, None], FP8_MAX)
    # E4M3 uses three fraction bits and a fixed step below the smallest normal.
    normal = tl.maximum(scaled, 2.0**-6).to(tl.int32, bitcast=True)
    unit_exponent = (normal >> 23) - 127 - 3
    unit = ((unit_exponent + 127) << 23).to(tl.float32, bitcast=True)
    inverse_unit = ((127 - unit_exponent) << 23).to(tl.float32, bitcast=True)
    rounded = tl.extra.cann.libdevice.nearbyint(scaled * inverse_unit)
    restored = rounded * unit * scale[:, None]
    sign = data.to(tl.int32, bitcast=True) & -2147483648
    signed = (restored.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True)
    result = tl.where(valid[:, None], signed, float("nan"))
    tl.store(output + offset, result, row[:, None] < rows)


def round_prefill_fp8_rows(transformed: torch.Tensor, *, output_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Round the FP32 Hadamard output to the IndexPool E4M3 grid."""
    rows = transformed.shape[0]
    result = torch.empty_like(transformed, dtype=output_dtype)
    tile_rows = 32
    _round_prefill_fp8_rows[(triton.cdiv(rows, tile_rows),)](
        transformed,
        result,
        rows,
        tile_rows,
        INDEX_DIM,
        INDEX_DIM**-0.5,
        FP8_MAX,
        _MIN_AMAX,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return result


@triton.jit(do_not_specialize=["first_token"])
def _read_index_history(
    cache,
    table,
    keys,
    positions,
    first_token,
    PAGE_ROWS: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    KEY_OFFSET: tl.constexpr,
    KEY_WORDS: tl.constexpr,
):
    row = tl.program_id(0)
    token = first_token + row
    page = tl.load(table + token // PAGE_ROWS).to(tl.int64)
    slot = page * PAGE_ROWS + token % PAGE_ROWS
    source = cache + slot * ROW_WORDS + KEY_OFFSET
    key = tl.arange(0, KEY_WORDS)
    tl.store(keys + row * KEY_WORDS + key, tl.load(source + key))
    coord = tl.arange(0, 8)
    valid_coord = coord < 6
    coord_words = tl.load(source + KEY_WORDS + coord, valid_coord, other=0)
    tl.store(positions + row * 6 + coord, coord_words, valid_coord)


def read_index_history(cache, table, block_size, first_token, count, key_offset, position_ndim):
    keys = torch.empty((count, 128), dtype=torch.bfloat16, device=cache.device)
    positions = torch.empty((count, 3), dtype=torch.int64, device=cache.device)
    _read_index_history[(count,)](
        cache.view(torch.int32),
        table,
        keys.view(torch.int32),
        positions.view(torch.int32),
        first_token,
        block_size,
        cache.size(1) // 4,
        key_offset // 4,
        64,
    )
    return keys, positions[:, 0] if position_ndim == 1 else positions.T


@triton.jit(do_not_specialize=["rows", "epsilon"])
def _index_pool_mean_square(keys, stats, rows, epsilon, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    channel = tl.arange(0, 128)
    valid = row < rows
    key = tl.load(keys + row[:, None] * 128 + channel[None, :], valid[:, None], other=0).to(tl.float32)
    mean = tl.sum(key * key, axis=1) * (1.0 / 128.0)
    tl.store(stats + row, mean + epsilon, valid)


def token_key_norm_inverse(keys, epsilon):
    rows = keys.shape[0]
    stats = torch.empty(rows, dtype=torch.float32, device=keys.device)
    tile = 16
    _index_pool_mean_square[(triton.cdiv(rows, tile),)](keys, stats, rows, epsilon, tile, enable_fp_fusion=False)
    return stats.rsqrt()


@lru_cache(maxsize=32)
def rope_constants(device, stream, theta, rotary_dim, sections, interleaved):
    frequency = theta ** (-torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
    if sections and interleaved:
        channels = torch.arange(rotary_dim // 2, device=device)
        height = (channels % 3 == 1) & (channels < sections[1] * 3)
        width = (channels % 3 == 2) & (channels < sections[2] * 3)
        return frequency, height, width
    return frequency, None, None


def _make_factors(positions, theta, rotary_dim, sections, interleaved, stream):
    frequency, height, width = rope_constants(positions.device, stream, theta, rotary_dim, sections, interleaved)
    phase = positions.float().unsqueeze(-1) * frequency
    if positions.ndim == 2:
        if interleaved:
            phase = torch.where(width, phase[2], torch.where(height, phase[1], phase[0]))
        else:
            phase = torch.cat([part[axis] for axis, part in enumerate(phase.split(sections, dim=-1))], dim=-1)
    return phase.cos(), phase.sin()


@triton.jit(do_not_specialize=["rows"])
def _index_pool_rope_phase(
    positions,
    frequency,
    phase,
    rows,
    ROW_STRIDE: tl.constexpr,
    AXIS_STRIDE: tl.constexpr,
    HEIGHT_LIMIT: tl.constexpr,
    WIDTH_LIMIT: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    row = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    col = tl.arange(0, 32)
    axis = tl.where((col % 3 == 2) & (col < WIDTH_LIMIT), 2, tl.where((col % 3 == 1) & (col < HEIGHT_LIMIT), 1, 0))
    position = tl.load(
        positions + row[:, None] * ROW_STRIDE + axis[None, :] * AXIS_STRIDE,
        row[:, None] < rows,
        other=0,
    ).to(tl.float32)
    inverse = tl.load(frequency + col)
    tl.store(phase + row[:, None] * 32 + col[None, :], position * inverse[None, :], row[:, None] < rows)


def make_factors(positions, theta, rotary_dim, sections, interleaved, stream):
    if (
        positions.device.type != "npu"
        or positions.dtype not in (torch.int32, torch.int64)
        or rotary_dim != 64
        or not 0 < positions.shape[-1] <= 4096
        or not (positions.ndim == 2 and positions.shape[0] == 3 and interleaved and sections)
    ):
        return _make_factors(positions, theta, rotary_dim, sections, interleaved, stream)
    frequency = rope_constants(positions.device, stream, theta, rotary_dim, sections, interleaved)[0]
    rows = positions.shape[-1]
    phase = torch.empty((rows, 32), dtype=torch.float32, device=positions.device)
    tile = 32
    _index_pool_rope_phase[(triton.cdiv(rows, tile),)](
        positions,
        frequency,
        phase,
        rows,
        positions.stride(-1),
        positions.stride(0),
        sections[1] * 3,
        sections[2] * 3,
        tile,
    )
    return phase.cos(), phase.sin()


@triton.jit(do_not_specialize=["Rows", "Start"])
def _index_pool_rotate_rows(
    Vectors,
    Cos,
    Sin,
    Out,
    Rows,
    Heads: tl.constexpr,
    Start,
    Step: tl.constexpr,
    Block: tl.constexpr,
    GroupMap,
    HistoryCos,
    HistorySin,
    MAPPED: tl.constexpr,
    WITH_HISTORY: tl.constexpr,
    NormWeight,
    NormInverse,
    APPLY_NORM: tl.constexpr,
):
    row = tl.program_id(0) * Block + tl.arange(0, Block)
    col = tl.arange(0, 64)
    angle = tl.arange(0, 32)
    base = row[:, None] * 128
    mask = row[:, None] < Rows
    rope = tl.load(Vectors + base + 64 + col[None, :], mask, other=0).to(tl.float32)
    if APPLY_NORM:
        inverse = tl.load(NormInverse + row, row < Rows, other=0)
        rotary_weight = tl.load(NormWeight + 64 + col)
        rope = (rope * inverse[:, None]) * rotary_weight[None, :]
        rope = tl.cast(rope, tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    even, odd = tl.split(tl.reshape(rope, (Block, 32, 2)))
    if MAPPED:
        pos = tl.load(GroupMap + row, row < Rows, other=0)
    else:
        pos = Start + row // Heads * Step
    offset = pos[:, None] * 32 + angle[None, :]
    if WITH_HISTORY:
        is_history = pos[:, None] < 0
        history_offset = (-1 - pos[:, None]) * 32 + angle[None, :]
        cos = tl.load(Cos + offset, mask & ~is_history, other=0)
        sin = tl.load(Sin + offset, mask & ~is_history, other=0)
        old_cos = tl.load(HistoryCos + history_offset, mask & is_history, other=0)
        old_sin = tl.load(HistorySin + history_offset, mask & is_history, other=0)
        cos = tl.where(is_history, old_cos, cos)
        sin = tl.where(is_history, old_sin, sin)
    else:
        cos = tl.load(Cos + offset, mask, other=0)
        sin = tl.load(Sin + offset, mask, other=0)
    even_rotated = even * cos - odd * sin
    odd_rotated = odd * cos + even * sin
    nope = tl.load(Vectors + base + col[None, :], mask, other=0)
    if APPLY_NORM:
        nope_weight = tl.load(NormWeight + col)
        nope = (nope * inverse[:, None]) * nope_weight[None, :]
    tl.store(Out + base + col[None, :], nope, mask)
    tl.store(Out + base + 64 + angle[None, :], even_rotated, mask)
    tl.store(Out + base + 96 + angle[None, :], odd_rotated, mask)


def apply_factors(vectors, factors, start=0, step=1, *, group_map=None, history_factors=None, normalization=None):
    out = torch.empty_like(vectors) if normalization is None else torch.empty_like(vectors, dtype=torch.bfloat16)
    heads = vectors.shape[1] if vectors.ndim == 3 else 1
    rows = vectors.shape[0] * heads
    block = 128 if rows >= 32768 else 32
    _index_pool_rotate_rows[(triton.cdiv(rows, block),)](
        vectors,
        *factors,
        out,
        rows,
        heads,
        start,
        step,
        block,
        vectors if group_map is None else group_map,
        *(factors if history_factors is None else history_factors),
        group_map is not None,
        history_factors is not None,
        *(normalization if normalization is not None else (vectors, vectors)),
        normalization is not None,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out


@triton.jit(do_not_specialize=["key_rows", "key_tiles", "query_rows"])
def _fuse_partial_index_scores(
    per_head,
    weights,
    positions,
    output,
    key_rows,
    key_tiles,
    query_rows,
    TILE: tl.constexpr,
):
    program = tl.program_id(0)
    query = program // key_tiles
    key = (program % key_tiles) * TILE + tl.arange(0, TILE)
    head = tl.arange(0, 16)
    live = key < key_rows
    head_scores = tl.load(
        per_head + (head[:, None] * query_rows + query) * key_rows + key[None, :],
        live[None, :],
        other=0,
    )
    head_weight = tl.load(weights + query * 16 + head)
    scores = tl.sum(tl.maximum(head_scores, 0.0) * head_weight[:, None], axis=0)
    visible_group = tl.load(positions + query).to(tl.int32) >> 2
    scores = tl.where(key < visible_group, scores, -float("inf"))
    tl.store(output + query * key_rows + key, scores, live)


@triton.jit(do_not_specialize=["count", "sentinel"])
def _mask_partial_topk_indices(values, selected, masked, count, sentinel, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    live = index < count
    score = tl.load(values + index, live, other=0)
    group = tl.load(selected + index, live, other=0)
    tl.store(masked + index, tl.where(score == -float("inf"), sentinel, group), live)


@triton.jit(do_not_specialize=["rows"])
def _expand_partial_index_candidates(ordered, positions, output, rows, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    row = index >> 11
    group = (index & 2047) >> 2
    lane = index & 3
    live = row < rows
    history = group < 511
    position = tl.load(positions + row, live, other=0).to(tl.int64)
    cutoff = position >> 2
    group_id = tl.load(ordered + row * 511 + group, live & history, other=0)
    history_id = tl.where(group_id < cutoff, (group_id << 2) + lane, -1)
    tail = (cutoff << 2) + lane
    candidate = tl.where(history, history_id, tl.where(tail <= position, tail, -1))
    tl.store(output + index, candidate, live)


def _select_partial_candidates(scores, positions, num_keys):
    values, selected = scores.topk(511, dim=-1)
    masked = torch.empty_like(selected)
    rows, budget = selected.shape
    tile = 512
    count = rows * budget
    _mask_partial_topk_indices[(triton.cdiv(count, tile),)](
        values,
        selected,
        masked,
        count,
        num_keys,
        tile,
    )
    ordered = masked.sort(dim=-1).values
    output = torch.empty((rows, 2048), dtype=selected.dtype, device=selected.device)
    _expand_partial_index_candidates[(triton.cdiv(rows * 2048, tile),)](
        ordered,
        positions,
        output,
        rows,
        tile,
    )
    return output


def select_partial_prefill_pool(query, weights, keys, positions, ratio, budget):
    """Select candidates for a partial Prefill chunk with the native TopK and Sort."""
    if (
        HAS_TRITON
        and query.device.type == "npu"
        and is_950()
        and ratio == 4
        and budget == 511
        and query.dtype == weights.dtype == keys.dtype == torch.float32
        and query.ndim == 3
        and query.shape[1:] == (16, INDEX_DIM)
        and 2 <= query.shape[0] < 16
        and query.is_contiguous()
        and weights.shape == query.shape[:2]
        and weights.is_contiguous()
        and keys.ndim == 2
        and keys.shape[1] == INDEX_DIM
        and keys.shape[0] >= 8192
        and keys.is_contiguous()
        and positions.shape == (query.shape[0],)
        and positions.is_contiguous()
        and positions.dtype in (torch.int32, torch.int64)
        and query.device == weights.device == keys.device == positions.device
    ):
        rows, num_keys = query.shape[0], keys.shape[0]
        per_head = torch.matmul(query.transpose(0, 1), keys.T)
        if per_head.is_contiguous():
            scores = torch.empty((rows, num_keys), device=query.device, dtype=torch.float32)
            tile = 512
            tiles = triton.cdiv(num_keys, tile)
            _fuse_partial_index_scores[(rows * tiles,)](
                per_head,
                weights,
                positions,
                scores,
                num_keys,
                tiles,
                rows,
                tile,
                enable_fp_fusion=False,
            )
            return _select_partial_candidates(scores, positions, num_keys)
    return sparse_candidates(query, weights, keys, positions, ratio, budget)


@triton.jit(do_not_specialize=["rows"])
def _build_li_tail(positions, output, rows, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    pos = tl.load(positions + row, row < rows, other=0).to(tl.int32)
    lane = tl.arange(0, 4)
    group_start = pos & -4
    candidates = group_start[:, None] + lane[None, :]
    tail = tl.where(candidates <= pos[:, None], candidates, -1)
    tl.store(output + row[:, None] * 4 + lane[None, :], tail, row[:, None] < rows)


def build_li_tail(positions):
    rows = positions.shape[0]
    output = torch.empty((rows, 4), dtype=torch.int32, device=positions.device)
    tile = 128
    _build_li_tail[(triton.cdiv(rows, tile),)](positions, output, rows, tile)
    return output


@triton.jit(do_not_specialize=["count", "sentinel"])
def _mask_group_ids(groups, output, count, sentinel, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = index < count
    group = tl.load(groups + index, valid, other=-1)
    sortable = tl.where(group >= 0, group, sentinel)
    tl.store(output + index, sortable, valid)


def mask_group_ids(groups, sentinel, dtype):
    count = groups.numel()
    result = torch.empty_like(groups, dtype=dtype)
    tile = 4096
    _mask_group_ids[(triton.cdiv(count, tile),)](groups, result, count, sentinel, tile)
    return result


@triton.jit(do_not_specialize=["rows", "sentinel"])
def _expand_group_tile(groups, tail, output, rows, sentinel, ROW_TILE: tl.constexpr):
    row = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    column = tl.arange(0, 512)
    lane = tl.arange(0, 4)
    group = tl.load(
        groups + row[:, None] * 511 + column[None, :], (row[:, None] < rows) & (column[None, :] < 511), other=sentinel
    ).to(tl.int32)
    current = tl.load(tail + row[:, None] * 4 + lane[None, :], row[:, None] < rows, other=-1)
    history = tl.where(group[:, :, None] == sentinel, -1, group[:, :, None] * 4 + lane[None, None, :])
    index = tl.where(column[None, :, None] < 511, history, current[:, None, :])
    out = row[:, None, None] * 2048 + column[None, :, None] * 4 + lane[None, None, :]
    tl.store(output + out, index, row[:, None, None] < rows)


@triton.jit(do_not_specialize=["rows", "sentinel"])
def _expand_group_rows(groups, tail, output, rows, sentinel, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    col = tl.arange(0, 512)
    valid_row = row < rows
    history = col < 511
    valid = valid_row[:, None] & history[None, :]
    group = tl.load(groups + row[:, None] * 511 + col[None, :], valid, other=sentinel).to(tl.int32)
    start = group * 4
    has_group = group != sentinel
    tail0 = tl.load(tail + row * 4, valid_row, other=-1)
    tail1 = tl.load(tail + row * 4 + 1, valid_row, other=-1)
    tail2 = tl.load(tail + row * 4 + 2, valid_row, other=-1)
    tail3 = tl.load(tail + row * 4 + 3, valid_row, other=-1)
    p0 = tl.where(history[None, :], tl.where(has_group, start, -1), tail0[:, None])
    p1 = tl.where(history[None, :], tl.where(has_group, start + 1, -1), tail1[:, None])
    p2 = tl.where(history[None, :], tl.where(has_group, start + 2, -1), tail2[:, None])
    p3 = tl.where(history[None, :], tl.where(has_group, start + 3, -1), tail3[:, None])
    packed = tl.join(tl.join(p0, p2), tl.join(p1, p3)).reshape(TILE, 2048)
    lane = tl.arange(0, 2048)
    tl.store(output + row[:, None] * 2048 + lane[None, :], packed, valid_row[:, None])


def expand_sorted_group_ids(groups, sentinel, tail):
    rows = groups.shape[0]
    output = torch.empty((rows, 2048), dtype=torch.int32, device=groups.device)
    row_tile = 8
    kernel = _expand_group_rows if rows >= 768 else _expand_group_tile
    kernel[(triton.cdiv(rows, row_tile),)](groups, tail, output, rows, sentinel, row_tile)
    return output


@triton.jit(do_not_specialize=["rows", "epsilon"])
def _index_pool_weighted_stats(values, probabilities, output, stats, rows, epsilon, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    channel = tl.arange(0, 128)
    source = row[:, None] * 512 + channel[None, :]
    valid = row < rows
    live = valid[:, None]
    p0 = tl.load(values + source, live, other=0) * tl.load(probabilities + source, live, other=0)
    p1 = tl.load(values + source + 128, live, other=0) * tl.load(probabilities + source + 128, live, other=0)
    p2 = tl.load(values + source + 256, live, other=0) * tl.load(probabilities + source + 256, live, other=0)
    p3 = tl.load(values + source + 384, live, other=0) * tl.load(probabilities + source + 384, live, other=0)
    # The four-lane CANN reduction pairs opposite lanes before the final add.
    pooled = (p0 + p2) + (p1 + p3)
    mean = tl.sum(pooled * pooled, axis=1) * (1.0 / 128.0)
    tl.store(stats + row, mean + epsilon, valid)
    tl.store(output + row[:, None] * 128 + channel[None, :], pooled, live)


def pool_with_inverse_norm(values, probabilities, epsilon):
    rows = values.shape[0]
    pooled = torch.empty((rows, 128), dtype=torch.float32, device=values.device)
    stats = torch.empty(rows, dtype=torch.float32, device=values.device)
    tile = 16
    _index_pool_weighted_stats[(triton.cdiv(rows, tile),)](
        values, probabilities, pooled, stats, rows, epsilon, tile, enable_fp_fusion=False
    )
    return pooled, stats.rsqrt()


@triton.jit(do_not_specialize=["first", "rows"])
def _build_full_recall_rows(output, first, rows, BUDGET: tl.constexpr, RATIO: tl.constexpr, TILE: tl.constexpr):
    width = (BUDGET + 1) * RATIO
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    row, column = index // width, index % width
    position = first + row
    cutoff = position // RATIO * RATIO
    tail = cutoff + column - BUDGET * RATIO
    candidate = tl.where(
        column < BUDGET * RATIO, tl.where(column < cutoff, column, -1), tl.where(tail <= position, tail, -1)
    )
    tl.store(output + index, candidate, row < rows)


def build_full_recall_rows(first, rows, budget, ratio, device, dtype):
    """Emit the same sorted history, padding and current tail as TopK."""
    output = torch.empty((rows, (budget + 1) * ratio), device=device, dtype=dtype)
    tile = 512
    _build_full_recall_rows[(triton.cdiv(output.numel(), tile),)](
        output,
        first,
        rows,
        budget,
        ratio,
        tile,
    )
    return output
