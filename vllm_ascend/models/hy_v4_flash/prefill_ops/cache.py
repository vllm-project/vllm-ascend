# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 Flash prefill ops cache."""

import torch
from torch.nn.functional import pad
from vllm.models.hy_v4_flash.reference import INDEX_DIM
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.device.device_config import is_950
from vllm_ascend.hisparse.c8 import C8_ROW_BYTES, dequantize_c8_rows, pack_c8_components

_WORD_BYTES = torch.int32.itemsize
_LATENT_WORDS = 512 // _WORD_BYTES
_ROPE_END = (512 + 64 * torch.bfloat16.itemsize) // _WORD_BYTES
_C8_END = C8_ROW_BYTES // _WORD_BYTES
_INDEX_END = _C8_END + INDEX_DIM * torch.bfloat16.itemsize // _WORD_BYTES
_POSITIONS_END = _INDEX_END + 3 * torch.int64.itemsize // _WORD_BYTES
_DECODE_ROWS = 4
_PREFILL_ROWS = 16
_PREFILL_THRESHOLD = 256


@triton.jit(do_not_specialize=["tokens", "position_axis_stride"])
def _write_c8_rows(
    cache,
    slots,
    latent,
    rope,
    scales,
    keys,
    positions,
    tokens,
    position_axis_stride,
    POSITION_TOKEN_STRIDE: tl.constexpr,
    HAS_INDEXER: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    LATENT_WORDS: tl.constexpr,
    ROPE_END: tl.constexpr,
    C8_END: tl.constexpr,
    INDEX_END: tl.constexpr,
    POSITIONS_END: tl.constexpr,
    norm_weight,
    norm_inverse,
    normalized_keys,
    APPLY_NORM: tl.constexpr,
):
    token = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    slot = tl.load(slots + token, token < tokens, -1).to(tl.int64)
    valid = (token < tokens) & (slot >= 0)
    word = tl.arange(0, triton.next_power_of_2(ROW_WIDTH))
    row = token[:, None]
    live = valid[:, None]
    lane = word[None, :]

    bits = tl.load(latent + row * LATENT_WORDS + lane, live & (lane < LATENT_WORDS), 0)
    bits |= tl.load(
        rope + row * (ROPE_END - LATENT_WORDS) + lane - LATENT_WORDS,
        live & (lane >= LATENT_WORDS) & (lane < ROPE_END),
        0,
    )
    bits |= tl.load(
        scales + row * (C8_END - ROPE_END) + lane - ROPE_END,
        live & (lane >= ROPE_END) & (lane < C8_END),
        0,
    )
    if HAS_INDEXER:
        if APPLY_NORM:
            channel = tl.arange(0, 128)
            coord = row * 128 + channel[None, :]
            token_valid = token[:, None] < tokens
            key = tl.load(keys + coord, token_valid, other=0).to(tl.float32)
            weight = tl.load(norm_weight + channel).to(tl.float32)
            inverse = tl.load(norm_inverse + token, token < tokens, other=0)
            key = tl.cast((key * inverse[:, None]) * weight[None, :], tl.bfloat16, fp_downcast_rounding="rtne")
            tl.store(normalized_keys + coord, key, token_valid)
            low, high = tl.split(tl.reshape(key, (ROWS, 64, 2)))
            low = low.to(tl.uint16, bitcast=True).to(tl.uint32)
            high = high.to(tl.uint16, bitcast=True).to(tl.uint32)
            channels = tl.arange(0, 64)
            tl.store(cache + slot[:, None] * ROW_WIDTH + C8_END + channels[None, :], low | (high << 16), live)
        else:
            bits |= tl.load(
                keys + row * (INDEX_END - C8_END) + lane - C8_END,
                live & (lane >= C8_END) & (lane < INDEX_END),
                0,
            )
        position_lane = lane - INDEX_END
        bits |= tl.load(
            positions + row * POSITION_TOKEN_STRIDE + position_lane // 2 * position_axis_stride + position_lane % 2,
            live & (lane >= INDEX_END) & (lane < POSITIONS_END),
            0,
        )
    write = live & (lane < ROW_WIDTH)
    if APPLY_NORM:
        write &= (lane < C8_END) | (lane >= INDEX_END)
    tl.store(cache + slot[:, None] * ROW_WIDTH + lane, bits, write)


@triton.jit(do_not_specialize=["tokens", "position_axis_stride"])
def _write_c8_segments(
    cache,
    slots,
    latent,
    rope,
    scales,
    keys,
    positions,
    tokens,
    position_axis_stride,
    POSITION_TOKEN_STRIDE: tl.constexpr,
    HAS_INDEXER: tl.constexpr,
    ROW_WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    LATENT_WORDS: tl.constexpr,
    ROPE_END: tl.constexpr,
    C8_END: tl.constexpr,
    INDEX_END: tl.constexpr,
    POSITIONS_END: tl.constexpr,
    norm_weight,
    norm_inverse,
    normalized_keys,
    APPLY_NORM: tl.constexpr,
):
    token = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    present = token < tokens
    slot = tl.load(slots + token, present, -1).to(tl.int64)
    live = (present & (slot >= 0))[:, None]
    dst = cache + slot[:, None] * ROW_WIDTH
    row = token[:, None]

    latent_lane = tl.arange(0, LATENT_WORDS)[None, :]
    payload = tl.load(latent + row * LATENT_WORDS + latent_lane, live, 0)
    tl.store(dst + latent_lane, payload, live)

    rope_lane = tl.arange(0, ROPE_END - LATENT_WORDS)[None, :]
    rotary = tl.load(rope + row * (ROPE_END - LATENT_WORDS) + rope_lane, live, 0)
    tl.store(dst + LATENT_WORDS + rope_lane, rotary, live)

    scale_lane = tl.arange(0, C8_END - ROPE_END)[None, :]
    scale = tl.load(scales + row * (C8_END - ROPE_END) + scale_lane, live, 0)
    tl.store(dst + ROPE_END + scale_lane, scale, live)

    key_lane = tl.arange(0, INDEX_END - C8_END)[None, :]
    if HAS_INDEXER:
        if APPLY_NORM:
            channel = tl.arange(0, (INDEX_END - C8_END) * 2)
            coord = row * ((INDEX_END - C8_END) * 2) + channel[None, :]
            index_key = tl.load(keys + coord, present[:, None], other=0).to(tl.float32)
            weight = tl.load(norm_weight + channel).to(tl.float32)
            inverse = tl.load(norm_inverse + token, present, other=0)
            index_key = tl.cast(
                (index_key * inverse[:, None]) * weight[None, :], tl.bfloat16, fp_downcast_rounding="rtne"
            )
            tl.store(normalized_keys + coord, index_key, present[:, None])
            low, high = tl.split(tl.reshape(index_key, (ROWS, INDEX_END - C8_END, 2)))
            low = low.to(tl.uint16, bitcast=True).to(tl.uint32)
            high = high.to(tl.uint16, bitcast=True).to(tl.uint32)
            key_bits = low | (high << 16)
        else:
            key_bits = tl.load(keys + row * (INDEX_END - C8_END) + key_lane, live, 0)
        tl.store(dst + C8_END + key_lane, key_bits, live)
    else:
        tl.store(dst + C8_END + key_lane, 0, live)

    if ROW_WIDTH > INDEX_END:
        tail = tl.arange(0, triton.next_power_of_2(ROW_WIDTH - INDEX_END))[None, :]
        if HAS_INDEXER:
            coord = row * POSITION_TOKEN_STRIDE + tail // 2 * position_axis_stride + tail % 2
            tail_bits = tl.load(positions + coord, live & (tail < POSITIONS_END - INDEX_END), 0)
        else:
            tail_bits = 0
        tl.store(dst + INDEX_END + tail, tail_bits, live & (tail < ROW_WIDTH - INDEX_END))


def write_c8_rows_triton(cache, slots, latent, rope, scales, keys=None, positions=None, *, index_norm=None):
    """Pack the quantizer outputs and token metadata directly into unique cache slots."""
    tokens = slots.numel()
    row_width = cache.shape[-1] // _WORD_BYTES
    rows = _PREFILL_ROWS if tokens >= _PREFILL_THRESHOLD else _DECODE_ROWS
    has_indexer = keys is not None
    if has_indexer:
        assert positions is not None
        positions = positions.to(torch.int64)
        if positions.stride(-1) != 1:
            positions = positions.contiguous()
        token_stride = positions.stride(-1) * 2
        axis_stride = positions.stride(0) * 2 if positions.ndim == 2 else 0
        position_words = positions.view(torch.int32)
    else:
        token_stride = 0
        axis_stride = 0
        position_words = None
    output_keys = None
    key_data = keys.contiguous().view(torch.int32) if has_indexer else None
    norm_weight = norm_inverse = None
    if index_norm is not None:
        assert has_indexer
        norm_weight, norm_inverse = index_norm
        output_keys = torch.empty_like(keys)
        key_data = keys
    # Keep the sparse-row path for small batches and layouts outside a 256-word tile.
    kernel = _write_c8_rows
    if _POSITIONS_END <= row_width <= 256 and tokens >= (512 if has_indexer else 1024):
        kernel = _write_c8_segments
        rows = 32
    kernel[(triton.cdiv(tokens, rows),)](
        cache.view(torch.int32),
        slots.contiguous(),
        latent.view(torch.int32),
        rope.contiguous().view(torch.int32),
        scales.view(torch.int32),
        key_data,
        position_words,
        tokens,
        axis_stride,
        token_stride,
        has_indexer,
        row_width,
        rows,
        _LATENT_WORDS,
        _ROPE_END,
        _C8_END,
        _INDEX_END,
        _POSITIONS_END,
        norm_weight,
        norm_inverse,
        output_keys,
        index_norm is not None,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return output_keys


def write_c8_rows(cache, slots, latent, rope, scales, keys=None, positions=None, *, index_norm=None):
    """Pack quantizer outputs and write them to their unique physical slots."""
    if not slots.numel():
        return
    if HAS_TRITON and cache.device.type == "npu" and cache.is_contiguous():
        if keys is not None:
            assert positions is not None
            positions = positions.to(torch.int64)
            if positions.stride(-1) != 1:
                # A singleton slice can report contiguous with a non-unit stride.
                positions = positions.clone(memory_format=torch.contiguous_format)
        return write_c8_rows_triton(cache, slots, latent, rope, scales, keys, positions, index_norm=index_norm)

    if index_norm is not None:
        weight, inverse = index_norm
        keys = ((keys.float() * inverse[:, None]) * weight).bfloat16()
    payload = pack_c8_components(latent, rope.contiguous(), scales)
    if keys is not None:
        assert positions is not None
        coords = positions[:, None].expand(-1, 3) if positions.ndim == 1 else positions.T
        coords = coords.to(torch.int64).contiguous().view(torch.uint8)
        payload = torch.cat((payload, keys.contiguous().view(torch.uint8), coords), dim=-1)

    rows = pad(payload, (0, cache.shape[-1] - payload.shape[-1]))
    # vLLM reserves page zero. All padding writes restore identical bytes there.
    rows = torch.where(slots[:, None] >= 0, rows, cache[:1])
    cache.index_copy_(0, slots.long().clamp_min(0), rows)
    if index_norm is not None:
        return keys


_LATENT = 512
_ROPE = 64


def gather_c8_rows(cache, block_table, block_size, positions):
    """Read selected C8 rows through the page table, zeroing padded positions."""
    if not positions.numel():
        return cache.new_empty((*positions.shape, _LATENT + _ROPE), dtype=torch.float32)
    logical = positions.clamp_min(0).long().flatten()
    pages = block_table.index_select(0, logical // block_size).long()
    slots = pages * block_size + logical % block_size
    packed = cache.index_select(0, slots)[:, :C8_ROW_BYTES].contiguous()
    rows = dequantize_c8_rows(packed).view(*positions.shape, _LATENT + _ROPE)
    return rows.masked_fill(positions.unsqueeze(-1) < 0, 0)


def gather_c8_qk(query, cache, block_table, block_size, positions):
    """Return FP32 QK scores and the selected 512-wide latent values."""
    rows = gather_c8_rows(cache, block_table, block_size, positions)
    scores = torch.bmm(query.float(), rows.transpose(1, 2))
    return scores, rows[..., :_LATENT].contiguous()


_PAGE_ROWS = 64
_CACHE_ROW_BYTES = 1000
_COPY_WORD_BYTES = 4


def supports_paged_cache_copy(cache, table, block_size):
    return (
        HAS_TRITON
        and cache.device.type == "npu"
        and is_950()
        and cache.dtype == torch.uint8
        and cache.ndim == 2
        and cache.is_contiguous()
        and cache.size(-1) == _CACHE_ROW_BYTES
        and cache.storage_offset() % _COPY_WORD_BYTES == 0
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
    words = row_bytes // _COPY_WORD_BYTES
    _copy_paged_cache_lanes[(triton.cdiv(rows, _PAGE_ROWS),)](
        cache.view(torch.int32),
        table,
        output.view(torch.int32),
        rows,
        _PAGE_ROWS,
        _CACHE_ROW_BYTES // _COPY_WORD_BYTES,
        first_byte // _COPY_WORD_BYTES,
        words,
        triton.next_power_of_2(words),
        num_warps=1,
    )
    return output
