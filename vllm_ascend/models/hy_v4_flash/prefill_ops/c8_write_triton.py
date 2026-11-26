# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4_flash.reference import INDEX_DIM
from vllm.triton_utils import tl, triton

from vllm_ascend.hisparse.c8 import C8_ROW_BYTES

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
    cache, slots, latent, rope, scales, keys, positions, tokens,
    position_axis_stride, POSITION_TOKEN_STRIDE: tl.constexpr,
    HAS_INDEXER: tl.constexpr, ROW_WIDTH: tl.constexpr, ROWS: tl.constexpr,
    LATENT_WORDS: tl.constexpr, ROPE_END: tl.constexpr, C8_END: tl.constexpr,
    INDEX_END: tl.constexpr, POSITIONS_END: tl.constexpr,
    norm_weight, norm_inverse, normalized_keys, APPLY_NORM: tl.constexpr,
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
            index_key = tl.cast((index_key * inverse[:, None]) * weight[None, :],
                                tl.bfloat16, fp_downcast_rounding="rtne")
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

def write_c8_rows(cache, slots, latent, rope, scales, keys=None, positions=None, *, index_norm=None):
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
