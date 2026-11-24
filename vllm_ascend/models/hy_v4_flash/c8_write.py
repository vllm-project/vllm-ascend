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
    WORDS: tl.constexpr,
    LATENT_WORDS: tl.constexpr,
    ROPE_END: tl.constexpr,
    C8_END: tl.constexpr,
    INDEX_END: tl.constexpr,
    POSITIONS_END: tl.constexpr,
):
    token = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    slot = tl.load(slots + token, token < tokens, -1).to(tl.int64)
    valid = (token < tokens) & (slot >= 0)
    word = tl.arange(0, WORDS)
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
    tl.store(cache + slot[:, None] * ROW_WIDTH + lane, bits, live & (lane < ROW_WIDTH))


def write_c8_rows(cache, slots, latent, rope, scales, keys=None, positions=None):
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
    _write_c8_rows[(triton.cdiv(tokens, rows),)](
        cache.view(torch.int32),
        slots.contiguous(),
        latent.view(torch.int32),
        rope.contiguous().view(torch.int32),
        scales.view(torch.int32),
        keys.contiguous().view(torch.int32) if has_indexer else None,
        position_words,
        tokens,
        axis_stride,
        token_stride,
        has_indexer,
        row_width,
        rows,
        triton.next_power_of_2(row_width),
        _LATENT_WORDS,
        _ROPE_END,
        _C8_END,
        _INDEX_END,
        _POSITIONS_END,
        num_warps=1,
    )
