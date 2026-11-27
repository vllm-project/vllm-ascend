# SPDX-License-Identifier: Apache-2.0
"""First-layer iHC with a shared embedding residual."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _hy4_first_ihc_pre(
    embedding,
    token_ids,
    gates,
    reduced,
    post,
    TOKENS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    CHANNELS: tl.constexpr,
    TILE: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    row_live = row < TOKENS
    live = row_live[:, None] & (col[None, :] < WIDTH)
    offset = row[:, None] * WIDTH + col[None, :]
    token = tl.load(token_ids + row, row_live, other=VOCAB_SIZE)
    token = tl.where((token >= 0) & (token < VOCAB_SIZE), token, VOCAB_SIZE)
    gate_row = token * (2 * CHANNELS)
    x = tl.load(embedding + offset, live, 0).to(tl.float32)
    mixed = tl.full((ROWS, TILE), 0, tl.float32)
    for channel in tl.static_range(CHANNELS):
        pre = tl.load(gates + gate_row + channel, row_live, 0)
        mixed = mixed + x * pre[:, None]
        if tl.program_id(1) == 0:
            p = tl.load(gates + gate_row + CHANNELS + channel, row_live, 0)
            tl.store(post + row * CHANNELS + channel, p, row_live)
    tl.store(reduced + offset, mixed, live)


@triton.jit
def _hy4_first_ihc_post(
    hidden,
    embedding,
    post,
    output,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
    CHANNELS: tl.constexpr,
    HIDDEN_ROW_STRIDE: tl.constexpr,
    HIDDEN_COL_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    live = (row[:, None] < TOKENS) & (col[None, :] < WIDTH)
    offset = row[:, None] * WIDTH + col[None, :]
    hidden_offset = row[:, None] * HIDDEN_ROW_STRIDE + col[None, :] * HIDDEN_COL_STRIDE
    x = tl.load(hidden + hidden_offset, live, 0).to(tl.float32)
    residual = tl.load(embedding + offset, live, 0).to(tl.float32)
    for channel in tl.static_range(CHANNELS):
        gate = tl.load(post + row * CHANNELS + channel, row < TOKENS, 0)
        dst = (row[:, None] * CHANNELS + channel) * WIDTH + col[None, :]
        tl.store(output + dst, residual + x * gate[:, None], live)


def first_ihc_pre(embedding, token_ids, gates):
    tokens, width = embedding.shape
    channels = gates.shape[1] // 2
    reduced = torch.empty_like(embedding)
    post = torch.empty((tokens, channels), dtype=torch.float32, device=embedding.device)
    rows, tile = 4, 1024
    _hy4_first_ihc_pre[(triton.cdiv(tokens, rows), triton.cdiv(width, tile))](
        embedding,
        token_ids,
        gates,
        reduced,
        post,
        tokens,
        gates.shape[0] - 1,
        width,
        channels,
        tile,
        rows,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return reduced, post


def first_ihc_post(hidden, embedding, post):
    tokens, width = embedding.shape
    channels = post.shape[1]
    output = torch.empty((tokens, channels, width), dtype=hidden.dtype, device=embedding.device)
    rows, tile = 4, 1024
    _hy4_first_ihc_post[(triton.cdiv(tokens, rows), triton.cdiv(width, tile))](
        hidden,
        embedding,
        post,
        output,
        tokens,
        width,
        channels,
        hidden.stride(0),
        hidden.stride(1),
        tile,
        rows,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return output
