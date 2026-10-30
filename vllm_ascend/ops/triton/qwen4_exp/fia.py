# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp masks using aligned 32-bit writes for four-token QSA groups."""

import torch
from vllm.triton_utils import tl, triton

QSA_FIA_MASK_WORD_TILE = 4096


@triton.jit(do_not_specialize=["kv_length", "mask_words", "word_stride"])
def _prepare_qsa_mask_words_kernel(
    compact_indices,
    visible_lens,
    words,
    scatter_indices,
    updates,
    has_selection,
    stride_row,
    stride_column,
    kv_length,
    mask_words,
    word_stride,
    GROUPS: tl.constexpr,
    UPDATE_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    WORD_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    visible = tl.minimum(tl.maximum(tl.load(visible_lens + row), 0), kv_length)
    group = tl.load(compact_indices + row * stride_row + columns * stride_column, columns < GROUPS, other=-1)
    valid = (columns < GROUPS) & (group >= 0) & (group < visible // 4)
    residual = visible % 4
    if GROUPS >= 512:
        if residual != 0:
            # Stable prefix rank matches GBSA's final full-group replacement.
            valid = valid & (tl.cumsum(valid.to(tl.int32), 0) < GROUPS)
    selected = (tl.sum(valid.to(tl.int32), 0) > 0) | (residual != 0)
    # All four mask bytes in a complete group have the same value. One int32
    # scatter replaces four costly unaligned byte scatters. Invalid entries get
    # distinct scratch destinations, avoiding contention on a shared sentinel.
    target = tl.where(valid, group, mask_words + columns)
    tail = (columns == GROUPS) & (residual != 0)
    target = tl.where(tail, visible // 4, target)
    value = tl.where(tail, (0x01010101 << (residual * 8)).to(tl.int32), 0)
    # An empty row attends token zero and is explicitly zeroed after FIA.
    dummy = (columns == 0) & (~selected)
    target = tl.where(dummy, 0, target)
    value = tl.where(dummy, 0x01010100, value)
    tl.store(scatter_indices + row * UPDATE_WIDTH + columns, target.to(tl.int64), columns < UPDATE_WIDTH)
    tl.store(updates + row * UPDATE_WIDTH + columns, value, columns < UPDATE_WIDTH)
    tl.store(has_selection + row, selected)
    # Bound on-chip vectors independently of KV length. Q tiling bounds the
    # global mask allocation, but cannot bound a single row's UB requirements.
    for word_start in range(0, word_stride, WORD_BLOCK):
        word = word_start + tl.arange(0, WORD_BLOCK)
        tl.store(words + row * word_stride + word, 0x01010101, word < word_stride)


@triton.jit
def _copy_qsa_fia_output_kernel(
    result,
    selected,
    output,
    stride_row,
    stride_head,
    stride_dim,
    HEAD_DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    CONTIGUOUS: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(result + row * WIDTH + columns, columns < WIDTH, other=0)
    valid = tl.load(selected + row)
    if CONTIGUOUS:
        offset = row * WIDTH + columns
    else:
        offset = row * stride_row + columns // HEAD_DIM * stride_head + columns % HEAD_DIM * stride_dim
    tl.store(output + offset, tl.where(valid, values, 0), columns < WIDTH)


def build_qsa_fia_mask(
    compact_indices: torch.Tensor, visible_lens: torch.Tensor, kv_length: int, mask_width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return blocked [T, KV] mask and nonempty rows, with complete initialization."""
    rows = compact_indices.shape[0]
    # Dense DMA stores must not share a 32-byte block with another row.
    mask_words = triton.cdiv(mask_width, 32) * 8
    groups = compact_indices.shape[1] - 3
    update_width = triton.cdiv(groups + 1, 8) * 8
    word_stride = mask_words + update_width
    words = torch.empty((rows, word_stride), dtype=torch.int32, device=compact_indices.device)
    indices = torch.empty((rows, update_width), dtype=torch.int64, device=compact_indices.device)
    updates = torch.empty((rows, update_width), dtype=torch.int32, device=compact_indices.device)
    selected = torch.empty((rows,), dtype=torch.bool, device=compact_indices.device)
    _prepare_qsa_mask_words_kernel[(rows,)](
        compact_indices,
        visible_lens,
        words,
        indices,
        updates,
        selected,
        compact_indices.stride(0),
        compact_indices.stride(1),
        kv_length=kv_length,
        mask_words=mask_words,
        word_stride=word_stride,
        GROUPS=groups,
        UPDATE_WIDTH=update_width,
        BLOCK=triton.next_power_of_2(update_width),
        WORD_BLOCK=min(QSA_FIA_MASK_WORD_TILE, triton.next_power_of_2(word_stride)),
    )
    words.scatter_(1, indices, updates)
    # Ascend's little-endian int32 representation is four bool bytes. This is
    # a dtype view, not a numerical conversion (which would collapse the bytes).
    mask = words[:, :mask_words].contiguous().view(torch.bool)
    return mask[:, :mask_width], selected


def copy_qsa_fia_output(result: torch.Tensor, selected: torch.Tensor, output: torch.Tensor) -> None:
    width = output.shape[1] * output.shape[2]
    _copy_qsa_fia_output_kernel[(output.shape[0], triton.cdiv(width, 1024))](
        result,
        selected,
        output,
        *output.stride(),
        HEAD_DIM=output.shape[2],
        WIDTH=width,
        BLOCK=1024,
        CONTIGUOUS=output.is_contiguous(),
    )
