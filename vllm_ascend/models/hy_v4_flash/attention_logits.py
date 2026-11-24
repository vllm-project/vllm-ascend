# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

_LOGIT_TILE = 512


@triton.jit(do_not_specialize=["rows", "width", "candidate_row_stride"])
def _build_mla_logits(
    scores,
    candidates,
    sink,
    logits,
    rows,
    width,
    candidate_row_stride,
    CANDIDATE_COL_STRIDE: tl.constexpr,
    SINK_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    TILE: tl.constexpr,
    SHORT_ROW: tl.constexpr,
):
    start = tl.program_id(0) * TILE
    offsets = start + tl.arange(0, TILE)
    row_width = width + 1
    if SHORT_ROW:
        row = offsets // row_width
        column = offsets % row_width
    else:
        # A tile spans at most two rows. Divide once per program.
        column = start % row_width + tl.arange(0, TILE)
        crosses_row = column >= row_width
        row = start // row_width + crosses_row.to(tl.int32)
        column = tl.where(crosses_row, column - row_width, column)
    live = row < rows
    is_sink = column == width
    is_candidate = live & ~is_sink

    candidate_offset = row // HEADS * candidate_row_stride + column * CANDIDATE_COL_STRIDE
    candidate = tl.load(candidates + candidate_offset, is_candidate, -1)
    score = tl.load(scores + offsets - row, is_candidate, 0.0)
    sink_score = tl.load(sink + row % HEADS * SINK_STRIDE, live & is_sink, 0.0)

    logit = tl.where(candidate >= 0, score * SCALE, float("-inf"))
    tl.store(logits + offsets, tl.where(is_sink, sink_score, logit), live)


def build_mla_logits(scores, candidates, sink, scale):
    """Build contiguous FP32 logits with the learned sink in the final column."""
    queries, heads, width = scores.shape
    logits = torch.empty((queries, heads, width + 1), dtype=torch.float32, device=scores.device)
    _build_mla_logits[(triton.cdiv(logits.numel(), _LOGIT_TILE),)](
        scores,
        candidates,
        sink,
        logits,
        queries * heads,
        width,
        *candidates.stride(),
        sink.stride(0),
        heads,
        scale,
        _LOGIT_TILE,
        width + 1 < _LOGIT_TILE,
        num_warps=1,
        multibuffer=False,
    )
    return logits
