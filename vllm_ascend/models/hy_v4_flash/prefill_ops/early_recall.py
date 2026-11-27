# SPDX-License-Identifier: Apache-2.0
"""Skip index scores when the recall budget fits every visible group."""

import torch
from vllm.triton_utils import tl, triton


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
