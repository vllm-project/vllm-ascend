# SPDX-License-Identifier: Apache-2.0
"""Decode candidate expansion adapted from HY4 Prefill index postprocessing."""

import torch
from vllm.triton_utils import tl, triton

_POOL_RATIO = 4
_POOL_BUDGET = 511
_CANDIDATES = (_POOL_BUDGET + 1) * _POOL_RATIO
_EXPAND_TILE = 512


@triton.jit(do_not_specialize=["rows"])
def _expand_decode_candidates(
    groups,
    visible,
    output,
    rows,
    TILE: tl.constexpr,
    RATIO: tl.constexpr,
    BUDGET: tl.constexpr,
    CANDIDATES: tl.constexpr,
):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    row = index // CANDIDATES
    column = index % CANDIDATES
    group = column // RATIO
    lane = column % RATIO
    live = row < rows
    position = tl.load(visible + row, live, other=-1).to(tl.int64)
    cutoff = position >> 2
    selected = tl.load(groups + row * BUDGET + group, live & (group < BUDGET), other=-1).to(tl.int64)
    history = tl.where((selected >= 0) & (selected < cutoff), selected * RATIO + lane, -1)
    tail = cutoff * RATIO + lane
    candidate = tl.where(group < BUDGET, history, tl.where(tail <= position, tail, -1))
    tl.store(output + index, candidate, live)


def expand_decode_candidates(groups, visible):
    """Keep the sorted LI order and return the original INT64 candidate contract."""
    rows = groups.shape[0]
    output = torch.empty((rows, _CANDIDATES), dtype=torch.int64, device=groups.device)
    if rows:
        _expand_decode_candidates[(triton.cdiv(rows * _CANDIDATES, _EXPAND_TILE),)](
            groups.contiguous(),
            visible.contiguous(),
            output,
            rows,
            _EXPAND_TILE,
            _POOL_RATIO,
            _POOL_BUDGET,
            _CANDIDATES,
        )
    return output
