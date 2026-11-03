# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental split-vocabulary greedy selection for Ascend NPU."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import (
    get_vectorcore_num,
    init_device_properties_triton,
)

_VOCAB_BLOCK_SIZE = 4096
_INDEX_SENTINEL = 2147483647


@triton.jit
def _argmax_partial(
    X,
    VALUES,
    INDICES,
    ROWS: tl.constexpr,
    VOCAB: tl.constexpr,
    STRIDE: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    SENTINEL: tl.constexpr,
):
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK)
    for task in tl.range(worker, ROWS * SPLITS, workers):
        row = task // SPLITS
        part = task % SPLITS
        cols = part * BLOCK + offsets
        valid = cols < VOCAB
        values = tl.load(X + row.to(tl.int64) * STRIDE + cols, valid, float("-inf")).to(tl.float32)
        is_nan = (values != values) & valid
        has_nan = tl.max(is_nan.to(tl.int32), axis=0) != 0
        maximum = tl.max(tl.where(is_nan, float("-inf"), values), axis=0)
        # Explicit smallest-index tie breaking also excludes padded -inf lanes.
        candidates = valid & tl.where(has_nan, is_nan, values == maximum)
        index = tl.min(tl.where(candidates, cols, SENTINEL), axis=0)
        tl.store(VALUES + task, tl.where(has_nan, float("nan"), maximum))
        tl.store(INDICES + task, index)


@triton.jit
def _argmax_finish(
    VALUES, INDICES, OUT, ROWS: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr, SENTINEL: tl.constexpr
):
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    parts = tl.arange(0, BLOCK)
    valid = parts < SPLITS
    for row in tl.range(worker, ROWS, workers):
        values = tl.load(VALUES + row * SPLITS + parts, valid, float("-inf"))
        indices = tl.load(INDICES + row * SPLITS + parts, valid, SENTINEL)
        is_nan = (values != values) & valid
        has_nan = tl.max(is_nan.to(tl.int32), axis=0) != 0
        maximum = tl.max(tl.where(is_nan, float("-inf"), values), axis=0)
        candidates = valid & tl.where(has_nan, is_nan, values == maximum)
        index = tl.min(tl.where(candidates, indices, SENTINEL), axis=0)
        tl.store(OUT + row, index.to(tl.int64))


def greedy_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Match torch.argmax(-1), including first-NaN and first-index ties.

    Unlike upstream #57770 this preserves NaNs rather than sanitizing them.
    The existing NPU fallback uses torch.argmax, so preserving it avoids a
    sampling behavior change. No collectives or tensor-to-host reads occur.
    """
    if (
        logits.device.type not in ("npu", "privateuseone")
        or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or logits.ndim != 2
        or logits.stride(1) != 1
        or not _VOCAB_BLOCK_SIZE <= logits.shape[1] < _INDEX_SENTINEL
    ):
        return logits.argmax(dim=-1)
    rows, vocab = logits.shape
    out = torch.empty(rows, dtype=torch.int64, device=logits.device)
    if rows == 0:
        return out
    splits = triton.cdiv(vocab, _VOCAB_BLOCK_SIZE)
    values = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
    indices = torch.empty((rows, splits), dtype=torch.int32, device=logits.device)
    try:
        cores = int(get_vectorcore_num())
    except AssertionError:
        init_device_properties_triton()
        cores = int(get_vectorcore_num())
    _argmax_partial[(min(cores, rows * splits),)](
        logits,
        values,
        indices,
        rows,
        vocab,
        logits.stride(0),
        splits,
        _VOCAB_BLOCK_SIZE,
        _INDEX_SENTINEL,
    )
    _argmax_finish[(min(cores, rows),)](
        values, indices, out, rows, splits, triton.next_power_of_2(splits), _INDEX_SENTINEL
    )
    return out
