# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""EAGLE-4 selected LM-head rows, without a materialized [batch, C, hidden]."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_dot(
    weight,
    hidden,
    indices,
    output,
    weight_row_stride: tl.constexpr,
    weight_col_stride: tl.constexpr,
    hidden_row_stride: tl.constexpr,
    hidden_col_stride: tl.constexpr,
    index_row_stride: tl.constexpr,
    index_col_stride: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    candidate = tl.program_id(1)
    token = tl.load(indices + row * index_row_stride + candidate * index_col_stride)
    offsets = tl.arange(0, BLOCK_H)
    w = tl.load(weight + token * weight_row_stride + offsets * weight_col_stride, offsets < HIDDEN_SIZE, 0)
    x = tl.load(hidden + row * hidden_row_stride + offsets * hidden_col_stride, offsets < HIDDEN_SIZE, 0)
    dot = tl.sum(w.to(tl.float32) * x.to(tl.float32), axis=0)
    rounded = dot.to(hidden.dtype.element_ty)
    tl.store(output + row * NUM_CANDIDATES + candidate, rounded.to(tl.float32))


def gather_dot(weight: torch.Tensor, hidden: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """One program per candidate; FP32 accumulation, then draft-dtype rounding."""
    rows, candidates = indices.shape
    output = torch.empty((rows, candidates), dtype=torch.float32, device=hidden.device)
    if rows and candidates:
        _gather_dot[(rows, candidates)](
            weight,
            hidden,
            indices,
            output,
            *weight.stride(),
            *hidden.stride(),
            *indices.stride(),
            candidates,
            hidden.shape[1],
            triton.next_power_of_2(hidden.shape[1]),
        )
    return output
