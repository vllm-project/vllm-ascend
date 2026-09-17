# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-vocabulary DSpark greedy reduction for Ascend.

Based on local-inference-lab/vllm PR #738.
Uses standard max/min reductions instead of a custom tuple reducer.
"""

import torch
from vllm.triton_utils import tl, triton

_BLOCK = 2048


@triton.jit
def _reduce_first_argmax(values, indices):
    sentinel = 2147483647
    is_nan = values != values

    first_nan_index = tl.min(
        tl.where(is_nan, indices, sentinel),
        axis=0,
    )
    has_nan = first_nan_index != sentinel

    numeric_values = tl.where(is_nan, -float("inf"), values)
    max_value = tl.max(numeric_values, axis=0)

    first_max_index = tl.min(
        tl.where(
            (~is_nan) & (values == max_value),
            indices,
            sentinel,
        ),
        axis=0,
    )

    result_index = tl.where(
        has_nan,
        first_nan_index,
        first_max_index,
    )
    result_value = tl.where(
        has_nan,
        float("nan"),
        max_value,
    )
    return result_value, result_index


@triton.jit
def _partial_argmax(
    base,
    bias,
    partial_values,
    partial_indices,
    base_row_stride,
    bias_row_stride,
    VOCAB: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)

    col = part * BLOCK + tl.arange(0, BLOCK)
    valid = col < VOCAB

    a = tl.load(
        base + row * base_row_stride + col,
        mask=valid,
        other=0,
    )
    b = tl.load(
        bias + row * bias_row_stride + col,
        mask=valid,
        other=0,
    )

    # Match the rounding of a materialized source-dtype torch.add.
    summed = (
        (a.to(tl.float32) + b.to(tl.float32))
        .to(base.dtype.element_ty)
        .to(tl.float32)
    )
    summed = tl.where(valid, summed, -float("inf"))
    indices = tl.where(valid, col, 2147483647)

    value, index = _reduce_first_argmax(summed, indices)

    offset = row * PARTS + part
    tl.store(partial_values + offset, value)
    tl.store(partial_indices + offset, index)


@triton.jit
def _finish_argmax(
    partial_values,
    partial_indices,
    output,
    output_stride,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    valid = part < PARTS

    values = tl.load(
        partial_values + row * PARTS + part,
        mask=valid,
        other=-float("inf"),
    )
    indices = tl.load(
        partial_indices + row * PARTS + part,
        mask=valid,
        other=2147483647,
    )

    _, result = _reduce_first_argmax(values, indices)
    tl.store(output + row * output_stride, result)


def scratch_shape(
    max_requests: int,
    vocabulary_size: int,
) -> tuple[int, int]:
    return max_requests, triton.cdiv(vocabulary_size, _BLOCK)


def sample_greedy_markov(
    base_logits: torch.Tensor,
    markov_bias: torch.Tensor,
    output: torch.Tensor,
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
) -> None:
    """Write the full-vocabulary greedy token directly into output.

    Inputs use matching FP16/BF16/FP32 dtypes and unit vocabulary strides.
    Output is an int32/int64 vector, potentially a strided token column.
    Scratch is contiguous FP32/int32 storage, disjoint from inputs/output.
    """
    if base_logits.ndim != 2 or base_logits.shape != markov_bias.shape:
        raise ValueError("Base logits and Markov bias must have equal 2D shapes")

    if (
        base_logits.dtype != markov_bias.dtype
        or base_logits.dtype
        not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise ValueError("Inputs must have matching FP16/BF16/FP32 dtypes")

    if base_logits.stride(1) != 1 or markov_bias.stride(1) != 1:
        raise ValueError("Vocabulary columns must have unit stride")

    rows, vocab = base_logits.shape
    if vocab <= 0:
        raise ValueError("Vocabulary size must be positive")

    if output.ndim != 1 or output.shape[0] != rows:
        raise ValueError("Output must contain one token ID per request")
    if output.dtype not in (torch.int32, torch.int64):
        raise ValueError("Output must use int32 or int64")

    parts = triton.cdiv(vocab, _BLOCK)
    if (
        partial_values.dtype != torch.float32
        or partial_indices.dtype != torch.int32
        or not partial_values.is_contiguous()
        or not partial_indices.is_contiguous()
        or min(partial_values.numel(), partial_indices.numel()) < rows * parts
    ):
        raise ValueError("Invalid or insufficient reduction scratch")

    if not (
        base_logits.device
        == markov_bias.device
        == output.device
        == partial_values.device
        == partial_indices.device
    ):
        raise ValueError("All tensors must be on the same device")

    if rows == 0:
        return

    _partial_argmax[(rows, parts)](
        base_logits,
        markov_bias,
        partial_values,
        partial_indices,
        base_logits.stride(0),
        markov_bias.stride(0),
        vocab,
        parts,
        _BLOCK,
    )
    _finish_argmax[(rows,)](
        partial_values,
        partial_indices,
        output,
        output.stride(0),
        parts,
        triton.next_power_of_2(parts),
    )