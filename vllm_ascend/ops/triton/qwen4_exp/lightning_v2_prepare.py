# SPDX-License-Identifier: Apache-2.0
"""Prepare request-batched QSA inputs for native LightningIndexerV2."""

import math

import torch
from vllm.triton_utils import tl, triton

_HEAD_DIM = 128
# Preserve the old BF16 weight, promoted to the FP32 input required by V2.
_WEIGHT_FRACTION_BITS = 11


@triton.jit
def _prepare_qsa_lightning_v2_inputs_kernel(
    lengths_ptr,
    starts_ptr,
    weights_ptr,
    cu_query_ptr,
    compressed_ptr,
    residual_ptr,
    rows,
    requests,
    weight_elements,
    length_stride,
    start_stride,
    COMPRESS_RATIO: tl.constexpr,
    SCORE_WEIGHT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    lengths = tl.load(lengths_ptr + offsets * length_stride, offsets < requests, other=0)
    # Sequence lengths are nonnegative, so truncation and floor agree.
    compressed = lengths // COMPRESS_RATIO
    residual = lengths - compressed * COMPRESS_RATIO
    tl.store(compressed_ptr + offsets, compressed, offsets < requests)
    tl.store(residual_ptr + offsets, residual, offsets < requests)
    starts = tl.load(starts_ptr + offsets * start_stride, offsets <= requests, other=0)
    tl.store(cu_query_ptr + offsets, tl.minimum(starts, rows), offsets <= requests)
    tl.store(weights_ptr + offsets, SCORE_WEIGHT, offsets < weight_elements)


def prepare_qsa_lightning_v2_inputs(
    query: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FP32 weights, clamped query boundaries, compressed and tail lengths.

    Query contents are not read or copied. Request-sized outputs preserve the
    integer input dtypes; all outputs are fresh contiguous tensors. Sequence
    lengths must be nonnegative and boundaries describe packed query suffixes.
    """
    if query.ndim != 3 or query.shape[2] != _HEAD_DIM or query.shape[1] == 0:
        raise ValueError("QSA LightningV2 preparation requires query [tokens, heads, 128]")
    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")
    if sequence_lengths.ndim != 1 or query_start_loc.ndim != 1:
        raise ValueError("QSA LightningV2 lengths and boundaries must be vectors")
    requests = sequence_lengths.numel()
    if query_start_loc.numel() < requests + 1:
        raise ValueError("QSA LightningV2 requires requests + 1 query boundaries")
    for tensor in (sequence_lengths, query_start_loc):
        if tensor.dtype not in (torch.int32, torch.int64) or tensor.device != query.device:
            raise ValueError("QSA LightningV2 metadata must be INT32/INT64 on the query device")
    rows, heads = query.shape[:2]
    weights = torch.empty((rows, heads), dtype=torch.float32, device=query.device)
    cu_query = torch.empty((requests + 1,), dtype=query_start_loc.dtype, device=query.device)
    compressed = torch.empty((requests,), dtype=sequence_lengths.dtype, device=query.device)
    residual = torch.empty_like(compressed)
    score_weight = math.ldexp(
        round(math.ldexp(1.0 / math.sqrt(_HEAD_DIM), _WEIGHT_FRACTION_BITS)), -_WEIGHT_FRACTION_BITS
    )
    # One vector tile covers small request batches; larger query weights scale
    # through the grid. Counts are runtime arguments, but the backend may still
    # specialize on their types and default scalar specialization classes.
    block_size = 256
    grid = (triton.cdiv(max(requests + 1, weights.numel()), block_size),)
    _prepare_qsa_lightning_v2_inputs_kernel[grid](
        sequence_lengths,
        query_start_loc,
        weights,
        cu_query,
        compressed,
        residual,
        rows,
        requests,
        weights.numel(),
        sequence_lengths.stride(0),
        query_start_loc.stride(0),
        COMPRESS_RATIO=compress_ratio,
        SCORE_WEIGHT=score_weight,
        BLOCK_SIZE=block_size,
    )
    return weights, cu_query, compressed, residual
