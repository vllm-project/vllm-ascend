# SPDX-License-Identifier: Apache-2.0
"""Prepare per-query visible KV lengths for Qwen4Exp Union Prefill."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_qsa_lengths_kernel(
    lengths,
    compressed_lengths,
    residual_lengths,
    requests,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < requests
    values = tl.load(lengths + offsets, mask=mask, other=0).to(tl.int32)
    tl.store(compressed_lengths + offsets, values // COMPRESS_RATIO, mask=mask)
    tl.store(residual_lengths + offsets, values % COMPRESS_RATIO, mask=mask)


@triton.jit
def _prepare_qsa_visible_lengths_kernel(
    positions,
    lengths,
    requests,
    visible_lengths,
    position_stride,
    length_stride,
    request_stride,
    requests_count,
):
    row = tl.program_id(0)
    request = tl.load(requests + row * request_stride).to(tl.int64)
    request = tl.where(request < 0, request + requests_count, request)
    valid_request = (request >= 0) & (request < requests_count)
    safe_request = tl.minimum(tl.maximum(request, 0), requests_count - 1)
    position = tl.load(positions + row * position_stride).to(tl.int32)
    length = tl.load(lengths + safe_request * length_stride, mask=valid_request, other=0).to(tl.int32)
    tl.store(visible_lengths + row, tl.minimum(tl.maximum(position + 1, 0), length))


def prepare_qsa_visible_lengths(
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    visible_lengths: torch.Tensor,
    rows: int,
) -> None:
    """Prepare only visibility for Prefill Union-GBSA once per model step."""
    if rows < 0 or rows > logical_positions.shape[0] or rows > token_to_req.shape[0]:
        raise ValueError("QSA prepared row count exceeds input capacity")
    if visible_lengths.shape[0] < rows:
        raise ValueError("QSA visible-length buffer is too small")
    if rows == 0:
        return
    _prepare_qsa_visible_lengths_kernel[(rows,)](
        logical_positions,
        seq_lens,
        token_to_req,
        visible_lengths,
        logical_positions.stride(0),
        seq_lens.stride(0),
        token_to_req.stride(0),
        seq_lens.shape[0],
        num_warps=1,
    )


def prepare_qsa_lengths(
    lengths: torch.Tensor,
    compressed_lengths: torch.Tensor,
    residual_lengths: torch.Tensor,
    requests: int,
    compress_ratio: int = 4,
) -> None:
    """Prepare LightningIndexer quotient and remainder arrays in one launch."""
    if requests < 0 or requests > lengths.shape[0]:
        raise ValueError("QSA prepared request count exceeds length capacity")
    if compressed_lengths.shape[0] < requests or residual_lengths.shape[0] < requests:
        raise ValueError("QSA prepared length buffers are too small")
    if requests == 0:
        return
    block = 128
    _prepare_qsa_lengths_kernel[(triton.cdiv(requests, block),)](
        lengths,
        compressed_lengths,
        residual_lengths,
        requests,
        COMPRESS_RATIO=compress_ratio,
        BLOCK=block,
        num_warps=1,
    )
