# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend kernel for request-major sampling-mask packing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.logger import logger
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors

_BLOCK_SIZE = 4096
_BITS_PER_BYTE = tl.constexpr(8)


@triton.jit
def _pack_sampling_mask_kernel_ascend(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    cu_num_logits_ptr,
    num_sampled_tokens_ptr,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    ROWS_PER_REQUEST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_row = tl.program_id(0)
    req_idx = output_row // ROWS_PER_REQUEST
    slot_idx = output_row % ROWS_PER_REQUEST
    is_active = slot_idx < tl.load(num_sampled_tokens_ptr + req_idx)
    source_row = tl.load(cu_num_logits_ptr + req_idx) + slot_idx
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        # Guard the load: inactive slots may have no source logits row.
        load_mask = (offsets < vocab_size) & is_active
        logits = tl.load(
            logits_ptr + source_row * logits_row_stride + offsets * logits_col_stride,
            mask=load_mask,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf"))
        keep_i32 = keep.to(tl.int32)
        # Triton-Ascend reduces an i1 tensor as a boolean. Cast before the sum
        # so counts contains the number of finite logits, not the tile count.
        count += tl.sum(keep_i32)

        bits = tl.trans(tl.reshape(keep_i32, (BLOCK_SIZE // _BITS_PER_BYTE, _BITS_PER_BYTE)))
        bit_weights = (1 << tl.arange(0, _BITS_PER_BYTE))[:, None]
        packed = tl.sum(bits * bit_weights, axis=0).to(tl.uint8)
        byte_offsets = start_idx // _BITS_PER_BYTE + tl.arange(0, BLOCK_SIZE // _BITS_PER_BYTE)
        tl.store(
            packed_mask_ptr + output_row * packed_mask_row_stride + byte_offsets,
            packed,
            mask=byte_offsets < tl.cdiv(vocab_size, _BITS_PER_BYTE),
        )

    tl.store(counts_ptr + output_row, count)


def sampling_mask_from_logits_npu(
    cls: type[SamplingMaskTensors],
    logits: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_sampled_tokens: torch.Tensor,
    max_num_kept: int,
    rows_per_request: int = 1,
) -> SamplingMaskTensors:
    """Pack committed supports into fixed request-major output slots.

    The caller supplies valid device-side row offsets and sampled counts,
    with each count bounded by rows_per_request and its logits span.
    Compact IDs remain zero-width to use upstream's exact packed-mask fallback.
    """
    if not isinstance(rows_per_request, int) or rows_per_request < 1:
        raise ValueError("rows_per_request must be a positive integer")
    if logits.ndim != 2 or num_sampled_tokens.ndim != 1:
        raise ValueError("expected 2-D logits and 1-D num_sampled_tokens")

    num_reqs = num_sampled_tokens.shape[0]
    if cu_num_logits.ndim != 1 or cu_num_logits.shape[0] < num_reqs + 1:
        raise ValueError("cu_num_logits must contain a row offset per request plus end")
    if logits.stride(1) != 1:
        logits = logits.contiguous()

    vocab_size = logits.shape[1]
    num_output_rows = num_reqs * rows_per_request
    packed_width = triton.cdiv(vocab_size, _BITS_PER_BYTE.value)
    packed_mask = torch.empty(
        (num_output_rows, packed_width),
        dtype=torch.uint8,
        device=logits.device,
    )
    counts = torch.empty(num_output_rows, dtype=torch.int32, device=logits.device)
    if num_output_rows:
        _pack_sampling_mask_kernel_ascend[(num_output_rows,)](
            logits,
            logits.stride(0),
            logits.stride(1),
            cu_num_logits,
            num_sampled_tokens,
            packed_mask,
            packed_mask.stride(0),
            counts,
            vocab_size,
            ROWS_PER_REQUEST=rows_per_request,
            BLOCK_SIZE=_BLOCK_SIZE,
        )

    # Keep compact token IDs zero-width so upstream tolists() decodes the exact
    # packed mask for every non-empty row. This avoids the cumsum-based dynamic
    # scatter that hangs on Triton-Ascend 3.2.2, but may increase CPU decoding work.
    logger.warning_once(
        "Compact sampling-mask token IDs are temporarily unsupported on Ascend; "
        "falling back to exact packed-mask decoding, which may increase CPU work.",
        scope="process",
    )
    token_ids = torch.empty((num_output_rows, 0), dtype=torch.int32, device=logits.device)
    return cls(token_ids, packed_mask, counts, vocab_size, rows_per_request)
