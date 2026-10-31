# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Read rollback-safe PLE history without Torch advanced indexing."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_ngram_context_kernel(
    context_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    context_stride_row,
    context_stride_col,
    mapping_stride,
    lengths_stride,
    tokens_stride_row,
    tokens_stride_col,
    num_reqs,
    eos_token_id,
    CONTEXT_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    if row < num_reqs:
        request_idx = tl.load(idx_mapping_ptr + row * mapping_stride).to(tl.int64)
        # Use the device length after speculative rejection correction. A CPU
        # scheduling upper bound could include tokens that have been rejected.
        context_end = tl.load(num_computed_tokens_ptr + request_idx * lengths_stride).to(tl.int64)
        positions = context_end - CONTEXT_LEN + cols
        valid = (cols < CONTEXT_LEN) & (positions >= 0)
        # Keep masked GM addresses in bounds too: short histories have negative
        # positions, and extra block lanes can extend beyond the history table.
        safe_positions = tl.where(valid, positions, 0)
        token_ids = tl.load(
            all_token_ids_ptr + request_idx * tokens_stride_row + safe_positions * tokens_stride_col,
            mask=valid,
            other=eos_token_id,
        ).to(context_ptr.dtype.element_ty)
    else:
        # No input read at all for padding, even when idx_mapping is empty.
        token_ids = tl.full((BLOCK_SIZE,), eos_token_id, context_ptr.dtype.element_ty)
    # Oldest -> newest, as in the upstream offsets [-CONTEXT_LEN, ..., -1].
    # Inactive request rows are overwritten with EOS, including reused rows.
    tl.store(
        context_ptr + row * context_stride_row + cols * context_stride_col,
        token_ids,
        mask=cols < CONTEXT_LEN,
    )


def gather_ngram_context(
    context: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
    num_reqs: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Fill the existing PLE context on the current device stream.

    ``context`` has one row per padded request. Request-state slots and exact
    computed lengths remain on device; no host read or intermediate tensor is
    needed. The caller retains the same persistent buffer used by model graphs.
    Active ``idx_mapping`` entries are valid request-state slots, as guaranteed
    by the model runner; padded entries are never read.
    """
    padded_reqs, context_len = context.shape
    if padded_reqs and context_len:
        _gather_ngram_context_kernel[(padded_reqs,)](
            context,
            idx_mapping,
            num_computed_tokens,
            all_token_ids,
            context.stride(0),
            context.stride(1),
            idx_mapping.stride(0),
            num_computed_tokens.stride(0),
            all_token_ids.stride(0),
            all_token_ids.stride(1),
            num_reqs,
            eos_token_id,
            CONTEXT_LEN=context_len,
            BLOCK_SIZE=triton.next_power_of_2(context_len),
        )
    return context
