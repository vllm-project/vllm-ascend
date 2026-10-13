# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

# Amortize bounds checks while keeping FP32 tiles within the local memory limit.
OUTPUT_BLOCK_BYTES = 16384


@triton.jit(do_not_specialize=["last_query", "source_tokens", "elements"])
def _write_output(
    source,
    destination,
    query_ends,
    last_query,
    source_tokens,
    elements,
    EMPTY_SOURCE: tl.constexpr,
    TOKEN_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tokens = offsets // TOKEN_WIDTH
    if EMPTY_SOURCE:
        # An empty NPU tensor can have a null data pointer. Avoid generating
        # even a fully masked source load for an all-padding graph batch.
        values = tl.full((BLOCK,), 0, destination.dtype.element_ty)
    elif tl.program_id(0) * BLOCK < source_tokens * TOKEN_WIDTH:
        # DP can pad a decode shard to another shard's prefill token count.
        # Do not issue a masked load whose entire block is beyond the source
        # allocation: the Ascend backend can still access that block address.
        valid_tokens = tl.minimum(tl.load(query_ends + last_query), source_tokens)
        values = tl.load(source + offsets, (offsets < elements) & (tokens < valid_tokens), other=0)
    else:
        values = tl.full((BLOCK,), 0, destination.dtype.element_ty)
    tl.store(destination + offsets, values, offsets < elements)


def write_recurrent_output(source: torch.Tensor, destination: torch.Tensor, query_ends: torch.Tensor) -> None:
    """Copy valid recurrent rows and zero all captured padding in one launch."""
    if destination.numel() == 0:
        return
    block_size = OUTPUT_BLOCK_BYTES // destination.element_size()
    _write_output[(triton.cdiv(destination.numel(), block_size),)](
        source,
        destination,
        query_ends,
        query_ends.numel() - 1,
        source.shape[1],
        destination.numel(),
        source.shape[1] == 0,
        source.shape[2] * source.shape[3],
        block_size,
    )
