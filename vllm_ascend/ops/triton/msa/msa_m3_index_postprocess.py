# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused TopK helpers around the A2/A3 AscendC MsaIndexScore operator."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

_SPARSE_BLOCK_SIZE = 128
_SCORE_ALIGNMENT = 16
_PREPARE_TILE = 256


@triton.jit
def _ascendc_decode_score_prepare_kernel(
    score_ptr,
    prepared_ptr,
    context_ptr,
    stride_sh,
    stride_sq,
    stride_sk,
    stride_ph,
    stride_pq,
    stride_pk,
    stride_context,
    logical_blocks,
    prepared_width,
    decode_query_len,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    query = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.program_id(2) * TILE + tl.arange(0, TILE)
    request = query // decode_query_len
    offset = query % decode_query_len
    context = tl.load(context_ptr + request * stride_context).to(tl.int32)
    visible_blocks = tl.maximum((context + offset + BLOCK_SIZE) // BLOCK_SIZE, 0)
    valid = (cols < logical_blocks) & (cols < visible_blocks)

    # Never load unwritten padding, including full-graph zero-length rows.
    scores = tl.load(
        score_ptr + head * stride_sh + query * stride_sq + cols * stride_sk,
        mask=valid,
        other=float("-inf"),
    )
    if INIT_BLOCKS > 0:
        scores = tl.where(valid & (cols < INIT_BLOCKS), 1.0e30, scores)
    if LOCAL_BLOCKS > 0:
        local_start = tl.maximum(visible_blocks - LOCAL_BLOCKS, 0)
        # Preserve the existing precedence: local blocks override init blocks.
        scores = tl.where(valid & (cols >= local_start), 1.0e29, scores)
    tl.store(
        prepared_ptr + head * stride_ph + query * stride_pq + cols * stride_pk,
        scores,
        mask=cols < prepared_width,
    )


@triton.jit
def _ascendc_decode_topk_postprocess_kernel(
    raw_ptr,
    indices_ptr,
    counts_ptr,
    context_ptr,
    stride_rh,
    stride_rq,
    stride_rk,
    stride_ih,
    stride_iq,
    stride_ik,
    stride_ch,
    stride_cq,
    stride_context,
    logical_blocks,
    decode_query_len,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    query = tl.program_id(0)
    head = tl.program_id(1)
    slots = tl.arange(0, TILE)
    context = tl.load(context_ptr + (query // decode_query_len) * stride_context).to(tl.int32)
    visible_blocks = tl.maximum((context + query % decode_query_len + BLOCK_SIZE) // BLOCK_SIZE, 0)
    visible_blocks = tl.minimum(visible_blocks, logical_blocks)
    raw = tl.load(
        raw_ptr + head * stride_rh + query * stride_rq + slots * stride_rk,
        mask=slots < TOPK,
        other=-1,
    )
    valid = (slots < TOPK) & (raw >= 0) & (raw < visible_blocks)
    tl.store(
        indices_ptr + head * stride_ih + query * stride_iq + slots * stride_ik,
        tl.where(valid, raw, -1).to(tl.int32),
        mask=slots < TOPK,
    )
    count = tl.sum(valid.to(tl.int32), axis=0)
    tl.store(counts_ptr + head * stride_ch + query * stride_cq, count)


def index_decode_topk_fused(
    score: torch.Tensor,
    context_lens: torch.Tensor,
    decode_query_len: int,
    topk: int,
    logical_block_count: int,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int32 block IDs and valid counts for A2/A3 decode."""
    if score.ndim != 3 or score.dtype != torch.float32:
        raise ValueError("Expected a [heads, query_tokens, score_width] float32 tensor")
    if context_lens.ndim != 1 or context_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected one int32/int64 context length per request")
    if score.device != context_lens.device:
        raise ValueError("Scores and context lengths must be on the same device")
    if decode_query_len <= 0 or score.shape[1] != context_lens.numel() * decode_query_len:
        raise ValueError("Decode tokens must equal request count times decode_query_len")
    if not 0 <= logical_block_count <= score.shape[-1] or topk <= 0:
        raise ValueError("Invalid logical block count or TopK")
    if init_blocks < 0 or local_blocks < 0:
        raise ValueError("Forced block counts must be non-negative")

    heads, tokens, width = score.shape
    if heads == 0 or tokens == 0:
        return (
            torch.empty((heads, tokens, topk), dtype=torch.int32, device=score.device),
            torch.empty((heads, tokens), dtype=torch.int32, device=score.device),
        )

    prepared_width = max(
        width,
        (topk + _SCORE_ALIGNMENT - 1) // _SCORE_ALIGNMENT * _SCORE_ALIGNMENT,
    )
    prepared = (
        score
        if score.is_contiguous() and prepared_width == width
        else torch.empty(
            (heads, tokens, prepared_width),
            dtype=score.dtype,
            device=score.device,
        )
    )
    _ascendc_decode_score_prepare_kernel[(tokens, heads, triton.cdiv(prepared_width, _PREPARE_TILE))](
        score,
        prepared,
        context_lens,
        *score.stride(),
        *prepared.stride(),
        context_lens.stride(0),
        logical_block_count,
        prepared_width,
        decode_query_len,
        INIT_BLOCKS=init_blocks,
        LOCAL_BLOCKS=local_blocks,
        BLOCK_SIZE=_SPARSE_BLOCK_SIZE,
        TILE=_PREPARE_TILE,
    )

    # TopK sees one contiguous, fully masked width; no score slice/copy is needed.
    _, raw_indices = torch.topk(prepared, k=topk, dim=-1)
    indices = torch.empty(
        (heads, tokens, topk),
        dtype=torch.int32,
        device=score.device,
    )
    counts = torch.empty(
        (heads, tokens),
        dtype=torch.int32,
        device=score.device,
    )
    _ascendc_decode_topk_postprocess_kernel[(tokens, heads)](
        raw_indices,
        indices,
        counts,
        context_lens,
        *raw_indices.stride(),
        *indices.stride(),
        *counts.stride(),
        context_lens.stride(0),
        logical_block_count,
        decode_query_len,
        TOPK=topk,
        BLOCK_SIZE=_SPARSE_BLOCK_SIZE,
        TILE=triton.next_power_of_2(topk),
    )
    return indices, counts
