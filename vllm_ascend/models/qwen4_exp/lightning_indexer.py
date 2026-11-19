# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
"""BF16 and dynamic INT8 LightningIndexer adapters for Qwen3.8 QSA."""

from __future__ import annotations

import math

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available

from vllm_ascend.ops.triton.qwen4_exp.c8 import quantize_qsa_query
from vllm_ascend.ops.triton.qwen4_exp.qsa import pack_qsa_group_indices_npu

from .quantization import qsa_c8_indexer_cache_views

_HEAD_DIM = 128
_COMPRESS_RATIO = 4
_TOKEN_TOPK = 2048
_SUPPORTED_HEAD_COUNTS = (8, 16, 24, 32, 64)
_QLI_METADATA_CACHE_KEY = "qsa_c8_indexer_metadata"


def is_qsa_lightning_cache_compatible(cache: torch.Tensor) -> bool:
    """Return whether a dynamically sized QSA page satisfies the LI ABI."""
    return bool(cache.ndim == 4 and cache.shape[0] > 0 and cache.shape[1] > 0 and cache.shape[2:] == (1, _HEAD_DIM))


def _validate_request_boundaries(
    query: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_to_req: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> None:
    """Validate structural request metadata without synchronizing the NPU."""
    if query_start_loc.ndim != 1:
        raise ValueError("QSA query_start_loc must be one-dimensional")
    if query_start_loc.shape[0] != sequence_lengths.shape[0] + 1:
        raise ValueError("QSA query_start_loc must have one boundary per request")
    if token_to_req.shape != (query.shape[0],):
        raise ValueError("QSA request mapping must match query rows")


def qsa_select_paged_tokens_lightning(
    query: torch.Tensor,
    compressed_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select compressed groups with BF16 LightningIndexer.

    Every packed query row is represented as an independent TND sequence.
    Request ownership and causal visibility come directly from QSA metadata;
    neither decode length nor speculative-token count is used as a phase
    heuristic.

    The upstream 2051-column output ABI is preserved while the compact
    515-column group selection occupies its prefix.
    """
    _validate_request_boundaries(
        query,
        query_start_loc,
        token_to_req,
        sequence_lengths,
    )
    if query.dtype != torch.bfloat16 or query.ndim != 3:
        raise ValueError("QSA LightningIndexer query must be three-dimensional BF16")
    if query.shape[1] > 64 or query.shape[2] != _HEAD_DIM:
        raise ValueError("QSA LightningIndexer requires at most 64 heads of width 128")
    if not is_qsa_lightning_cache_compatible(compressed_key_cache):
        raise ValueError(
            "QSA LightningIndexer requires a [pages,page_size,1,128] cache "
            "with positive pages and page_size, "
            f"got shape={tuple(compressed_key_cache.shape)} "
            f"stride={compressed_key_cache.stride()}"
        )
    if compress_ratio != _COMPRESS_RATIO or token_topk != _TOKEN_TOPK:
        raise ValueError("QSA LightningIndexer requires ratio=4 and token_topk=2048")
    if block_table.ndim != 2 or sequence_lengths.shape != (block_table.shape[0],):
        raise ValueError("QSA LightningIndexer request metadata has invalid shapes")
    if query_positions.shape != token_to_req.shape:
        raise ValueError("QSA LightningIndexer positions must match query rows")

    rows = query.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=query.device)
    elif out.shape != (rows, output_width):
        raise ValueError("QSA LightningIndexer output has an invalid shape")
    if not rows:
        return out

    # Keep unused ABI columns deterministic; MTP copies complete rows.
    out.fill_(-1)
    block_topk = token_topk // compress_ratio
    packed_width = block_topk + compress_ratio - 1
    score_weight = 1.0 / math.sqrt(query.shape[2])
    row_requests = token_to_req.to(torch.int64)
    row_sequence_lengths = sequence_lengths[row_requests]
    visible_groups = torch.minimum(
        (query_positions + 1).floor_divide(compress_ratio),
        row_sequence_lengths.floor_divide(compress_ratio),
    ).to(torch.int32)
    row_block_table = block_table[row_requests]
    query_cu_seqlens = torch.arange(
        1,
        rows + 1,
        dtype=torch.int32,
        device=query.device,
    )
    original_heads = query.shape[1]
    padded_heads = next(heads for heads in _SUPPORTED_HEAD_COUNTS if heads >= original_heads)
    weights = torch.full(
        (rows, original_heads),
        score_weight,
        dtype=query.dtype,
        device=query.device,
    )
    if padded_heads != original_heads:
        query = torch.cat(
            (
                query,
                torch.zeros(
                    (rows, padded_heads - original_heads, query.shape[2]),
                    dtype=query.dtype,
                    device=query.device,
                ),
            ),
            dim=1,
        )
        weights = torch.cat(
            (
                weights,
                torch.zeros(
                    (rows, padded_heads - original_heads),
                    dtype=query.dtype,
                    device=query.device,
                ),
            ),
            dim=1,
        )
    groups, _ = torch.ops.npu.npu_lightning_indexer.default(
        query,
        compressed_key_cache,
        weights,
        actual_seq_lengths_query=query_cu_seqlens,
        actual_seq_lengths_key=visible_groups,
        block_table=row_block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=block_topk,
        sparse_mode=0,
        pre_tokens=9223372036854775807,
        next_tokens=9223372036854775807,
        return_value=False,
    )
    groups = groups.squeeze(1)
    pack_qsa_group_indices_npu(
        groups,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
        out[:, :packed_width],
    )
    return out


def _prepare_quantized_indexer_metadata(
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    group_topk: int,
    compress_ratio: int,
    max_seqlen_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = None
    cache_key = (block_table, token_to_req, query_positions, sequence_lengths, group_topk, compress_ratio, max_seqlen_k)
    context = get_forward_context() if is_forward_context_available() else None
    if context is not None and context.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE:
        # The scheduler keeps these tensors unchanged during a forward.
        # Tensor identity distinguishes cache groups, and each MTP substep
        # gets its own context. PIECEWISE capture can split producers from
        # consumers, so metadata stays local in that mode.
        cache = context.additional_kwargs.setdefault(_QLI_METADATA_CACHE_KEY, {})
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    rows = query_positions.shape[0]
    requests = token_to_req.long()
    visible_groups = torch.minimum(
        (query_positions + 1).clamp_min(0).floor_divide(compress_ratio),
        sequence_lengths[requests].floor_divide(compress_ratio),
    ).to(torch.int32)
    cu_query = torch.arange(rows + 1, dtype=torch.int32, device=query_positions.device)
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=64,
        num_heads_k=1,
        head_dim=_HEAD_DIM,
        topk=group_topk,
        batch_size=rows,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
        quant_mode=2,
        cu_seqlens_q=cu_query,
        seqused_k=visible_groups,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=0,
        cmp_ratio=1,
    )
    inputs = cu_query, visible_groups, block_table[requests].contiguous(), metadata
    if cache is not None:
        cache[cache_key] = inputs
    return inputs


def qsa_select_paged_tokens_quantized(
    query: torch.Tensor,
    compressed_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """INT8 QLI-v2 with explicit per-row visibility in compressed-key units."""
    _validate_request_boundaries(query, query_start_loc, token_to_req, sequence_lengths)
    if query.ndim != 3 or query.dtype != torch.bfloat16 or query.shape[-1] != _HEAD_DIM:
        raise ValueError("QSA INT8 indexer requires BF16 queries with width 128")
    if not 0 < query.shape[1] <= 64 or token_topk != _TOKEN_TOPK or compress_ratio != _COMPRESS_RATIO:
        raise ValueError("QSA INT8 indexer requires 1..64 heads, topk=2048 and ratio=4")
    if block_table.ndim != 2 or not block_table.shape[1] or sequence_lengths.shape != (block_table.shape[0],):
        raise ValueError("QSA INT8 indexer requires request page tables and sequence lengths")
    if query_positions.shape != (query.shape[0],):
        raise ValueError("QSA INT8 indexer positions must match query rows")
    key, key_scales = qsa_c8_indexer_cache_views(compressed_key_cache)
    if key.shape[2:] != (1, _HEAD_DIM):
        raise ValueError("QSA INT8 indexer requires one KV head of width 128")
    rows, original_heads = query.shape[:2]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=query.device)
    if out.shape != (rows, output_width) or out.dtype != torch.int32:
        raise ValueError("QSA INT8 indexer output must match the packed INT32 ABI")
    out.fill_(-1)
    if not rows:
        return out
    query_codes, query_scales = quantize_qsa_query(query)
    # A2/A3 QLI-v2 requires 64 query heads and FP16 scales/weights. All
    # additional heads have zero weight, preserving the unpadded score sum.
    padded_query = torch.zeros((rows, 64, _HEAD_DIM), dtype=torch.int8, device=query.device)
    padded_query[:, :original_heads].copy_(query_codes)
    padded_scales = torch.ones((rows, 64), dtype=torch.float16, device=query.device)
    padded_scales[:, :original_heads].copy_(query_scales)
    weights = torch.zeros((rows, 64), dtype=torch.float16, device=query.device)
    weights[:, :original_heads] = 1.0 / math.sqrt(_HEAD_DIM)
    # Preserve explicit causal visibility in compressed-key units. The
    # per-row lengths, page table and CANN metadata are shared across layers.
    group_topk = token_topk // compress_ratio
    cu_query, visible_groups, row_table, metadata = _prepare_quantized_indexer_metadata(
        block_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        group_topk,
        compress_ratio,
        block_table.shape[1] * key.shape[1],
    )
    common = dict(
        quant_mode=2,
        cu_seqlens_q=cu_query,
        seqused_k=visible_groups,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=0,
        cmp_ratio=1,
    )
    groups, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query=padded_query,
        key=key,
        weights=weights,
        query_dequant_scale=padded_scales,
        key_dequant_scale=key_scales,
        topk=group_topk,
        block_table=row_table,
        metadata=metadata,
        return_value=0,
        **common,
    )
    pack_qsa_group_indices_npu(
        groups.squeeze(1),
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
        out[:, : group_topk + compress_ratio - 1],
    )
    return out
