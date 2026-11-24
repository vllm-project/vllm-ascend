# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
"""BF16 and dynamic INT8 LightningIndexer adapters for Qwen3.8 QSA."""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn.functional as F
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

from vllm_ascend.ops.triton.qwen4_exp.c8 import QLI_V2_QUERY_HEADS, prepare_qli_metadata_inputs, prepare_qli_query
from vllm_ascend.ops.triton.qwen4_exp.qsa import pack_qsa_group_indices_npu

from .quantization import qsa_c8_indexer_cache_views

_HEAD_DIM = 128
_COMPRESS_RATIO = 4
_TOKEN_TOPK = 2048
_LI_V1_HEAD_COUNTS = (8, 16, 24, 32, 64)
# BF16 spacing at the old weight 1/sqrt(128) is 2**-11.
_LI_V2_WEIGHT_FRACTION_BITS = 11
_QLI_METADATA_CACHE_KEY = "qsa_c8_indexer_metadata"


class QSAIndexerMetadataProtocol(Protocol):
    """Metadata fields consumed by the Ascend sparse QSA implementation."""

    num_actual_tokens: int
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor


def qsa_select_paged_tokens_lightning_v1(
    query: torch.Tensor,
    compressed_key_cache: torch.Tensor,
    attn_metadata: QSAIndexerMetadataProtocol,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select QSA groups using explicit per-query causal lengths on A2."""
    rows, heads, head_dim = query.shape
    if query.dtype != torch.bfloat16 or head_dim != _HEAD_DIM or not 0 < heads <= _LI_V1_HEAD_COUNTS[-1]:
        raise ValueError("QSA BF16 LightningIndexer V1 requires 1..64 heads of width 128")
    if out is None:
        out = torch.empty((rows, token_topk + compress_ratio - 1), dtype=torch.int32, device=query.device)
    out.fill_(-1)
    if not rows:
        return out

    positions = attn_metadata.logical_positions[:rows]
    token_to_req = attn_metadata.token_to_req[:rows]
    lengths = attn_metadata.seq_lens
    cu_query, visible_groups, row_table = prepare_qli_metadata_inputs(
        attn_metadata.block_table, token_to_req, positions, lengths, compress_ratio
    )
    # V1 needs a supported head count. Zero-weight heads preserve the score;
    # per-row visibility preserves causality in mixed prefill and MTP batches.
    padded_heads = next(count for count in _LI_V1_HEAD_COUNTS if count >= heads)
    weights = torch.full((rows, heads), 1.0 / math.sqrt(head_dim), dtype=query.dtype, device=query.device)
    if padded_heads > heads:
        query = F.pad(query, (0, 0, 0, padded_heads - heads))
        weights = F.pad(weights, (0, padded_heads - heads))
    block_topk = token_topk // compress_ratio
    groups, _ = torch.ops.npu.npu_lightning_indexer.default(
        query,
        compressed_key_cache,
        weights,
        actual_seq_lengths_query=cu_query[1:],
        actual_seq_lengths_key=visible_groups,
        block_table=row_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=block_topk,
        sparse_mode=0,
        pre_tokens=torch.iinfo(torch.int64).max,
        next_tokens=torch.iinfo(torch.int64).max,
        return_value=False,
    )
    pack_qsa_group_indices_npu(
        groups.squeeze(1),
        positions,
        lengths,
        token_to_req,
        compress_ratio,
        token_topk,
        out[:rows, : block_topk + compress_ratio - 1],
    )
    return out


def qsa_select_paged_tokens_lightning_v2(
    query: torch.Tensor,
    compressed_key_cache: torch.Tensor,
    attn_metadata: QSAIndexerMetadataProtocol,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select QSA groups with native request-batched LightningIndexerV2.

    Preserve the original request batch and page table.
    The QSA producer must provide packed, contiguous query suffixes: local row i
    of request r has logical position seq_lens[r] - query_len[r] + i. The upstream
    QSA builder establishes this relationship. Request boundaries start at zero
    and their final live boundary equals the number of live query rows. Positions
    and token_to_req must describe those same rows; arbitrary reordering is not
    supported. These are value-level preconditions, not host tensor reads here.

    Keep padded request capacity and derive the active ranges from device
    query boundaries. Clamp boundaries to the supplied query rows so padded
    requests have zero query length when the caller trims the query tensor.
    Generate the Liv2 schedule from the same boundaries used by the kernel.

    Preserve the caller's INT32 output buffer and fill unused entries with -1.
    For Qwen3.8 (token_topk=2048, compress_ratio=4), each 2051-column row
    stores 512 logical group IDs and up to three original-token tail IDs
    in its first 515 columns.
    Later MTP steps must continue to reuse the buffer without invoking selection.
    """
    # CANN checks even singleton-axis strides strictly. Cache views may have
    # an arbitrary stride on the size-one head axis; normalize metadata only,
    # preserving storage, storage offset and the physical stride between pages.
    compressed_key_cache = canonicalize_singleton_dim_strides(compressed_key_cache)
    # Request counts and packing buffers follow the QSA builder contract.
    # Leave operator input and tiling validation to the upstream implementation.
    rows = query.shape[0]
    requests = attn_metadata.seq_lens.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=query.device)
    out.fill_(-1)
    if not rows:
        return out

    # Register CANN Torch operators lazily in the executing worker.
    import cann_ops_transformer  # noqa: F401

    # Metadata uses original-token lengths. Only request-sized arrays are
    # materialized; there is no token-wise length or page-table gather.
    lengths = attn_metadata.seq_lens[:requests]
    compressed_lengths = lengths.floor_divide(compress_ratio)
    residual_lengths = lengths.remainder(compress_ratio)
    common = dict(
        cu_seqlens_q=attn_metadata.query_start_loc[: requests + 1].clamp_max(rows),
        seqused_k=compressed_lengths,
        cmp_residual_k=residual_lengths,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=compress_ratio,
    )
    block_topk = token_topk // compress_ratio
    # MRV2 prepares one schedule outside the layer loop. A caller that trims
    # the query further must regenerate it for the new boundary instead.
    metadata = getattr(attn_metadata, "qsa_lightning_v2_metadata", None)
    if metadata is None or rows != attn_metadata.num_actual_tokens:
        metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
            query.shape[1],
            1,
            _HEAD_DIM,
            block_topk,
            batch_size=requests,
            **common,
        )
    # Promote the old BF16-representable uniform weight to FP32. Keep its
    # effective value while removing zero-head padding supported by native V2.
    score_weight = math.ldexp(
        round(math.ldexp(1.0 / math.sqrt(_HEAD_DIM), _LI_V2_WEIGHT_FRACTION_BITS)),
        -_LI_V2_WEIGHT_FRACTION_BITS,
    )
    weights = torch.full(
        (rows, query.shape[1]),
        score_weight,
        dtype=torch.float32,
        device=query.device,
    )
    groups, _ = torch.ops.cann_ops_transformer.lightning_indexer(
        query,
        compressed_key_cache,
        weights,
        block_topk,
        block_table=attn_metadata.block_table[:requests],
        metadata=metadata,
        return_value=0,
        **common,
    )
    packed_width = block_topk + compress_ratio - 1
    pack_qsa_group_indices_npu(
        groups.squeeze(1),
        attn_metadata.logical_positions[:rows],
        lengths,
        attn_metadata.token_to_req[:rows],
        compress_ratio,
        token_topk,
        out[:rows, :packed_width],
    )
    return out


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
    cu_query, visible_groups, row_table = prepare_qli_metadata_inputs(
        block_table, token_to_req, query_positions, sequence_lengths, compress_ratio
    )
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=QLI_V2_QUERY_HEADS,
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
    inputs = cu_query, visible_groups, row_table, metadata
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
    if not 0 < query.shape[1] <= QLI_V2_QUERY_HEADS or token_topk != _TOKEN_TOPK or compress_ratio != _COMPRESS_RATIO:
        raise ValueError("QSA INT8 indexer requires 1..64 heads, topk=2048 and ratio=4")
    if block_table.ndim != 2 or not block_table.shape[1] or sequence_lengths.shape != (block_table.shape[0],):
        raise ValueError("QSA INT8 indexer requires request page tables and sequence lengths")
    if query_positions.shape != (query.shape[0],):
        raise ValueError("QSA INT8 indexer positions must match query rows")
    key, key_scales = qsa_c8_indexer_cache_views(compressed_key_cache)
    if key.shape[2:] != (1, _HEAD_DIM):
        raise ValueError("QSA INT8 indexer requires one KV head of width 128")
    rows = query.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=query.device)
    if out.shape != (rows, output_width) or out.dtype != torch.int32:
        raise ValueError("QSA INT8 indexer output must match the packed INT32 ABI")
    out.fill_(-1)
    if not rows:
        return out
    padded_query, padded_scales, weights = prepare_qli_query(query)
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
