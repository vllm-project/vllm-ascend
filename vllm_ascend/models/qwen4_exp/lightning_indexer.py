# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
"""BF16 LightningIndexer adapter for Qwen3.8 QSA."""

from __future__ import annotations

import math

import torch
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

from vllm_ascend.ops.lightning_indexer_v2 import (
    lightning_indexer_v2,
    lightning_indexer_v2_metadata,
)
from vllm_ascend.ops.triton.qwen4_exp.qsa import pack_qsa_group_indices_npu

_HEAD_DIM = 128
_COMPRESS_RATIO = 4
_TOKEN_TOPK = 2048
_SUPPORTED_HEAD_COUNTS = (8, 16, 24, 32, 64)


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

# LI V2 accepts page sizes that are multiples of 16 in [16, 1024].
_LI_V2_MIN_PAGE_SIZE = 16
_LI_V2_MAX_PAGE_SIZE = 1024
_LI_V2_PAGE_ALIGNMENT = 16
# BF16 spacing at the old weight 1/sqrt(128) is 2**-11.
_LI_V2_WEIGHT_FRACTION_BITS = 11


def _validate_qsa_lightning_v2_inputs(
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
    *,
    num_actual_requests: int | None = None,
    num_actual_tokens: int | None = None,
) -> tuple[int, int]:
    """Validate tensor metadata without device reads; return live request/row counts."""
    if query.ndim != 3 or query.dtype != torch.bfloat16:
        raise ValueError("QSA LI V2 query must be three-dimensional BF16")
    if not 1 <= query.shape[1] <= 64 or query.shape[2] != _HEAD_DIM:
        raise ValueError("QSA LI V2 requires 1..64 heads of width 128")
    if token_topk != _TOKEN_TOPK or compress_ratio != _COMPRESS_RATIO:
        raise ValueError("QSA LI V2 requires token_topk=2048 and compress_ratio=4")
    if (
        not is_qsa_lightning_cache_compatible(compressed_key_cache)
        or compressed_key_cache.dtype != query.dtype
        or not _LI_V2_MIN_PAGE_SIZE <= compressed_key_cache.shape[1] <= _LI_V2_MAX_PAGE_SIZE
        or compressed_key_cache.shape[1] % _LI_V2_PAGE_ALIGNMENT
        or compressed_key_cache.stride(1) != _HEAD_DIM
        or compressed_key_cache.stride(2) != _HEAD_DIM
        or compressed_key_cache.stride(3) != 1
    ):
        raise ValueError(
            "QSA LI V2 requires a dense BF16 [pages,page_size,1,128] cache within each page; "
            f"got shape={tuple(compressed_key_cache.shape)} stride={compressed_key_cache.stride()}"
        )
    if sequence_lengths.ndim != 1 or query_start_loc.ndim != 1 or block_table.ndim != 2:
        raise ValueError("QSA LI V2 request metadata has invalid ranks")
    if any(t.dtype != torch.int32 for t in (sequence_lengths, query_start_loc, block_table)):
        raise ValueError("QSA LI V2 request lengths, boundaries and page table must be INT32")
    if token_to_req.dtype not in (torch.int32, torch.int64) or query_positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("QSA LI V2 row mappings and positions must be integer tensors")
    rows = query.shape[0]
    if token_to_req.shape != (rows,) or query_positions.shape != (rows,):
        raise ValueError("QSA LI V2 row metadata must match query rows")
    tensors = (compressed_key_cache, block_table, token_to_req, query_positions,
               sequence_lengths, query_start_loc)
    if any(t.device != query.device for t in tensors):
        raise ValueError("QSA LI V2 inputs must be on the same device")
    if any(count is not None and not isinstance(count, int)
           for count in (num_actual_requests, num_actual_tokens)):
        raise ValueError("QSA LI V2 live counts must be host integers")
    requests = sequence_lengths.shape[0] if num_actual_requests is None else num_actual_requests
    live_rows = rows if num_actual_tokens is None else num_actual_tokens
    if not 0 <= requests <= sequence_lengths.shape[0] or not 0 <= live_rows <= rows:
        raise ValueError("QSA LI V2 live counts exceed metadata capacity")
    if block_table.shape[0] < requests or query_start_loc.shape[0] < requests + 1:
        raise ValueError("QSA LI V2 metadata does not cover the live requests")
    if num_actual_requests is None:
        _validate_request_boundaries(query, query_start_loc, token_to_req, sequence_lengths)
        if block_table.shape[0] != requests:
            raise ValueError("QSA LI V2 padded requests require explicit host counts")
    if live_rows and (not requests or not block_table.shape[1]):
        raise ValueError("QSA LI V2 nonempty queries require a request page table")
    output_width = token_topk + compress_ratio - 1
    if out is not None and (
        out.shape != (rows, output_width) or out.dtype != torch.int32 or out.device != query.device
    ):
        raise ValueError("QSA LI V2 output must match query rows, INT32 dtype and device")
    return requests, live_rows


def qsa_select_paged_tokens_lightning_v2(
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
    *,
    num_actual_requests: int | None = None,
    num_actual_tokens: int | None = None,
) -> torch.Tensor:
    """Select QSA groups with native request-batched LightningIndexerV2.

    Unlike the old selector, preserve the original request batch and page table.
    The QSA producer must provide packed, contiguous query suffixes: local row i
    of request r has logical position seq_lens[r] - query_len[r] + i. The upstream
    QSA builder establishes this relationship. Request boundaries start at zero
    and their final live boundary equals the number of live query rows. Positions
    and token_to_req must describe those same rows; arbitrary reordering is not
    supported. These are value-level preconditions, not host tensor reads here.

    Optional host counts trim graph-padding requests/rows before V2. Supply
    them from existing CPU scheduling metadata, not by reading NPU tensors.
    Without counts, all input rows and requests are assumed live. Live requests
    and rows must occupy contiguous prefixes of the corresponding inputs.

    Preserve the original [query_rows,2051] INT32 output ABI, including the
    caller's buffer identity. The first 515 columns hold 512 logical group IDs
    and up to three original-token tail IDs; all other/padded entries are -1.
    Later MTP steps must continue to reuse the buffer without invoking selection.
    """
    # CANN checks even singleton-axis strides strictly. Cache views may have
    # an arbitrary stride on the size-one head axis; normalize metadata only,
    # preserving storage, storage offset and the physical stride between pages.
    compressed_key_cache = canonicalize_singleton_dim_strides(compressed_key_cache)
    requests, live_rows = _validate_qsa_lightning_v2_inputs(
        query, compressed_key_cache, block_table, token_to_req, query_positions,
        sequence_lengths, query_start_loc, token_topk, compress_ratio, out,
        num_actual_requests=num_actual_requests,
        num_actual_tokens=num_actual_tokens,
    )
    rows = query.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=query.device)
    out.fill_(-1)
    if not live_rows:
        return out

    # Metadata uses original-token lengths. Only these two request-sized arrays
    # are materialized; there is no token-wise length or page-table gather.
    lengths = sequence_lengths[:requests]
    compressed_lengths = lengths.floor_divide(compress_ratio)
    residual_lengths = lengths.remainder(compress_ratio)
    common = dict(
        cu_seqlens_q=query_start_loc[:requests + 1],
        seqused_k=compressed_lengths,
        cmp_residual_k=residual_lengths,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=compress_ratio,
    )
    block_topk = token_topk // compress_ratio
    metadata = lightning_indexer_v2_metadata(
        query.shape[1], 1, _HEAD_DIM, block_topk,
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
        (live_rows, query.shape[1]), score_weight,
        dtype=torch.float32, device=query.device,
    )
    groups, _ = lightning_indexer_v2(
        query[:live_rows], compressed_key_cache, weights, block_topk,
        block_table=block_table[:requests],
        metadata=metadata,
        return_value=0,
        **common,
    )
    packed_width = block_topk + compress_ratio - 1
    pack_qsa_group_indices_npu(
        groups.squeeze(1), query_positions[:live_rows], lengths,
        token_to_req[:live_rows], compress_ratio, token_topk,
        out[:live_rows, :packed_width],
    )
    return out
