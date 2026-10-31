# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
"""BF16 LightningIndexer adapter for Qwen3.8 QSA."""

from __future__ import annotations

import math

import torch

from vllm_ascend.ops.triton.qwen4_exp.qsa import (
    expand_qsa_block_indices_e3,
    expand_qsa_block_indices_npu,
)

_PA_PAGE_SIZE = 192
_HEAD_DIM = 128
_COMPRESS_RATIO = 4
_TOKEN_TOPK = 2048


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
    *,
    use_e3: bool = False,
) -> torch.Tensor:
    """Select compressed groups with BF16 LightningIndexer.

    Every packed query row is represented as an independent TND sequence.
    Request ownership and causal visibility come directly from QSA metadata;
    neither decode length nor speculative-token count is used as a phase
    heuristic.
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
    if compressed_key_cache.ndim != 4 or compressed_key_cache.shape[1:] != (
        _PA_PAGE_SIZE,
        1,
        _HEAD_DIM,
    ):
        raise ValueError("QSA LightningIndexer requires a [pages,192,1,128] cache")
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

    block_topk = token_topk // compress_ratio
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
    weights = torch.full(
        (rows, query.shape[1]),
        score_weight,
        dtype=torch.float32,
        device=query.device,
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
    expand = expand_qsa_block_indices_e3 if use_e3 else expand_qsa_block_indices_npu
    expand(
        groups,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
        out,
    )
    return out
