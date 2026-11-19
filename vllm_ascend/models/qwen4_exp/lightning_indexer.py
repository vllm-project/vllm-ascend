# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
"""CANN Torch LightningIndexerV2 adapter for Qwen3.8 QSA."""

from __future__ import annotations

import math
from typing import Protocol

import torch
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

from vllm_ascend.ops.triton.qwen4_exp.qsa import pack_qsa_group_indices_npu

_HEAD_DIM = 128
# BF16 spacing at the old weight 1/sqrt(128) is 2**-11.
_LI_V2_WEIGHT_FRACTION_BITS = 11


class QSAIndexerMetadataProtocol(Protocol):
    """Metadata fields consumed by the Ascend sparse QSA implementation."""

    num_actual_tokens: int
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    token_to_req: torch.Tensor
    logical_positions: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor


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
