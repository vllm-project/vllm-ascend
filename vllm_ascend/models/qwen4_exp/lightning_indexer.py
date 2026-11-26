# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
"""CANN Torch LightningIndexerV2 adapter for Qwen3.8 QSA."""

from __future__ import annotations

import math
from typing import Any, Protocol, cast

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


class _DCPIndexerMetadata(QSAIndexerMetadataProtocol, Protocol):
    dcp_interleave: int


def qsa_lightning_indexer_args(
    metadata: QSAIndexerMetadataProtocol, rows: int, compress_ratio: int, dcp_inputs=None
) -> dict[str, Any]:
    """Use the same causal bounds for the LI-v2 schedule and the operator."""
    common: dict[str, Any] = {"layout_q": "TND", "layout_k": "PA_BBND"}
    if dcp_inputs is None:
        lengths = metadata.seq_lens
        common.update(
            cu_seqlens_q=metadata.query_start_loc[: lengths.shape[0] + 1].clamp_max(rows),
            seqused_k=lengths.floor_divide(compress_ratio),
            cmp_residual_k=lengths.remainder(compress_ratio),
            mask_mode=3,
            cmp_ratio=compress_ratio,
        )
    else:
        cu_q, local_lengths, _ = dcp_inputs
        common.update(cu_seqlens_q=cu_q, seqused_k=local_lengths, mask_mode=0, cmp_ratio=1)
    return common


def qsa_select_paged_tokens_lightning_v2(
    query: torch.Tensor,
    compressed_key_cache: torch.Tensor,
    attn_metadata: QSAIndexerMetadataProtocol,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    *,
    dcp_group=None,
) -> torch.Tensor:
    """Select QSA groups with BF16 LightningIndexerV2.

    DCP searches the local compressed prefix per query and merges candidates
    with distributed_topk. The unsharded path preserves the original request batch.
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
    stores up to 512 winning group IDs (rank-local under DCP), followed by
    up to three global residual token IDs in its first 515 columns.
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

    sharded = dcp_group is not None and dcp_group.world_size > 1
    inputs = None
    if sharded:
        from vllm_ascend.models.qwen4_exp.qsa_cp import qsa_dcp_indexer_inputs

        inputs = (
            getattr(attn_metadata, "qsa_dcp_indexer_inputs", None) if rows == attn_metadata.num_actual_tokens else None
        )
        if inputs is None:
            inputs = qsa_dcp_indexer_inputs(attn_metadata, compress_ratio, rows)
        table = inputs[2]
    else:
        table = attn_metadata.block_table[:requests]
    common = qsa_lightning_indexer_args(attn_metadata, rows, compress_ratio, inputs)
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
            batch_size=table.shape[0],
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
    groups, scores = torch.ops.cann_ops_transformer.lightning_indexer(
        query,
        compressed_key_cache,
        weights,
        block_topk,
        block_table=table,
        metadata=metadata,
        return_value=int(sharded),
        **common,
    )
    if sharded:
        from vllm_ascend.attention.context_parallel.sharded_indexer import distributed_topk
        from vllm_ascend.attention.utils import build_valid_topk_mask

        # Indexer query heads are replicated across TP. Exchange only candidates.
        valid = build_valid_topk_mask(groups, common["seqused_k"][:, None, None])
        group_interleave = cast(_DCPIndexerMetadata, attn_metadata).dcp_interleave // compress_ratio
        groups = distributed_topk(
            groups.masked_fill(~valid, -1),
            scores.masked_fill(~valid, -torch.inf),
            dcp_group,
            block_topk,
            group_interleave,
            return_local=True,
        )
    packed_width = block_topk + compress_ratio - 1
    pack_qsa_group_indices_npu(
        groups.squeeze(1),
        attn_metadata.logical_positions[:rows],
        attn_metadata.seq_lens,
        attn_metadata.token_to_req[:rows],
        compress_ratio,
        token_topk,
        out[:rows, :packed_width],
    )
    return out
