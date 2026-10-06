# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qualified A5 QLI/QSLI adapters."""

from __future__ import annotations

from vllm_ascend.ops.dsv41_a5 import dsl as ops
from vllm_ascend.ops.triton.prepare_indexer_indices import (
    prepare_indexer_indices,
)
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, wait_for_device_metadata

from .quantization import mxfp4_quantize_e8m0


def _prepare_indices(
    selected,
    positions,
    compress_ratio,
    *,
    topk_lengths,
    indices_output,
):
    return prepare_indexer_indices(
        selected,
        positions,
        compress_ratio,
        indices_output=indices_output,
        lengths_output=topk_lengths,
    )


def _common(query, weights, source_cache, source_metadata, compress_ratio):
    op_metadata = source_metadata.qli_metadata
    if op_metadata is None:
        raise RuntimeError("A5 QLI metadata was not built")
    wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(op_metadata))
    q_data, q_scale = mxfp4_quantize_e8m0(query)
    query_start_loc = source_metadata.query_start_loc
    return (
        q_data,
        q_scale.unflatten(-1, (2, 2)).contiguous(),
        weights.float().contiguous(),
        source_cache[0],
        source_cache[1].unflatten(-1, (2, 2)),
        dict(
            cu_seqlens_q=query_start_loc,
            seqused_k=source_metadata.cache_seq_lens,
            cmp_residual_k=(source_metadata.cmp_residual if compress_ratio != 1 else None),
            block_table=source_metadata.block_table,
            metadata=op_metadata,
            max_seqlen_q=-1,
            mask_mode=3,
            cmp_ratio=compress_ratio,
            layout_q="TND",
            return_value=False,
        ),
    )


def _qli(
    query,
    weights,
    positions,
    source_cache,
    source_metadata,
    *,
    topk,
    compress_ratio,
    is_candidate_source,
    candidate_topk_blocks,
    candidate_block_size,
    candidates,
    candidate_lengths,
    topk_lengths,
    indices_output,
):
    q, qs, w, k, ks, common = _common(
        query,
        weights,
        source_cache,
        source_metadata,
        compress_ratio,
    )
    common["layout_k"] = "PA_BBND"
    indices, _, candidate_out, candidate_length = ops.quant_lightning_indexer(
        q,
        k,
        w,
        qs,
        ks,
        topk,
        1,
        candidate_topk_blocks=(candidate_topk_blocks if is_candidate_source else -1),
        candidate_block_size=(candidate_block_size if is_candidate_source else -1),
        **common,
    )
    if is_candidate_source:
        candidate_lengths.copy_(candidate_length)
    selected = _prepare_indices(
        indices.squeeze(1),
        positions,
        compress_ratio,
        topk_lengths=topk_lengths,
        indices_output=indices_output,
    )
    return selected, candidate_out if is_candidate_source else candidates


def _qsli(
    query,
    weights,
    positions,
    source_cache,
    source_metadata,
    *,
    topk,
    compress_ratio,
    candidate_block_size,
    candidates,
    candidate_lengths,
    topk_lengths,
    indices_output,
):
    if len(source_cache) != 3:
        raise RuntimeError("A5 QSLI requires its source's folded K/scale twin")
    q, qs, w, _, _, common = _common(
        query,
        weights,
        source_cache,
        source_metadata,
        compress_ratio,
    )
    common.pop("metadata")
    metadata = ops.quant_sparse_lightning_indexer_metadata(
        candidate_lengths,
        cu_seqlens_q=common["cu_seqlens_q"],
        seqused_k=common["seqused_k"],
        cmp_residual_k=common["cmp_residual_k"],
        batch_size=source_metadata.num_reqs,
        max_seqlen_q=-1,
        max_seqlen_k=-1,
        num_heads_q=query.shape[1],
        num_heads_k=1,
        head_dim=query.shape[2],
        topk=topk,
        quant_mode=1,
        candidate_block_size=candidate_block_size,
        mask_mode=3,
        cmp_ratio=compress_ratio,
        layout_q="TND",
        layout_k="PA_BBND",
    )
    common["metadata"] = metadata
    common["layout_k"] = "PA_BBND"
    indices, _ = ops.quant_sparse_lightning_indexer(
        q,
        source_cache[2].squeeze(2),
        w,
        qs,
        candidates,
        candidate_lengths,
        topk,
        1,
        candidate_block_size,
        **common,
    )
    selected = _prepare_indices(
        indices.squeeze(1),
        positions,
        compress_ratio,
        topk_lengths=topk_lengths,
        indices_output=indices_output,
    )
    return selected, candidates


def run_a5_indexer(
    query,
    weights,
    positions,
    source_cache,
    source_metadata,
    *,
    topk,
    compress_ratio,
    is_candidate_source,
    uses_candidate_filter,
    candidate_topk_blocks,
    candidate_block_size,
    candidates,
    candidate_lengths,
    topk_lengths,
    indices_output,
):
    if uses_candidate_filter:
        return _qsli(
            query,
            weights,
            positions,
            source_cache,
            source_metadata,
            topk=topk,
            compress_ratio=compress_ratio,
            candidate_block_size=candidate_block_size,
            candidates=candidates,
            candidate_lengths=candidate_lengths,
            topk_lengths=topk_lengths,
            indices_output=indices_output,
        )
    return _qli(
        query,
        weights,
        positions,
        source_cache,
        source_metadata,
        topk=topk,
        compress_ratio=compress_ratio,
        is_candidate_source=is_candidate_source,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        candidates=candidates,
        candidate_lengths=candidate_lengths,
        topk_lengths=topk_lengths,
        indices_output=indices_output,
    )
