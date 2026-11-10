# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed-quant cache and attention ABI selected by the A5 device adaptor."""

from __future__ import annotations

import torch

from vllm_ascend.ops import packaged_attention as ops
from vllm_ascend.ops.triton.build_window_indices import (
    build_window_indices_triton,
)
from vllm_ascend.ops.triton.packed_cache_slot_mapping import build_packed_cache_slot_mapping
from vllm_ascend.ops.triton.prepare_indexer_indices import (
    prepare_indexer_indices,
)
from vllm_ascend.ops.triton.quantize_mxfp4_indexer import (
    quantize_mxfp4_indexer,
    write_mxfp4_indexer_cache,
)
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, wait_for_device_metadata


def _common(query, weights, source_cache, source_metadata, compress_ratio, quantized_query, query_scale):
    op_metadata = source_metadata.qli_metadata
    if op_metadata is None:
        raise RuntimeError("A5 QLI metadata was not built")
    wait_for_device_metadata(DeviceMetadataStage.INDEXER, id(op_metadata))
    if quantized_query is None:
        quantized_query, query_scale = quantize_mxfp4_indexer(query)
    query_start_loc = source_metadata.query_start_loc
    return (
        quantized_query,
        query_scale.unflatten(-1, (2, 2)).contiguous(),
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
    quantized_query,
    query_scale,
):
    q, qs, w, k, ks, common = _common(
        query,
        weights,
        source_cache,
        source_metadata,
        compress_ratio,
        quantized_query,
        query_scale,
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
    selected = prepare_indexer_indices(
        indices.squeeze(1),
        positions,
        compress_ratio,
        lengths_output=topk_lengths,
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
    quantized_query,
    query_scale,
):
    if len(source_cache) != 3:
        raise RuntimeError("A5 QSLI requires its source's folded K/scale twin")
    q, qs, w, _, _, common = _common(
        query,
        weights,
        source_cache,
        source_metadata,
        compress_ratio,
        quantized_query,
        query_scale,
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
        num_heads_q=q.shape[1],
        num_heads_k=1,
        head_dim=q.shape[2] * 2,
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
    selected = prepare_indexer_indices(
        indices.squeeze(1),
        positions,
        compress_ratio,
        lengths_output=topk_lengths,
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
    quantized_query=None,
    query_scale=None,
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
            quantized_query=quantized_query,
            query_scale=query_scale,
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
        quantized_query=quantized_query,
        query_scale=query_scale,
    )


def build_smla_metadata(length_rows: torch.Tensor, cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    """Build the fixed A5 mixed-quant SMLA launch metadata."""
    return ops.mixed_quant_sparse_flash_mla_metadata(
        length_rows,
        length_rows,
        cu_seqlens_q=cu_seqlens_q,
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        quant_mode=1,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )


def _resolve_window_indices(q, metadata, window_size):
    indices = metadata.swa.ori_sparse_indices
    lengths = metadata.swa.ori_topk_length
    if indices is None or lengths is None:
        return build_window_indices_triton(
            metadata.positions[: q.shape[0]],
            window_size,
        )
    return indices[: q.shape[0]], lengths[: q.shape[0]]


def qsmla(
    q,
    ori_kv,
    cmp_kv,
    metadata,
    compressed_indices,
    *,
    window_size,
    sinks,
    softmax_scale,
    compressed_lengths=None,
):
    ori_indices, ori_lengths = _resolve_window_indices(q, metadata, window_size)
    has_cmp = cmp_kv is not None
    if has_cmp:
        cmp_indices = compressed_indices[:, None, :].to(torch.int32).contiguous()
        cmp_lengths = compressed_lengths
    else:
        cmp_indices = None
        cmp_lengths = None
    task_metadata = metadata.swa.smla_metadata
    wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(task_metadata))
    output, _ = ops.mixed_quant_sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_indices,
        cmp_sparse_indices=cmp_indices,
        ori_block_table=metadata.swa.block_table,
        cmp_block_table=metadata.attention.block_table if has_cmp else None,
        cu_seqlens_q=metadata.swa.query_start_loc,
        seqused_ori_kv=metadata.swa.seq_lens,
        seqused_cmp_kv=metadata.attention.cache_seq_lens if has_cmp else None,
        ori_topk_length=ori_lengths,
        cmp_topk_length=cmp_lengths if has_cmp else None,
        sinks=sinks.detach().float().contiguous(),
        metadata=task_metadata,
        quant_mode=1,
        softmax_scale=softmax_scale,
        layout_q="TND",
        layout_kv="PA_BBND",
        return_softmax_lse=False,
    )
    return output


def write_attention_cache(
    cache: torch.Tensor,
    flat_slots: torch.Tensor,
    values: torch.Tensor,
    *,
    kind: str,
) -> None:
    if kind == "cmp":
        cache_arg = cache
        group_size = 16
        quant_mode = "mxfp4_bf16"
    elif kind == "win":
        cache_arg = cache.view(torch.float8_e4m3fn)
        group_size = 32
        quant_mode = "mxfp8_bf16"
    else:
        raise ValueError(f"unsupported A5 cache kind: {kind}")
    torch.ops._C_ascend.kv_compress_epilog_v2(
        cache_arg,
        values.contiguous(),
        flat_slots.contiguous(),
        quant_group_size=group_size,
        quant_mode=quant_mode,
        round_scale=True,
        x_scale=1.0,
    )


def write_index_cache(
    cache: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    coordinates: torch.Tensor,
    values: torch.Tensor,
) -> None:
    write_mxfp4_indexer_cache(values, coordinates, cache[0], cache[1])


class MixedQuantPackedCacheOps:
    """Packed mixed-quant cache ABI selected by the hardware adaptor."""

    build_packed_cache_slot_mapping = staticmethod(build_packed_cache_slot_mapping)
    build_window_indices = staticmethod(build_window_indices_triton)
    build_smla_metadata = staticmethod(build_smla_metadata)
    qsmla = staticmethod(qsmla)
    run_a5_indexer = staticmethod(run_a5_indexer)
    write_attention_cache = staticmethod(write_attention_cache)
    write_index_cache = staticmethod(write_index_cache)
    mixed_quant_sparse_flash_mla_metadata = staticmethod(ops.mixed_quant_sparse_flash_mla_metadata)
    quant_lightning_indexer_metadata = staticmethod(ops.quant_lightning_indexer_metadata)
