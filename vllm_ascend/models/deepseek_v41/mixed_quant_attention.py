# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed-quant cache and attention ABI selected by the A5 device adaptor."""

from __future__ import annotations

import torch

import torch_npu  # noqa: F401
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

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
    compress_ratio=None,
    index_topk=None,
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


INT8_NOPE_HEAD_DIM = 448
INT8_ROPE_HEAD_DIM = 64
INT8_KV_RECORD_BYTES = 608
INT8_QUANT_GROUP_SIZE = 64
INT8_NOPE_GROUPS = INT8_NOPE_HEAD_DIM // INT8_QUANT_GROUP_SIZE
INT8_METADATA_SIZE = 2048
def pack_int8_kv(values: torch.Tensor) -> torch.Tensor:
    """Quantize one RoPE-applied latent row set into 608-byte INT8 records."""
    if values.ndim != 2 or values.shape[-1] != INT8_NOPE_HEAD_DIM + INT8_ROPE_HEAD_DIM:
        raise ValueError(f"INT8 CSA expects [tokens, 512], got {tuple(values.shape)}")
    tokens = values.shape[0]
    nope = values[:, :INT8_NOPE_HEAD_DIM].float().contiguous()
    rope = values[:, INT8_NOPE_HEAD_DIM:].to(torch.bfloat16).contiguous()
    groups = nope.view(tokens, INT8_NOPE_GROUPS, INT8_QUANT_GROUP_SIZE)
    amax = groups.abs().amax(dim=-1).clamp_min(1e-8)
    scale = (amax / 127.0).to(torch.bfloat16)
    quantized = (
        torch.round(groups / scale.float().unsqueeze(-1))
        .clamp(-127, 127)
        .to(torch.int8)
    )

    packed = torch.zeros((tokens, INT8_KV_RECORD_BYTES), dtype=torch.int8, device=values.device)
    packed_u8 = packed.view(torch.uint8)
    packed_bf16 = packed.view(torch.bfloat16)
    packed_bf16[:, :INT8_ROPE_HEAD_DIM] = rope
    packed_u8[:, 128:576] = quantized.reshape(tokens, INT8_NOPE_HEAD_DIM).view(torch.uint8)
    packed_bf16[:, 288:295] = scale
    return packed


@triton.jit(do_not_specialize=["tokens", "width"])
def _int8_write_packed_cache(
    cache,
    updates,
    slots,
    tokens,
    width,
    pages: tl.constexpr,
    page_size: tl.constexpr,
    page_stride: tl.constexpr,
    row_stride: tl.constexpr,
    update_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    paired: tl.constexpr,
    columns: tl.constexpr,
):
    columns_idx = tl.arange(0, columns)
    for token in range(tl.program_id(0), tokens, tl.num_programs(0)):
        first = tl.load(slots + token * slot_stride).to(tl.int64)
        if paired:
            block = first
            offset = tl.load(slots + token * slot_stride + 1).to(tl.int64)
        else:
            block = first // page_size
            offset = first % page_size
        if (first >= 0) & (block < pages) & (offset >= 0) & (offset < page_size):
            value = tl.load(updates + token * update_stride + columns_idx, columns_idx < width, other=0)
            address = block * page_stride + offset * row_stride + columns_idx
            tl.store(cache + address, value, columns_idx < width)


def _int8_write_packed_cache_tensor(cache: torch.Tensor, updates: torch.Tensor, slots: torch.Tensor) -> None:
    """Write packed rows while preserving the allocator's physical-page stride."""
    if cache.ndim != 4 or cache.shape[2] != 1 or cache.stride(-1) != 1:
        raise ValueError("Cache must be [pages, page_size, 1, width] with contiguous rows")
    if updates.ndim != 2 or updates.shape[-1] != cache.shape[-1]:
        raise ValueError("Packed updates must be [tokens, cache_width]")
    if updates.dtype != cache.dtype or updates.stride(-1) != 1:
        raise ValueError("Packed updates must match the cache dtype and be row-contiguous")
    if slots.ndim != 1 or slots.shape[0] != updates.shape[0]:
        raise ValueError("Slots must be one-dimensional with one entry per update row")
    tokens = updates.shape[0]
    if tokens == 0:
        return
    _int8_write_packed_cache[(min(get_vectorcore_num(), tokens),)](
        cache,
        updates,
        slots,
        tokens,
        cache.shape[-1],
        cache.shape[0],
        cache.shape[1],
        cache.stride(0),
        cache.stride(1),
        updates.stride(0),
        slots.stride(0),
        False,
        triton.next_power_of_2(cache.shape[-1]),
        multibuffer=False,
    )


def int8_write_attention_cache(
    cache: torch.Tensor,
    flat_slots: torch.Tensor,
    values: torch.Tensor,
    *,
    kind: str,
) -> None:
    """Pack one SWA/compressed latent batch and scatter it into the INT8 cache."""
    if kind not in ("win", "cmp"):
        raise ValueError(f"unsupported INT8 cache kind: {kind}")
    packed = pack_int8_kv(values)
    slots = flat_slots[: packed.shape[0]].contiguous()
    _int8_write_packed_cache_tensor(cache, packed, slots)


def _int8_pad_sparse_indices(indices: torch.Tensor, topk: int) -> torch.Tensor:
    if indices.shape[-1] < topk:
        indices = torch.nn.functional.pad(indices, (0, topk - indices.shape[-1]), value=-1)
    return indices[:, None, :].to(torch.int32).contiguous()


def int8_qsmla(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor | None,
    metadata,
    compressed_indices: torch.Tensor,
    *,
    window_size: int,
    sinks: torch.Tensor | None,
    softmax_scale: float,
    compressed_lengths: torch.Tensor | None = None,
    compress_ratio: int = 1,
    index_topk: int | None = None,
) -> torch.Tensor:
    """Run the INT8 CSA operator for one layer."""
    import cann_ops_transformer  # noqa: F401

    if cmp_kv is None:
        raise ValueError("INT8 CSA requires a compressed KV cache")
    topk = index_topk if index_topk is not None else compressed_indices.shape[-1]
    cmp_indices = _int8_pad_sparse_indices(compressed_indices, topk)
    task_metadata = metadata.swa.smla_metadata
    if task_metadata is None:
        raise RuntimeError("INT8 CSA metadata was not built")
    wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(task_metadata))
    num_reqs = metadata.swa.num_reqs
    ori_block_table = metadata.swa.block_table[:num_reqs]
    ori_seq_lens = metadata.swa.seq_lens[:num_reqs]
    if metadata.swa.replay_block_table is not None:
        # Bounded replay keeps only the retained SWA prefix in the cache;
        # rebase the original KV view exactly as the bf16 fallback does.
        ori_kv = ori_kv[metadata.swa.replay_cache_blocks, metadata.swa.replay_cache_offsets]
        ori_kv = ori_kv.reshape(-1, metadata.swa.storage_block_size, 1, ori_kv.shape[-1])
        ori_block_table = metadata.swa.replay_block_table[:num_reqs]
        ori_seq_lens = metadata.swa.replay_seq_lens[:num_reqs]
    ori_mask_mode = metadata.swa.ori_mask_mode
    win_left = metadata.swa.ori_win_left if ori_mask_mode == 4 else -1
    win_right = metadata.swa.ori_win_right if ori_mask_mode == 4 else -1
    cmp_residual = metadata.attention.cmp_residual
    output, _ = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla(
        q,
        quant_mode=1,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=metadata.attention.block_table[:num_reqs],
        cu_seqlens_q=metadata.swa.query_start_loc[: num_reqs + 1],
        seqused_ori_kv=ori_seq_lens,
        seqused_cmp_kv=metadata.attention.cache_seq_lens[:num_reqs],
        cmp_residual_kv=(
            None if compress_ratio == 1 or cmp_residual is None else cmp_residual[:num_reqs]
        ),
        sinks=None if sinks is None else sinks.detach().float().contiguous(),
        metadata=task_metadata,
        rope_head_dim=INT8_ROPE_HEAD_DIM,
        softmax_scale=softmax_scale,
        cmp_ratio=compress_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=3,
        ori_win_left=win_left,
        ori_win_right=win_right,
        layout_q="TND",
        layout_kv="PA_BBND",
        topk_value_mode=1,
        return_softmax_lse=False,
    )
    return output


def int8_build_smla_metadata(
    *,
    num_heads_q: int,
    head_dim: int,
    num_reqs: int,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    cmp_seq_lens: torch.Tensor,
    cmp_residual: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_ori_kv: int,
    max_seqlen_cmp_kv: int,
    index_topk: int,
    compress_ratio: int,
    ori_mask_mode: int,
    ori_win_left: int,
    ori_win_right: int,
) -> torch.Tensor:
    """Build the fixed 2048-int32 launch metadata for the INT8 CSA path."""
    import cann_ops_transformer  # noqa: F401

    return torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(
        num_heads_q,
        1,
        head_dim,
        1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_ori_kv=seq_lens,
        seqused_cmp_kv=cmp_seq_lens,
        cmp_residual_kv=cmp_residual if compress_ratio != 1 else None,
        batch_size=num_reqs,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv,
        max_seqlen_cmp_kv=max_seqlen_cmp_kv,
        cmp_topk=index_topk,
        rope_head_dim=INT8_ROPE_HEAD_DIM,
        cmp_ratio=compress_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=3,
        ori_win_left=ori_win_left if ori_mask_mode == 4 else -1,
        ori_win_right=ori_win_right if ori_mask_mode == 4 else -1,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )


class Int8MixedQuantPackedCacheOps:
    """Packed INT8 CSA cache ABI selected by the device adaptor."""

    uses_a5_packed_cache = False
    uses_int8_packed_cache = True
    metadata_size = INT8_METADATA_SIZE
    pack_int8_kv = staticmethod(pack_int8_kv)
    write_attention_cache = staticmethod(int8_write_attention_cache)
    qsmla = staticmethod(int8_qsmla)
    build_smla_metadata = staticmethod(int8_build_smla_metadata)
