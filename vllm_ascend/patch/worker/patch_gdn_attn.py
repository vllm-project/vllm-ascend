# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import torch
import vllm.v1.attention.backends.gdn_attn as gdn_attn
from vllm.logger import logger

from vllm_ascend.ops.triton.gdn_chunk_meta import (
    _build_seq_lens,
    _validate_cu_seqlens,
    build_chunk_meta_device,
)
from vllm_ascend.utils import is_310p

_GDN_CHUNK_SIZE = 64
# Keep this aligned with solve_tril.LARGE_BLOCK_T in ops/triton/fla/solve_tril.py.
_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2
_GDN_CUMSUM_WORKING_SET = 2**18

_IS_PATCHED = False
_ORIGINAL_BUILD = gdn_attn.GDNAttentionMetadataBuilder.build
_ORIGINAL_INIT_THRESHOLD = gdn_attn.GDNAttentionMetadataBuilder._init_reorder_batch_threshold


@dataclass
class GDNChunkedPrefillMetadata:
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor
    _buffer_slot: object | None = None


@dataclass
class GDNCausalConv1dHostMetadata:
    query_start_loc_cpu: torch.Tensor
    cache_indices_cpu: torch.Tensor
    has_initial_state_cpu: torch.Tensor
    _buffer_slot: object | None = None


@dataclass
class GDNPrefillFallbackMeta:
    causal_conv1d: GDNCausalConv1dHostMetadata
    chunk: GDNChunkedPrefillMetadata


@dataclass
class _GDNChunkedPrefillBufferSlot:
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor


@dataclass
class _GDNCausalConv1dHostBufferSlot:
    cache_indices_cpu: torch.Tensor
    has_initial_state_cpu: torch.Tensor


@dataclass
class _GDNChunkMetaSizeInfo:
    num_seqs: int
    num_chunk_indices_chunk64: int
    num_chunk_indices_large_block: int
    num_block_indices_cumsum: int


@dataclass
class _GDNChunkMetaShapeInfo(_GDNChunkMetaSizeInfo):
    chunk_counts_chunk64: torch.Tensor
    chunk_counts_large_block: torch.Tensor
    chunk_counts_cumsum: torch.Tensor


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _prepare_chunk_counts_cpu(cu_seqlens_cpu: torch.Tensor, chunk_size: int) -> torch.Tensor:
    lens = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    return torch.div(lens + chunk_size - 1, chunk_size, rounding_mode="floor")


def _fill_chunk_indices_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    cursor = 0
    compact_seq_idx = 0
    for num_chunks in chunk_counts.tolist():
        if num_chunks <= 0:
            continue
        # `prepare_chunk_indices` compacts away zero-length sequences, so the
        # sequence index here must follow the same compact numbering.
        out[cursor : cursor + num_chunks, 0].fill_(compact_seq_idx)
        out[cursor : cursor + num_chunks, 1] = torch.arange(
            num_chunks,
            dtype=out.dtype,
        )
        cursor += num_chunks
        compact_seq_idx += 1
    return cursor


def _fill_chunk_offsets_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    out[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts, dim=0, out=out[1 : chunk_counts.numel() + 1])
    return chunk_counts.numel() + 1


def _fill_update_chunk_offsets_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    out[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(
            chunk_counts + 1,
            dim=0,
            out=out[1 : chunk_counts.numel() + 1],
        )
    return chunk_counts.numel() + 1


def _fill_final_chunk_indices_cpu(out: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts + 1, dim=0, out=out[: chunk_counts.numel()])
        out[: chunk_counts.numel()].sub_(1)
    return chunk_counts.numel()


def _build_chunk_meta_shape_info(builder, cu_seqlens_cpu: torch.Tensor) -> _GDNChunkMetaShapeInfo:
    chunk_counts_chunk64 = _prepare_chunk_counts_cpu(
        cu_seqlens_cpu,
        builder._ascend_gdn_chunk_size,
    )
    chunk_counts_large_block = _prepare_chunk_counts_cpu(
        cu_seqlens_cpu,
        builder._ascend_gdn_large_block_size,
    )
    chunk_counts_cumsum = _prepare_chunk_counts_cpu(
        cu_seqlens_cpu,
        builder._ascend_gdn_cumsum_block_size,
    )
    return _GDNChunkMetaShapeInfo(
        num_seqs=chunk_counts_chunk64.numel(),
        num_chunk_indices_chunk64=int(chunk_counts_chunk64.sum().item()),
        num_chunk_indices_large_block=int(chunk_counts_large_block.sum().item()),
        num_block_indices_cumsum=int(chunk_counts_cumsum.sum().item()),
        chunk_counts_chunk64=chunk_counts_chunk64,
        chunk_counts_large_block=chunk_counts_large_block,
        chunk_counts_cumsum=chunk_counts_cumsum,
    )


def _count_chunk_indices_cpu(seq_lens_cpu: torch.Tensor, chunk_size: int) -> int:
    return int(
        torch.div(
            seq_lens_cpu + chunk_size - 1,
            chunk_size,
            rounding_mode="floor",
        )
        .sum()
        .item()
    )


def _build_chunk_meta_size_info(builder, cu_seqlens_cpu: torch.Tensor) -> _GDNChunkMetaSizeInfo:
    seq_lens_cpu = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    return _GDNChunkMetaSizeInfo(
        num_seqs=seq_lens_cpu.numel(),
        num_chunk_indices_chunk64=_count_chunk_indices_cpu(
            seq_lens_cpu,
            builder._ascend_gdn_chunk_size,
        ),
        num_chunk_indices_large_block=_count_chunk_indices_cpu(
            seq_lens_cpu,
            builder._ascend_gdn_large_block_size,
        ),
        num_block_indices_cumsum=_count_chunk_indices_cpu(
            seq_lens_cpu,
            builder._ascend_gdn_cumsum_block_size,
        ),
    )


def _allocate_chunk_meta_cpu_tensors(shape_info: _GDNChunkMetaSizeInfo) -> dict[str, torch.Tensor]:
    return {
        "chunk_indices_chunk64": torch.empty(
            (shape_info.num_chunk_indices_chunk64, 2),
            dtype=torch.int32,
        ),
        "chunk_offsets_chunk64": torch.empty(
            (shape_info.num_seqs + 1,),
            dtype=torch.int32,
        ),
        "update_chunk_offsets_chunk64": torch.empty(
            (shape_info.num_seqs + 1,),
            dtype=torch.int32,
        ),
        "final_chunk_indices_chunk64": torch.empty(
            (shape_info.num_seqs,),
            dtype=torch.int32,
        ),
        "chunk_indices_large_block": torch.empty(
            (shape_info.num_chunk_indices_large_block, 2),
            dtype=torch.int32,
        ),
        "block_indices_cumsum": torch.empty(
            (shape_info.num_block_indices_cumsum, 2),
            dtype=torch.int32,
        ),
    }


def _slice_chunk_meta_slot_tensors(
    slot: _GDNChunkedPrefillBufferSlot,
    shape_info: _GDNChunkMetaSizeInfo,
) -> dict[str, torch.Tensor]:
    return {
        "chunk_indices_chunk64": slot.chunk_indices_chunk64[: shape_info.num_chunk_indices_chunk64],
        "chunk_offsets_chunk64": slot.chunk_offsets_chunk64[: shape_info.num_seqs + 1],
        "update_chunk_offsets_chunk64": slot.update_chunk_offsets_chunk64[: shape_info.num_seqs + 1],
        "final_chunk_indices_chunk64": slot.final_chunk_indices_chunk64[: shape_info.num_seqs],
        "chunk_indices_large_block": slot.chunk_indices_large_block[: shape_info.num_chunk_indices_large_block],
        "block_indices_cumsum": slot.block_indices_cumsum[: shape_info.num_block_indices_cumsum],
    }


def _fill_chunk_meta_cpu_tensors(
    tensors: dict[str, torch.Tensor],
    shape_info: _GDNChunkMetaShapeInfo,
) -> None:
    _fill_chunk_indices_cpu(
        tensors["chunk_indices_chunk64"],
        shape_info.chunk_counts_chunk64,
    )
    _fill_chunk_offsets_cpu(
        tensors["chunk_offsets_chunk64"],
        shape_info.chunk_counts_chunk64,
    )
    _fill_update_chunk_offsets_cpu(
        tensors["update_chunk_offsets_chunk64"],
        shape_info.chunk_counts_chunk64,
    )
    _fill_final_chunk_indices_cpu(
        tensors["final_chunk_indices_chunk64"],
        shape_info.chunk_counts_chunk64,
    )
    _fill_chunk_indices_cpu(
        tensors["chunk_indices_large_block"],
        shape_info.chunk_counts_large_block,
    )
    _fill_chunk_indices_cpu(
        tensors["block_indices_cumsum"],
        shape_info.chunk_counts_cumsum,
    )


def _fill_chunk_meta_device_tensors(
    builder,
    cu_seqlens: torch.Tensor,
    tensors: dict[str, torch.Tensor],
) -> None:
    seq_lens = None
    validate_inputs = True
    if cu_seqlens.device.type == "npu":
        _validate_cu_seqlens(cu_seqlens, builder._ascend_gdn_chunk_size)
        assert builder._ascend_gdn_large_block_size > 0
        assert builder._ascend_gdn_cumsum_block_size > 0
        seq_lens = _build_seq_lens(cu_seqlens)
        validate_inputs = False
    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=builder._ascend_gdn_chunk_size,
        out_chunk_indices=tensors["chunk_indices_chunk64"],
        out_chunk_offsets=tensors["chunk_offsets_chunk64"],
        out_update_chunk_offsets=tensors["update_chunk_offsets_chunk64"],
        out_final_chunk_indices=tensors["final_chunk_indices_chunk64"],
        seq_lens=seq_lens,
        validate_inputs=validate_inputs,
    )
    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=builder._ascend_gdn_large_block_size,
        out_chunk_indices=tensors["chunk_indices_large_block"],
        seq_lens=seq_lens,
        validate_inputs=validate_inputs,
    )
    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=builder._ascend_gdn_cumsum_block_size,
        out_chunk_indices=tensors["block_indices_cumsum"],
        seq_lens=seq_lens,
        validate_inputs=validate_inputs,
    )


def _build_chunked_prefill_metadata(
    builder,
    tensors: dict[str, torch.Tensor],
    *,
    slot: _GDNChunkedPrefillBufferSlot | None = None,
) -> GDNChunkedPrefillMetadata:
    return GDNChunkedPrefillMetadata(
        chunk_indices_chunk64=tensors["chunk_indices_chunk64"],
        chunk_offsets_chunk64=tensors["chunk_offsets_chunk64"],
        update_chunk_offsets_chunk64=tensors["update_chunk_offsets_chunk64"],
        final_chunk_indices_chunk64=tensors["final_chunk_indices_chunk64"],
        chunk_indices_large_block=tensors["chunk_indices_large_block"],
        block_indices_cumsum=tensors["block_indices_cumsum"],
        _buffer_slot=slot,
    )


def _get_gdn_num_heads(builder) -> int:
    hf_text_config = getattr(builder.vllm_config.model_config, "hf_text_config", None)
    if hf_text_config is not None and hasattr(hf_text_config, "linear_num_value_heads"):
        return hf_text_config.linear_num_value_heads // builder.vllm_config.parallel_config.tensor_parallel_size
    return builder.vllm_config.model_config.get_num_attention_heads(builder.vllm_config.parallel_config)


def _allocate_chunked_prefill_slot(builder, device: torch.device):
    max_num_batched_tokens = builder.vllm_config.scheduler_config.max_num_batched_tokens
    max_num_seqs = builder.vllm_config.scheduler_config.max_num_seqs
    return _GDNChunkedPrefillBufferSlot(
        chunk_indices_chunk64=torch.empty(
            (max_num_batched_tokens, 2),
            dtype=torch.int32,
            device=device,
        ),
        chunk_offsets_chunk64=torch.empty(
            (max_num_seqs + 1,),
            dtype=torch.int32,
            device=device,
        ),
        update_chunk_offsets_chunk64=torch.empty(
            (max_num_seqs + 1,),
            dtype=torch.int32,
            device=device,
        ),
        final_chunk_indices_chunk64=torch.empty(
            (max_num_seqs,),
            dtype=torch.int32,
            device=device,
        ),
        chunk_indices_large_block=torch.empty(
            (max_num_batched_tokens, 2),
            dtype=torch.int32,
            device=device,
        ),
        block_indices_cumsum=torch.empty(
            (max_num_batched_tokens, 2),
            dtype=torch.int32,
            device=device,
        ),
    )


def _ensure_chunk_meta_state(builder, device: torch.device) -> None:
    if getattr(builder, "_ascend_gdn_chunk_meta_initialized", False):
        return
    builder._ascend_gdn_chunk_meta_initialized = True
    builder._ascend_gdn_chunk_meta_device = device
    builder._ascend_gdn_chunk_size = _GDN_CHUNK_SIZE
    builder._ascend_gdn_large_block_size = _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE
    gdn_num_heads = _get_gdn_num_heads(builder)
    cumsum_chunks = max(1, _GDN_CUMSUM_WORKING_SET // (gdn_num_heads * builder._ascend_gdn_chunk_size))
    builder._ascend_gdn_cumsum_block_size = _next_power_of_2(cumsum_chunks)
    builder._ascend_gdn_chunked_prefill_pool_idx = -1
    builder._ascend_gdn_chunked_prefill_pool = []
    if device.type != "cpu":
        builder._ascend_gdn_chunked_prefill_pool = [
            _allocate_chunked_prefill_slot(builder, device),
            _allocate_chunked_prefill_slot(builder, device),
        ]


def _build_spec_sequence_masks_cpu(builder, num_decode_draft_tokens_cpu: torch.Tensor | None) -> torch.Tensor | None:
    if (
        not getattr(builder, "use_spec_decode", False)
        or num_decode_draft_tokens_cpu is None
        or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0].sum().item() == 0
    ):
        return None
    return num_decode_draft_tokens_cpu >= 0


def _build_non_spec_query_start_loc_cpu(
    builder,
    attn_metadata,
    common_attn_metadata,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
) -> torch.Tensor | None:
    if attn_metadata.num_prefills <= 0:
        return None

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    spec_sequence_masks_cpu = _build_spec_sequence_masks_cpu(builder, num_decode_draft_tokens_cpu)
    if spec_sequence_masks_cpu is None:
        return query_start_loc_cpu

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
    non_spec_query_start_loc_cpu = torch.zeros(
        non_spec_query_lens_cpu.numel() + 1,
        dtype=query_start_loc_cpu.dtype,
    )
    torch.cumsum(
        non_spec_query_lens_cpu,
        dim=0,
        out=non_spec_query_start_loc_cpu[1:],
    )
    return non_spec_query_start_loc_cpu


def _allocate_causal_conv1d_host_slot(
    builder,
    device: torch.device,
) -> _GDNCausalConv1dHostBufferSlot:
    max_num_seqs = builder.vllm_config.scheduler_config.max_num_seqs
    return _GDNCausalConv1dHostBufferSlot(
        cache_indices_cpu=torch.empty(
            max_num_seqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=device.type != "cpu",
        ),
        has_initial_state_cpu=torch.empty(
            max_num_seqs,
            dtype=torch.bool,
            device="cpu",
            pin_memory=device.type != "cpu",
        ),
    )


def _ensure_causal_conv1d_host_meta_state(builder, device: torch.device) -> None:
    if getattr(builder, "_ascend_gdn_causal_conv1d_host_meta_initialized", False):
        return
    builder._ascend_gdn_causal_conv1d_host_meta_initialized = True
    builder._ascend_gdn_causal_conv1d_host_pool_idx = -1
    builder._ascend_gdn_causal_conv1d_host_pool = []
    if device.type != "cpu":
        builder._ascend_gdn_causal_conv1d_host_pool = [
            _allocate_causal_conv1d_host_slot(builder, device),
            _allocate_causal_conv1d_host_slot(builder, device),
        ]


def _acquire_causal_conv1d_host_slot(builder) -> _GDNCausalConv1dHostBufferSlot:
    pool = builder._ascend_gdn_causal_conv1d_host_pool
    builder._ascend_gdn_causal_conv1d_host_pool_idx = (builder._ascend_gdn_causal_conv1d_host_pool_idx + 1) % len(pool)
    return pool[builder._ascend_gdn_causal_conv1d_host_pool_idx]


def _copy_to_pinned_cpu(
    tensor: torch.Tensor,
    pinned_buffer: torch.Tensor | None,
) -> torch.Tensor:
    if tensor.device.type == "cpu":
        return tensor

    assert pinned_buffer is not None
    cpu_tensor = pinned_buffer[: tensor.numel()]
    cpu_tensor.copy_(
        tensor.reshape(-1),
        non_blocking=True,
    )
    return cpu_tensor


def _build_non_spec_causal_conv1d_host_meta(
    builder,
    attn_metadata,
    non_spec_query_start_loc_cpu: torch.Tensor,
) -> GDNCausalConv1dHostMetadata:
    assert attn_metadata.num_prefills > 0
    if attn_metadata.non_spec_state_indices_tensor is None:
        raise RuntimeError(
            "Expected attn_metadata.non_spec_state_indices_tensor for patched GDN non-spec prefill path."
        )
    if attn_metadata.has_initial_state is None:
        raise RuntimeError("Expected attn_metadata.has_initial_state for patched GDN non-spec prefill path.")

    slot = None
    if (
        attn_metadata.non_spec_state_indices_tensor.device.type != "cpu"
        or attn_metadata.has_initial_state.device.type != "cpu"
    ):
        slot = _acquire_causal_conv1d_host_slot(builder)

    cache_indices_cpu = _copy_to_pinned_cpu(
        attn_metadata.non_spec_state_indices_tensor,
        None if slot is None else slot.cache_indices_cpu,
    )
    has_initial_state_cpu = _copy_to_pinned_cpu(
        attn_metadata.has_initial_state,
        None if slot is None else slot.has_initial_state_cpu,
    )

    return GDNCausalConv1dHostMetadata(
        query_start_loc_cpu=non_spec_query_start_loc_cpu,
        cache_indices_cpu=cache_indices_cpu,
        has_initial_state_cpu=has_initial_state_cpu,
        _buffer_slot=slot,
    )


def _build_non_spec_chunked_prefill_meta_cpu(builder, cu_seqlens_cpu: torch.Tensor) -> GDNChunkedPrefillMetadata:
    shape_info = _build_chunk_meta_shape_info(builder, cu_seqlens_cpu)
    tensors = _allocate_chunk_meta_cpu_tensors(shape_info)
    _fill_chunk_meta_cpu_tensors(tensors, shape_info)
    return _build_chunked_prefill_metadata(builder, tensors)


def _build_non_spec_chunked_prefill_meta(
    builder,
    cu_seqlens_cpu: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> GDNChunkedPrefillMetadata:
    device = builder._ascend_gdn_chunk_meta_device
    if device.type == "cpu":
        return _build_non_spec_chunked_prefill_meta_cpu(builder, cu_seqlens_cpu)

    shape_info = _build_chunk_meta_size_info(builder, cu_seqlens_cpu)
    builder._ascend_gdn_chunked_prefill_pool_idx = (builder._ascend_gdn_chunked_prefill_pool_idx + 1) % len(
        builder._ascend_gdn_chunked_prefill_pool
    )
    slot = builder._ascend_gdn_chunked_prefill_pool[builder._ascend_gdn_chunked_prefill_pool_idx]
    tensors = _slice_chunk_meta_slot_tensors(slot, shape_info)
    _fill_chunk_meta_device_tensors(builder, cu_seqlens, tensors)
    return _build_chunked_prefill_metadata(builder, tensors, slot=slot)


def _compute_all_mode_metadata(builder, attn_metadata, m):
    """Compute all-mode prefix caching metadata and attach to attn_metadata.

    In "all" mode, each sequence may span multiple blocks. We compute:
    - SOURCE pool slots (where initial state was last written)
    - DEST pool slots (where final state will be written) — overrides
      non_spec_state_indices_tensor
    - Block indices for scatter of intermediate block boundary states
    - Chunk offsets for mapping FLA h-tensor to block boundaries

    All fields are batch-wide [num_seqs]; forward code slices with
    [:num_decodes] / [num_decodes:] as needed.
    """
    block_size = builder.kv_cache_spec.block_size
    chunk_size = _GDN_CHUNK_SIZE
    num_decodes = attn_metadata.num_decodes
    num_prefills = attn_metadata.num_prefills
    num_seqs = num_decodes + num_prefills
    device = m.query_start_loc.device

    if attn_metadata.spec_state_indices_tensor is not None:
        raise AssertionError(
            "spec decode batches must be rerouted to align-mode metadata "
            "before all-mode metadata computation"
        )

    # In "all" mode, mamba_get_block_table_tensor returns the full table
    block_table_2d = m.block_table_tensor[:num_seqs]  # [num_seqs, max_blocks]

    seq_lens = m.seq_lens[:num_seqs]
    query_lens = (
        m.query_start_loc[1:num_seqs + 1] - m.query_start_loc[:num_seqs]
    )
    context_lens = seq_lens - query_lens

    # DEST block index: last block containing scheduled tokens
    block_idx_last_scheduled = (seq_lens - 1) // block_size
    # First block that needs writing (first block with new tokens)
    block_idx_first_scheduled = context_lens // block_size

    # SOURCE pool slots: block containing the last computed token
    has_context = context_lens > 0
    source_block_idx = torch.where(
        has_context,
        (context_lens - 1) // block_size,
        torch.zeros_like(context_lens),  # placeholder for gather
    )
    block_state_indices = torch.where(
        has_context,
        block_table_2d.gather(
            1, source_block_idx.unsqueeze(1).long()
        ).squeeze(1),
        torch.full((num_seqs,), -1, dtype=block_table_2d.dtype, device=device),
    )

    # DEST pool slots: block containing the last scheduled token
    dest_slots = block_table_2d.gather(
        1, block_idx_last_scheduled.unsqueeze(1).long()
    ).squeeze(1)

    # Override non_spec_state_indices_tensor:
    # upstream set it to block_table[:, 0] (block 0), we need DEST block
    attn_metadata.non_spec_state_indices_tensor = dest_slots

    # Prefill chunk computation for intermediate state scatter
    # Each decode seq contributes 1 chunk to the FLA h tensor
    chunks_per_block = block_size // chunk_size
    prefill_chunk_start = num_decodes
    prefill_chunk_offsets = None
    scatter_src_indices = torch.empty(0, dtype=torch.long, device=device)
    scatter_dst_slots = torch.empty(0, dtype=torch.long, device=device)
    if num_prefills > 0:
        prefill_query_lens = query_lens[num_decodes:]
        prefill_context_lens = context_lens[num_decodes:]
        prefill_block_first = block_idx_first_scheduled[num_decodes:]
        prefill_block_last = block_idx_last_scheduled[num_decodes:]
        prefill_chunk_counts = (
            (prefill_query_lens + chunk_size - 1) // chunk_size
        )
        offsets = torch.zeros(
            num_prefills + 1, dtype=torch.long, device=device
        )
        torch.cumsum(prefill_chunk_counts, dim=0, out=offsets[1:])
        prefill_chunk_offsets = offsets

        unaligned_prefills = torch.nonzero(
            prefill_context_lens.remainder(block_size) != 0,
            as_tuple=False,
        ).flatten()
        if unaligned_prefills.numel() > 0:
            bad_idx = int(unaligned_prefills[0].item())
            raise AssertionError(
                "Scheduler must guarantee block-aligned context for all-mode "
                f"scatter: seq_idx={bad_idx}, "
                f"context_len={int(prefill_context_lens[bad_idx].item())}, "
                f"block_size={block_size}"
            )

        scatter_counts = torch.clamp(
            prefill_block_last - prefill_block_first,
            min=0,
        ).to(torch.long)
        scatter_seq_ids = torch.repeat_interleave(
            torch.arange(num_prefills, device=device, dtype=torch.long),
            scatter_counts,
        )
        if scatter_seq_ids.numel() > 0:
            scatter_prefix = torch.cumsum(scatter_counts, dim=0) - scatter_counts
            local_offsets = (
                torch.arange(scatter_seq_ids.numel(), device=device, dtype=torch.long)
                - torch.repeat_interleave(scatter_prefix, scatter_counts)
            )
            scatter_rows = block_table_2d[num_decodes:].index_select(0, scatter_seq_ids)
            scatter_block_indices = (
                prefill_block_first.to(torch.long).index_select(0, scatter_seq_ids)
                + local_offsets
            )
            scatter_dst_slots = scatter_rows.gather(
                1, scatter_block_indices.unsqueeze(1)
            ).squeeze(1).to(torch.long)
            scatter_src_indices = (
                prefill_chunk_offsets[:-1].index_select(0, scatter_seq_ids)
                + prefill_chunk_start
                + (local_offsets + 1) * chunks_per_block
            )
            valid_scatter = torch.nonzero(
                scatter_dst_slots >= 0,
                as_tuple=False,
            ).flatten()
            if valid_scatter.numel() != scatter_dst_slots.numel():
                scatter_dst_slots = scatter_dst_slots.index_select(0, valid_scatter)
                scatter_src_indices = scatter_src_indices.index_select(0, valid_scatter)

    attn_metadata.is_all_mode = True
    attn_metadata.mamba_block_size = block_size
    attn_metadata.all_mode_chunk_size = chunk_size
    attn_metadata.block_table_2d = block_table_2d
    # SOURCE: pool slot of last-computed block (for reading initial state)
    attn_metadata.block_state_indices = block_state_indices
    attn_metadata.block_idx_first_scheduled_token = block_idx_first_scheduled
    attn_metadata.block_idx_last_scheduled_token = block_idx_last_scheduled
    attn_metadata.num_computed_tokens_all = context_lens
    attn_metadata.prefill_chunk_start = prefill_chunk_start
    attn_metadata.prefill_chunk_offsets = prefill_chunk_offsets
    attn_metadata.scatter_src_indices_tensor = scatter_src_indices
    attn_metadata.scatter_dst_slots_tensor = scatter_dst_slots


def _warn_all_mode_spec_fallback_once() -> None:
    global _FALLBACK_ALL_MODE_SPEC_WARNED
    if _FALLBACK_ALL_MODE_SPEC_WARNED:
        return
    logger.warning(
        "mamba_cache_mode='all' does not support speculative decode yet; "
        "falling back to align-mode metadata for this batch."
    )
    _FALLBACK_ALL_MODE_SPEC_WARNED = True


def _patched_build(
    self,
    common_prefix_len: int,
    common_attn_metadata,
    num_accepted_tokens: torch.Tensor | None = None,
    num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    fast_build: bool = False,
):
    cache_config = self.vllm_config.cache_config

    def _build_with_cache_mode(cache_mode: str | None = None):
        saved_mode = cache_config.mamba_cache_mode
        if cache_mode is not None:
            cache_config.mamba_cache_mode = cache_mode
        try:
            return _ORIGINAL_BUILD(
                self,
                common_prefix_len,
                common_attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
                fast_build=fast_build,
            )
        finally:
            cache_config.mamba_cache_mode = saved_mode

    attn_metadata = _build_with_cache_mode()
    attn_metadata.non_spec_prefill_fallback_meta = None
    attn_metadata.is_all_mode = False

    mamba_cache_mode = cache_config.mamba_cache_mode
    if mamba_cache_mode == "all":
        if attn_metadata.spec_state_indices_tensor is not None:
            _warn_all_mode_spec_fallback_once()
            attn_metadata = _build_with_cache_mode("align")
            attn_metadata.non_spec_prefill_fallback_meta = None
            attn_metadata.is_all_mode = False
        else:
            _compute_all_mode_metadata(self, attn_metadata, common_attn_metadata)

    if attn_metadata.num_prefills <= 0:
        return attn_metadata

    _ensure_chunk_meta_state(self, common_attn_metadata.query_start_loc.device)
    _ensure_causal_conv1d_host_meta_state(
        self,
        common_attn_metadata.query_start_loc.device,
    )
    non_spec_query_start_loc_cpu = _build_non_spec_query_start_loc_cpu(
        self,
        attn_metadata,
        common_attn_metadata,
        num_decode_draft_tokens_cpu,
    )
    assert non_spec_query_start_loc_cpu is not None
    if attn_metadata.non_spec_query_start_loc is None:
        raise RuntimeError("Expected attn_metadata.non_spec_query_start_loc for patched GDN non-spec prefill path.")
    attn_metadata.non_spec_prefill_fallback_meta = GDNPrefillFallbackMeta(
        causal_conv1d=_build_non_spec_causal_conv1d_host_meta(
            self,
            attn_metadata,
            non_spec_query_start_loc_cpu,
        ),
        chunk=_build_non_spec_chunked_prefill_meta(
            self,
            non_spec_query_start_loc_cpu,
            attn_metadata.non_spec_query_start_loc,
        ),
    )
    return attn_metadata


def _init_reorder_batch_threshold(
    self,
    reorder_batch_threshold: int | None = 1,
    supports_spec_as_decode: bool = False,
    supports_dcp_with_varlen: bool = False,
) -> None:
    _ORIGINAL_INIT_THRESHOLD(
        self,
        reorder_batch_threshold,
        supports_spec_as_decode,
        supports_dcp_with_varlen,
    )
    if self.reorder_batch_threshold != 1:
        speculative_config = self.vllm_config.speculative_config
        if (
            speculative_config is not None
            and speculative_config.num_speculative_tokens is not None
            and hasattr(speculative_config, "method")
            and speculative_config.method == "dflash"
        ):
            self.reorder_batch_threshold = 1 + speculative_config.num_speculative_tokens


if not _IS_PATCHED and not is_310p():
    gdn_attn.GDNChunkedPrefillMetadata = GDNChunkedPrefillMetadata
    gdn_attn.GDNCausalConv1dHostMetadata = GDNCausalConv1dHostMetadata
    gdn_attn.GDNPrefillFallbackMeta = GDNPrefillFallbackMeta
    gdn_attn.GDNAttentionMetadataBuilder.build = _patched_build
    gdn_attn.GDNAttentionMetadataBuilder._init_reorder_batch_threshold = _init_reorder_batch_threshold
    _IS_PATCHED = True
