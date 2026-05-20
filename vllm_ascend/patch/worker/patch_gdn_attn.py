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
    """Operator arguments consumed by the GDN chunked-prefill FLA kernels."""

    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor
    _buffer_slot: object | None = None


@dataclass
class GDNCausalConv1dHostMetadata:
    """CPU-side operator arguments consumed by the GDN causal-conv1d path."""

    query_start_loc_cpu: torch.Tensor
    cache_indices_cpu: torch.Tensor
    has_initial_state_cpu: torch.Tensor
    _buffer_slot: object | None = None


@dataclass
class GDNNonSpecPrefillOperatorMetadata:
    """Prebuilt metadata consumed by GDN non-spec prefill operators."""

    causal_conv1d: GDNCausalConv1dHostMetadata
    chunk: GDNChunkedPrefillMetadata


@dataclass
class _GDNChunkedPrefillBufferSlot:
    """Reusable device buffers sized for the scheduler maximums."""

    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor


@dataclass
class _GDNCausalConv1dHostBufferSlot:
    """Reusable pinned CPU buffers for async device-to-host metadata copies."""

    cache_indices_cpu: torch.Tensor
    has_initial_state_cpu: torch.Tensor


@dataclass
class _GDNChunkOperatorArgSizes:
    """Row counts needed by each chunk metadata tensor for one batch."""

    num_seqs: int
    num_chunk_indices_chunk64: int
    num_chunk_indices_large_block: int
    num_block_indices_cumsum: int


@dataclass
class _GDNChunkOperatorArgCpuCounts(_GDNChunkOperatorArgSizes):
    """CPU chunk counts plus the derived operator-argument tensor sizes."""

    chunk_counts_chunk64: torch.Tensor
    chunk_counts_large_block: torch.Tensor
    chunk_counts_cumsum: torch.Tensor


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _prepare_chunk_counts_cpu(cu_seqlens_cpu: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Return per-sequence chunk counts from CPU cumulative sequence offsets.

    Args:
        cu_seqlens_cpu: CPU cumulative sequence offsets, shape
            ``[num_seqs + 1]``.
        chunk_size: Number of tokens per chunk.

    Returns:
        A CPU tensor of shape ``[num_seqs]`` containing
        ``ceil(seq_len / chunk_size)`` for each sequence.
    """
    seq_lens_cpu = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    return torch.div(seq_lens_cpu + chunk_size - 1, chunk_size, rounding_mode="floor")


def _fill_chunk_indices_cpu(chunk_indices: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    """Fill CPU chunk-index rows for one chunk size.

    Args:
        chunk_indices: Output tensor of shape ``[num_chunks, 2]``.
        chunk_counts: Per-sequence chunk counts, shape ``[num_seqs]``.

    Returns:
        Number of rows written. Each row is
        ``[compact_sequence_index, chunk_index]`` and zero-length sequences are
        skipped to match the runtime FLA helper.
    """
    write_offset = 0
    compact_sequence_index = 0
    for num_chunks in chunk_counts.tolist():
        if num_chunks <= 0:
            continue
        # `prepare_chunk_indices` compacts away zero-length sequences, so the
        # sequence index here must follow the same compact numbering.
        chunk_indices[write_offset : write_offset + num_chunks, 0].fill_(compact_sequence_index)
        chunk_indices[write_offset : write_offset + num_chunks, 1] = torch.arange(
            num_chunks,
            dtype=chunk_indices.dtype,
        )
        write_offset += num_chunks
        compact_sequence_index += 1
    return write_offset


def _fill_chunk_offsets_cpu(chunk_offsets: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    """Fill CPU prefix offsets into chunk-index rows.

    Args:
        chunk_offsets: Output prefix tensor, shape ``[num_seqs + 1]``.
        chunk_counts: Per-sequence chunk counts, shape ``[num_seqs]``.

    Returns:
        Number of prefix entries written.
    """
    chunk_offsets[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(chunk_counts, dim=0, out=chunk_offsets[1 : chunk_counts.numel() + 1])
    return chunk_counts.numel() + 1


def _fill_update_chunk_offsets_cpu(update_chunk_offsets: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    """Fill CPU update-path prefix offsets.

    Args:
        update_chunk_offsets: Output prefix tensor, shape ``[num_seqs + 1]``.
        chunk_counts: Per-sequence chunk counts, shape ``[num_seqs]``.

    Returns:
        Number of prefix entries written. Each sequence contributes its normal
        chunk count plus one extra recurrent state-update row.
    """
    update_chunk_offsets[0] = 0
    if chunk_counts.numel() > 0:
        torch.cumsum(
            chunk_counts + 1,
            dim=0,
            out=update_chunk_offsets[1 : chunk_counts.numel() + 1],
        )
    return chunk_counts.numel() + 1


def _fill_final_chunk_indices_cpu(final_chunk_indices: torch.Tensor, chunk_counts: torch.Tensor) -> int:
    """Fill CPU final update-row indices.

    Args:
        final_chunk_indices: Output tensor, shape ``[num_seqs]``.
        chunk_counts: Per-sequence chunk counts, shape ``[num_seqs]``.

    Returns:
        Number of entries written. Entry ``i`` is the last update row reserved
        for sequence ``i``.
    """
    if chunk_counts.numel() > 0:
        torch.cumsum(
            chunk_counts + 1,
            dim=0,
            out=final_chunk_indices[: chunk_counts.numel()],
        )
        final_chunk_indices[: chunk_counts.numel()].sub_(1)
    return chunk_counts.numel()


def _build_chunk_operator_arg_cpu_counts(builder, cu_seqlens_cpu: torch.Tensor) -> _GDNChunkOperatorArgCpuCounts:
    """Build CPU chunk counts and derived tensor sizes for all GDN chunk args.

    Args:
        builder: Patched GDN metadata builder carrying the three chunk sizes.
        cu_seqlens_cpu: CPU cumulative offsets for non-spec prefill requests.

    Returns:
        Per-chunk-size counts plus the output row counts needed to allocate and
        fill chunk64, large-block, and cumsum operator arguments on CPU.
    """
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
    return _GDNChunkOperatorArgCpuCounts(
        num_seqs=chunk_counts_chunk64.numel(),
        num_chunk_indices_chunk64=int(chunk_counts_chunk64.sum().item()),
        num_chunk_indices_large_block=int(chunk_counts_large_block.sum().item()),
        num_block_indices_cumsum=int(chunk_counts_cumsum.sum().item()),
        chunk_counts_chunk64=chunk_counts_chunk64,
        chunk_counts_large_block=chunk_counts_large_block,
        chunk_counts_cumsum=chunk_counts_cumsum,
    )


def _count_chunk_indices_cpu(seq_lens_cpu: torch.Tensor, chunk_size: int) -> int:
    """Return the number of rows needed by prepare_chunk_indices.

    Args:
        seq_lens_cpu: CPU sequence lengths, shape ``[num_seqs]``.
        chunk_size: Number of tokens per chunk.

    Returns:
        Total number of chunk-index rows for this chunk size.
    """
    return int(
        torch.div(
            seq_lens_cpu + chunk_size - 1,
            chunk_size,
            rounding_mode="floor",
        )
        .sum()
        .item()
    )


def _build_chunk_operator_arg_sizes(builder, cu_seqlens_cpu: torch.Tensor) -> _GDNChunkOperatorArgSizes:
    """Build only the tensor sizes needed by device-side metadata filling.

    Args:
        builder: Patched GDN metadata builder carrying the three chunk sizes.
        cu_seqlens_cpu: CPU cumulative offsets for non-spec prefill requests.

    Returns:
        Row counts used to slice reusable device slots before Triton/Torch fills
        the actual metadata values on NPU.
    """
    seq_lens_cpu = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    return _GDNChunkOperatorArgSizes(
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


def _allocate_chunk_meta_cpu_tensors(arg_sizes: _GDNChunkOperatorArgSizes) -> dict[str, torch.Tensor]:
    """Allocate CPU tensors for all GDN chunked-prefill operator arguments.

    Args:
        arg_sizes: Exact row counts for the current non-spec prefill batch.

    Returns:
        A dictionary keyed by metadata field name. The tensors are empty and
        must be filled before being wrapped in ``GDNChunkedPrefillMetadata``.
    """
    return {
        "chunk_indices_chunk64": torch.empty(
            (arg_sizes.num_chunk_indices_chunk64, 2),
            dtype=torch.int32,
        ),
        "chunk_offsets_chunk64": torch.empty(
            (arg_sizes.num_seqs + 1,),
            dtype=torch.int32,
        ),
        "update_chunk_offsets_chunk64": torch.empty(
            (arg_sizes.num_seqs + 1,),
            dtype=torch.int32,
        ),
        "final_chunk_indices_chunk64": torch.empty(
            (arg_sizes.num_seqs,),
            dtype=torch.int32,
        ),
        "chunk_indices_large_block": torch.empty(
            (arg_sizes.num_chunk_indices_large_block, 2),
            dtype=torch.int32,
        ),
        "block_indices_cumsum": torch.empty(
            (arg_sizes.num_block_indices_cumsum, 2),
            dtype=torch.int32,
        ),
    }


def _slice_chunk_meta_slot_tensors(
    slot: _GDNChunkedPrefillBufferSlot,
    arg_sizes: _GDNChunkOperatorArgSizes,
) -> dict[str, torch.Tensor]:
    """Take current-batch views from reusable device metadata buffers.

    Args:
        slot: One reusable buffer slot allocated to scheduler maximum sizes.
        arg_sizes: Exact row counts for the current non-spec prefill batch.

    Returns:
        A dictionary of device tensor views sized to the current batch. The
        views keep ``slot`` storage and are filled in place by the device path.
    """
    return {
        "chunk_indices_chunk64": slot.chunk_indices_chunk64[: arg_sizes.num_chunk_indices_chunk64],
        "chunk_offsets_chunk64": slot.chunk_offsets_chunk64[: arg_sizes.num_seqs + 1],
        "update_chunk_offsets_chunk64": slot.update_chunk_offsets_chunk64[: arg_sizes.num_seqs + 1],
        "final_chunk_indices_chunk64": slot.final_chunk_indices_chunk64[: arg_sizes.num_seqs],
        "chunk_indices_large_block": slot.chunk_indices_large_block[: arg_sizes.num_chunk_indices_large_block],
        "block_indices_cumsum": slot.block_indices_cumsum[: arg_sizes.num_block_indices_cumsum],
    }


def _fill_chunk_meta_cpu_tensors(
    tensors: dict[str, torch.Tensor],
    cpu_counts: _GDNChunkOperatorArgCpuCounts,
) -> None:
    """Fill all CPU GDN chunked-prefill metadata tensors.

    Args:
        tensors: CPU tensors allocated by ``_allocate_chunk_meta_cpu_tensors``.
        cpu_counts: Per-chunk-size counts and derived row counts.

    Returns:
        None. The tensors are mutated in place and then passed to the GDN FLA
        operators through ``GDNChunkedPrefillMetadata``.
    """
    _fill_chunk_indices_cpu(
        tensors["chunk_indices_chunk64"],
        cpu_counts.chunk_counts_chunk64,
    )
    _fill_chunk_offsets_cpu(
        tensors["chunk_offsets_chunk64"],
        cpu_counts.chunk_counts_chunk64,
    )
    _fill_update_chunk_offsets_cpu(
        tensors["update_chunk_offsets_chunk64"],
        cpu_counts.chunk_counts_chunk64,
    )
    _fill_final_chunk_indices_cpu(
        tensors["final_chunk_indices_chunk64"],
        cpu_counts.chunk_counts_chunk64,
    )
    _fill_chunk_indices_cpu(
        tensors["chunk_indices_large_block"],
        cpu_counts.chunk_counts_large_block,
    )
    _fill_chunk_indices_cpu(
        tensors["block_indices_cumsum"],
        cpu_counts.chunk_counts_cumsum,
    )


def _fill_chunk_meta_device_tensors(
    builder,
    cu_seqlens: torch.Tensor,
    tensors: dict[str, torch.Tensor],
) -> None:
    """Fill all device GDN chunked-prefill metadata tensors.

    Args:
        builder: Patched GDN metadata builder carrying chunk sizes.
        cu_seqlens: NPU cumulative offsets for non-spec prefill requests.
        tensors: Device tensor views from ``_slice_chunk_meta_slot_tensors``.

    Returns:
        None. The tensors are filled in place by ``build_chunk_meta_device``.
        ``cu_seqlens`` is validated once, and the derived sequence lengths are
        reused for all three chunk sizes.
    """
    device_seq_lens = None
    validate_inputs = True
    if cu_seqlens.device.type == "npu":
        # Validate the shared cu_seqlens once, then reuse its derived sequence
        # lengths for all chunk sizes to avoid redundant device slicing.
        _validate_cu_seqlens(cu_seqlens, builder._ascend_gdn_chunk_size)
        assert builder._ascend_gdn_large_block_size > 0
        assert builder._ascend_gdn_cumsum_block_size > 0
        device_seq_lens = _build_seq_lens(cu_seqlens)
        validate_inputs = False
    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=builder._ascend_gdn_chunk_size,
        out_chunk_indices=tensors["chunk_indices_chunk64"],
        out_chunk_offsets=tensors["chunk_offsets_chunk64"],
        out_update_chunk_offsets=tensors["update_chunk_offsets_chunk64"],
        out_final_chunk_indices=tensors["final_chunk_indices_chunk64"],
        seq_lens=device_seq_lens,
        validate_inputs=validate_inputs,
    )
    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=builder._ascend_gdn_large_block_size,
        out_chunk_indices=tensors["chunk_indices_large_block"],
        seq_lens=device_seq_lens,
        validate_inputs=validate_inputs,
    )
    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=builder._ascend_gdn_cumsum_block_size,
        out_chunk_indices=tensors["block_indices_cumsum"],
        seq_lens=device_seq_lens,
        validate_inputs=validate_inputs,
    )


def _build_chunked_prefill_metadata(
    tensors: dict[str, torch.Tensor],
    *,
    slot: _GDNChunkedPrefillBufferSlot | None = None,
) -> GDNChunkedPrefillMetadata:
    """Wrap filled tensors in the metadata object consumed by GDN operators.

    Args:
        tensors: Filled chunk metadata tensors keyed by field name.
        slot: Optional owner of reusable device storage. Keeping this reference
            prevents the backing buffers from being reused too early.

    Returns:
        A ``GDNChunkedPrefillMetadata`` instance ready for operator calls.
    """
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


def _allocate_chunked_prefill_slot(builder, device: torch.device) -> _GDNChunkedPrefillBufferSlot:
    """Allocate one reusable device slot for GDN chunked-prefill metadata.

    Args:
        builder: GDN metadata builder with scheduler maximum batch sizes.
        device: Target device for the operator metadata tensors.

    Returns:
        A slot sized to ``max_num_batched_tokens`` and ``max_num_seqs``.
    """
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
    """Initialize chunk metadata state on the patched builder.

    Args:
        builder: GDN metadata builder being patched.
        device: Device where chunk metadata should be produced.

    Returns:
        None. The builder receives chunk-size constants and, for non-CPU
        devices, a small pool of reusable metadata buffer slots.
    """
    if getattr(builder, "_ascend_gdn_chunk_meta_initialized", False):
        return
    builder._ascend_gdn_chunk_meta_initialized = True
    builder._ascend_gdn_chunk_meta_device = device
    builder._ascend_gdn_chunk_size = _GDN_CHUNK_SIZE
    builder._ascend_gdn_large_block_size = _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE
    num_gdn_value_heads = _get_gdn_num_heads(builder)
    # The cumsum chunk size is chosen so the per-head working set stays under
    # the tuned solve-tril budget, rounded up for Triton-friendly tiling.
    cumsum_chunk_size = max(
        1,
        _GDN_CUMSUM_WORKING_SET // (num_gdn_value_heads * builder._ascend_gdn_chunk_size),
    )
    builder._ascend_gdn_cumsum_block_size = _next_power_of_2(cumsum_chunk_size)
    builder._ascend_gdn_chunked_prefill_pool_idx = -1
    builder._ascend_gdn_chunked_prefill_pool = []
    if device.type != "cpu":
        builder._ascend_gdn_chunked_prefill_pool = [
            _allocate_chunked_prefill_slot(builder, device),
            _allocate_chunked_prefill_slot(builder, device),
        ]


def _has_spec_decode_drafts(num_decode_draft_tokens_cpu: torch.Tensor) -> bool:
    # The upstream builder only enters the spec path when at least one request
    # has a positive draft-token count; all-zero draft counts are no-op spec.
    return bool(torch.gt(num_decode_draft_tokens_cpu, 0).any().item())


def _build_spec_sequence_masks_cpu(builder, num_decode_draft_tokens_cpu: torch.Tensor | None) -> torch.Tensor | None:
    if (
        not getattr(builder, "use_spec_decode", False)
        or num_decode_draft_tokens_cpu is None
        or not _has_spec_decode_drafts(num_decode_draft_tokens_cpu)
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
    # The returned view is kept alive through the metadata object's buffer slot.
    cpu_view = pinned_buffer[: tensor.numel()]
    cpu_view.copy_(
        tensor.reshape(-1),
        non_blocking=True,
    )
    return cpu_view


def _build_non_spec_causal_conv1d_host_meta(
    builder,
    attn_metadata,
    non_spec_query_start_loc_cpu: torch.Tensor,
) -> GDNCausalConv1dHostMetadata:
    assert attn_metadata.num_prefills > 0
    non_spec_state_indices = attn_metadata.non_spec_state_indices_tensor
    has_initial_state = attn_metadata.has_initial_state

    host_slot = None
    if non_spec_state_indices.device.type != "cpu" or has_initial_state.device.type != "cpu":
        host_slot = _acquire_causal_conv1d_host_slot(builder)

    cache_indices_cpu = _copy_to_pinned_cpu(
        non_spec_state_indices,
        None if host_slot is None else host_slot.cache_indices_cpu,
    )
    has_initial_state_cpu = _copy_to_pinned_cpu(
        has_initial_state,
        None if host_slot is None else host_slot.has_initial_state_cpu,
    )

    return GDNCausalConv1dHostMetadata(
        query_start_loc_cpu=non_spec_query_start_loc_cpu,
        cache_indices_cpu=cache_indices_cpu,
        has_initial_state_cpu=has_initial_state_cpu,
        _buffer_slot=host_slot,
    )


def _build_non_spec_chunked_prefill_meta_cpu(
    builder,
    non_spec_query_start_loc_cpu: torch.Tensor,
) -> GDNChunkedPrefillMetadata:
    """Build GDN chunked-prefill operator metadata entirely on CPU.

    Args:
        builder: Patched GDN metadata builder carrying chunk sizes.
        non_spec_query_start_loc_cpu: CPU cumulative offsets for non-spec
            prefill requests.

    Returns:
        Filled chunked-prefill metadata whose tensors live on CPU.
    """
    cpu_counts = _build_chunk_operator_arg_cpu_counts(builder, non_spec_query_start_loc_cpu)
    tensors = _allocate_chunk_meta_cpu_tensors(cpu_counts)
    _fill_chunk_meta_cpu_tensors(tensors, cpu_counts)
    return _build_chunked_prefill_metadata(tensors)


def _build_non_spec_chunked_prefill_meta(
    builder,
    non_spec_query_start_loc_cpu: torch.Tensor,
    non_spec_query_start_loc: torch.Tensor,
) -> GDNChunkedPrefillMetadata:
    """Build GDN chunked-prefill operator metadata for non-spec prefill.

    Args:
        builder: Patched GDN metadata builder carrying state and chunk sizes.
        non_spec_query_start_loc_cpu: CPU cumulative offsets used to calculate
            exact tensor row counts.
        non_spec_query_start_loc: Runtime cumulative offsets on the target
            device, used to fill metadata values on device.

    Returns:
        ``GDNChunkedPrefillMetadata`` containing chunk64, large-block, and
        cumsum metadata tensors for the GDN operators.
    """
    device = builder._ascend_gdn_chunk_meta_device
    if device.type == "cpu":
        return _build_non_spec_chunked_prefill_meta_cpu(builder, non_spec_query_start_loc_cpu)

    arg_sizes = _build_chunk_operator_arg_sizes(builder, non_spec_query_start_loc_cpu)
    builder._ascend_gdn_chunked_prefill_pool_idx = (builder._ascend_gdn_chunked_prefill_pool_idx + 1) % len(
        builder._ascend_gdn_chunked_prefill_pool
    )
    slot = builder._ascend_gdn_chunked_prefill_pool[builder._ascend_gdn_chunked_prefill_pool_idx]
    tensors = _slice_chunk_meta_slot_tensors(slot, arg_sizes)
    _fill_chunk_meta_device_tensors(builder, non_spec_query_start_loc, tensors)
    return _build_chunked_prefill_metadata(tensors, slot=slot)


def _patched_build(
    self,
    common_prefix_len: int,
    common_attn_metadata,
    num_accepted_tokens: torch.Tensor | None = None,
    num_decode_draft_tokens_cpu: torch.Tensor | None = None,
    fast_build: bool = False,
):
    attn_metadata = _ORIGINAL_BUILD(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        fast_build=fast_build,
    )
    attn_metadata.non_spec_prefill_operator_meta = None
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
    attn_metadata.non_spec_prefill_operator_meta = GDNNonSpecPrefillOperatorMetadata(
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
    gdn_attn.GDNNonSpecPrefillOperatorMetadata = GDNNonSpecPrefillOperatorMetadata
    gdn_attn.GDNAttentionMetadataBuilder.build = _patched_build
    gdn_attn.GDNAttentionMetadataBuilder._init_reorder_batch_threshold = _init_reorder_batch_threshold
    _IS_PATCHED = True
