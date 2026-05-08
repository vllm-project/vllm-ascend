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

from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm_ascend.ops.triton.gdn_chunk_meta import (
    _build_seq_lens,
    _validate_cu_seqlens,
    build_chunk_meta_device,
)

_GDN_CHUNK_SIZE = 64
# Keep this aligned with solve_tril.LARGE_BLOCK_T in ops/triton/fla/solve_tril.py.
_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2
_GDN_CUMSUM_WORKING_SET = 2**18


def _stable_argsort_for_npu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.int32)
    return torch.argsort(tensor, stable=True)


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
            "Expected attn_metadata.non_spec_state_indices_tensor for Ascend GDN non-spec prefill path."
        )
    if attn_metadata.has_initial_state is None:
        raise RuntimeError("Expected attn_metadata.has_initial_state for Ascend GDN non-spec prefill path.")

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


class AscendGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.spec_sequence_masks_cpu: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.bool,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.spec_sequence_indices_cpu: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.non_spec_sequence_indices_cpu: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.spec_sequence_indices: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int64,
            device=device,
        )
        self.non_spec_sequence_indices: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int64,
            device=device,
        )

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        super()._init_reorder_batch_threshold(
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

    def _copy_sequence_indices_to_device(
        self,
        spec_sequence_masks_cpu: torch.Tensor,
        num_spec_decodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = spec_sequence_masks_cpu.numel()
        num_non_spec_decodes = num_reqs - num_spec_decodes

        spec_indices_cpu = self.spec_sequence_indices_cpu[:num_spec_decodes]
        spec_indices_cpu.copy_(
            torch.nonzero(spec_sequence_masks_cpu, as_tuple=True)[0],
        )
        spec_indices = self.spec_sequence_indices[:num_spec_decodes]
        spec_indices.copy_(spec_indices_cpu, non_blocking=True)

        non_spec_indices_cpu = self.non_spec_sequence_indices_cpu[:num_non_spec_decodes]
        non_spec_indices_cpu.copy_(
            torch.nonzero(~spec_sequence_masks_cpu, as_tuple=True)[0],
        )
        non_spec_indices = self.non_spec_sequence_indices[:num_non_spec_decodes]
        non_spec_indices.copy_(non_spec_indices_cpu, non_blocking=True)

        return spec_indices, non_spec_indices

    def _attach_non_spec_prefill_fallback_meta(
        self,
        attn_metadata: GDNAttentionMetadata,
        common_attn_metadata: CommonAttentionMetadata,
        non_spec_query_start_loc_cpu: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        attn_metadata.non_spec_prefill_fallback_meta = None
        if attn_metadata.num_prefills <= 0:
            return attn_metadata

        _ensure_chunk_meta_state(self, common_attn_metadata.query_start_loc.device)
        _ensure_causal_conv1d_host_meta_state(
            self,
            common_attn_metadata.query_start_loc.device,
        )
        if non_spec_query_start_loc_cpu is None:
            raise RuntimeError("Expected non_spec_query_start_loc_cpu for Ascend GDN non-spec prefill path.")
        if attn_metadata.non_spec_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.non_spec_query_start_loc for Ascend GDN non-spec prefill path.")

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

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        context_lens_tensor = m.compute_num_computed_tokens()
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

        spec_sequence_masks_cpu: torch.Tensor | None = None
        spec_sequence_indices: torch.Tensor | None = None
        non_spec_sequence_indices: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            num_reqs = num_decode_draft_tokens_cpu.numel()
            spec_sequence_masks_cpu = self.spec_sequence_masks_cpu[:num_reqs]
            torch.ge(
                num_decode_draft_tokens_cpu,
                0,
                out=spec_sequence_masks_cpu,
            )
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = self.spec_sequence_masks[:num_reqs]
                spec_sequence_masks.copy_(spec_sequence_masks_cpu, non_blocking=True)
                spec_sequence_indices, non_spec_sequence_indices = self._copy_sequence_indices_to_device(
                    spec_sequence_masks_cpu,
                    num_spec_decodes,
                )

        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
                m,
                decode_threshold=1,
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            assert spec_sequence_masks_cpu is not None
            assert spec_sequence_indices is not None
            assert non_spec_sequence_indices is not None

            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            num_spec_decode_tokens = query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens

            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                spec_state_indices_tensor = torch.index_select(
                    block_table_tensor[:, : self.num_spec + 1],
                    0,
                    spec_sequence_indices,
                )
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = _stable_argsort_for_npu(spec_token_masks)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = torch.index_select(
                    block_table_tensor[:, : self.num_spec + 1],
                    0,
                    spec_sequence_indices,
                )
                non_spec_state_indices_tensor = torch.index_select(
                    block_table_tensor[:, 0],
                    0,
                    non_spec_sequence_indices,
                )
                spec_query_lens = torch.index_select(
                    query_lens,
                    0,
                    spec_sequence_indices,
                )
                non_spec_query_lens = torch.index_select(
                    query_lens,
                    0,
                    non_spec_sequence_indices,
                )

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    spec_query_lens,
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    non_spec_query_lens,
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = torch.index_select(
                num_accepted_tokens,
                0,
                spec_sequence_indices,
            )

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        if num_prefills > 0:
            from vllm.model_executor.layers.fla.ops.index import (
                prepare_chunk_indices,
                prepare_chunk_offsets,
            )
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            gpu_device = query_start_loc.device
            chunk_indices = prepare_chunk_indices(
                non_spec_query_start_loc_cpu,
                FLA_CHUNK_SIZE,
            ).to(device=gpu_device, non_blocking=True)
            chunk_offsets = prepare_chunk_offsets(
                non_spec_query_start_loc_cpu,
                FLA_CHUNK_SIZE,
            ).to(device=gpu_device, non_blocking=True)

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                assert non_spec_sequence_indices is not None
                has_initial_state = torch.index_select(
                    has_initial_state,
                    0,
                    non_spec_sequence_indices,
                )
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
                non_spec_query_start_loc_cpu,
                device=query_start_loc.device,
            )
        else:
            has_initial_state = None

        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor,
                non_blocking=True,
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)

            self.spec_sequence_masks[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes],
                non_blocking=True,
            )
            spec_sequence_masks = self.spec_sequence_masks[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx,
                non_blocking=True,
            )
            non_spec_token_indx = self.non_spec_token_indx[: non_spec_token_indx.size(0)]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx,
                non_blocking=True,
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]

            self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                spec_query_start_loc,
                non_blocking=True,
            )
            spec_num_query_tokens = spec_query_start_loc[-1]
            spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            self.num_accepted_tokens[:num_spec_decodes].copy_(
                num_accepted_tokens,
                non_blocking=True,
            )
            num_accepted_tokens = self.num_accepted_tokens[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor,
                non_blocking=True,
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[:batch_size]
            non_spec_state_indices_tensor[num_decodes:].fill_(NULL_BLOCK_ID)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc,
                non_blocking=True,
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return self._attach_non_spec_prefill_fallback_meta(
            attn_metadata,
            common_attn_metadata,
            non_spec_query_start_loc_cpu,
        )


class AscendGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder]:
        return AscendGDNAttentionMetadataBuilder
