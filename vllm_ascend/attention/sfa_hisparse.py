# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend SFA execution over upstream HiSparse's resident and hot cache pool."""

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch.overrides import TorchFunctionMode
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup, SparseMLAIndexGroupBuilder
from vllm.v1.attention.ops.metadata import compute_token_to_req_indices
from vllm.v1.hisparse.runtime import HiSparsePrefillStagingPlan
from vllm.v1.hisparse.runtime import build_hisparse_prefill_staging_plan as build_core_prefill_plan
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    needs_pcp_kv_gather,
)
from vllm_ascend.hisparse import cache_ops
from vllm_ascend.hisparse.c8 import C8_ROW_BYTES
from vllm_ascend.hisparse.pcp import prepare_pcp_query_metadata

# Bound the temporary D2D packing workspace independently of prefill length.
_ATTENTION_CHUNK_ROWS = 32


def finish_hisparse_kv_update(cache, metadata) -> None:
    """Publish real writes; capture forwards have no connector step to mirror.

    V2 runs both FULL-graph warmup and capture with runtime mode NONE.
    Preserve the device KV writes in those forwards, but leave Host mirror
    submission to the real connector step after graph replay.
    """
    if not getattr(metadata, "hisparse_for_capture", False):
        cache.finish_kv_update()


class _PrefillColumnSelection(TorchFunctionMode):
    """Keep a local workaround for the NPU singleton-column indexing defect.

    torch-npu 2.10.0.post7 / CANN 9.2.1 can interleave rows for ``x[:, ids]``
    when x has one column. index_select has the same semantics and returns
    correct rows. Remove this scope when the supported device stack is fixed.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.Tensor.__getitem__:
            tensor, index = args
            if (
                tensor.device.type == "npu"
                and tensor.ndim == 2
                and isinstance(index, tuple)
                and len(index) == 2
                and isinstance(index[0], slice)
                and index[0] == slice(None)
                and isinstance(index[1], torch.Tensor)
                and index[1].ndim == 1
            ):
                return torch.index_select(tensor, 1, index[1])
        return func(*args, **(kwargs or {}))


class AscendHiSparsePrefillStagingPlan(HiSparsePrefillStagingPlan):
    def ensure_gpu_sources(self, resident_state_rows, state_indices, resident_block_size) -> None:
        # Core retains source-key caching, representative-page selection and
        # current Resident/Host masks. The override affects only this call.
        with _PrefillColumnSelection():
            super().ensure_gpu_sources(resident_state_rows, state_indices, resident_block_size)


def build_hisparse_prefill_staging_plan(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    staging_block_capacity: int,
) -> AscendHiSparsePrefillStagingPlan:
    plan = build_core_prefill_plan(block_table, seq_lens, block_size, staging_block_capacity)
    return AscendHiSparsePrefillStagingPlan(
        block_table=plan.block_table,
        row_ids=plan.row_ids,
        dst_rows=plan.dst_rows,
        miss_mask=plan.miss_mask,
        block_size=plan.block_size,
    )


@dataclass
class AscendHiSparsePrefillMetadata:
    host_staging_plan: AscendHiSparsePrefillStagingPlan


@dataclass
class AscendHiSparseMetadata(AscendSFAMetadata):
    req_id_per_token: torch.Tensor | None = None
    num_reqs: int = 0
    prefill: AscendHiSparsePrefillMetadata | None = None
    hisparse_for_capture: bool = False


class AscendHiSparseMetadataBuilder(AscendSFAMetadataBuilder):
    def build_for_cudagraph_capture(self, common_attn_metadata, **kwargs):
        metadata = cast(AscendHiSparseMetadata, super().build_for_cudagraph_capture(common_attn_metadata, **kwargs))
        metadata.hisparse_for_capture = True
        return metadata

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kwargs):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, metadata_cls=AscendHiSparseMetadata, **kwargs)
        self._request_row_buffers: dict[int | None, torch.Tensor] = {}
        self._pcp_query_buffers: dict[int | None, tuple[torch.Tensor, torch.Tensor]] = {}
        self._pcp_request_capacity = 2 * vllm_config.scheduler_config.max_num_seqs
        self._pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self._row_capacity = max(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )

    def _build(
        self,
        common_attn_metadata,
        draft_index=None,
        pcp_context=None,
    ):
        metadata = cast(AscendHiSparseMetadata, super()._build(common_attn_metadata, draft_index, pcp_context))
        metadata.num_reqs = common_attn_metadata.num_reqs
        if draft_index not in self._request_row_buffers:
            self._request_row_buffers[draft_index] = torch.empty(
                self._row_capacity, device=self.device, dtype=torch.int32
            )
        # Use Core's device-boundary mapping with Ascend's padded input extent.
        # Keep separate backing storage for each captured MTP graph owner.
        metadata.req_id_per_token = compute_token_to_req_indices(
            metadata.query_start_loc,
            self._request_row_buffers[draft_index],
            int(common_attn_metadata.query_start_loc_cpu[-1]),
            metadata.num_input_tokens,
        )
        if pcp_context is not None:
            if draft_index not in self._pcp_query_buffers:
                self._pcp_query_buffers[draft_index] = (
                    torch.empty(self._pcp_request_capacity, dtype=torch.int32, device=self.device),
                    torch.empty(1, dtype=torch.int32, device=self.device),
                )
            states, valid_rows = self._pcp_query_buffers[draft_index]
            local_states = pcp_context.local_request_state_indices
            assert local_states is not None
            prepare_pcp_query_metadata(metadata, states, valid_rows, local_states, self._pcp_size)
        if metadata.num_prefills:
            seq_lens_cpu = metadata.seq_lens_cpu
            if seq_lens_cpu is None:
                # Capacity may use an upper bound; the staging plan receives
                # current device lengths to decide the actual copied rows.
                seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            if seq_lens_cpu is None:
                raise RuntimeError("HiSparse prefill requires scheduler sequence-length bounds on CPU")
            prefill_lens = seq_lens_cpu[metadata.num_decodes :]
            capacity = int(((prefill_lens + metadata.block_size - 1) // metadata.block_size).sum())
            metadata.prefill = AscendHiSparsePrefillMetadata(
                build_hisparse_prefill_staging_plan(
                    metadata.block_table[metadata.num_decodes :],
                    metadata.seq_lens[metadata.num_decodes :],
                    metadata.block_size,
                    capacity,
                )
            )
        return metadata


class AscendHiSparseImpl(AscendSFAImpl):
    """Keep HiSparse residency decisions on device, with an SFA layout bridge."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.enable_sparse_sfa_turboquant:
            raise ValueError("Ascend HiSparse does not support TurboQuant")
        if self.enable_sparse_sfa_c8:
            if (
                self.c8_cache_dtype != torch.float8_e4m3fn
                or (self.kv_lora_rank, self.qk_rope_head_dim) != (512, 64)
                or self.sfa_qsfa_tile_size != 128
            ):
                raise ValueError("HiSparse C8 requires native E4M3FN NoPE512/RoPE64 with FP32 scales")
        parallel = self.vllm_config.parallel_config
        if parallel.decode_context_parallel_size > 1:
            raise ValueError("Ascend HiSparse does not support DCP")
        if parallel.prefill_context_parallel_size > 1:
            if not self.enable_sparse_sfa_c8 or not self.is_pcp_decode_sharded:
                raise ValueError("HiSparse PCP requires C8 and request-sharded target decode")
        builder = kwargs.get("index_group_builder")
        if builder is None:
            if self.topk_indices_buffer is None:
                raise ValueError("Ascend HiSparse requires a model sparse-index group")
            builder = SparseMLAIndexGroupBuilder(self.topk_indices_buffer)
        if self.enable_sparse_sfa_c8:
            row_width = C8_ROW_BYTES
        else:
            row_width = self.kv_lora_rank + self.qk_rope_head_dim
        self.index_group, self.index_group_index = builder.register_layer(
            self.has_indexer and not self.skip_topk,
            self.vllm_config,
            row_width=row_width,
            kv_cache_dtype=self.vllm_config.cache_config.cache_dtype,
        )
        if not isinstance(self.index_group, HiSparseMLAIndexGroup):
            raise RuntimeError("Ascend HiSparse was initialized without a HiSparse index group")
        self.hisparse_cache = self.index_group.cache(self.index_group_index)
        self.topk = self.index_group.logical_topk_indices.shape[-1]
        if self.topk % 128:
            raise ValueError("Ascend HiSparse SFA requires top-k to be divisible by 128")
        max_rows = max(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            self.vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        self._kv_staging_capacity = (max_rows + 127) // 128 * 128

    def _fused_preprocess_type(self):
        # Reuse SFA's supported quantization/weight preparation. The HiSparse
        # handle supplies the final Resident cache and current write mapping.
        if self.enable_sparse_sfa_c8:
            from vllm_ascend.quantization.methods import AscendW8A8MXFP8DynamicLinearMethod

            # Native Prolog ties its C8 payload dtype to the weight scheme.
            # INT8 weights cannot write this bridge's E4M3FN cache; retain the
            # existing native projection/quantizer path for that combination.
            method = self._get_layer_quant_method(self.fused_qkv_a_proj)
            if type(method) is AscendW8A8MXFP8DynamicLinearMethod:
                return super()._fused_preprocess_type()
        return None

    def _compose_sfa_kv_cache(self, kv_cache):
        return (kv_cache,)

    def _workspace(self, dtype):
        key = ("ascend_hisparse_attention", self.topk, self.kv_lora_rank, self.qk_rope_head_dim, dtype)

        def allocate():
            device = self.index_group.logical_topk_indices.device
            shape = (_ATTENTION_CHUNK_ROWS, self.topk, 1)
            return (
                torch.empty((*shape, self.kv_lora_rank), dtype=dtype, device=device),
                torch.empty((*shape, self.qk_rope_head_dim), dtype=dtype, device=device),
                torch.arange(_ATTENTION_CHUNK_ROWS * self.topk // 128, dtype=torch.int32, device=device).view(
                    _ATTENTION_CHUNK_ROWS, self.topk // 128
                ),
                torch.arange(self.topk, dtype=torch.int32, device=device).view(1, 1, self.topk),
                torch.arange(1, _ATTENTION_CHUNK_ROWS + 1, dtype=torch.int32, device=device),
            )

        return current_workspace_manager().get_persistent_resource(key, allocate)

    def _kv_workspace(self, dtype):
        key = (
            "ascend_hisparse_kv_preprocess",
            self._kv_staging_capacity,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            dtype,
        )

        def allocate():
            device = self.index_group.logical_topk_indices.device
            shape = (self._kv_staging_capacity // 128, 128, 1)
            return (
                torch.empty((*shape, self.kv_lora_rank), dtype=dtype, device=device),
                torch.empty((*shape, self.qk_rope_head_dim), dtype=dtype, device=device),
                torch.arange(self._kv_staging_capacity, dtype=torch.int64, device=device),
            )

        return current_workspace_manager().get_persistent_resource(key, allocate)

    def _c8_length_workspace(self):
        def allocate():
            device = self.index_group.logical_topk_indices.device
            return tuple(torch.empty(1, dtype=torch.int32, device=device) for _ in range(2))

        return current_workspace_manager().get_persistent_resource(("ascend_hisparse_c8_lengths",), allocate)

    def _pcp_prolog_workspace(self):
        def allocate():
            device = self.index_group.logical_topk_indices.device
            return (
                torch.empty(
                    (self._kv_staging_capacity // 128, 128, 1, C8_ROW_BYTES),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
                torch.arange(self._kv_staging_capacity, dtype=torch.int64, device=device),
            )

        # Layers execute sequentially on the same stream. Share scratch across
        # layers while keeping its address fixed for graph capture/replay.
        key = ("ascend_hisparse_pcp_prolog", self._kv_staging_capacity)
        return current_workspace_manager().get_persistent_resource(key, allocate)

    def _prolog_slot_mapping(self, mapping, num_input_rows):
        context = get_forward_context()
        scope = getattr(context, "_ascend_hisparse_resident_i64", None)
        if scope is None:
            scope = context._ascend_hisparse_resident_i64 = {}
        # Exact slot aliases across layers share one cast. The captured cast
        # reads current slots again on replay, even when the address is stable.
        key = (
            mapping.data_ptr(),
            mapping.numel(),
            num_input_rows,
            mapping.device,
            torch.npu.current_stream().npu_stream,
        )
        cached = scope.get(key)
        if cached is None:
            slots = mapping.to(torch.int64)
            # Core returns the effective write prefix. Prolog computes queries
            # for every input row and skips cache writes at -1 slots. Graph
            # replay's changing padding is already set by the slot producer.
            if mapping.numel() < num_input_rows:
                slots = torch.nn.functional.pad(slots, (0, num_input_rows - mapping.numel()), value=-1)
            cached = (mapping.untyped_storage(), slots)
            scope[key] = cached
        return cached[1]

    def _sfa_preprocess_prolog_v3(self, hidden_states, kv_cache, cos, sin, slot_mapping, *, attn_metadata=None):
        if attn_metadata is None:
            raise ValueError("HiSparse Prolog requires current row-validity metadata")
        rows = hidden_states.shape[0]
        if self.pcp_enabled and needs_pcp_kv_gather(attn_metadata, self.is_pcp_decode_sharded):
            # Prolog writes local rows before the collective. Its cache indices
            # cannot use the rank-major global Resident write mapping yet.
            scratch, local_slots = self._pcp_prolog_workspace()
            return super()._sfa_preprocess_prolog_v3(
                hidden_states,
                (scratch,),
                cos,
                sin,
                local_slots[:rows],
                attn_metadata=attn_metadata,
            )
        cache = self.hisparse_cache
        destination, mapping, _ = cache.write_target(rows, slot_mapping.numel())
        result = super()._sfa_preprocess_prolog_v3(
            hidden_states,
            (destination.unsqueeze(2),),
            cos,
            sin,
            self._prolog_slot_mapping(mapping, rows),
            attn_metadata=attn_metadata,
        )
        finish_hisparse_kv_update(cache, attn_metadata)
        return result

    def _sync_pcp_kv_cache(self, kv_cache, attn_metadata):
        scratch, _ = self._pcp_prolog_workspace()
        rows = attn_metadata.num_input_tokens
        packed = scratch.view(torch.int8).view(-1, scratch.shape[-1])[:rows]
        # Preserve C8 latent, BF16 RoPE and FP32 scales byte-for-byte. The
        # collective has the same rank-major ordering as gathered slot mapping.
        gathered = get_pcp_group().all_gather(packed, dim=0)
        cache = self.hisparse_cache
        destination, mapping, count = cache.write_target(gathered.shape[0], gathered.shape[0])
        payload = gathered[:count].view(torch.uint8)
        cache_ops.scatter_packed_kv(
            payload,
            payload[:, :0],
            destination.view(torch.uint8),
            mapping,
            attn_metadata.hisparse_num_valid_write_rows,
        )
        finish_hisparse_kv_update(cache, attn_metadata)

    def forward(self, layer_name, hidden_states, kv_cache, attn_metadata, output=None):
        if self.enable_sparse_sfa_c8:
            self._c8_length_workspace()
            if self.pcp_enabled:
                self._pcp_prolog_workspace()
        else:
            self._workspace(hidden_states.dtype)
        if self.qk_rope_head_dim and not self.enable_sparse_sfa_c8:
            # Include preprocessing scratch in profiling and allocate it before
            # graph capture, including dummy forwards without attention metadata.
            self._kv_workspace(hidden_states.dtype)
        self.index_group.prepare_for_batch(self.index_group_index, attn_metadata)
        return super().forward(layer_name, hidden_states, kv_cache, attn_metadata, output)

    def _exec_kv_local(self, kv_no_split, cos, sin, kv_cache, slots, attn_metadata):
        assert self.kv_a_layernorm is not None
        if self.enable_sparse_sfa_c8:
            from vllm_ascend.hisparse.c8 import pack_c8_components

            rope, latent, scales = super()._exec_kv_local(kv_no_split, cos, sin, kv_cache, slots, attn_metadata)
            rows = latent.numel() // self.kv_lora_rank
            packed = pack_c8_components(latent.contiguous(), rope.contiguous(), scales.contiguous())
            cache = self.hisparse_cache
            destination, mapping, count = cache.write_target(rows, slots.numel())
            packed = packed[:count]
            # Reuse the existing byte-capable scatter; no fused C8 kernel or
            # payload conversion is needed for the baseline data path.
            cache_ops.scatter_packed_kv(
                packed,
                packed[:, :0],
                destination.view(torch.uint8),
                mapping,
                getattr(attn_metadata, "hisparse_num_valid_write_rows", attn_metadata.query_start_loc[-1:]),
            )
            finish_hisparse_kv_update(cache, attn_metadata)
            return rope, latent, scales
        values = kv_no_split.reshape(-1, self.kv_lora_rank + self.qk_rope_head_dim)
        num_tokens = values.shape[0]
        if self.qk_rope_head_dim:
            if num_tokens > self._kv_staging_capacity:
                raise ValueError("HiSparse KV preprocessing exceeds its profiled token capacity")
            latent_cache, rope_cache, local_slots = self._kv_workspace(values.dtype)
            # Match ordinary SFA's fused RMSNorm/RoPE rounding. Separate RMSNorm
            # can differ by a BF16 ULP near a rounding midpoint. Local slots
            # isolate padding from the real resident/hot destination below.
            super()._exec_kv_local(
                values, cos, sin, (latent_cache, rope_cache), local_slots[:num_tokens], attn_metadata
            )
            latent = latent_cache.view(-1, self.kv_lora_rank)[:num_tokens]
            rope = rope_cache.view(-1, self.qk_rope_head_dim)[:num_tokens]
        else:
            # The native NoPE path uses the layer's normalization implementation.
            latent = self.kv_a_layernorm(values)
            rope = values[:, self.kv_lora_rank :]
        cache = self.hisparse_cache
        destination, mapping, num_rows = cache.write_target(values.shape[0], slots.numel())
        num_valid_rows = attn_metadata.query_start_loc[-1:]
        cache_ops.scatter_packed_kv(latent[:num_rows], rope[:num_rows], destination, mapping, num_valid_rows)
        # The common connector mirrors resident rows after this write. Draft
        # rows are deferred until the next step, and FULL graphs are mirrored
        # by finish_forward after replay.
        finish_hisparse_kv_update(cache, attn_metadata)
        return rope, latent

    def _store_packed_kv_cache(self, k_pe, k_nope, knope_scale, kv_cache, slot_mapping_sfa, attn_metadata):
        # exec_kv has already written through the current HiSparse destination
        # and row mapping. The ordinary C8 store uses source slots instead,
        # which would duplicate the write and can overwrite unrelated slab rows.
        return None

    def _c8_physical_attention(self, query, query_rope, cache, indices, *, indices_are_mapped=False, lengths=None):
        """Adapt native packed rows to the existing quantized SFA operation.

        Keep compaction stable, including duplicates. Decode lengths describe
        binding/graph geometry; Prefill retains current device writes. Neither
        path introduces a Host read or a new stream/event policy.
        """
        from vllm_ascend.hisparse.c8 import c8_storage_view, compact_c8_indices, map_c8_physical_indices

        query_ends, key_length = self._c8_length_workspace() if lengths is None else lengths
        scale = self.scale
        view, block, stride_rows, capacity = c8_storage_view(cache)
        rows = query.shape[0]
        if query.ndim != 3 or query.dtype != torch.bfloat16 or query.shape[-1] != self.kv_lora_rank or rows <= 0:
            raise ValueError("HiSparse C8 query must be nonempty BF16 [queries, heads, 512]")
        if query_rope.shape != (*query.shape[:-1], 64) or query_rope.dtype != query.dtype:
            raise ValueError("HiSparse C8 requires matching BF16 query RoPE")
        if indices.ndim != 2 or indices.shape[0] != rows or indices.shape[1] <= 0 or indices.shape[1] % 128:
            raise ValueError("HiSparse C8 top-k width must be positive and divisible by 128")
        if query.device.type != "npu" or any(
            tensor.device != query.device for tensor in (query_rope, cache, indices, query_ends, key_length)
        ):
            raise ValueError("HiSparse C8 inputs must share one NPU")
        if any(tensor.shape != (1,) or tensor.dtype != torch.int32 for tensor in (query_ends, key_length)):
            raise ValueError("HiSparse C8 lengths must be int32 scalar tensors")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("HiSparse C8 attention scale must be finite and positive")
        mapped = (
            indices
            if indices_are_mapped
            else compact_c8_indices(map_c8_physical_indices(indices, block, stride_rows, capacity))
        )
        packed_query = torch.cat((query, query_rope), dim=-1)
        if lengths is None:
            query_ends.fill_(rows)
            key_length.fill_(view.shape[0])
        result = torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention_vllm(
            query=packed_query,
            key=view,
            value=view,
            sparse_indices=mapped.unsqueeze(1),
            scale_value=scale,
            sparse_block_size=1,
            actual_seq_lengths_query=query_ends,
            actual_seq_lengths_kv=key_length,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=0,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            key_quant_mode=2,
            value_quant_mode=2,
            rope_head_dim=64,
            return_softmax_lse=False,
        )
        if isinstance(result, tuple):
            result = result[0]
        # Native C8 clears empty query/head ranges, including graph padding.
        # Avoid rescanning indices and allocating a second masked output.
        return result

    def _attention_physical(
        self,
        query,
        query_rope,
        packed_cache,
        physical_topk,
        *,
        indices_are_mapped=False,
        lengths=None,
    ):
        if self.enable_sparse_sfa_c8:
            return self._c8_physical_attention(
                query,
                query_rope,
                packed_cache,
                physical_topk,
                indices_are_mapped=indices_are_mapped,
                lengths=lengths,
            )
        from vllm_ascend.hisparse.cache_ops import pack_attention_rows

        latent, rope, block_table, indices, query_ends = self._workspace(query.dtype)
        positions = indices.view(1, -1).to(torch.float32)
        outputs = []
        for start in range(0, query.shape[0], _ATTENTION_CHUNK_ROWS):
            end = min(start + _ATTENTION_CHUNK_ROWS, query.shape[0])
            count = end - start
            # SFA's causal mode derives the valid selection length from KV
            # lengths. Move invalid entries to the tail while preserving the
            # selected order and duplicate multiplicity. Reordering valid KV
            # changes BF16 reductions. Unique FP32 position keys are exact for
            # supported top-k sizes and avoid integer ArgSort on AiCPU.
            physical = physical_topk[start:end]
            keys = positions + (physical < 0).to(torch.float32) * physical.shape[-1]
            physical = physical.gather(1, keys.sort(dim=-1).indices)
            valid = physical >= 0
            valid_lengths = valid.sum(dim=-1, dtype=torch.int32)
            selected_latent, selected_rope = latent[:count], rope[:count]
            pack_attention_rows(packed_cache, physical, selected_latent, selected_rope, self.kv_lora_rank)
            selected_indices = torch.where(valid[:, None, :], indices, -1)
            result = torch.ops._C_ascend.npu_sparse_flash_attention(
                query=query[start:end],
                key=selected_latent.view(-1, 128, 1, self.kv_lora_rank),
                value=selected_latent.view(-1, 128, 1, self.kv_lora_rank),
                sparse_indices=selected_indices,
                scale_value=self.scale,
                sparse_block_size=1,
                block_table=block_table[:count],
                actual_seq_lengths_query=query_ends[:count],
                actual_seq_lengths_kv=valid_lengths.clamp_min(1),
                query_rope=query_rope[start:end] if self.qk_rope_head_dim else None,
                key_rope=selected_rope.view(-1, 128, 1, self.qk_rope_head_dim) if self.qk_rope_head_dim else None,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
            )
            if isinstance(result, tuple):
                result = result[0]
            result = result.masked_fill((valid_lengths == 0)[:, None, None], 0)
            outputs.append(result)
        return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]

    def _try_c8_resident_prefill(self, query, query_rope, logical_topk, metadata, query_ends, key_lengths):
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_actual_tokens - num_decode_tokens
        if (
            not self.enable_sparse_sfa_c8
            or metadata.hisparse_for_capture
            or query_rope is None
            or num_prefill_tokens <= 0
            or metadata.num_reqs != num_decodes + metadata.num_prefills
            or not (query.shape[0] == logical_topk.shape[0] == num_prefill_tokens)
        ):
            return None
        cache = self.hisparse_cache
        view = cache.view
        state_indices = cache.runtime.request_state_indices
        if (
            not cache.all_context_pages_resident
            or cache.dummy_batch
            or cache.draft_layer
            or view is None
            or cache.block_table is None
            or state_indices is None
            or state_indices.numel() < metadata.num_reqs
            or query_ends.shape != (metadata.num_reqs,)
            or key_lengths.shape != (metadata.num_reqs,)
        ):
            return None
        if (
            get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.NONE
            or torch.npu.is_current_stream_capturing()
        ):
            return None

        # Read the exact layer's live Resident table after the connector updates
        # it. The dense HMA alias includes gaps, so scale only its physical pages.
        page_stride, remainder = divmod(view.attention_block_stride, view.block_size)
        storage = view.attention_cache
        if (
            remainder
            or page_stride < 1
            or not storage.is_contiguous()
            or view.cache.stride(0) != view.attention_block_stride * view.cache.stride(1)
            or storage.size(0) <= (view.cache.size(0) - 1) * page_stride
        ):
            return None
        table = cache.block_table.index_select(0, state_indices[num_decodes : metadata.num_reqs].to(torch.long))
        table.clamp_min_(0).mul_(page_stride)
        if num_decodes:
            query_ends = query_ends[num_decodes:] - num_decode_tokens
            key_lengths = key_lengths[num_decodes:]
        output = super()._execute_sparse_flash_attention_process(
            query,
            query_rope,
            (storage.unsqueeze(2),),
            logical_topk.unsqueeze(1),
            metadata,
            query_ends,
            key_lengths,
            block_table=table,
        )
        logger.info_once("[hisparse] Resident paged Prefill active: mixed=%s", bool(num_decodes), scope="process")
        return output

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        metadata = attn_metadata
        logical = topk_indices.squeeze(1) if topk_indices.ndim == 3 else topk_indices
        assert metadata.req_id_per_token is not None
        assert metadata.query_start_loc is not None
        if metadata.num_prefills:
            rows = torch.arange(logical.shape[0], dtype=torch.int32, device=logical.device)
            logical = torch.where(rows[:, None] < metadata.query_start_loc[-1], logical, -1)
        # Decode's resolver already masks graph padding from this same device
        # scalar. Keep that validation at the producer of physical indices.
        self.index_group.set_logical_topk_ready(self.index_group_index)
        outputs = []
        num_decode = metadata.num_decode_tokens
        cache = self.hisparse_cache
        if num_decode:
            assert cache.source_block_table is not None
            # Keep mixed prefill/decode batches on the NPU residency resolver.
            # The generic index-group resident shortcut uses the CUDA sparse
            # index converter when a batch also contains prefill tokens.
            physical = cache.swap_in(
                metadata.req_id_per_token[:num_decode],
                block_table=cache.source_block_table,
                logical_topk_indices=logical[:num_decode],
                block_size=metadata.block_size,
                num_valid_rows=metadata.query_start_loc[-1:],
            )
            lengths = None
            if self.enable_sparse_sfa_c8:
                query_lengths, key_length = cache.runtime.cache_ops.attention_lengths
                lengths = query_lengths[num_decode : num_decode + 1], key_length
            outputs.append(
                self._attention_physical(
                    ql_nope[:num_decode],
                    q_pe[:num_decode] if q_pe is not None else None,
                    self.index_group.physical_kv_cache(self.index_group_index),
                    physical,
                    indices_are_mapped=cache.runtime.cache_ops.attention_capacity > 0,
                    lengths=lengths,
                )
            )
        if metadata.num_prefills:
            query = ql_nope[num_decode:]
            query_rope = q_pe[num_decode:] if q_pe is not None else None
            prefill_output = self._try_c8_resident_prefill(
                query, query_rope, logical[num_decode:], metadata, actual_seq_lengths_query, actual_seq_lengths_key
            )
            if prefill_output is None:
                host = cache.runtime.host_cache
                host = host.view(-1, metadata.prefill.host_staging_plan.block_size, host.shape[-1])
                staged, staged_table, req_ids = self.index_group.stage_prefill_rows(
                    self.index_group_index, host, metadata
                )
                selected = logical[num_decode:]
                nonnegative = selected.clamp_min(0)
                block_ids = nonnegative.to(torch.int64) // metadata.block_size
                request_tables = staged_table[req_ids.to(torch.int64)]
                physical = torch.gather(request_tables, 1, block_ids) * metadata.block_size
                physical = physical + nonnegative % metadata.block_size
                physical = physical.masked_fill(selected < 0, -1).to(torch.int32)
                prefill_output = self._attention_physical(query, query_rope, staged, physical)
            outputs.append(prefill_output)
        if not outputs:
            return torch.zeros_like(ql_nope)
        result = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        if result.shape[0] < ql_nope.shape[0]:
            result = torch.cat((result, torch.zeros_like(ql_nope[result.shape[0] :])), dim=0)
        return result
