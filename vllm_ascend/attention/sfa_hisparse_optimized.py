# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend SFA execution over upstream HiSparse's resident and hot cache pool."""

from dataclasses import dataclass

import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup, SparseMLAIndexGroupBuilder
from vllm.v1.hisparse.runtime import HiSparsePrefillStagingPlan, build_hisparse_prefill_staging_plan
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_ascend.attention.hisparse_resident_state import resident_eager_transport_allowed
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.ops.hisparse_resident import begin_resident_forward, resident_topk_or_none
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def _hisparse_uniform_mtp_group_size(
    common_attn_metadata,
    metadata,
    vllm_config,
    uniform_backend: bool,
    draft_index: int | None,
) -> int:
    """Certify only the current CPU query layout for fixed MTP3 decode graphs.

    Adaptive verification may change device boundaries independently of CPU
    ones. Exclude it and non-uniform graph backends rather than transferring
    a device scalar. This fact never includes sequence lengths or KV addresses.
    """
    spec = vllm_config.speculative_config
    if (
        not uniform_backend
        or draft_index is not None
        or vllm_config.model_config.enforce_eager
        or vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
        or spec is None
        or spec.method != "mtp"
        or spec.num_speculative_tokens != 3
        or getattr(spec, "enable_adaptive_verification", False)
        or getattr(spec, "parallel_drafting", False)
        or metadata.num_prefills != 0
        or common_attn_metadata.max_query_len != 4
    ):
        return 0
    count = common_attn_metadata.num_reqs
    rows = common_attn_metadata.num_actual_tokens
    bounds = common_attn_metadata.query_start_loc_cpu
    if (
        rows not in (16, 32, 64)
        or count * 4 != rows
        or metadata.num_input_tokens != rows
        or not isinstance(bounds, torch.Tensor)
        or bounds.device.type != "cpu"
        or bounds.dtype not in (torch.int32, torch.int64)
        or bounds.ndim != 1
        or bounds.numel() < count + 1
    ):
        return 0
    # The device check above is required: this reads an existing CPU mirror.
    values = bounds[: count + 1].tolist()
    return 4 if all(value == index * 4 for index, value in enumerate(values)) else 0


# Bound the temporary D2D packing workspace independently of prefill length.
_ATTENTION_CHUNK_ROWS = 32


@dataclass
class AscendHiSparsePrefillMetadata:
    host_staging_plan: HiSparsePrefillStagingPlan


@dataclass
class AscendHiSparseMetadata(AscendSFAMetadata):
    req_id_per_token: torch.Tensor | None = None
    num_reqs: int = 0
    hisparse_uniform_query_group_size: int = 0
    prefill: AscendHiSparsePrefillMetadata | None = None


class AscendHiSparseMetadataBuilder(AscendSFAMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kwargs):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, metadata_cls=AscendHiSparseMetadata, **kwargs)
        self._request_row_buffers: dict[int | None, torch.Tensor] = {}
        self._row_capacity = max(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )

    def _build(self, common_attn_metadata, draft_index=None, pcp_context=None):
        metadata = super()._build(common_attn_metadata, draft_index, pcp_context)
        metadata.num_reqs = common_attn_metadata.num_reqs
        metadata.hisparse_uniform_query_group_size = _hisparse_uniform_mtp_group_size(
            common_attn_metadata,
            metadata,
            self.vllm_config,
            self.get_cudagraph_support(self.vllm_config, self.kv_cache_spec) == AttentionCGSupport.UNIFORM_BATCH,
            draft_index,
        )
        if draft_index not in self._request_row_buffers:
            self._request_row_buffers[draft_index] = torch.empty(
                self._row_capacity, device=self.device, dtype=torch.int32
            )
        metadata.req_id_per_token = self._request_row_buffers[draft_index][: metadata.num_input_tokens]
        from vllm_ascend.ops.triton.hisparse_request_rows import try_build_request_rows

        if not try_build_request_rows(metadata.query_start_loc, metadata.req_id_per_token, metadata.num_reqs):
            rows = torch.arange(metadata.num_input_tokens, device=self.device, dtype=torch.int32)
            metadata.req_id_per_token.copy_(
                torch.searchsorted(metadata.query_start_loc[1:].contiguous(), rows, right=True)
            )
            metadata.req_id_per_token.clamp_(max=max(metadata.num_reqs - 1, 0))
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
    """Keep HiSparse residency on device and read packed KV directly in SFA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.enable_sparse_sfa_turboquant:
            raise ValueError("Ascend HiSparse does not support TurboQuant")
        if self.enable_sparse_sfa_c8:
            from vllm_ascend.ops.hisparse_c8 import C8_PAYLOAD_FORMAT

            if get_ascend_device_type() != AscendDeviceType.A5 or self.c8_cache_dtype != torch.float8_e4m3fn:
                raise ValueError("HiSparse C8 requires the native A5 FP8 E4M3 cache format")
            self.hisparse_c8_payload_format = C8_PAYLOAD_FORMAT
        if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
            raise ValueError("HiSparse direct SFA currently requires MLA latent 512 and RoPE 64")
        parallel = self.vllm_config.parallel_config
        if parallel.prefill_context_parallel_size > 1 or parallel.decode_context_parallel_size > 1:
            raise ValueError("Ascend HiSparse does not support context parallelism")
        builder = kwargs.get("index_group_builder")
        if builder is None:
            if self.topk_indices_buffer is None:
                raise ValueError("Ascend HiSparse requires a model sparse-index group")
            builder = SparseMLAIndexGroupBuilder(self.topk_indices_buffer)
        from vllm_ascend.patch.worker.patch_hisparse_streams import install

        install()
        self.index_group, self.index_group_index = builder.register_layer(
            self.has_indexer and not self.skip_topk,
            self.vllm_config,
            head_size=656 if self.enable_sparse_sfa_c8 else self.kv_lora_rank + self.qk_rope_head_dim,
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
        self._resident_hisparse_enabled = resident_eager_transport_allowed(
            self.vllm_config.model_config.enforce_eager,
            get_ascend_device_type() == AscendDeviceType.A5,
            self.vllm_config.kv_transfer_config,
        )

    def _fused_preprocess_type(self):
        # Only the native A5 packed C8 contract has a Prolog bridge. Keep the
        # parent's eligibility/weight preparation and the original BF16 path.
        if self.enable_sparse_sfa_c8:
            return super()._fused_preprocess_type()
        return None

    def _prolog_c8_workspace(self):
        # Layers on the same workspace lane use the compute stream in order.
        # Scatter consumes this scratch before another layer can overwrite it;
        # Host mirroring reads the final resident cache, never this scratch.
        # The workspace manager isolates concurrent ubatches/lanes. No token
        # values, resident slots, or layer-local cache addresses are cached.
        key = ("ascend_hisparse_prolog_c8", self._kv_staging_capacity)

        def allocate():
            device = self.index_group.logical_topk_indices.device
            return (
                torch.empty(
                    (self._kv_staging_capacity // 128, 128, 1, 656),
                    device=device,
                    dtype=torch.float8_e4m3fn,
                ),
                torch.arange(self._kv_staging_capacity, device=device, dtype=torch.int64),
            )

        return current_workspace_manager().get_persistent_resource(key, allocate)

    def _sfa_preprocess_prolog_v3(
        self,
        hidden_states,
        kv_cache,
        cos,
        sin,
        slot_mapping,
        *,
        attn_metadata=None,
    ):
        from vllm_ascend.ops.hisparse_optimized import scatter_packed_kv

        if not self.enable_sparse_sfa_c8 or attn_metadata is None:
            raise RuntimeError("HiSparse Prolog requires current C8 attention metadata")
        rows = hidden_states.shape[0]
        if rows > self._kv_staging_capacity:
            raise ValueError("HiSparse Prolog exceeds its profiled token capacity")
        staging, local_slots = self._prolog_c8_workspace()
        result = super()._sfa_preprocess_prolog_v3(
            hidden_states,
            (staging,),
            cos,
            sin,
            local_slots[:rows],
            attn_metadata=attn_metadata,
        )
        # Prolog emits the existing opaque 656-byte C8 row. Preserve FP8 NoPE,
        # BF16 RoPE, and FP32 scales byte for byte when writing the current
        # layer's strided Resident destination. Device valid_rows masks graph
        # padding at replay; local Prolog indices are always safe scratch rows.
        payload = staging.view(torch.uint8).view(-1, 656)
        cache = self.hisparse_cache
        destination, mapping, count = cache.write_target(rows, slot_mapping.numel())
        scatter_packed_kv(
            payload[:count, :512],
            payload[:count, 512:],
            destination.view(torch.uint8),
            mapping,
            attn_metadata.query_start_loc[-1:],
        )
        cache.finish_kv_update()
        return result

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
                torch.full((_ATTENTION_CHUNK_ROWS,), self.topk, dtype=torch.int32, device=device),
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

    def forward(self, layer_name, hidden_states, kv_cache, attn_metadata, output=None):
        if not self.enable_sparse_sfa_c8:
            self._workspace(hidden_states.dtype)
        if self.enable_sparse_sfa_c8:
            self._c8_length_workspace()
            if self.preprocess_type.name == "PROLOG_V3":
                self._prolog_c8_workspace()
        if self.qk_rope_head_dim and not self.enable_sparse_sfa_c8:
            # Include preprocessing scratch in profiling and allocate it before
            # graph capture, including dummy forwards without attention metadata.
            self._kv_workspace(hidden_states.dtype)
        self.index_group.prepare_for_batch(self.index_group_index, attn_metadata)
        runtime = self.hisparse_cache.runtime
        if runtime.is_group_leader:
            from vllm_ascend.ops.hisparse_c8_scope import begin_c8_mapping_scope

            begin_c8_mapping_scope(
                runtime.index_group,
                get_forward_context(),
                enabled=self.enable_sparse_sfa_c8 and attn_metadata is not None and attn_metadata.num_prefills == 0,
                cache_handles=self.index_group.caches,
            )
        if runtime.is_group_leader and runtime.cache_ops is not None:
            # Each captured graph owns this choice. Replay selection retains
            # upstream uniform-query ownership; current row/state/KV values
            # still arrive through the original group-local device buffers.
            runtime.cache_ops.uniform_query_group_size = (
                getattr(attn_metadata, "hisparse_uniform_query_group_size", 0)
                if self.enable_sparse_sfa_c8
                and (
                    get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
                    or (
                        get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE
                        and torch.npu.is_current_stream_capturing()
                    )
                )
                else 0
            )
        begin_resident_forward(
            self.index_group,
            self.index_group_index,
            attn_metadata,
            enforce_eager=self._resident_hisparse_enabled,
        )
        return super().forward(layer_name, hidden_states, kv_cache, attn_metadata, output)

    def exec_kv(self, kv_no_split, cos, sin, kv_cache, slots, attn_metadata):
        from vllm_ascend.ops.hisparse_optimized import scatter_packed_kv

        assert self.kv_a_layernorm is not None
        if self.enable_sparse_sfa_c8:
            from vllm_ascend.attention.sfa_v1 import custom_kv_rmsnorm_rope
            from vllm_ascend.ops.triton.hisparse_c8_fused import scatter_c8_components

            # Match the normal SFA owner: the native RoPE path consumes BNSD.
            values = kv_no_split.reshape(-1, self.num_kv_heads, 1, self.kv_lora_rank + self.qk_rope_head_dim)
            rope, latent, scales = custom_kv_rmsnorm_rope(
                values,
                self.kv_a_layernorm.weight,
                cos,
                sin,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                dst_type=self.c8_cache_dtype,
                tile_size=self.sfa_qsfa_tile_size,
            )
            # Native A5 exposes RoPE/scale as opaque FP8-typed byte views.
            rows = latent.numel() // self.kv_lora_rank
            destination, mapping, count = self.hisparse_cache.write_target(rows, slots.numel())
            scatter_c8_components(
                latent.reshape(rows, -1)[:count].contiguous(),
                rope.view(torch.uint8).reshape(rows, 128)[:count].contiguous(),
                scales.view(torch.uint8).reshape(rows, 16)[:count].contiguous(),
                destination,
                mapping,
                attn_metadata.query_start_loc[-1:],
            )
            self.hisparse_cache.finish_kv_update()
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
            torch_npu.npu_kv_rmsnorm_rope_cache(
                values.view(num_tokens, self.num_kv_heads, 1, self.kv_lora_rank + self.qk_rope_head_dim),
                self.kv_a_layernorm.weight,
                cos,
                sin,
                local_slots[:num_tokens],
                rope_cache,
                latent_cache,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA",
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
        scatter_packed_kv(latent[:num_rows], rope[:num_rows], destination, mapping, num_valid_rows)
        # The common connector mirrors resident rows after this write. Draft
        # rows are deferred until the next step, and FULL graphs are mirrored
        # by finish_forward after replay.
        cache.finish_kv_update()
        return rope, latent

    def _store_parallel_kv(
        self,
        k_pe,
        k_nope,
        knope_scale,
        fused_kv_no_split,
        kv_ag_handles,
        kv_cache,
        slot_mapping_sfa,
        attn_metadata,
        full_gather_o_proj_enabled,
    ):
        if self.enable_sparse_sfa_c8:
            if kv_ag_handles or full_gather_o_proj_enabled:
                raise ValueError("HiSparse C8 requires local KV preprocessing without a parallel KV gather")
            # exec_kv already wrote the runtime-selected resident/hot destination.
            return k_pe, k_nope
        return super()._store_parallel_kv(
            k_pe,
            k_nope,
            knope_scale,
            fused_kv_no_split,
            kv_ag_handles,
            kv_cache,
            slot_mapping_sfa,
            attn_metadata,
            full_gather_o_proj_enabled,
        )

    def _c8_length_workspace(self):
        key = ("ascend_hisparse_c8_lengths", id(self))

        def allocate():
            device = self.index_group.logical_topk_indices.device
            values = torch.empty(2, dtype=torch.int32, device=device)
            return values[:1], values[1:]

        return current_workspace_manager().get_persistent_resource(key, allocate)

    def _c8_attention_metadata(self, cache, rows):
        if not 0 <= rows <= self._kv_staging_capacity:
            raise ValueError("HiSparse C8 attention exceeds its profiled query capacity")
        query_end, key_length = self._c8_length_workspace()
        # The upstream manager separates ubatches/lanes; each instance owns its
        # two scalars. Same-lane consumers follow the compute stream order.
        # Query packing writes the current lengths in its own captured kernel,
        # before the same-stream SFA consumer. No additional fills are needed.
        return query_end, key_length

    def _attention_physical(
        self,
        query,
        query_rope,
        packed_cache,
        physical_topk,
        *,
        use_eager_compaction=False,
        _c8_prepared=False,
        _mapping_scope=None,
    ):
        if self.enable_sparse_sfa_c8:
            from vllm_ascend.ops.hisparse_c8 import c8_physical_attention

            query_end, key_length = self._c8_attention_metadata(packed_cache, query.shape[0])
            return c8_physical_attention(
                query,
                query_rope,
                packed_cache,
                physical_topk,
                self.scale,
                query_end,
                key_length,
                _prepared_indices=_c8_prepared,
                _mapping_scope=_mapping_scope,
            )
        return self._attention_physical_bridge(query, query_rope, packed_cache, physical_topk)

    def _attention_physical_bridge(self, query, query_rope, packed_cache, physical_topk):
        """Retain the materializing implementation for explicit oracle comparisons."""
        from vllm_ascend.ops.hisparse_optimized import pack_attention_rows

        latent, rope, block_table, indices, query_ends, seq_lens = self._workspace(query.dtype)
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
            valid_lengths = (physical >= 0).sum(dim=-1, dtype=torch.int32)
            selected_latent, selected_rope = latent[:count], rope[:count]
            pack_attention_rows(packed_cache, physical, selected_latent, selected_rope, self.kv_lora_rank)
            selected_indices = torch.where(physical[:, None, :] >= 0, indices, -1)
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
            result = result.masked_fill(~(physical >= 0).any(dim=-1)[:, None, None], 0)
            outputs.append(result)
        return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]

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
        # The pure C8 Decode resolver masks request IDs after waiting for
        # logical_topk_ready, on its existing copy stream. A second full TopK
        # mask per layer has no consumer in this path. Retain it for Prefill,
        # mixed batches, BF16 and the eager resident shortcut.
        if not (self.enable_sparse_sfa_c8 and metadata.num_prefills == 0 and not self._resident_hisparse_enabled):
            rows = torch.arange(logical.shape[0], dtype=torch.int32, device=logical.device)
            logical = torch.where(rows[:, None] < metadata.query_start_loc[-1], logical, -1)
        self.index_group.set_logical_topk_ready(self.index_group_index)
        outputs = []
        num_decode = metadata.num_decode_tokens
        if num_decode:
            cache = self.hisparse_cache
            assert cache.source_block_table is not None
            # Keep mixed prefill/decode batches on the NPU residency resolver.
            # The generic index-group resident shortcut uses the CUDA sparse
            # index converter when a batch also contains prefill tokens.
            physical = resident_topk_or_none(
                self.index_group,
                self.index_group_index,
                metadata,
                logical[:num_decode],
                enforce_eager=self._resident_hisparse_enabled and metadata.num_prefills == 0 and num_decode <= 64,
            )
            mapping_scope = None
            if physical is None:
                physical = cache.swap_in(
                    metadata.req_id_per_token[:num_decode],
                    block_table=cache.source_block_table,
                    logical_topk_indices=logical[:num_decode],
                    block_size=metadata.block_size,
                    num_valid_rows=metadata.query_start_loc[-1:],
                )
                if metadata.num_prefills == 0:
                    from vllm_ascend.ops.hisparse_c8_scope import current_c8_mapping_scope

                    mapping_scope = current_c8_mapping_scope(cache.runtime.index_group, get_forward_context())
            outputs.append(
                self._attention_physical(
                    ql_nope[:num_decode],
                    q_pe[:num_decode] if q_pe is not None else None,
                    self.index_group.physical_kv_cache(self.index_group_index),
                    physical,
                    use_eager_compaction=metadata.num_prefills == 0,
                    _mapping_scope=mapping_scope,
                )
            )
        if metadata.num_prefills:
            host_cache = self.hisparse_cache.runtime.host_cache
            staged, staged_table, req_ids = self.index_group.stage_prefill_rows(
                self.index_group_index,
                host_cache.view(-1, metadata.block_size, host_cache.shape[-1]),
                metadata,
            )
            selected = logical[num_decode:]
            physical = None
            if self.enable_sparse_sfa_c8:
                from vllm_ascend.ops.hisparse_c8_staged import prepare_staged_c8_indices

                physical = prepare_staged_c8_indices(selected, staged_table, req_ids, staged, metadata.block_size)
            prepared = physical is not None
            if not prepared:
                block_ids = selected.clamp_min(0).to(torch.int64) // metadata.block_size
                request_tables = staged_table[req_ids.to(torch.int64)]
                physical = torch.gather(request_tables, 1, block_ids) * metadata.block_size
                physical = physical + selected.clamp_min(0) % metadata.block_size
                physical = physical.masked_fill(selected < 0, -1).to(torch.int32)
            outputs.append(
                self._attention_physical(
                    ql_nope[num_decode:],
                    q_pe[num_decode:] if q_pe is not None else None,
                    staged,
                    physical,
                    _c8_prepared=prepared,
                )
            )
        if not outputs:
            return torch.zeros_like(ql_nope)
        result = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        if result.shape[0] < ql_nope.shape[0]:
            result = torch.cat((result, torch.zeros_like(ql_nope[result.shape[0] :])), dim=0)
        return result
