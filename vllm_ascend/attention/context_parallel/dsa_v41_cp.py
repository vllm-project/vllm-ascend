# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 replicated-cache adapters for MRV2 PCP."""

import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any, ClassVar

import torch
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.attention import dsa_v41 as v41
from vllm_ascend.attention.context_parallel.common_cp import is_pcp_decode_sharding_enabled
from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSAPCPMetadataBuilder,
)
from vllm_ascend.attention.dsa_v41 import (
    AscendDSAV41AttentionMetadataBuilder,
    AscendDSAV41FoldedIndexMetadataBuilder,
    AscendDSAV41Impl,
    AscendDSAV41IndexerMetadataBuilder,
    AscendDSAV41Metadata,
    AscendDSAV41MetadataBuilder,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, is_pcp_enabled, uses_pcp_sharded_layout
from vllm_ascend.core.kv_cache_interface import get_kv_cache_compression_ratio, get_storage_block_size
from vllm_ascend.ops.rope_dsv4 import get_full_cos_and_sin_dsa_for_layer


def get_v41_cp_classes():
    # Read the current model config: DSpark's target uses PCP, while its
    # replicated draft is constructed under a separate PCP=1 config.
    use_pcp = is_pcp_enabled()
    if use_pcp:
        return AscendDSAV41PCPMetadataBuilder, AscendDSAV41PCPImpl
    return AscendDSAV41MetadataBuilder, AscendDSAV41Impl


class _ReplicatedCacheMetadataBuilder(AscendDSAV41MetadataBuilder):
    """Keep global cache metadata independent from local query buffers."""

    _global_metadata_builder_cls = AscendDSAV41MetadataBuilder

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, *, build_local_compressor_metadata=False):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            build_compressor_metadata=build_local_compressor_metadata,
        )
        self._global_builder = self._global_metadata_builder_cls(
            kv_cache_spec, layer_names, vllm_config, device, build_query_metadata=False
        )

    def prepare_source_rope(self):
        # Both metadata owners need source tables before MRV2 warmup/capture.
        super().prepare_source_rope()
        self._global_builder.prepare_source_rope()

    def enable_device_metadata(self):
        super().enable_device_metadata()
        self._global_builder.enable_device_metadata()

    def take_device_metadata_tasks(self):
        return (
            *self._global_builder.take_device_metadata_tasks(),
            *super().take_device_metadata_tasks(),
        )

    @contextmanager
    def defer_device_metadata(self, *, in_graph: bool = False):
        # Enter the global guard first: the outer enable method enables both.
        with (
            self._global_builder.defer_device_metadata(in_graph=in_graph),
            super().defer_device_metadata(in_graph=in_graph),
        ):
            yield

    def _build_global_metadata(self, common_prefix_len, common, fast_build, kwargs):
        global_kwargs = dict(kwargs)
        shared = kwargs.get("common_v41_metadata")
        if shared is not None:
            global_kwargs["common_v41_metadata"] = shared.setdefault("cp_global", {})
        batch_shared = kwargs.get("common_v41_batch_metadata")
        if batch_shared is not None:
            global_kwargs["common_v41_batch_metadata"] = batch_shared.setdefault("cp_global", {})
        return self._global_builder.build(common_prefix_len, common, fast_build, **global_kwargs)


class _ReplicatedCacheImpl(AscendDSAV41Impl):
    def _global_layer_metadata(self, metadata_by_prefix):
        global_by_prefix = {}
        # The runner also includes DSpark's native DSA metadata in this map.
        # Resolve only the cache planes consumed by this target layer.
        for prefix in (
            self.swa_prefix,
            self.long_kv_source_prefix,
            self.index_k_source_prefix,
            self.compressor_state_prefix,
        ):
            if prefix is None:
                continue
            metadata = metadata_by_prefix[prefix]
            global_by_prefix[prefix] = metadata.global_metadata
        return self._get_layer_metadata(global_by_prefix)


@dataclass(kw_only=True)
class AscendDSAV41PCPMetadata(AscendDSAV41Metadata):
    """Rank-local query metadata with a canonical replicated-cache view."""

    local_num_tokens_after_padding: int
    # Maps scheduler token order to the rank-major gathered layout.
    hidden_restore_idx: torch.Tensor

    @classmethod
    def from_local_metadata(
        cls,
        local: AscendDSAV41Metadata,
        global_metadata: AscendDSAV41Metadata,
        local_num_tokens_after_padding: int,
        hidden_restore_idx: torch.Tensor,
    ) -> "AscendDSAV41PCPMetadata":
        values = {field.name: getattr(local, field.name) for field in fields(AscendDSAV41Metadata)}
        values["global_metadata"] = global_metadata
        return cls(
            **values,
            local_num_tokens_after_padding=local_num_tokens_after_padding,
            hidden_restore_idx=hidden_restore_idx,
        )


class AscendDSAV41PCPMetadataBuilder(_ReplicatedCacheMetadataBuilder):
    """Build PCP-local queries and scheduler-global cache-write controls."""

    _request_capacity_factor: ClassVar[int] = 2

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if vllm_config.parallel_config.decode_context_parallel_size != 1:
            raise NotImplementedError("V4.1 PCP currently requires DCP=1 for replicated caches.")
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, build_local_compressor_metadata=True)
        self._pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.use_pcp = self._pcp_world_size > 1
        self._pcp_rank = get_pcp_group().rank_in_group
        self._is_decode_sharded = is_pcp_decode_sharding_enabled(vllm_config)
        self._max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._hidden_restore_idx_buffer = torch.empty(self._max_tokens, dtype=torch.int64, device=device)

        # DualChunkSwap doubles prefill rows; decode graphs also pad requests.
        max_reqs = vllm_config.scheduler_config.max_num_seqs
        capture_sizes = vllm_config.compilation_config.cudagraph_capture_sizes or ()
        graph_reqs = max(capture_sizes, default=0)
        self._resize_request_buffers(self, max(self._request_capacity_factor * max_reqs, graph_reqs) + 1)
        self._resize_request_buffers(self._global_builder, max(max_reqs, graph_reqs) + 1)

        self._pcp_context_owner = self
        self._pcp_rope_buffers: dict[str, tuple[torch.Tensor, ...]] = {}

    @staticmethod
    def prepare_shared_buffers(builders) -> None:
        """Choose stable batch buffer owners before warmup and graph capture.

        A runtime cache alone is insufficient: a different first builder must
        not change the addresses referenced by an already captured graph.
        """
        by_config: dict[int, list[AscendDSAV41PCPMetadataBuilder]] = {}
        for builder in builders:
            by_config.setdefault(id(builder.vllm_config), []).append(builder)
        for config_builders in by_config.values():
            config_builders.sort(key=lambda builder: tuple(builder.layer_names))
            owner = config_builders[0]
            rope_buffers: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
            for builder in config_builders:
                builder._pcp_context_owner = owner
                for layer_name, buffers in builder._pcp_rope_buffers.items():
                    key = (buffers[0].data_ptr(), buffers[1].data_ptr())
                    # The model config fixes capacity; the source tables fix
                    # the remaining dimensions of these preallocated buffers.
                    builder._pcp_rope_buffers[layer_name] = rope_buffers.setdefault(key, buffers)

    @staticmethod
    def _resize_request_buffers(builder, capacity: int) -> None:
        # Resize before graph capture.
        for name in ("_seq_lens", "_cache_seq_lens", "_cmp_residual"):
            buffer = getattr(builder, name)
            setattr(builder, name, buffer.new_zeros(capacity))
        if builder._build_compressor_metadata:
            # C2 metadata rows: 0=start position, 1=valid token count,
            # 2=output row base (reserved), 3=input row base, 4=state block ID.
            builder._c2_ring_metadata = builder._c2_ring_metadata.new_zeros(5 * capacity)

    def prepare_source_rope(self) -> None:
        super().prepare_source_rope()
        if self._cache_kind != "swa" or self._pcp_rope_buffers:
            return
        by_table: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        for cache_name in self.layer_names:
            layer_name = cache_name.removesuffix(".swa_cache") + ".attn"
            full_cos, full_sin = get_full_cos_and_sin_dsa_for_layer(layer_name)
            key = (full_cos.data_ptr(), full_sin.data_ptr())
            buffers = by_table.get(key)
            if buffers is None:
                # Keep Q/KV RoPE buffers graph-stable and independent of the draft.
                shape = (self._max_tokens, *full_cos.shape[1:])
                # Full tables, local Q cos/sin, global KV cos/sin.
                buffers = (
                    full_cos,
                    full_sin,
                    full_cos.new_empty(shape),
                    full_sin.new_empty(shape),
                    full_cos.new_empty(shape),
                    full_sin.new_empty(shape),
                )
                by_table[key] = buffers
            self._pcp_rope_buffers[layer_name] = buffers

    def _build_pcp_rope_views(self, common: AscendCommonAttentionMetadata, *, global_cache: bool, shared_rope=None):
        if self._cache_kind != "swa":
            return None
        assert self._pcp_rope_buffers, "Prepare V4.1 PCP RoPE before building attention metadata"
        num_tokens = common.num_input_tokens
        gathered = {} if shared_rope is None else shared_rope
        positions = gathered.get("positions")
        if positions is None:
            positions = common.positions[:num_tokens].long()
            gathered["positions"] = positions
        cos_views, sin_views = {}, {}
        for layer_name, buffers in self._pcp_rope_buffers.items():
            full_cos, full_sin = buffers[:2]
            key = (full_cos.data_ptr(), full_sin.data_ptr())
            views = gathered.get(key)
            if views is None:
                offset = 4 if global_cache else 2
                cos, sin = buffers[offset : offset + 2]
                cos, sin = cos[:num_tokens], sin[:num_tokens]
                if num_tokens:
                    indices = positions.reshape(-1, 1, 1, 1).expand(num_tokens, 1, 1, full_cos.shape[-1])
                    torch.gather(full_cos, 0, indices, out=cos)
                    torch.gather(full_sin, 0, indices, out=sin)
                views = (cos, sin)
                gathered[key] = views
            cos_views[layer_name], sin_views[layer_name] = views
        return cos_views, sin_views

    def _build_empty_local_metadata(self, common: AscendCommonAttentionMetadata) -> AscendDSAV41Metadata:
        # Skip SMLA/QLI metadata ops on ranks with no queries.
        self._device_metadata_tasks = ()
        self._smla_metadata.zero_()
        self._qli_metadata.zero_()
        if self._a5_smla_metadata is not None:
            self._a5_smla_metadata.zero_()
        num_tokens, num_reqs = common.num_input_tokens, common.num_reqs
        is_state = self._cache_kind == "compressor_state"
        slots = self._slot_mapping if is_state else self._slot_mapping_2d
        return AscendDSAV41Metadata(
            block_table=common.block_table_tensor[:num_reqs],
            query_start_loc=common.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=common.query_start_loc_cpu,
            seq_lens=self._seq_lens[:num_reqs].zero_(),
            seq_lens_cpu=common.seq_lens_cpu,
            cache_seq_lens=self._cache_seq_lens[:num_reqs].zero_(),
            slot_mapping=slots[:num_tokens].fill_(-1),
            flat_slot_mapping=self._flat_slot_mapping[:num_tokens].fill_(-1),
            compress_ratio=get_kv_cache_compression_ratio(self.kv_cache_spec),
            storage_block_size=get_storage_block_size(self.kv_cache_spec),
            logical_block_size=self.kv_cache_spec.block_size,
            is_compressor_state=is_state,
            cache_kind=self._cache_kind,
            positions=common.positions[:num_tokens],
            num_input_tokens=num_tokens,
            num_reqs=num_reqs,
            attn_state=common.attn_state,
            is_prefilling=common.is_prefilling,
            causal=common.causal,
        )

    def _prepare_graph_pcp_context(
        self,
        pcp_context,
        full_graph_mode: bool,
    ):
        """Prepare graph-stable V4.1 tensors without changing the PCP batch layout."""
        global_batch = pcp_context.global_batch
        num_actual_tokens = global_batch.num_tokens

        # FULL graphs replay the fixed padded batch, so the restore indices
        # must cover every padded row. Other modes run the DSA cache update
        # with the actual token extent, and compact indices avoid restoring
        # placeholder rows that would be trimmed before the cache write anyway.
        num_restore_tokens = global_batch.num_tokens_after_padding if full_graph_mode else num_actual_tokens
        # Use the preallocated buffer to keep the address fixed for graph replay.
        hidden_restore_idx = self._hidden_restore_idx_buffer[:num_restore_tokens]
        hidden_restore_idx[:num_actual_tokens].copy_(pcp_context.hidden_restore_idx[:num_actual_tokens])
        hidden_restore_idx[num_actual_tokens:].zero_()

        # The upstream dummy path only invalidates gathered slot mappings.
        # DSA also consumes the scheduler-global mapping, so clear it here.
        if global_batch.is_dummy:
            pcp_context.global_slot_mappings.fill_(-1)

        return dataclasses.replace(
            pcp_context,
            hidden_restore_idx=hidden_restore_idx,
        )

    def _build_local_common_attn_metadata(
        self,
        pcp_context,
        common_attn_metadata: AscendCommonAttentionMetadata,
        cache_group_idx: int,
        has_prefill: bool,
    ) -> AscendCommonAttentionMetadata:
        num_local_padded_tokens = common_attn_metadata.num_input_tokens
        # Sharded decode owns different requests per rank, so it uses the same
        # rank-local slot slice as prefill. Replicated decode writes every rank.
        if has_prefill or self._is_decode_sharded:
            gathered_slot_mapping = common_attn_metadata.slot_mapping
            if pcp_context.global_batch.is_dummy:
                gathered_slot_mapping.fill_(-1)
            local_slot_mapping = gathered_slot_mapping.view(
                self._pcp_world_size,
                num_local_padded_tokens,
            )[self._pcp_rank]
        else:
            # Decode tokens are replicated, so every rank must write its own
            # SWA cache. The gathered mapping masks these writes off rank 0.
            local_slot_mapping = pcp_context.global_slot_mappings[cache_group_idx]
            if pcp_context.global_batch.is_dummy:
                local_slot_mapping.fill_(-1)
            assert local_slot_mapping.shape[0] == num_local_padded_tokens
        return common_attn_metadata.replace(
            slot_mapping=local_slot_mapping,
        )

    def build(
        self,
        common_prefix_len,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build=False,
        num_actual_reqs: int | None = None,
        **kwargs: Any,
    ) -> AscendDSAV41Metadata:
        pcp_context = common_attn_metadata.pcp_context
        pcp_cache_group_idx = common_attn_metadata.pcp_cache_group_idx
        if not uses_pcp_sharded_layout(common_attn_metadata, self.use_pcp):
            # Without a PCP context the batch runs replicated on every rank,
            # like a PCP=1 model (e.g. a replicated draft decode), so build the
            # ordinary V4.1 metadata that the impl executes without PCP.
            if num_actual_reqs is not None:
                kwargs["num_actual_reqs"] = num_actual_reqs
            return super().build(common_prefix_len, common_attn_metadata, fast_build, **kwargs)
        assert pcp_context is not None
        assert pcp_cache_group_idx is not None
        batch_shared = kwargs.get("common_v41_batch_metadata")
        if batch_shared is None:
            batch_shared = {}
        # Query overrides do not change PCP request layout. Query-dependent
        # outputs are scoped separately by the base metadata builder.
        shared = batch_shared.setdefault(("v41_pcp_specialized", id(self.vllm_config), None), {})
        full_graph_mode = bool(kwargs.get("full_graph_mode", False))
        signature = (
            common_attn_metadata.num_reqs,
            common_attn_metadata.num_input_tokens,
            common_attn_metadata.num_actual_tokens,
            num_actual_reqs,
            full_graph_mode,
            bool(kwargs.get("skip_ring_state_update", False)),
        )
        cached_signature = shared.setdefault("signature", signature)
        if cached_signature != signature:
            raise ValueError("V4.1 PCP metadata sharing requires matching batch signatures")
        mode = shared.get("mode")
        if mode is None:
            has_prefill = bool(pcp_context.global_batch.is_prefilling_np.any())
            mode = (has_prefill, has_prefill or self._is_decode_sharded)
            shared["mode"] = mode
        has_prefill, needs_global_metadata = mode
        self._global_builder.take_device_metadata_tasks()

        if needs_global_metadata:
            prepared_context = shared.get("pcp_context")
            if prepared_context is None:
                prepared_context = self._pcp_context_owner._prepare_graph_pcp_context(
                    pcp_context, full_graph_mode=full_graph_mode
                )
                shared["pcp_context"] = prepared_context
            pcp_context = prepared_context
            assert pcp_context is not None
            global_common = shared.get("global_common")
            if global_common is None:
                global_common = AscendDSAPCPMetadataBuilder._build_global_common_attn_metadata(
                    pcp_context, pcp_cache_group_idx, common_attn_metadata
                )
                if full_graph_mode:
                    global_common = AscendDSAPCPMetadataBuilder._build_graph_common_attn_metadata(
                        global_common, pcp_context.global_batch.num_reqs
                    )
                shared["global_common"] = global_common
            else:
                global_common = global_common.replace(
                    block_table_tensor=pcp_context.global_block_tables[pcp_cache_group_idx],
                    slot_mapping=pcp_context.global_slot_mappings[pcp_cache_group_idx],
                    max_seq_len=common_attn_metadata.max_seq_len,
                    causal=common_attn_metadata.causal,
                )

        local_common = self._build_local_common_attn_metadata(
            pcp_context, common_attn_metadata, pcp_cache_group_idx, has_prefill
        )
        if full_graph_mode:
            offsets = shared.get("local_graph_offsets")
            if offsets is None:
                local_common = AscendDSAPCPMetadataBuilder._build_graph_common_attn_metadata(
                    local_common, num_actual_reqs
                )
                offsets = (local_common.query_start_loc, local_common.query_start_loc_cpu)
                shared["local_graph_offsets"] = offsets
            else:
                local_common = local_common.replace(query_start_loc=offsets[0], query_start_loc_cpu=offsets[1])

        kwargs["skip_ring_state_update"] = (
            bool(kwargs.get("skip_ring_state_update", False)) or pcp_context.global_batch.is_dummy
        )
        kwargs["common_v41_batch_metadata"] = shared
        if needs_global_metadata:
            global_kwargs = dict(kwargs)
            global_kwargs.update(
                num_actual_reqs=pcp_context.global_batch.num_reqs,
                build_query=False,
                update_cache=True,
                build_compressor_metadata=True,
            )
            global_rope = self._build_pcp_rope_views(
                global_common, global_cache=True, shared_rope=shared.setdefault("global_rope", {})
            )
            if global_rope is not None:
                global_kwargs["rope_views"] = global_rope
            global_metadata = self._build_global_metadata(common_prefix_len, global_common, fast_build, global_kwargs)

        if needs_global_metadata and local_common.num_actual_tokens == 0:
            local_metadata = self._build_empty_local_metadata(local_common)
        else:
            local_kwargs = dict(kwargs)
            local_kwargs.update(
                num_actual_reqs=local_common.num_reqs if num_actual_reqs is None else num_actual_reqs,
                build_query=True,
                update_cache=not needs_global_metadata,
                build_compressor_metadata=not needs_global_metadata,
            )
            for name in ("common_v41_metadata", "common_v41_batch_metadata"):
                cache = kwargs.get(name)
                if cache is not None:
                    local_kwargs[name] = cache.setdefault("pcp_local", {})
            local_rope = self._build_pcp_rope_views(
                local_common, global_cache=False, shared_rope=shared.setdefault("local_rope", {})
            )
            if local_rope is not None:
                local_kwargs["rope_views"] = local_rope
            local_metadata = super().build(common_prefix_len, local_common, fast_build, **local_kwargs)

        if not needs_global_metadata:
            return local_metadata
        return AscendDSAV41PCPMetadata.from_local_metadata(
            local_metadata, global_metadata, local_common.num_input_tokens, pcp_context.hidden_restore_idx
        )


class AscendDSAV41PCPStateMetadataBuilder(
    AscendDSAV41PCPMetadataBuilder,
    v41.AscendDSAV41StateMetadataBuilder,
):
    _global_metadata_builder_cls = v41.AscendDSAV41StateMetadataBuilder


class AscendDSAV41PCPSWAMetadataBuilder(
    AscendDSAV41PCPMetadataBuilder,
    v41.AscendDSAV41SWAMetadataBuilder,
):
    _global_metadata_builder_cls = v41.AscendDSAV41SWAMetadataBuilder


class AscendDSAV41PCPAttentionMetadataBuilder(
    AscendDSAV41PCPMetadataBuilder,
    AscendDSAV41AttentionMetadataBuilder,
):
    _global_metadata_builder_cls = AscendDSAV41AttentionMetadataBuilder


class AscendDSAV41PCPIndexerMetadataBuilder(
    AscendDSAV41PCPMetadataBuilder,
    AscendDSAV41IndexerMetadataBuilder,
):
    _global_metadata_builder_cls = AscendDSAV41IndexerMetadataBuilder


class AscendDSAV41PCPFoldedIndexMetadataBuilder(
    AscendDSAV41PCPMetadataBuilder,
    AscendDSAV41FoldedIndexMetadataBuilder,
):
    _global_metadata_builder_cls = AscendDSAV41FoldedIndexMetadataBuilder


class AscendDSAV41PCPImpl(_ReplicatedCacheImpl):
    """Update replicated caches once, then run rank-local V4.1 attention."""

    supports_pcp: ClassVar[bool] = True

    @staticmethod
    def _gather_and_restore_hidden_states(
        hidden_states: torch.Tensor,
        metadata: AscendDSAV41PCPMetadata,
    ) -> torch.Tensor:
        assert hidden_states.shape[0] == metadata.local_num_tokens_after_padding
        gathered = get_pcp_group().all_gather(hidden_states.contiguous(), dim=0)
        return torch.index_select(gathered, 0, metadata.hidden_restore_idx)

    def _prepare_inputs_and_caches(self, attn, hidden_states, metadata, metadata_by_prefix):
        if not isinstance(metadata.swa, AscendDSAV41PCPMetadata):
            return super()._prepare_inputs_and_caches(attn, hidden_states, metadata, metadata_by_prefix)
        # Empty query ranks also join hidden and weight collectives.
        global_hidden_states = self._gather_and_restore_hidden_states(hidden_states, metadata.swa)
        projector = attn.dsa_attn.dsa_attn.impl
        projector._parallel_o_proj.all_gather_weights()
        global_metadata = self._global_layer_metadata(metadata_by_prefix)
        return self._preprocess(attn, hidden_states, global_hidden_states, metadata, global_metadata)

    def forward(self, attn, positions, hidden_states, output=None):
        # Every layer has SWA metadata carrying the batch's PCP and prefill state.
        metadata = (
            get_forward_context().attn_metadata[self.swa_prefix]
            if get_forward_context().attn_metadata is not None
            else None
        )
        if isinstance(metadata, AscendDSAV41PCPMetadata):
            needs_global_metadata = True
            assert metadata.global_metadata is not None
            has_prefill = metadata.global_metadata.num_prefills > 0
        else:
            needs_global_metadata = False
            has_prefill = False
        projector = attn.dsa_attn.dsa_attn.impl
        # V4 owns the quantized O modules and their batch-scoped collective mode.
        # V4.1 already projects padded output even on empty query ranks.
        with projector._parallel_o_proj.batch(needs_global_metadata=needs_global_metadata, has_prefill=has_prefill):
            return super().forward(attn, positions, hidden_states, output)
