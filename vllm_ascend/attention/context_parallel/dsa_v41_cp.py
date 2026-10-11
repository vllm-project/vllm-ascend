# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 replicated-cache adapters for legacy TP-token CP and MRV2 PCP."""

from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from vllm.distributed import get_pcp_group, get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.attention.context_parallel.common_cp import is_pcp_decode_sharding_enabled
from vllm_ascend.attention.context_parallel.dsa_cp import (
    AscendDSACPMetadataBuilder,
    AscendDSAPCPMetadataBuilder,
    restore_tp_heads,
)
from vllm_ascend.attention.dsa_v41 import (
    AscendDSAV41Impl,
    AscendDSAV41Metadata,
    AscendDSAV41MetadataBuilder,
    _config_value,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, enable_pcp
from vllm_ascend.core.kv_cache_interface import get_kv_cache_compression_ratio, get_storage_block_size
from vllm_ascend.ops.rope_dsv4 import get_full_cos_and_sin_dsa_for_layer
from vllm_ascend.utils import enable_dsa_cp

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext


def get_v41_cp_classes():
    use_dsa_cp = enable_dsa_cp()
    # Read the current model config: DSpark's target uses PCP, while its
    # replicated draft is constructed under a separate PCP=1 config.
    use_pcp = enable_pcp()
    if use_dsa_cp and use_pcp:
        raise ValueError("Legacy DSACP and PCP cannot be enabled at the same time.")
    if use_dsa_cp:
        return AscendDSAV41CPMetadataBuilder, AscendDSAV41CPImpl
    if use_pcp:
        return AscendDSAV41PCPMetadataBuilder, AscendDSAV41PCPImpl
    return AscendDSAV41MetadataBuilder, AscendDSAV41Impl


class _ReplicatedCacheMetadataBuilder(AscendDSAV41MetadataBuilder):
    """Keep global cache metadata independent from local query buffers."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, *, build_local_compressor_metadata=False):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            build_compressor_metadata=build_local_compressor_metadata,
        )
        self._global_builder = AscendDSAV41MetadataBuilder(
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


# =============================================================================
# Legacy DSV4.1 DSACP implementation
# =============================================================================


class AscendDSAV41CPMetadataBuilder(_ReplicatedCacheMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # SMLA consumes INT32 offsets at a fixed address during graph replay.
        self._cp_query_start_loc = self._seq_lens.new_zeros(self._seq_lens.numel() + 1)

    # Reuse Legacy DSACP's request intersection and causal-prefix calculation.
    _local_token_range = staticmethod(AscendDSACPMetadataBuilder._local_token_range)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
        common = common_attn_metadata
        global_metadata = self._build_global_metadata(common_prefix_len, common, fast_build, kwargs)
        seq_lens_cpu = (
            common._seq_lens_cpu if getattr(common, "_seq_lens_cpu", None) is not None else common.seq_lens_cpu
        )
        start, end, per_rank, padded, qsl, seq_lens = AscendDSACPMetadataBuilder._build_local_token_metadata(
            self,
            common.num_reqs,
            common.num_input_tokens,
            common.query_start_loc_cpu,
            seq_lens_cpu,
            is_noncausal=not bool(getattr(common, "causal", True)),
        )
        actual_end = min(end, common.num_actual_tokens)
        actual_start = min(start, actual_end)
        # Padding participates in the output exchange, not in cache reads.
        qsl = qsl.clamp_max(actual_end - actual_start).to(self._cp_query_start_loc.dtype)
        query_start_loc = self._cp_query_start_loc[: qsl.numel()]
        query_start_loc.copy_(qsl.pin_memory(), non_blocking=True)
        # Device lengths are authoritative after speculative rejection; the
        # CPU mirror may still be an upper bound. Remove only the query suffix
        # beyond this rank's token interval from each request's device length.
        query_ends = common.query_start_loc_cpu[1 : common.num_reqs + 1]
        suffix = query_ends - query_ends.clamp(min=actual_start, max=actual_end)
        if not bool(getattr(common, "causal", True)):
            suffix = torch.zeros_like(suffix)
        local_seq_lens = (
            common.seq_lens[: common.num_reqs] - suffix.pin_memory().to(common.seq_lens.device, non_blocking=True)
        ).clamp_min(0)
        local_seq_lens = torch.where(query_start_loc[1:] > query_start_loc[:-1], local_seq_lens, 0)
        local_common = common.replace(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=qsl,
            seq_lens=local_seq_lens,
            seq_lens_cpu=seq_lens,
            num_actual_tokens=actual_end - actual_start,
            num_input_tokens=actual_end - actual_start,
            positions=common.positions[actual_start:actual_end],
            slot_mapping=common.slot_mapping[actual_start:actual_end],
            max_query_len=int((qsl[1:] - qsl[:-1]).max()) if common.num_reqs else 0,
            max_seq_len=int(seq_lens.max()) if common.num_reqs else 0,
        )
        kwargs["num_query_heads"] = _config_value(self.vllm_config.model_config.hf_text_config, "num_attention_heads")
        if global_metadata.cos is not None and global_metadata.sin is not None:
            # Q owns a contiguous token slice of the global KV batch. Reuse
            # that slice: a second cached RoPE gather would overwrite the
            # process-wide buffer still referenced by global KV metadata.
            kwargs["rope_views"] = (
                global_metadata.cos[actual_start:actual_end],
                global_metadata.sin[actual_start:actual_end],
            )
        if global_metadata.ori_sparse_indices is not None:
            kwargs["ori_sparse_indices"] = global_metadata.ori_sparse_indices[actual_start:actual_end]
        local = super().build(common_prefix_len, local_common, fast_build, **kwargs)
        return replace(local, global_metadata=global_metadata, cp_token_range=(start, end, per_rank, padded))


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


class AscendDSAV41CPImpl(_ReplicatedCacheImpl):
    def _prepare_inputs_and_caches(self, attn, hidden_states, metadata, metadata_by_prefix):
        global_metadata = self._global_layer_metadata(metadata_by_prefix)
        start, _, _, _ = metadata.swa.cp_token_range
        return self._preprocess(attn, hidden_states[start:], hidden_states, metadata, global_metadata)

    def _indexer_hidden_states(self, hidden_states, metadata):
        start, _, _, _ = metadata.swa.cp_token_range
        return hidden_states[start : start + metadata.swa.num_actual_tokens]

    def _select_sparse_indices(self, attn, hidden_states, qr, positions, cos, sin, metadata, prepared_indexer=None):
        if not self.role.has_long_context:
            return None
        if not self.role.is_index_source:
            shared = attn.shared_state
            # ``hidden_states`` still owns the full pre-CP token batch here,
            # while ``qr`` was projected from this rank's local query slice.
            # SparseFlashMla requires cmp_sparse_indices.T to match q.T.
            return shared.topk_indices[: qr.shape[0]]
        return super()._select_sparse_indices(attn, hidden_states, qr, positions, cos, sin, metadata, prepared_indexer)

    def _project_output(self, attn, output, hidden_states, metadata, *, projected):
        _, _, per_rank, _ = metadata.swa.cp_token_range
        padded = output
        if output.shape[0] != per_rank:
            padded = output.new_zeros((per_rank, output.shape[1], output.shape[2]))
            padded[: output.shape[0]] = output
        exchanged = restore_tp_heads(padded, get_tp_group())
        # DSpark's DP/FlashComm padding can extend hidden states beyond the
        # metadata token interval. Restore that suffix before O projection so
        # its TP reduction keeps the caller's padded output layout.
        if exchanged.shape[0] < hidden_states.shape[0]:
            padded_exchange = exchanged.new_zeros((hidden_states.shape[0], *exchanged.shape[1:]))
            padded_exchange[: exchanged.shape[0]] = exchanged
            exchanged = padded_exchange
        # The inherited V4 module owns quantized weights and TP projection logic.
        attn.dsa_attn.dsa_attn.impl._forward_o_proj(exchanged[: hidden_states.shape[0]], projected)
        return projected


# =============================================================================
# MRV2 DSV4.1-PCP implementation
# =============================================================================


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

    consumes_pcp_context: ClassVar[bool] = True
    _request_capacity_factor: ClassVar[int] = 2

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if vllm_config.parallel_config.decode_context_parallel_size != 1:
            raise NotImplementedError("V4.1 PCP currently requires DCP=1 for replicated caches.")
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, build_local_compressor_metadata=True)
        self._pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
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

        self._pcp_rope_buffers: dict[str, tuple[torch.Tensor, ...]] = {}

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

    def _build_pcp_rope_views(self, common: AscendCommonAttentionMetadata, *, global_cache: bool):
        if self._cache_kind != "swa":
            return None
        assert self._pcp_rope_buffers, "Prepare V4.1 PCP RoPE before building attention metadata"
        num_tokens = common.num_input_tokens
        positions = common.positions[:num_tokens].long()
        cos_views, sin_views = {}, {}
        gathered: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
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

    def build(
        self,
        common_prefix_len,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build=False,
        pcp_context: "AscendPCPAttentionContext | None" = None,
        pcp_cache_group_idx: int | None = None,
        num_actual_reqs: int | None = None,
        **kwargs: Any,
    ) -> AscendDSAV41Metadata:
        assert pcp_context is not None, "V4.1 PCP requires the MRV2 global batch context"
        assert pcp_cache_group_idx is not None

        # Use the global batch so all ranks choose the same prefill/decode path.
        has_prefill = bool(pcp_context.global_batch.is_prefilling_np.any())
        # Sharded decodes have distinct queries but still need complete caches.
        needs_global_metadata = has_prefill or self._is_decode_sharded
        self._global_builder.take_device_metadata_tasks()

        if needs_global_metadata:
            pcp_context = AscendDSAPCPMetadataBuilder._prepare_graph_pcp_context(self, pcp_context)
            global_common = AscendDSAPCPMetadataBuilder._build_global_common_attn_metadata(
                pcp_context, pcp_cache_group_idx, common_attn_metadata
            )
        local_common = AscendDSAPCPMetadataBuilder._build_local_common_attn_metadata(
            self, pcp_context, common_attn_metadata, pcp_cache_group_idx, has_prefill
        )

        # Match graph capture shapes; dummy batches must not advance the C2 ring.
        if kwargs.get("full_graph_mode", False):
            local_common = AscendDSAPCPMetadataBuilder._build_graph_common_attn_metadata(local_common, num_actual_reqs)
            if needs_global_metadata:
                global_common = AscendDSAPCPMetadataBuilder._build_graph_common_attn_metadata(
                    global_common, pcp_context.global_batch.num_reqs
                )
        kwargs["skip_ring_state_update"] = (
            bool(kwargs.get("skip_ring_state_update", False)) or pcp_context.global_batch.is_dummy
        )

        # Keep global KV RoPE separate from rank-local Q positions.
        if needs_global_metadata:
            global_kwargs = dict(kwargs)
            global_kwargs["num_actual_reqs"] = pcp_context.global_batch.num_reqs
            global_rope = self._build_pcp_rope_views(global_common, global_cache=True)
            if global_rope is not None:
                global_kwargs["rope_views"] = global_rope
            global_metadata = self._build_global_metadata(common_prefix_len, global_common, fast_build, global_kwargs)

        # Empty Query ranks skip query metadata ops but still retain the global cache view.
        if needs_global_metadata and local_common.num_actual_tokens == 0:
            local_metadata = self._build_empty_local_metadata(local_common)
        else:
            local_kwargs = dict(kwargs)
            local_kwargs["num_actual_reqs"] = local_common.num_reqs if num_actual_reqs is None else num_actual_reqs
            # C2 state updates belong to the cache metadata owner.
            local_kwargs["build_compressor_metadata"] = not needs_global_metadata
            for name in ("common_v41_metadata", "common_v41_batch_metadata"):
                shared = kwargs.get(name)
                if shared is not None:
                    local_kwargs[name] = shared.setdefault("pcp_local", {})
            local_rope = self._build_pcp_rope_views(local_common, global_cache=False)
            if local_rope is not None:
                local_kwargs["rope_views"] = local_rope
            local_metadata = super().build(common_prefix_len, local_common, fast_build, **local_kwargs)

        # Replicated decode needs no hidden gather or separate cache metadata.
        if not needs_global_metadata:
            return local_metadata
        return AscendDSAV41PCPMetadata.from_local_metadata(
            local_metadata, global_metadata, local_common.num_input_tokens, pcp_context.hidden_restore_idx
        )


class AscendDSAV41PCPImpl(_ReplicatedCacheImpl):
    """Update replicated caches once, then run rank-local V4.1 attention."""

    supports_pcp: ClassVar[bool] = True

    @contextmanager
    def _o_proj_batch(self, projector, *, needs_global_metadata: bool):
        projector._pcp_o_proj_use_full_weight = projector.enable_pcp_o_proj_weight_sharding and needs_global_metadata
        try:
            # V4.1 bypasses DSA forward; prepare persistent aliases during warmup.
            if projector.enable_pcp_o_proj_weight_sharding:
                projector._get_pcp_o_proj_weight_switches()
            yield
        finally:
            for layer, method, state in projector._pcp_o_proj_weight_switches or ():
                if state.handles:
                    method.wait_weight_all_gather(state)
                if projector._pcp_o_proj_use_full_weight:
                    method.switch_weight(layer, state, use_full_weight=False)
            projector._pcp_o_proj_use_full_weight = False

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
        projector._maybe_all_gather_pcp_o_proj_weights()
        global_metadata = self._global_layer_metadata(metadata_by_prefix)
        return self._preprocess(attn, hidden_states, global_hidden_states, metadata, global_metadata)

    def forward(self, attn, positions, hidden_states, output=None):
        metadata = get_forward_context().attn_metadata
        needs_global_metadata = metadata is not None and isinstance(metadata[self.swa_prefix], AscendDSAV41PCPMetadata)
        projector = attn.dsa_attn.dsa_attn.impl
        with self._o_proj_batch(projector, needs_global_metadata=needs_global_metadata):
            return super().forward(attn, positions, hidden_states, output)
