# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA prefill context parallelism using the PCP group and MRV2 batches."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.distributed as dist
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group, get_tp_group
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention import dsa_v1
from vllm_ascend.attention.context_parallel.common_cp import is_pcp_decode_sharding_enabled
from vllm_ascend.attention.dsa_attn_kv_plan import (
    get_dsa_attn_kv_plan,
)
from vllm_ascend.attention.dsa_v1 import (
    AscendDSAImpl,
    AscendDSAMetadata,
    AscendDSAMetadataBuilder,
    DSAMetadataDict,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    uses_pcp_sharded_layout,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorMetadata
from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerMetadata
from vllm_ascend.quantization.methods import AscendLinearScheme
from vllm_ascend.utils import enable_pcp_o_proj_weight_sharding
from vllm_ascend.weight_switch import (
    WeightLoadPartition,
    WeightSwitchConfig,
    WeightSwitchMixin,
)
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext


@dataclass(kw_only=True)
class AscendDSAPCPMetadata(AscendDSAMetadata):
    """Rank-local DSA metadata with its canonical cache-update view."""

    local_num_tokens_after_padding: int
    hidden_restore_idx: torch.Tensor
    global_dsa_metadata: AscendDSAMetadata
    pcp_has_global_prefill: bool = False

    @classmethod
    def from_local_metadata(
        cls,
        local_metadata: AscendDSAMetadata,
        local_num_tokens_after_padding: int,
        hidden_restore_idx: torch.Tensor,
        global_dsa_metadata: AscendDSAMetadata,
        pcp_has_global_prefill: bool = False,
    ) -> "AscendDSAPCPMetadata":
        return cls(
            num_actual_tokens=local_metadata.num_actual_tokens,
            num_decodes=local_metadata.num_decodes,
            num_decode_tokens=local_metadata.num_decode_tokens,
            num_prefills=local_metadata.num_prefills,
            head_dim=local_metadata.head_dim,
            attn_state=local_metadata.attn_state,
            req_metadata=local_metadata.req_metadata,
            reshape_cache_event=local_metadata.reshape_cache_event,
            hadamard=local_metadata.hadamard,
            local_num_tokens_after_padding=local_num_tokens_after_padding,
            hidden_restore_idx=hidden_restore_idx,
            global_dsa_metadata=global_dsa_metadata,
            pcp_has_global_prefill=pcp_has_global_prefill,
        )


class AscendDSAPCPMetadataBuilder(AscendDSAMetadataBuilder):
    """Build ordinary decode metadata or PCP prefill cache metadata."""

    # DualChunkSwap expands each prefill into at most two local rows.
    _request_capacity_factor: ClassVar[int] = 2
    _global_metadata_builder_cls = dsa_v1.AscendDSAMetadataBuilder

    def __init__(
        self,
        kv_cache_spec: AscendMLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
        )
        self._global_metadata_builder = self._global_metadata_builder_cls(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls=AscendDSAMetadata,
        )
        self._pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.use_pcp = self._pcp_world_size > 1
        self._pcp_rank = get_pcp_group().rank_in_group
        self._is_decode_sharded = is_pcp_decode_sharding_enabled(vllm_config)
        self._hidden_restore_idx_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._global_rope_buffers: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendDSAPCPMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def _prepare_graph_pcp_context(
        self,
        pcp_context: "AscendPCPAttentionContext",
        full_graph_mode: bool,
    ) -> "AscendPCPAttentionContext":
        """Prepare graph-stable DSA tensors without changing the PCP batch layout."""
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

        return replace(
            pcp_context,
            hidden_restore_idx=hidden_restore_idx,
        )

    @staticmethod
    def _build_graph_common_attn_metadata(
        common_attn_metadata: AscendCommonAttentionMetadata,
        num_actual_reqs: int | None,
    ) -> AscendCommonAttentionMetadata:
        """Restore DSA graph request metadata after PCP partitioning.

        PCP preserves the fixed graph token shape, but represents the padded
        local rows as zero-token requests. DSA requires the query offsets to
        cover every decode graph token, so expand the padded query offsets in
        both the NPU and CPU metadata views. A regular decode has one query
        token per request, while MTP verification has a fixed K+1 query
        length. Padded slot mappings are invalidated separately.

        Regular decode example: 2 actual requests in a graph batch padded to 4::

            After PCP partition:
                query_start_loc = [0, 1, 2, 2, 2]

            After DSA restoration:
                query_start_loc = [0, 1, 2, 3, 4]

        With MTP=1, each request verifies K+1=2 tokens in the same graph::

            After PCP partition:
                query_start_loc = [0, 2, 4, 4, 4]

            After DSA restoration:
                query_start_loc = [0, 2, 4, 6, 8]

        ``num_actual_reqs`` remains 2, and the two padded slot mappings remain
        invalid.
        """
        if num_actual_reqs is None:
            return common_attn_metadata
        num_reqs = common_attn_metadata.num_reqs
        if num_reqs <= num_actual_reqs:
            return common_attn_metadata

        is_prefilling = common_attn_metadata.is_prefilling
        if is_prefilling is not None and bool(is_prefilling.any()):
            return common_attn_metadata

        # FULL_DECODE_ONLY uses the same query length for every request.
        query_len = common_attn_metadata.num_input_tokens // num_reqs
        padded_offsets = slice(num_actual_reqs + 1, num_reqs + 1)
        padding_start = (num_actual_reqs + 1) * query_len
        padding_end = (num_reqs + 1) * query_len

        # Expand NPU and CPU query offsets to cover padded graph tokens.
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        torch.arange(
            padding_start,
            padding_end,
            query_len,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
            out=query_start_loc[padded_offsets],
        )
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu.new_empty((num_reqs + 1,))
        query_start_loc_cpu[: num_actual_reqs + 1].copy_(
            common_attn_metadata.query_start_loc_cpu[: num_actual_reqs + 1]
        )
        torch.arange(
            padding_start,
            padding_end,
            query_len,
            dtype=query_start_loc_cpu.dtype,
            out=query_start_loc_cpu[padded_offsets],
        )

        return common_attn_metadata.replace(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
        )

    @staticmethod
    def _build_global_common_attn_metadata(
        pcp_context: "AscendPCPAttentionContext",
        cache_group_idx: int,
        local_common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendCommonAttentionMetadata:
        global_batch = pcp_context.global_batch
        num_reqs = global_batch.num_reqs_after_padding
        return AscendCommonAttentionMetadata(
            query_start_loc=global_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(global_batch.query_start_loc_np),
            seq_lens=global_batch.seq_lens[:num_reqs],
            seq_lens_cpu=torch.from_numpy(global_batch.seq_lens_np)[:num_reqs],
            seq_lens_cpu_upper_bound=global_batch.seq_lens_cpu_upper_bound[:num_reqs],
            num_computed_tokens_cpu=torch.from_numpy(global_batch.num_computed_tokens_np),
            num_reqs=num_reqs,
            num_actual_tokens=global_batch.num_tokens,
            max_query_len=int(global_batch.num_scheduled_tokens.max()),
            max_seq_len=local_common_attn_metadata.max_seq_len,
            block_table_tensor=pcp_context.global_block_tables[cache_group_idx],
            slot_mapping=pcp_context.global_slot_mappings[cache_group_idx],
            causal=local_common_attn_metadata.causal,
            dcp_local_seq_lens=global_batch.dcp_local_seq_lens,
            positions=global_batch.positions,
            attn_state=global_batch.attn_state,
            num_input_tokens=global_batch.num_tokens_after_padding,
            is_prefilling=torch.from_numpy(global_batch.is_prefilling_np),
        )

    def _build_local_common_attn_metadata(
        self,
        pcp_context: "AscendPCPAttentionContext",
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

    def _build_local_dsa_metadata(
        self,
        common_prefix_len: int,
        local_common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool,
        num_actual_reqs: int | None,
        common_ratio_to_sas_metadata: dict[Any, Any],
        full_graph_mode: bool = False,
        update_cache: bool = True,
    ) -> AscendDSAMetadata:
        # Cooperative dispatch reaches the concrete role's request builder.
        # PCP only supplies the local layout and whether it owns cache writes.
        # A captured graph still runs its padded queries, so their metadata
        # must be refreshed even when this rank owns no tokens.
        if local_common_attn_metadata.query_start_loc_cpu[-1] > 0:
            return super().build(
                common_prefix_len,
                local_common_attn_metadata,
                fast_build,
                num_actual_reqs=num_actual_reqs,
                common_ratio_to_sas_metadata=common_ratio_to_sas_metadata,
                full_graph_mode=full_graph_mode,
                update_cache=update_cache,
            )

        # Empty ranks still participate in the global cache update collectives.
        self._device_metadata_tasks = ()
        self.common_ratio_to_sas_metadata = common_ratio_to_sas_metadata
        return self.metadata_cls(  # type: ignore[call-arg]
            num_actual_tokens=0,
            head_dim=self.model_config.get_head_size(),
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=0,
            attn_state=local_common_attn_metadata.attn_state,
            req_metadata=None,
            hadamard=self._global_metadata_builder.hadamard,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        num_actual_reqs: int | None = None,
        common_ratio_to_sas_metadata: dict[Any, Any] | None = None,
        full_graph_mode: bool = False,
        **kwargs: Any,
    ) -> AscendDSAMetadata:
        pcp_context = common_attn_metadata.pcp_context
        pcp_cache_group_idx = common_attn_metadata.pcp_cache_group_idx
        assert common_ratio_to_sas_metadata is not None
        if not uses_pcp_sharded_layout(common_attn_metadata, self.use_pcp):
            # Without a PCP context the batch runs replicated on every rank,
            # like a PCP=1 model (e.g. a replicated draft decode), so build the
            # ordinary DSA metadata that the impl executes without PCP.
            return super().build(
                common_prefix_len,
                common_attn_metadata,
                fast_build,
                num_actual_reqs=num_actual_reqs,
                common_ratio_to_sas_metadata=common_ratio_to_sas_metadata,
                full_graph_mode=full_graph_mode,
                **kwargs,
            )
        assert pcp_context is not None
        assert pcp_cache_group_idx is not None
        has_prefill = bool(pcp_context.global_batch.is_prefilling_np.any())
        # Replicated decode stays on the local metadata path. Sharded decode
        # owns different requests per rank, so it still needs global metadata.
        needs_global_metadata = has_prefill or self._is_decode_sharded
        use_specialized_metadata = self._use_specialized_metadata()
        # The standard caller creates a fresh cache for each build invocation.
        # Global and rank-local layouts must never share metadata entries.
        shared = common_ratio_to_sas_metadata if use_specialized_metadata else {}
        # Reuse the base DSA computation cache even on the full-builder path.
        # Only graph-stable PCP preparation is restricted to specialized builds.
        global_shared = common_ratio_to_sas_metadata.setdefault("pcp_global", {})
        local_shared = common_ratio_to_sas_metadata.setdefault("pcp_local", {})
        if needs_global_metadata:
            prepared = shared.get("pcp_context")
            if prepared is None:
                prepared = self._prepare_graph_pcp_context(
                    pcp_context,
                    full_graph_mode=full_graph_mode,
                )
                shared["pcp_context"] = prepared
            pcp_context = prepared
            global_common_attn_metadata = shared.get("pcp_global_common")
            if global_common_attn_metadata is None:
                global_common_attn_metadata = self._build_global_common_attn_metadata(
                    pcp_context,
                    pcp_cache_group_idx,
                    common_attn_metadata,
                )
                global_common_attn_metadata = self._build_graph_common_attn_metadata(
                    global_common_attn_metadata,
                    pcp_context.global_batch.num_reqs,
                )
                shared["pcp_global_common"] = global_common_attn_metadata
            # Physical cache addresses and causality remain group-specific.
            global_common_attn_metadata = global_common_attn_metadata.replace(
                block_table_tensor=pcp_context.global_block_tables[pcp_cache_group_idx],
                slot_mapping=pcp_context.global_slot_mappings[pcp_cache_group_idx],
                causal=common_attn_metadata.causal,
            )
            # num_prefills can miss short prefills; prevent local PCP RoPE
            # from overwriting the global RoPE buffer. Sharded decode also
            # cannot share that buffer, because each rank has different positions.
            global_dsa_metadata = self._global_metadata_builder.build(
                common_prefix_len,
                global_common_attn_metadata,
                fast_build,
                num_actual_reqs=pcp_context.global_batch.num_reqs,
                common_ratio_to_sas_metadata=global_shared,
                build_query=not use_specialized_metadata,
                can_use_rope_cache=False,
                full_graph_mode=full_graph_mode,
            )
            if (
                not has_prefill
                and self._is_decode_sharded
                and self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
            ):
                # Keep the global RoPE at graph-stable addresses of its own.
                req_metadata = global_dsa_metadata.req_metadata
                assert req_metadata is not None
                if req_metadata.cos is not None:
                    rope = global_shared.get("staged_rope") if use_specialized_metadata else None
                    if rope is None:
                        rope = req_metadata.cos.copy_to_buffers(
                            self._global_rope_buffers, self._hidden_restore_idx_buffer.numel()
                        )
                        if use_specialized_metadata:
                            global_shared["staged_rope"] = rope
                    req_metadata.cos, req_metadata.sin = rope
        local_common_attn_metadata = self._build_local_common_attn_metadata(
            pcp_context,
            common_attn_metadata,
            pcp_cache_group_idx,
            has_prefill,
        )
        if local_common_attn_metadata.num_actual_tokens == 0:
            # The placeholder request of an empty rank is graph padding too.
            num_actual_reqs = 0
        padded_local = shared.get("pcp_local_offsets")
        if padded_local is None:
            local_common_attn_metadata = self._build_graph_common_attn_metadata(
                local_common_attn_metadata,
                num_actual_reqs,
            )
            shared["pcp_local_offsets"] = (
                local_common_attn_metadata.query_start_loc,
                local_common_attn_metadata.query_start_loc_cpu,
            )
        else:
            local_common_attn_metadata = local_common_attn_metadata.replace(
                query_start_loc=padded_local[0],
                query_start_loc_cpu=padded_local[1],
            )
        local_dsa_metadata = self._build_local_dsa_metadata(
            common_prefix_len,
            local_common_attn_metadata,
            fast_build,
            num_actual_reqs=num_actual_reqs,
            common_ratio_to_sas_metadata=local_shared,
            full_graph_mode=full_graph_mode,
            update_cache=not use_specialized_metadata or not needs_global_metadata,
        )
        # Base build stores the local namespace on self; expose both layouts
        # to the runner for the next cache group, including empty local ranks.
        self.common_ratio_to_sas_metadata = common_ratio_to_sas_metadata
        # Replicated decode tokens stay in scheduler order on every PCP rank.
        # The local metadata therefore describes the canonical cache update
        # directly and can use the ordinary non-PCP DSA execution path.
        if not needs_global_metadata:
            return local_dsa_metadata

        return AscendDSAPCPMetadata.from_local_metadata(
            local_dsa_metadata,
            local_common_attn_metadata.num_input_tokens,
            pcp_context.hidden_restore_idx,
            global_dsa_metadata,
            pcp_has_global_prefill=has_prefill,
        )

    def enable_device_metadata(self) -> None:
        self._device_metadata_enabled = True
        self._global_metadata_builder.enable_device_metadata()

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        global_tasks = self._global_metadata_builder.take_device_metadata_tasks()
        return (
            tuple(task for task in global_tasks if task.stage == DeviceMetadataStage.COMPRESSOR)
            + super().take_device_metadata_tasks()
        )


class AscendDSAPCPStateMetadataBuilder(AscendDSAPCPMetadataBuilder, dsa_v1.AscendDSAStateMetadataBuilder):
    """PCP layout handling with the state backend's metadata requirements."""

    _global_metadata_builder_cls = dsa_v1.AscendDSAStateMetadataBuilder


class AscendDSAPCPSWAMetadataBuilder(AscendDSAPCPMetadataBuilder, dsa_v1.AscendDSASWAMetadataBuilder):
    """PCP layout handling with the swa backend's metadata requirements."""

    _global_metadata_builder_cls = dsa_v1.AscendDSASWAMetadataBuilder


class AscendDSAPCPAttentionMetadataBuilder(AscendDSAPCPMetadataBuilder, dsa_v1.AscendDSAAttentionMetadataBuilder):
    """PCP layout handling with the attention backend's metadata requirements."""

    _global_metadata_builder_cls = dsa_v1.AscendDSAAttentionMetadataBuilder


class AscendDSAPCPIndexerMetadataBuilder(AscendDSAPCPMetadataBuilder, dsa_v1.AscendDSAIndexerMetadataBuilder):
    """PCP layout handling with the indexer backend's metadata requirements."""

    _global_metadata_builder_cls = dsa_v1.AscendDSAIndexerMetadataBuilder


class AscendDSAPCPImpl(AscendDSAImpl):
    """Use ordinary decode updates or global PCP prefill cache updates."""

    supports_pcp: ClassVar[bool] = True
    o_proj_full_pools: ClassVar[dict[Any, torch.Tensor]] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # PCP prepares replicated caches before local attention, leaving no
        # cache-update work for the auxiliary stream to overlap.
        self.multistream_dsv4_dsa_overlap = False

        # V4 and V4.1 share this projector; the batch context selects its O path.
        self.enable_pcp_o_proj_weight_sharding = enable_pcp_o_proj_weight_sharding()
        if not self.enable_pcp_o_proj_weight_sharding:
            return

        self._pcp_o_proj_weight_switches = None
        self._pcp_decode_sharded = is_pcp_decode_sharding_enabled(self.vllm_config)
        # Fall back to ordinary O weights if groups or quantization cannot be sharded.
        pcp_size = get_pcp_group().world_size
        try:
            if self.n_local_groups % pcp_size != 0:
                raise ValueError(f"n_local_groups={self.n_local_groups} is not divisible by pcp_size={pcp_size}")
            self._get_pcp_weight_switch_method(self.wo_a)
            self._get_pcp_weight_switch_method(self.wo_b)
        except (RuntimeError, ValueError) as exc:
            logger.warning_once(
                "DSA-PCP O-projection weight sharding is unavailable (%s); using unsharded O-projection weights.",
                exc,
            )
            self.enable_pcp_o_proj_weight_sharding = False
            return

        # Configure PCP weight loading before the model weights are loaded.
        self._prepare_pcp_o_proj_weight_shards()
        # Sharded decode needs exchange buffers allocated before graph capture.
        if self._pcp_decode_sharded:
            self._initialize_pcp_o_proj_exchange_buffers()

    @staticmethod
    def _get_pcp_weight_switch_method(layer: torch.nn.Module) -> WeightSwitchMixin:
        quant_method = layer.quant_method
        linear_method = getattr(quant_method, "quant_method", quant_method)
        if not isinstance(linear_method, WeightSwitchMixin) or not linear_method.supports_weight_switch:
            raise RuntimeError(
                "DSA-PCP O-proj weight sharding requires a weight-switch capable method, "
                f"got {type(linear_method).__name__}."
            )
        return linear_method

    def _prepare_pcp_o_proj_weight_shards(self) -> None:
        pcp_group = get_pcp_group()
        if self.n_local_groups % pcp_group.world_size != 0:
            raise ValueError(
                "DSA-PCP O-proj weight sharding requires n_local_groups to be "
                "divisible by the PCP size, got "
                f"n_local_groups={self.n_local_groups}, pcp_size={pcp_group.world_size}."
            )

        self.pcp_o_proj_weight_switch_config = WeightSwitchConfig.from_group(pcp_group)
        load_partition = WeightLoadPartition.from_nested_groups(get_tp_group(), pcp_group)
        self.wo_a_pcp_weight_method = self._get_pcp_weight_switch_method(self.wo_a)
        self.wo_b_pcp_weight_method = self._get_pcp_weight_switch_method(self.wo_b)
        self.wo_a_pcp_weight_method.prepare_layer_for_parallel_output_weight_load(
            self.wo_a,
            self.pcp_o_proj_weight_switch_config,
            load_partition,
        )
        self.wo_b_pcp_weight_method.prepare_layer_for_parallel_weight_load(
            self.wo_b,
            self.pcp_o_proj_weight_switch_config,
            load_partition,
        )

        # wo_a post-processing reshapes the raw output shard into a batched
        # group layout. Keep its layer/method metadata aligned with the
        # PCP-local resident shard; the attention implementation itself keeps
        # the TP-local group count because runtime all-gather restores it.
        local_groups = self.n_local_groups // pcp_group.world_size
        self.wo_a.n_local_groups = local_groups
        if hasattr(self.wo_a_pcp_weight_method, "n_local_groups"):
            self.wo_a_pcp_weight_method.n_local_groups = local_groups

    def _initialize_pcp_o_proj_exchange_buffers(self) -> None:
        """Allocate activation-typed collective storage before graph capture."""
        pcp_size = get_pcp_group().world_size
        compilation_config = self.vllm_config.compilation_config
        decode_query_len = 1 + self.vllm_config.num_speculative_tokens
        capacity = max(
            self.vllm_config.scheduler_config.max_num_seqs * decode_query_len,
            compilation_config.max_cudagraph_capture_size or 0,
            max(compilation_config.cudagraph_capture_sizes or [0]),
        )
        input_size = self.n_local_heads * self.head_dim
        output_size = self.wo_b.output_size_per_partition
        dtype = self.vllm_config.model_config.dtype
        device = self.wo_b.weight.device
        # Compatible layers share storage because O projections execute in
        # sequence. Forward slices the active bucket without reallocating or
        # sending the unused capacity; all HCCL addresses stay graph-stable.
        shapes = {
            "_pcp_o_proj_ag_in_buf": (capacity, input_size),
            "_pcp_o_proj_ag_out_buf": (pcp_size * capacity, input_size),
            "_pcp_o_proj_rs_in_buf": (pcp_size * capacity, output_size),
            "_pcp_o_proj_rs_out_buf": (capacity, output_size),
        }
        for name, shape in shapes.items():
            key = ("dsa_pcp_o_proj_decode", name, device, dtype, shape)
            buffer = self.o_proj_full_pools.get(key)
            if buffer is None:
                buffer = torch.empty(shape, dtype=dtype, device=device)
                self.o_proj_full_pools[key] = buffer
            setattr(self, name, buffer)

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...] | None,
        attn_metadata: DSAMetadataDict | None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        metadata = next(iter(attn_metadata.values())) if attn_metadata is not None else None
        pcp_metadata = metadata if isinstance(metadata, AscendDSAPCPMetadata) else None
        # Global prefill must select the same collectives on every rank;
        # a one-token prefill shard can look like DecodeOnly locally.
        has_prefill = pcp_metadata is not None and pcp_metadata.pcp_has_global_prefill
        with self._o_proj_batch(needs_global_metadata=pcp_metadata is not None, has_prefill=has_prefill):
            result = super().forward(layer_name, hidden_states, kv_cache, attn_metadata, output)
            # V4 returns after cache preparation on empty query ranks. They
            # still project other owners' requests with their resident O groups.
            # Keep this on the successful path, outside exception cleanup.
            if (
                self._pcp_o_proj_use_activation_exchange
                and pcp_metadata is not None
                and pcp_metadata.num_actual_tokens == 0
            ):
                o_proj_input = hidden_states.new_zeros(self._get_o_proj_input_shape(attn_metadata))
                self._forward_o_proj(o_proj_input, output)
                output.zero_()
            return result

    @contextmanager
    def _o_proj_batch(
        self,
        needs_global_metadata: bool,
        has_prefill: bool,
    ) -> Generator[None, None, None]:
        """Own the O-projection mode and weight lifetime for V4 and V4.1.

        Callers translate their metadata into global batch facts before cache
        preparation. Prefill gathers weights, sharded decode exchanges
        activations, and replicated decode/profile uses resident local weights.
        """
        try:
            self._pcp_o_proj_use_activation_exchange = (
                self.enable_pcp_o_proj_weight_sharding
                and needs_global_metadata
                and self._pcp_decode_sharded
                and not has_prefill
            )
            self._pcp_o_proj_use_full_weight = (
                self.enable_pcp_o_proj_weight_sharding
                and needs_global_metadata
                and not self._pcp_o_proj_use_activation_exchange
            )
            # V4.1 calls O projection directly rather than DSA forward. Prepare
            # persistent aliases during warmup, before either caller captures.
            if self.enable_pcp_o_proj_weight_sharding:
                self._get_pcp_o_proj_weight_switches()
            yield
        finally:
            # An empty query rank or an early return may skip O projection.
            # Drain launched weight gathers and restore resident shards.
            if self.enable_pcp_o_proj_weight_sharding:
                for layer, method, state in self._pcp_o_proj_weight_switches or ():
                    if state.handles:
                        method.wait_weight_all_gather(state)
                    if self._pcp_o_proj_use_full_weight:
                        method.switch_weight(layer, state, use_full_weight=False)

    def _maybe_all_gather_pcp_o_proj_weights(self) -> None:
        if not self._pcp_o_proj_use_full_weight:
            return

        for _, method, state in self._get_pcp_o_proj_weight_switches():
            method.all_gather_weight(state, self.pcp_o_proj_weight_switch_config)

    def _get_pcp_o_proj_weight_switches(self):
        if self._pcp_o_proj_weight_switches is None:
            weight_switches = []
            for name, layer, method in (
                ("wo_a", self.wo_a, self.wo_a_pcp_weight_method),
                ("wo_b", self.wo_b, self.wo_b_pcp_weight_method),
            ):
                state = method.enable_weight_switch(
                    layer,
                    self.pcp_o_proj_weight_switch_config,
                    pool=AscendDSAPCPImpl.o_proj_full_pools,
                    pool_key_prefix=(type(method).__qualname__, name, "dsa_pcp_o_proj"),
                )
                weight_switches.append((layer, method, state))
            self._pcp_o_proj_weight_switches = tuple(weight_switches)
        return self._pcp_o_proj_weight_switches

    def _project_o_proj_with_local_weights(self, group_input: torch.Tensor) -> torch.Tensor:
        """Project only the O groups owned by this PCP rank, then reduce TP."""
        pcp_group = get_pcp_group()
        projected = self._wo_a_bmm(group_input).reshape(group_input.shape[0], -1)
        linear_method = self._get_pcp_weight_switch_method(self.wo_b)
        if isinstance(linear_method, AscendLinearScheme):
            tp_group = get_tp_group()
            # The complete O-projection bias belongs to only one TP/PCP shard.
            bias_rank = 0 if tp_group.rank_in_group == 0 and pcp_group.rank_in_group == 0 else 1
            bias = self.wo_b.bias if bias_rank == 0 and not self.wo_b.skip_bias_add else None
            partial_output = linear_method.apply(self.wo_b, projected, bias=bias, tp_rank=bias_rank)
            if self.wo_b.reduce_results and tp_group.world_size > 1:
                partial_output = tp_group.all_reduce(partial_output)
        else:
            partial_output = self.wo_b(projected)
        return partial_output

    def _forward_o_proj_with_local_weights(self, o_proj_input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        # Decode tokens and their KV cache updates are replicated across PCP
        # ranks, so each rank projects the same attention output locally.
        pcp_group = get_pcp_group()
        num_tokens = o_proj_input.shape[0]
        groups_per_rank = self.n_local_groups // pcp_group.world_size
        group_input = o_proj_input.reshape(num_tokens, self.n_local_groups, -1)
        group_input = group_input.narrow(1, pcp_group.rank_in_group * groups_per_rank, groups_per_rank).contiguous()

        output[...] = pcp_group.all_reduce(self._project_o_proj_with_local_weights(group_input))
        return output

    def _forward_o_proj_with_sharded_tokens(self, o_proj_input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Gather decode tokens, project local O groups, and scatter sums."""
        pcp_group = get_pcp_group()
        num_tokens = o_proj_input.shape[0]
        groups_per_rank = self.n_local_groups // pcp_group.world_size
        group_hidden_dim = self.n_local_heads * self.head_dim // self.n_local_groups
        local_rows = self._pcp_o_proj_ag_in_buf[:num_tokens]
        local_rows.copy_(o_proj_input.reshape(num_tokens, -1))
        global_rows = self._pcp_o_proj_ag_out_buf[: pcp_group.world_size * num_tokens]
        # AllToAll reduces payload but adds a channel-packing transpose; measured
        # small-batch decode showed no clear benefit, so use token AllGather.
        dist.all_gather_into_tensor(global_rows, local_rows, group=pcp_group.device_group)
        # Rank-major padded rows match ReduceScatter's token ownership. Select
        # resident O groups after gathering to avoid packing channels to send.
        group_input = global_rows.view(pcp_group.world_size * num_tokens, self.n_local_groups, group_hidden_dim)
        group_input = group_input.narrow(1, pcp_group.rank_in_group * groups_per_rank, groups_per_rank).contiguous()
        partial_output = self._project_o_proj_with_local_weights(group_input)
        scatter_input = self._pcp_o_proj_rs_in_buf[: pcp_group.world_size * num_tokens]
        scatter_input.copy_(partial_output)
        reduced = self._pcp_o_proj_rs_out_buf[:num_tokens]
        dist.reduce_scatter_tensor(reduced, scatter_input, group=pcp_group.device_group)
        output.copy_(reduced)
        return output

    def _forward_o_proj(self, o_proj_input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if not self.enable_pcp_o_proj_weight_sharding:
            return super()._forward_o_proj(o_proj_input, output)
        if self._pcp_o_proj_use_activation_exchange:
            return self._forward_o_proj_with_sharded_tokens(o_proj_input, output)
        if not self._pcp_o_proj_use_full_weight:
            return self._forward_o_proj_with_local_weights(o_proj_input, output)

        weight_switches = self._get_pcp_o_proj_weight_switches()
        for layer, method, state in weight_switches:
            method.wait_weight_all_gather(state)
            method.switch_weight(layer, state, use_full_weight=True)
        try:
            return super()._forward_o_proj(o_proj_input, output)
        finally:
            for layer, method, state in weight_switches:
                method.switch_weight(layer, state, use_full_weight=False)

    def _gather_and_restore_hidden_states(
        self,
        hidden_states: torch.Tensor,
        metadata: AscendDSAPCPMetadata,
    ) -> torch.Tensor:
        """All-gather one padded rank slice and restore scheduler token order."""
        gathered_hidden_states = get_pcp_group().all_gather(
            hidden_states.contiguous(),
            dim=0,
        )
        return torch.index_select(
            gathered_hidden_states,
            0,
            metadata.hidden_restore_idx,
        )

    def _update_global_swa_cache(
        self,
        layer_name: str,
        global_hidden_states: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        swa_metadata: AscendDSAMetadata,
    ) -> None:
        """Update the replicated SWA cache from the canonical global batch."""
        req_metadata = dsa_v1._require_req_metadata(swa_metadata)
        assert req_metadata.slot_mapping is not None

        kv = self.kv_norm(self.wkv(global_hidden_states))
        assert self.rope_head_dim is not None
        kv = kv.view(
            -1,
            1,
            self.nope_head_dim + self.rope_head_dim,
        )
        # cos/sin/slot_mapping are built at the restored batch length -- padded
        # for FULL graphs, actual tokens otherwise -- so they already match the
        # kv rows and need no trimming here.
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            req_metadata.cos[layer_name],
            req_metadata.sin[layer_name],
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )
        get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
            swa_kv_cache,
            kv,
            req_metadata.slot_mapping,
        )

    def _update_global_compressor_cache(
        self,
        global_hidden_states: torch.Tensor,
        metadata: AscendCompressorMetadata,
        compress_kv_cache: torch.Tensor,
        state_cache: torch.Tensor,
    ) -> None:
        """Update the DSA compressor cache from the canonical global batch."""
        assert self.compressor is not None
        compressed_kv, compress_slot_mapping = self.compressor(
            hidden_states=global_hidden_states,
            state_cache=state_cache,
            metadata=metadata,
        )
        if compressed_kv.shape[0] > 0:
            get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                compress_kv_cache,
                compressed_kv,
                compress_slot_mapping,
            )

    def _update_global_indexer_cache(
        self,
        global_hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
    ) -> None:
        """Update the Indexer cache from the canonical global batch."""
        indexer = self.indexer
        assert indexer is not None
        if indexer.skip_topk:
            return
        indexer.update_cache(
            hidden_states=global_hidden_states,
            kv_cache=kv_cache,
            metadata=metadata,
        )

    def _prepare_caches_before_attention(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataDict,
    ) -> bool:
        """Use ordinary decode updates or restore the global prefill batch."""
        pcp_metadata = next(iter(attn_metadata.values()))
        if not isinstance(pcp_metadata, AscendDSAPCPMetadata):
            return super()._prepare_caches_before_attention(
                layer_name,
                hidden_states,
                kv_cache,
                attn_metadata,
            )
        global_hidden_states = self._gather_and_restore_hidden_states(
            hidden_states,
            pcp_metadata,
        )

        global_dsa_metadata_by_prefix = {}
        for cache_prefix, metadata in attn_metadata.items():
            assert isinstance(metadata, AscendDSAPCPMetadata)
            global_dsa_metadata_by_prefix[cache_prefix] = metadata.global_dsa_metadata
        global_layer_metadata = self._get_layer_metadata(
            layer_name,
            global_dsa_metadata_by_prefix,
        )

        cmp_kv, swa_kv, state_cache, _, _, _ = DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)

        self._update_global_swa_cache(
            layer_name,
            global_hidden_states,
            swa_kv,
            global_layer_metadata.swa,
        )
        if self.compress_ratio > 1:
            compressor_metadata = global_layer_metadata.compressor
            assert compressor_metadata is not None
            assert cmp_kv is not None
            assert state_cache is not None
            self._update_global_compressor_cache(
                global_hidden_states,
                compressor_metadata,
                cmp_kv,
                state_cache,
            )

            if self.compress_ratio == 4:
                indexer_metadata = global_layer_metadata.indexer
                assert indexer_metadata is not None
                self._update_global_indexer_cache(
                    global_hidden_states,
                    kv_cache,
                    indexer_metadata,
                )
        # Every rank must join the weight gathers, including ranks that return
        # before local attention. Launch after the cache collectives to overlap
        # local attention on nonempty ranks.
        self._maybe_all_gather_pcp_o_proj_weights()
        return True

    def _get_o_proj_input_shape(
        self,
        attn_metadata: DSAMetadataDict | None,
    ) -> tuple[int, int, int]:
        if attn_metadata is None:
            return super()._get_o_proj_input_shape(attn_metadata)
        pcp_metadata = next(iter(attn_metadata.values()))
        if not isinstance(pcp_metadata, AscendDSAPCPMetadata):
            return super()._get_o_proj_input_shape(attn_metadata)
        return (
            pcp_metadata.local_num_tokens_after_padding,
            self.n_local_heads,
            self.head_dim,
        )
