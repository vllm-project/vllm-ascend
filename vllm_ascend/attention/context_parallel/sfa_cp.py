from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeVar

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group, get_pcp_group, get_tp_group
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.kv_cache_interface import AttentionSpec

import vllm_ascend.ops.triton.dcp.dcp_a2a  # noqa: F401
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import (
    DCPImplMixin,
    DCPMetadataBuilderMixin,
    build_pcp_ordered_slot_mapping,
)
from vllm_ascend.attention.context_parallel.sfa_dcp_utils import (
    _index_select_kv_blocks,
    as_byte_view,
    build_sfa_dcp_replicated_block_table,
    build_sfa_dcp_replicated_slot_mapping,
    get_sfa_dcp_local_block_table,
    get_sfa_dcp_max_local_block_table_cols,
    get_sfa_max_num_reqs,
    get_sfa_pcp_global_metadata,
)
from vllm_ascend.attention.context_parallel.sharded_indexer import use_local_topk_output
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    _view_cache_as_operator_pages,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    PreprocessType,
    is_dcp_enabled,
    split_decodes_and_prefills,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.utils import (
    enable_sfa_dcp_force_tmajor_restore,
    enable_sfa_dcp_sharded_indexer,
    is_pd_decode_recompute_scheduler_enabled,
)


class DCPGatherContext(NamedTuple):
    """State needed to finish an async DCP query or KV all-gather."""

    gathered: torch.Tensor
    handle: torch.distributed.Work | None
    restore_perm: tuple[int, ...] | None
    split_sizes: tuple[int, ...]


@dataclass
class DCPContext:
    # Keep the complete main-cache mapping. Consumers that operate on the
    # rank-local model input take a view of the required prefix.
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    kv_gather_block_ids: torch.Tensor | None = None
    kv_gather_block_table: torch.Tensor | None = None
    gather_context: DCPGatherContext | None = None


@dataclass
class AscendSFADCPMetadata(AscendSFAMetadata):
    """SFA metadata fields used only by the DCP execution path."""

    dcp_context: DCPContext | None = None


M = TypeVar("M", bound=AscendSFAMetadata)


# SFA DCP replicated-indexer layout:
#
# - LightningIndexer cache is replicated on every DCP rank so index selection
#   can run against the full sequence and keep the same sparse topk semantics as
#   non-DCP SFA.
# - SFA KV cache remains DCP-local to preserve the KV memory saving. The sparse
#   topk indices produced from the replicated indexer view are remapped to local
#   KV indices before calling sparse flash attention.
# - BlockTable only owns the DCP-local physical layout. This builder derives its
#   replicated block table and slot mapping on demand, then stores the original
#   DCP-local view in metadata.dcp_context for KV writes and SFA reads. The
#   independent indexer builder derives its own replicated view from common
#   metadata using the same stateless address helpers.
# - The replicated view uses the same logical/kernel block size as BlockTable,
#   including hybrid block splitting.
class AscendSFADCPMetadataBuilder(
    DCPMetadataBuilderMixin,
    AscendSFAMetadataBuilder,
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = True,
    ):
        metadata_cls = metadata_cls or AscendSFADCPMetadata
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls,
            supports_dcp_with_varlen,
        )
        self.use_dcp = is_dcp_enabled()
        self.cp_kv_cache_interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        assert self.dcp_size > 1, "AscendSFADCPMetadataBuilder requires DCP world size > 1."
        if self.cp_kv_cache_interleave_size <= 0:
            raise RuntimeError(f"Invalid cp_kv_cache_interleave_size: {self.cp_kv_cache_interleave_size}")

        # PCP represents each prefill with two local DualChunkSwap segments,
        # while decode requests remain one local row. A mixed batch can
        # therefore require up to twice max_num_seqs rows on each PCP rank.
        # Full-graph FIA padding can append one additional dummy request.
        max_num_reqs = get_sfa_max_num_reqs(vllm_config)
        self.dcp_local_seq_lens_buf = torch.empty(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self._local_block_table_buffers: dict[int, torch.Tensor] = {}
        self.replicated_view_block_size = self.kernel_block_size
        physical_block_size = getattr(kv_cache_spec, "block_table_block_size", None) or kv_cache_spec.block_size
        if physical_block_size % self.replicated_view_block_size != 0:
            raise RuntimeError(
                "SFA replicated view requires the KV cache block size "
                f"({physical_block_size}) to be divisible by "
                f"{self.replicated_view_block_size}."
            )
        self.blocks_per_phys_block = physical_block_size // self.replicated_view_block_size
        max_num_input_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_model_len = vllm_config.model_config.max_model_len
        total_cp_size = self.dcp_size
        # The generic vLLM BlockTable may expose global-width storage, while
        # the DCP physical KV layout only populates rank-local block columns.
        self.max_local_block_table_cols = get_sfa_dcp_max_local_block_table_cols(
            max_model_len,
            physical_block_size,
            total_cp_size,
            self.blocks_per_phys_block,
        )
        max_replicated_block_table_cols = self.max_local_block_table_cols * total_cp_size
        self.block_table_replicated_view_buf: torch.Tensor = torch.empty(
            (max_num_reqs, max_replicated_block_table_cols),
            dtype=torch.int32,
            device=device,
        )
        self.arange_buffer: torch.Tensor = torch.arange(
            max_replicated_block_table_cols,
            dtype=torch.int32,
            device=device,
        )
        dcp_group = get_dcp_group()
        collective_ranks = sorted(dcp_group.ranks)
        # ``dcp_group.ranks`` is in logical PCP-first DCP order, but
        # ``torch.distributed.new_group`` sorts its global ranks and AllGather
        # writes payload segments in that ProcessGroup order. For example,
        # with PCP=2, TP=4, and DCP=8:
        #   logical DCP ranks:       [0, 4, 1, 5, 2, 6, 3, 7]
        #   AllGather segment ranks: [0, 1, 2, 3, 4, 5, 6, 7]
        # Logical DCP rank positions therefore address gathered segments
        # [0, 4, 1, 5, 2, 6, 3, 7]. Store this permutation for block-table
        # offsets so the large gathered KV tensor can stay in HCCL order.
        self.dcp_collective_rank_order = torch.tensor(
            [collective_ranks.index(rank) for rank in dcp_group.ranks],
            dtype=torch.int32,
            device=device,
        )
        self.slot_mapping_replicated_view_buf: torch.Tensor = torch.empty(
            (max_num_input_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def _get_dcp_local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        return get_dcp_local_seq_lens(
            seq_lens,
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
        )

    def _get_dcp_local_block_table(self, block_table: torch.Tensor, num_reqs: int) -> torch.Tensor:
        local = block_table[:num_reqs, : self.max_local_block_table_cols]
        if local.is_contiguous():
            return get_sfa_dcp_local_block_table(block_table, num_reqs, self.max_local_block_table_cols)
        # Metadata is rebuilt outside FULL capture. Refresh the captured compact
        # storage, keeping distinct source tables (including draft steps) separate.
        key = block_table.data_ptr()
        out = self._local_block_table_buffers.get(key)
        if out is None:
            out = torch.empty(
                (self.dcp_local_seq_lens_buf.numel(), self.max_local_block_table_cols),
                dtype=block_table.dtype,
                device=block_table.device,
            )
            self._local_block_table_buffers[key] = out
        if num_reqs > out.shape[0]:
            raise ValueError(f"DCP block table request count {num_reqs} exceeds capacity {out.shape[0]}")
        return get_sfa_dcp_local_block_table(
            block_table,
            num_reqs,
            self.max_local_block_table_cols,
            out=out[:num_reqs],
        )

    def _ensure_replicated_view_buffers(
        self,
        num_reqs: int,
        num_input_tokens: int,
        local_block_table_cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_table_cols = local_block_table_cols * self.dcp_size
        if (
            self.block_table_replicated_view_buf.shape[0] < num_reqs
            or self.block_table_replicated_view_buf.shape[1] < block_table_cols
        ):
            raise RuntimeError(
                f"Replicated view buffer is too small: "
                f"block_table_replicated_view_buf.shape={self.block_table_replicated_view_buf.shape}, "
                f"num_reqs={num_reqs}, block_table_cols={block_table_cols}"
            )
        if self.slot_mapping_replicated_view_buf.shape[0] < num_input_tokens:
            raise RuntimeError(
                f"Replicated view buffer is too small: "
                f"slot_mapping_replicated_view_buf.shape={self.slot_mapping_replicated_view_buf.shape}, "
                f"num_input_tokens={num_input_tokens}"
            )
        return (
            self.block_table_replicated_view_buf[:num_reqs, :block_table_cols],
            self.arange_buffer[:block_table_cols],
            self.slot_mapping_replicated_view_buf[:num_input_tokens],
        )

    def _build_block_table_replicated_view(
        self,
        dcp_block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = dcp_block_table.shape[0]
        local_block_table_cols = dcp_block_table.shape[1]
        block_table_replicated_view, replicated_col_idx, _ = self._ensure_replicated_view_buffers(
            num_reqs,
            0,
            local_block_table_cols,
        )

        return build_sfa_dcp_replicated_block_table(
            dcp_block_table,
            seq_lens,
            block_table_replicated_view,
            replicated_col_idx,
            self.dcp_size,
            self.blocks_per_phys_block,
        )

    def _build_slot_mapping_replicated_view(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        block_table_replicated_view: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        local_block_table_cols = block_table_replicated_view.shape[1] // self.dcp_size
        _, _, slot_mapping_replicated_view = self._ensure_replicated_view_buffers(
            num_reqs,
            num_input_tokens,
            local_block_table_cols,
        )
        return build_sfa_dcp_replicated_slot_mapping(
            common_attn_metadata,
            block_table_replicated_view,
            slot_mapping_replicated_view,
            self.replicated_view_block_size,
            self.device,
        )

    def _build_compact_kv_gather_metadata(
        self,
        dcp_block_table: torch.Tensor,
        *,
        global_dcp_block_table: torch.Tensor | None = None,
        global_dcp_num_blocks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the compact cross-DCP KV view used by prefill attention."""
        valid_block_ids, compact_block_table = dcp_block_table.flatten().unique(return_inverse=True)
        compact_block_table = compact_block_table.view_as(dcp_block_table)
        num_blocks = valid_block_ids.shape[0]
        dcp_collective_rank_order = self.dcp_collective_rank_order[: self.dcp_size]
        remapped_block_table = (
            compact_block_table.unsqueeze(-1)
            + (dcp_collective_rank_order * num_blocks).view(1, 1, -1).to(dcp_block_table)
        ).reshape(dcp_block_table.shape[0], -1)
        return valid_block_ids, remapped_block_table.to(torch.int32)

    def _build_with_metadata_view(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        build_metadata: Callable[[], AscendSFAMetadata],
        global_dcp_block_table: torch.Tensor | None = None,
        global_dcp_num_blocks: torch.Tensor | None = None,
    ) -> AscendSFAMetadata:
        dcp_slot_mapping = common_attn_metadata.slot_mapping
        full_dcp_block_table = common_attn_metadata.block_table_tensor
        num_reqs = common_attn_metadata.num_reqs
        dcp_block_table = self._get_dcp_local_block_table(full_dcp_block_table, num_reqs)
        block_table_replicated_view = self._build_block_table_replicated_view(
            dcp_block_table,
            common_attn_metadata.seq_lens,
        )
        slot_mapping_replicated_view = self._build_slot_mapping_replicated_view(
            common_attn_metadata,
            block_table_replicated_view,
        )

        common_attn_metadata.slot_mapping = slot_mapping_replicated_view
        common_attn_metadata.block_table_tensor = block_table_replicated_view
        try:
            metadata = build_metadata()
        finally:
            common_attn_metadata.slot_mapping = dcp_slot_mapping
            common_attn_metadata.block_table_tensor = full_dcp_block_table

        assert isinstance(metadata, AscendSFADCPMetadata)
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens
        if dcp_local_seq_lens is None:
            dcp_local_seq_lens = self._get_dcp_local_seq_lens(metadata.seq_lens)
        local_seq_lens_src = dcp_local_seq_lens[:num_reqs].to(
            device=self.device,
            dtype=torch.int32,
            non_blocking=True,
        )
        self.dcp_local_seq_lens_buf[:num_reqs].copy_(local_seq_lens_src, non_blocking=True)
        local_seq_lens = self.dcp_local_seq_lens_buf[:num_reqs]

        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            treat_short_extends_as_decodes=(
                self.use_dcp and is_pd_decode_recompute_scheduler_enabled(self.vllm_config)
            ),
        )
        kv_gather_block_ids = None
        kv_gather_block_table = None
        if (num_prefills > 0 or metadata.pcp_has_global_prefill) and not enable_sfa_dcp_sharded_indexer(
            self.vllm_config
        ):
            kv_gather_block_ids, kv_gather_block_table = self._build_compact_kv_gather_metadata(
                dcp_block_table,
                global_dcp_block_table=global_dcp_block_table,
                global_dcp_num_blocks=global_dcp_num_blocks,
            )
        metadata.dcp_context = DCPContext(
            slot_mapping=dcp_slot_mapping,
            block_table=dcp_block_table,
            seq_lens=local_seq_lens,
            kv_gather_block_ids=kv_gather_block_ids,
            kv_gather_block_table=kv_gather_block_table,
        )
        metadata.num_decodes = num_decodes
        metadata.num_decode_tokens = num_decode_tokens
        metadata.num_prefills = num_prefills
        return metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        **kwargs,
    ) -> AscendSFAMetadata:
        if attn_state not in {
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        }:
            raise NotImplementedError("Currently we only support building dummy metadata for DecodeOnly state")

        attn_metadata = self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )
        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendSFAPCPDCPMetadataBuilder(AscendSFADCPMetadataBuilder):
    """Adds PCP's main-cache token layout to the DCP metadata view."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = True,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls or AscendSFADCPMetadata,
            supports_dcp_with_varlen,
        )
        pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        max_num_input_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.pcp_indexer_slot_mapping_buf = torch.empty(
            (max_num_input_tokens * pcp_size,),
            dtype=torch.int32,
            device=device,
        )

    def _build_compact_kv_gather_metadata(
        self,
        dcp_block_table: torch.Tensor,
        *,
        global_dcp_block_table: torch.Tensor | None = None,
        global_dcp_num_blocks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if global_dcp_block_table is None:
            return super()._build_compact_kv_gather_metadata(dcp_block_table)

        # PCP ranks can have different local request rows, even with zeroed
        # block-table tails. Use the global request view so every DCP sender
        # packs the same block IDs in the same order.
        if global_dcp_num_blocks is None:
            raise ValueError("PCP+DCP compact KV metadata requires valid block counts for the global block table.")
        valid_mask = self.arange_buffer[: global_dcp_block_table.shape[1]].unsqueeze(
            0
        ) < global_dcp_num_blocks.unsqueeze(1)
        # Select allocated entries without modifying shared tables or
        # introducing a block-0 placeholder for unused columns.
        valid_block_ids = global_dcp_block_table[valid_mask].unique()
        compact_block_table = torch.searchsorted(valid_block_ids, dcp_block_table.contiguous())
        num_blocks = valid_block_ids.shape[0]
        dcp_collective_rank_order = self.dcp_collective_rank_order[: self.dcp_size]
        remapped_block_table = (
            compact_block_table.unsqueeze(-1)
            + (dcp_collective_rank_order * num_blocks).view(1, 1, -1).to(dcp_block_table)
        ).reshape(dcp_block_table.shape[0], -1)
        return valid_block_ids, remapped_block_table.to(torch.int32)

    def _build_pcp_ordered_indexer_slot_mapping(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        pcp_context: Any,
        pcp_cache_group_idx: int,
    ) -> torch.Tensor:
        """Build this receiver's Indexer addresses in PCP token order."""
        global_common_attn_metadata = get_sfa_pcp_global_metadata(
            common_attn_metadata,
            pcp_context,
            pcp_cache_group_idx,
        )
        dcp_block_table = self._get_dcp_local_block_table(
            global_common_attn_metadata.block_table_tensor,
            global_common_attn_metadata.num_reqs,
        )
        replicated_block_table = self._build_block_table_replicated_view(
            dcp_block_table,
            global_common_attn_metadata.seq_lens,
        )
        global_slot_mapping = self._build_slot_mapping_replicated_view(
            global_common_attn_metadata,
            replicated_block_table,
        )

        return build_pcp_ordered_slot_mapping(
            global_slot_mapping,
            pcp_context,
            self.pcp_indexer_slot_mapping_buf,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendSFAMetadata:
        pcp_context = common_attn_metadata.pcp_context
        pcp_cache_group_idx = common_attn_metadata.pcp_cache_group_idx
        if pcp_context is None:
            # Decode-only graph capture builds dummy metadata without running
            # PCP's prefill gather, so no PCP context is available or needed.
            return super().build(
                common_prefix_len,
                common_attn_metadata,
                fast_build,
                **kwargs,
            )
        pcp_ordered_indexer_slot_mapping = None
        if bool(pcp_context.global_batch.is_prefilling_np.any()):
            if pcp_cache_group_idx is None:
                raise RuntimeError("PCP+DCP prefill requires the PCP cache-group index.")
            pcp_ordered_indexer_slot_mapping = self._build_pcp_ordered_indexer_slot_mapping(
                common_attn_metadata, pcp_context, pcp_cache_group_idx
            )
        global_dcp_block_table = None
        global_dcp_num_blocks = None
        if pcp_cache_group_idx is not None:
            global_dcp_block_table = self._get_dcp_local_block_table(
                pcp_context.global_block_tables[pcp_cache_group_idx],
                pcp_context.global_batch.num_reqs,
            )
            if pcp_context.global_block_table_num_blocks is not None:
                global_dcp_num_blocks = pcp_context.global_block_table_num_blocks[pcp_cache_group_idx]
        metadata = self._build_with_metadata_view(
            common_attn_metadata,
            lambda: self._build(common_attn_metadata, draft_index=None, pcp_context=pcp_context),
            global_dcp_block_table=global_dcp_block_table,
            global_dcp_num_blocks=global_dcp_num_blocks,
        )
        assert isinstance(metadata, AscendSFADCPMetadata)
        if pcp_ordered_indexer_slot_mapping is not None:
            metadata.pcp_slot_mapping = pcp_ordered_indexer_slot_mapping
        if metadata.num_prefills or metadata.pcp_has_global_prefill:
            assert metadata.dcp_context is not None
            group = get_pcp_group()
            num_tokens = metadata.num_input_tokens
            rank_slots = metadata.dcp_context.slot_mapping[: group.world_size * num_tokens].view(
                group.world_size, num_tokens
            )
            num_decode_tokens = metadata.num_decode_tokens
            local_slots = rank_slots[group.rank_in_group].contiguous()
            if num_decode_tokens and group.rank_in_group != 0:
                # Replicated decode slots are masked outside rank 0, but each
                # rank still writes its locally computed decode KV.
                local_slots = torch.cat((rank_slots[0, :num_decode_tokens], local_slots[num_decode_tokens:]))
            metadata.pcp_prolog_local_slots = local_slots
            metadata.pcp_prolog_global_slots = rank_slots[:, num_decode_tokens:].reshape(-1)
            # Native KV gather keeps one decode prefix before all prefill rows.
            metadata.slot_mapping = metadata.pcp_prolog_global_slots
            if num_decode_tokens:
                metadata.slot_mapping = torch.cat((rank_slots[0, :num_decode_tokens], metadata.slot_mapping))
        return metadata


class AscendSFADCPImpl(DCPImplMixin, AscendSFAImpl):
    """DCP execution, inheriting the shared ordinary and PCP preprocessing."""

    can_return_lse_for_decode: bool = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Legacy MLAPO has no DCP cache writer; K3.1 PROLOG_V3 uses local owned slots.
        self.enable_mlapo = False
        self._sharded_indexer = enable_sfa_dcp_sharded_indexer(self.vllm_config)
        # Shared Top-K buffers carry the same coordinates on every layer.
        self._local_topk = use_local_topk_output(
            self._sharded_indexer,
            get_pcp_group().world_size,
            False,  # Legacy TP-based DSA-CP was removed from main.
            turboquant=self.enable_sparse_sfa_turboquant,
        )
        self._dcp_interleave_size = self.vllm_config.parallel_config.cp_kv_cache_interleave_size
        if self._dcp_interleave_size <= 0:
            raise RuntimeError(f"Invalid cp_kv_cache_interleave_size: {self._dcp_interleave_size}")
        self._dcp_index_topk = 0
        for config in (
            getattr(self.vllm_config.model_config, "hf_text_config", None),
            getattr(self.vllm_config.model_config, "hf_config", None),
        ):
            index_topk = getattr(config, "index_topk", None)
            if isinstance(index_topk, int) and index_topk > 0:
                self._dcp_index_topk = index_topk
                break
        if self._dcp_index_topk <= 0:
            raise RuntimeError("index_topk must be set in the model config for DCP SFA.")

    def _fused_preprocess_type(self) -> PreprocessType | None:
        # K3.1 without PCP writes owner-masked DCP slots directly through PROLOG_V3.
        if self.is_kimi_k31 and not self.pcp_enabled:
            return super()._fused_preprocess_type()
        return None

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
        *,
        defer_wait: bool = False,
    ) -> tuple | Callable[[], tuple]:
        if self.pcp_enabled and self._has_prefill(attn_metadata):
            assert isinstance(attn_metadata, AscendSFADCPMetadata)
            assert attn_metadata.dcp_context is not None
            slots = attn_metadata.dcp_context.slot_mapping
        return super().exec_kv(kv_no_split, cos, sin, kv_cache, slots, attn_metadata, defer_wait=defer_wait)

    def _get_sfa_kv_slot_mapping(self, attn_metadata: M) -> torch.Tensor:
        assert isinstance(attn_metadata, AscendSFADCPMetadata)
        assert attn_metadata.dcp_context is not None
        slots = attn_metadata.dcp_context.slot_mapping
        if self.pcp_enabled and self._has_prefill(attn_metadata):
            return attn_metadata.slot_mapping
        return slots[: attn_metadata.num_input_tokens]

    def _on_query_ready(self, ql_nope: torch.Tensor, q_pe: torch.Tensor | None, attn_metadata: M) -> None:
        # Sharded indexers use local KV and gathered Q even during prefill.
        if self._has_prefill(attn_metadata) and not self._sharded_indexer:
            return
        assert isinstance(attn_metadata, AscendSFADCPMetadata)
        assert attn_metadata.dcp_context is not None
        attn_metadata.dcp_context.gather_context = self._start_dcp_query_gather(ql_nope, q_pe)

    def _on_kv_cache_written(self, kv_cache: tuple[torch.Tensor, ...], attn_metadata: M) -> None:
        if self._has_prefill(attn_metadata) and not self._sharded_indexer:
            self._record_dcp_kv_gather_context(kv_cache, attn_metadata)

    @staticmethod
    def _has_prefill(attn_metadata: M) -> bool:
        # PCP ranks with no local prefill still join the DCP KV gather and use
        # the same attention path as ranks that received prefill tokens.
        return attn_metadata.num_prefills > 0 or attn_metadata.pcp_has_global_prefill

    def _record_dcp_kv_gather_context(
        self,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
    ) -> None:
        """Start the compact KV all-gather used by prefill/mixed DCP batches."""
        if not self._has_prefill(attn_metadata):
            return
        assert isinstance(attn_metadata, AscendSFADCPMetadata)
        assert attn_metadata.dcp_context is not None, "DCP SFA requires attn_metadata.dcp_context."
        assert self.dcp_group is not None, "DCP SFA requires dcp_group when dcp_size > 1."

        valid_block_ids = attn_metadata.dcp_context.kv_gather_block_ids
        block_table = attn_metadata.dcp_context.kv_gather_block_table
        assert valid_block_ids is not None and block_table is not None
        cache = kv_cache[0]
        if self.qk_rope_head_dim == 0:
            cache = _view_cache_as_operator_pages(cache, attn_metadata.block_size)
        kv = _index_select_kv_blocks(cache, valid_block_ids)
        split_sizes: tuple[int, ...]
        if self.uses_packed_sfa_main_cache or self.qk_rope_head_dim == 0:
            # Packed SFA formats store nope, rope, and quantization data in one
            # SFA KV cache. The remaining cache entries belong to the indexer
            # and must not participate in the DCP SFA KV all-gather.
            gather_input = kv.contiguous()
            split_sizes = (kv.shape[-1],)
        else:
            if len(kv_cache) < 2:
                raise RuntimeError("DCP SFA KV all-gather requires nope and rope KV caches.")
            key_rope = _index_select_kv_blocks(kv_cache[1], valid_block_ids)
            if kv.shape[:-1] != key_rope.shape[:-1] or kv.dtype != key_rope.dtype:
                raise RuntimeError(
                    "Cannot fuse DCP KV gather for KV/nope and KV/rope caches with "
                    f"shapes {tuple(kv.shape)} / {tuple(key_rope.shape)} and dtypes {kv.dtype} / {key_rope.dtype}."
                )
            gather_input = torch.cat([kv, key_rope], dim=-1).contiguous()
            split_sizes = (kv.shape[-1], key_rope.shape[-1])
        attn_metadata.dcp_context.gather_context = self._start_dcp_gather(
            gather_input,
            dim=0,
            split_sizes=split_sizes,
        )

    def _start_dcp_gather(
        self,
        x: torch.Tensor,
        dim: int,
        split_sizes: tuple[int, ...],
    ) -> DCPGatherContext:
        # HCCL does not transport FP8 directly; gather the packed cache as bytes.
        transport = as_byte_view(x)
        gathered, handle, restore_perm = self._all_gather_dim_async(transport, dim)
        if transport.dtype != x.dtype:
            gathered = gathered.view(x.dtype)
        return DCPGatherContext(
            gathered=gathered,
            handle=handle,
            restore_perm=restore_perm,
            split_sizes=split_sizes,
        )

    @staticmethod
    def _finish_dcp_packed_gather(
        context: DCPGatherContext,
        keep_view: bool = False,
    ) -> torch.Tensor:
        if context.handle is not None:
            context.handle.wait()
        gathered = context.gathered
        if context.restore_perm is not None:
            gathered = gathered.permute(context.restore_perm)
            if not keep_view:
                gathered = gathered.contiguous()
        return gathered

    @classmethod
    def _finish_dcp_gather(
        cls,
        context: DCPGatherContext,
        keep_view: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        gathered = cls._finish_dcp_packed_gather(context, keep_view=keep_view)
        return torch.split(gathered, context.split_sizes, dim=-1)

    def _finish_dcp_kv_gather(self, kv_cache, attn_metadata: AscendSFADCPMetadata) -> tuple[torch.Tensor, ...]:
        dcp_context = attn_metadata.dcp_context
        assert dcp_context is not None
        gather_context = dcp_context.gather_context
        dcp_context.gather_context = None
        if gather_context is None:
            # Normal forward starts this after cache writes to overlap indexer
            # work. Direct attention callers still need to start it here.
            self._record_dcp_kv_gather_context(kv_cache, attn_metadata)
            gather_context = dcp_context.gather_context
            dcp_context.gather_context = None
        assert gather_context is not None
        return self._finish_dcp_gather(gather_context)

    def _all_gather_dim_async(
        self,
        x: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, torch.distributed.Work | None, tuple[int, ...] | None]:
        assert self.dcp_group is not None
        if dim == 0:
            gathered, handle = all_gather_async(x.contiguous(), self.dcp_group)
            return gathered, handle, None

        perm = (dim, *[i for i in range(x.dim()) if i != dim])
        restore_perm = tuple(perm.index(i) for i in range(x.dim()))
        gathered, handle = all_gather_async(x.permute(perm).contiguous(), self.dcp_group)
        return gathered, handle, restore_perm

    def _remap_sparse_indices(self, topk_indices: torch.Tensor) -> torch.Tensor:
        if self.dcp_size <= 1:
            return topk_indices

        topk_count = topk_indices.shape[-1]
        if topk_count > self._dcp_index_topk:
            raise RuntimeError(
                f"topk_indices last dimension ({topk_count}) exceeds configured index_topk ({self._dcp_index_topk})."
            )
        if topk_indices.numel() == 0:
            return topk_indices

        from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton

        return remap_sparse_indices_triton(topk_indices, self.dcp_size, self.dcp_rank, self._dcp_interleave_size)

    def _merge_dcp_outputs(
        self,
        sfa_output: torch.Tensor,
        softmax_lse: torch.Tensor,
        softmax_sum: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The replicated MTP draft disables logical PCP, but still shares the
        # target's physical PCP ranks and DCP-sharded KV cache.
        if get_pcp_group().world_size > 1:
            pcp_group = get_pcp_group()
            tp_group = get_tp_group()
            tp_size = tp_group.world_size if self.dcp_size > pcp_group.world_size else 1
            return torch.ops.vllm.dcp_a2a_fused(
                sfa_output,
                softmax_lse,
                tp_size,
                1,
                tp_group.unique_name,
                pcp_group.unique_name,
                softmax_sum=softmax_sum,
            )

        assert self.dcp_group is not None, "DCP output All2All requires dcp_group when dcp_size > 1."
        return torch.ops.vllm.dcp_a2a_fused(
            sfa_output,
            softmax_lse,
            self.dcp_size,
            1,
            self.dcp_group.unique_name,
            softmax_sum=softmax_sum,
        )

    def _start_dcp_query_gather(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor | None,
    ) -> DCPGatherContext:
        if q_pe is None:
            q_pe = ql_nope[..., :0]
        assert self.dcp_group is not None, "DCP query gather requires dcp_group when dcp_size > 1."
        if ql_nope.shape[:-1] != q_pe.shape[:-1] or ql_nope.dtype != q_pe.dtype:
            raise RuntimeError(
                "Cannot fuse DCP query gather for ql_nope/q_pe with "
                f"shapes {tuple(ql_nope.shape)} / {tuple(q_pe.shape)} "
                f"and dtypes {ql_nope.dtype} / {q_pe.dtype}."
            )

        # Query heads are replicated across physical PCP ranks, including
        # when the MTP draft has logical PCP disabled. Gather only distinct
        # TP head shards; DCP == PCP already owns one complete TP shard.
        from vllm_ascend.ops.triton.query_gather_prep import prep_query_head_major

        fused_q = prep_query_head_major(ql_nope, q_pe)
        pcp_size = get_pcp_group().world_size
        if pcp_size > 1 and self.dcp_size == pcp_size:
            gathered, handle = fused_q, None
        else:
            group = get_tp_group() if pcp_size > 1 else self.dcp_group
            gathered, handle = all_gather_async(fused_q, group)
        return DCPGatherContext(
            gathered=gathered,
            handle=handle,
            restore_perm=(1, 0, 2),
            split_sizes=(ql_nope.shape[-1], q_pe.shape[-1]),
        )

    def _mask_nope_dcp_padding(self, output, query_lens, softmax_max=None, softmax_sum=None):
        if self.qk_rope_head_dim == 0:
            valid = torch.arange(output.shape[0], device=output.device) < query_lens[-1]
            output = output.masked_fill(~valid[:, None, None], 0)
            if softmax_max is not None:
                softmax_max = softmax_max.masked_fill(~valid[None, :, None], 0)
                softmax_sum = softmax_sum.masked_fill(~valid[None, :, None], 1)
        return output, softmax_max, softmax_sum

    def _execute_tq_dcp_sfa(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
    ):
        # Mirrors AscendSFAImpl for the non-DCP case, and additionally asks for
        # softmax LSE so decode can merge partial results across CP ranks.
        dcp_context = attn_metadata.dcp_context

        if self._has_prefill(attn_metadata):
            # Prefill keeps Q local and attends over the gathered KV, so no LSE.
            gathered_kv_cache = self._finish_dcp_kv_gather(kv_cache, attn_metadata)
            block_table = dcp_context.kv_gather_block_table
            assert block_table is not None
            attn_out, _, _ = self._turboquant_sfa(
                self._tq_rotate_query(ql_nope, q_pe),
                gathered_kv_cache[0],
                topk_indices,
                block_table,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                sparse_mode=3,
            )
            assert self.turboquant is not None
            output = self.turboquant.inverse(attn_out)
            return self._mask_nope_dcp_padding(output, actual_seq_lengths_query)[0]

        # Decode attends over this rank's KV shard only, so the partial results
        # are recombined by _merge_dcp_outputs using the LSE.
        gather_context = dcp_context.gather_context
        dcp_context.gather_context = None
        if gather_context is None:
            gather_context = self._start_dcp_query_gather(ql_nope, q_pe)
        topk_indices = self._remap_sparse_indices(topk_indices)
        ql_nope, q_pe = self._finish_dcp_gather(gather_context)
        attn_out, softmax_max, softmax_sum = self._turboquant_sfa(
            self._tq_rotate_query(ql_nope, q_pe),
            kv_cache[0],
            topk_indices,
            dcp_context.block_table,
            actual_seq_lengths_query,
            dcp_context.seq_lens,
            sparse_mode=0,
            return_softmax_lse=True,
        )
        assert softmax_max is not None
        assert softmax_sum is not None
        attn_out, softmax_max, softmax_sum = self._mask_nope_dcp_padding(
            attn_out, actual_seq_lengths_query, softmax_max, softmax_sum
        )
        softmax_lse = softmax_max.to(torch.float32) + torch.log(softmax_sum.to(torch.float32))
        softmax_lse = softmax_lse.permute(1, 0, 2).reshape(softmax_lse.shape[1], -1, 1)
        output_dtype = attn_out.dtype
        output = self._merge_dcp_outputs(attn_out, softmax_lse)
        # The merge runs in Hadamard space, so rotate back once at the end.
        assert self.turboquant is not None
        output = self.turboquant.inverse(output)
        return output.to(output_dtype)

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
        assert attn_metadata.dcp_context is not None, "DCP SFA requires attn_metadata.dcp_context."
        assert self.dcp_group is not None, "DCP SFA requires dcp_group when dcp_size > 1."
        dcp_context = attn_metadata.dcp_context
        if self.qk_rope_head_dim == 0:
            kv_cache = (_view_cache_as_operator_pages(kv_cache[0], attn_metadata.block_size), *kv_cache[1:])
            if q_pe is None and self.enable_sparse_sfa_turboquant:
                q_pe = ql_nope[..., :0]
        if self.enable_sparse_sfa_turboquant:
            return self._execute_tq_dcp_sfa(
                ql_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
            )
        if self._has_prefill(attn_metadata) and not self._sharded_indexer:
            # Splitting the packed KV buffer leaves non-contiguous inner strides.
            gathered_kv_cache = tuple(t.contiguous() for t in self._finish_dcp_kv_gather(kv_cache, attn_metadata))
            block_table = dcp_context.kv_gather_block_table
            assert block_table is not None
            # The gathered KV cache is complete, so each rank can attend with
            # its local Q heads/tokens directly; no Q all-gather, sparse-index
            # remap, LSE, or output all-to-all merge is required.
            attn_output = DeviceOperator.execute_sparse_flash_attention_process(
                self,
                ql_nope,
                q_pe,
                gathered_kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                block_table=block_table,
                sparse_mode=3,
                return_lse=False,
            )
            return self._mask_nope_dcp_padding(attn_output, actual_seq_lengths_query)[0]

        gather_context = dcp_context.gather_context
        dcp_context.gather_context = None
        if gather_context is None:
            gather_context = self._start_dcp_query_gather(ql_nope, q_pe)
        if not self._local_topk:
            topk_indices = self._remap_sparse_indices(topk_indices)
        # Keep the gathered query as a strided view of the head-major gather
        # output. The split query fragments handed to npu_sparse_flash_
        # attention have always been strided views (torch.split never copies),
        # so the old permute+contiguous call only materialized the t-major
        # [T, H, D] storage behind the views;
        # additional-config sfa_dcp_force_tmajor_restore=true keeps the
        # t-major materialization selectable for comparison runs.
        packed_query_kwargs = {}
        if kv_cache[0].dtype in (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2):
            # Quantized SFA consumes the complete gathered query on A3/A5.
            # The adaptor materializes contiguous TND storage when needed.
            packed_query_kwargs["packed_query"] = self._finish_dcp_packed_gather(
                gather_context,
                keep_view=not enable_sfa_dcp_force_tmajor_restore(),
            )
        else:
            ql_nope, q_pe = self._finish_dcp_gather(
                gather_context,
                keep_view=not enable_sfa_dcp_force_tmajor_restore(),
            )
        sfa_output, softmax_max, softmax_sum = DeviceOperator.execute_sparse_flash_attention_process(
            self,
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            dcp_context.seq_lens,
            block_table=dcp_context.block_table,
            # The indexer already applies global causal visibility, including
            # distributed local candidates on the sharded-indexer path.
            # After DCP remaps topk indices to local KV positions, local KV
            # length no longer shares the same coordinate system as global
            # query length, so SFA must not apply its right-down causal crop.
            sparse_mode=0,
            return_lse=True,
            **packed_query_kwargs,
        )
        sfa_output, softmax_max, softmax_sum = self._mask_nope_dcp_padding(
            sfa_output, actual_seq_lengths_query, softmax_max, softmax_sum
        )
        # Form LSE while packing on every device; keep max/sum in FP32.
        softmax_lse = softmax_max
        sum_for_pack = softmax_sum.permute(1, 0, 2).reshape(softmax_sum.shape[1], -1, 1)
        softmax_lse = softmax_lse.permute(1, 0, 2).reshape(softmax_lse.shape[1], -1, 1)
        output_dtype = sfa_output.dtype
        output = self._merge_dcp_outputs(
            sfa_output,
            softmax_lse,
            softmax_sum=sum_for_pack,
        )
        return output.to(output_dtype)
