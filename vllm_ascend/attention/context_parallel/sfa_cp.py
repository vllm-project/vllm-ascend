from collections.abc import Callable
from typing import Any, TypeVar

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.kv_cache_interface import AttentionSpec

import vllm_ascend.ops.triton.dcp.dcp_a2a  # noqa: F401
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import (
    DCPMetadataBuilderMixin,
    build_pcp_ordered_slot_mapping,
)
from vllm_ascend.attention.context_parallel.sfa_dcp_utils import (
    build_sfa_dcp_replicated_block_table,
    build_sfa_dcp_replicated_slot_mapping,
    get_sfa_dcp_local_block_table,
    get_sfa_dcp_max_local_block_table_cols,
    get_sfa_pcp_global_metadata,
)
from vllm_ascend.attention.sfa_v1 import (
    AscendSFADCPMetadata,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    DCPContext,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    is_dcp_enabled,
    split_decodes_and_prefills,
)
from vllm_ascend.utils import (
    is_pd_decode_recompute_scheduler_enabled,
)

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
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        if vllm_config.parallel_config.prefill_context_parallel_size > 1:
            max_num_reqs *= 2
        max_num_reqs += 1
        self.dcp_local_seq_lens_buf = torch.empty(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.replicated_view_block_size = self.kernel_block_size
        if kv_cache_spec.block_size % self.replicated_view_block_size != 0:
            raise RuntimeError(
                "SFA replicated view requires the KV cache block size "
                f"({kv_cache_spec.block_size}) to be divisible by "
                f"{self.replicated_view_block_size}."
            )
        self.blocks_per_phys_block = kv_cache_spec.block_size // self.replicated_view_block_size
        max_num_input_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_model_len = vllm_config.model_config.max_model_len
        total_cp_size = self.dcp_size
        # The generic vLLM BlockTable may expose global-width storage, while
        # the DCP physical KV layout only populates rank-local block columns.
        self.max_local_block_table_cols = get_sfa_dcp_max_local_block_table_cols(
            max_model_len,
            kv_cache_spec.block_size,
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
        return get_sfa_dcp_local_block_table(
            block_table,
            num_reqs,
            self.max_local_block_table_cols,
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
        if num_prefills > 0 or metadata.pcp_has_global_prefill:
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
