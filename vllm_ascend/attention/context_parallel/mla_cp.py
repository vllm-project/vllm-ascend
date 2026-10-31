from dataclasses import dataclass
from math import lcm
from typing import NamedTuple

import numpy as np
import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

# isort: off
from vllm_ascend.attention.mla_v1 import (
    ChunkedContextMetadata,
    DecodeMLAPreprocessResult,
    AscendMLADecodeMetadata,
    AscendMLAImpl,
    AscendMLAMetadata,
    AscendMLAMetadataBuilder,
)
# isort: on

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.context_parallel.common_cp import (
    CPKVScope,
    DCPImplMixin,
    DCPMetadataBuilderMixin,
    use_history_current_split_decode,
)
from vllm_ascend.attention.utils import MLAPO_MAX_SUPPORTED_TOKENS, AscendCommonAttentionMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.ops.triton.dcp.dcp_a2a import fused_dcp_lse_combine
from vllm_ascend.utils import cp_decode_comm_stream, weak_ref_tensors


class MLASplitAttentionGraphParams(NamedTuple):
    """One split attention task and its layer metadata identity."""

    attention_params: tuple
    attention_kind: CPKVScope
    layer_name: str


@dataclass
class DCPChunkedContextMetadata(ChunkedContextMetadata):
    """MLA chunk metadata for DCP-local context shards."""

    padded_chunk_seq_lens_npu: torch.Tensor = None
    padded_local_chunk_seq_lens: list[list[int]] | None = None
    local_context_lens_allranks: list[list[int]] | None = None
    padded_local_cu_seq_lens: torch.Tensor = None
    cu_seq_lens_lst: list[list[int]] | None = None
    chunk_size: int | None = None
    padded_local_cu_seq_lens_lst: list[list[int]] | None = None
    pcp_global_req_indices: list[int] | None = None
    pcp_global_block_table: torch.Tensor | None = None
    # Per-chunk offsets in the gathered buffer, indexed by logical DCP rank.
    kv_rank_offsets: list[list[int]] | None = None


@dataclass
class AscendMLADCPDecodeMetadata(AscendMLADecodeMetadata):
    """MLA decode metadata fields used only by DCP."""

    cp_seq_len: list[int] | None = None
    dcp_mtp_attn_mask: torch.Tensor = None
    cp_history_seq_len: list[int] | None = None

    def update_dcp_seq_lens_cpu(
        self,
        seq_lens_cpu: torch.Tensor,
        dcp_local_seq_lens_cpu: torch.Tensor,
        query_lens_cpu: torch.Tensor,
        *,
        dcp_size: int,
        dcp_rank: int,
        cp_kv_cache_interleave_size: int,
    ) -> None:
        """Consume producer-local lengths and derive MLA's cached history."""
        self.cp_seq_len = dcp_local_seq_lens_cpu.tolist()
        # Queries must be subtracted globally before partitioning history.
        history_lens = (seq_lens_cpu - query_lens_cpu).clamp(min=0)
        cp_history_seq_len = get_dcp_local_seq_lens(
            history_lens,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        ).tolist()
        assert self.actual_seq_lengths_q is not None
        num_padded = len(self.actual_seq_lengths_q) - len(cp_history_seq_len)
        self.cp_history_seq_len = cp_history_seq_len + [0] * num_padded


class AscendMlaDCPMetadataBuilder(
    DCPMetadataBuilderMixin,
    AscendMLAMetadataBuilder,
):
    """Build MLA metadata for decode context parallelism."""

    decode_metadata_cls = AscendMLADCPDecodeMetadata
    # Non-causal block drafters (e.g. DSpark) attend to all DCP-local KV.
    # Causal multi-token queries still need history/current split attention.
    supports_non_causal_multi_token_dcp = True

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        workspace_size = AscendMLAMetadataBuilder.determine_chunked_prefill_workspace_size(vllm_config)
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size > 1 and parallel_config.decode_context_parallel_size > 1:
            chunk_alignment = lcm(
                vllm_config.cache_config.block_size,
                parallel_config.cp_kv_cache_interleave_size * parallel_config.decode_context_parallel_size,
            )
            # The shared plan reserves two local fragments per global PCP
            # request. After halving the budget, each request must still fit
            # one aligned DCP chunk.
            min_workspace_size = 2 * vllm_config.scheduler_config.max_num_seqs * chunk_alignment
            workspace_size = max(workspace_size, min_workspace_size)
        return workspace_size

    def __init__(
        self,
        kv_cache_spec: AscendMLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendMLAMetadata] | None = None,
        supports_dcp_with_varlen: bool = True,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen)
        # ProcessGroup payloads follow sorted global ranks. Keep logical DCP
        # rank order for attention and translate it into source offsets below.
        dcp_ranks = get_dcp_group().ranks
        collective_ranks = sorted(dcp_ranks)
        self.dcp_collective_rank_order = [collective_ranks.index(rank) for rank in dcp_ranks]
        self.cp_local_block_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        self.cp_virtual_block_size = self.cp_local_block_size * self.dcp_size
        self.block_size = (self.block_size * self.cp_virtual_block_size) // np.gcd(
            self.block_size,
            self.cp_virtual_block_size,
        )

    def _get_pcp_prefill_kv_inputs(
        self,
    ) -> tuple[torch.Tensor | None, list[int] | None, torch.Tensor | None]:
        """Build the shared KV send inputs for PCP+DCP chunked prefill.

        Return global sequence-length bounds, local prefill-to-global request
        indices, and the global block table. The bounds give all DCP senders
        the same padded KV ranges per chunk. After all-gather, each local
        fragment selects its source request and consumes only its own history.
        Return three None values when no global PCP prefill view is needed.

        Example:
            With PCP=2 on rank 0 and no decode requests:
            - A: computed=100, query=12, global seq_len=112.
              Its two local fragments have history lengths [100, 109].
            - B: computed=200, query=20, global seq_len=220.
              Its two local fragments have history lengths [200, 215].

            Local rows are [A_first, A_last, B_first, B_last]. The result is:
            - gather_lens: [112, 220], one bound per global request.
            - local_rows: [0, 0, 1, 1], one index per local prefill fragment.
            - gather_block_table: shape (2, num_block_cols), in [A, B] order.

            The bounds include the current query; local attention still uses
            each fragment's actual history length. Short replicated prefills
            may have only one local fragment, so row counts need not be 2:1.
        """
        context = getattr(self, "_pcp_context", None)
        if not self.use_pcp or context is None or not context.global_batch.is_prefilling_np.any():
            return None, None, None
        if self._pcp_cache_group_idx is None or context.local_to_global_req_indices is None:
            raise ValueError("PCP+DCP chunks require global tables and the upstream request mapping.")

        batch = context.global_batch
        # The global sequence end bounds every fragment's history. Local
        # attention still consumes only its own seq_lens - query_lens.
        gather_lens = torch.from_numpy(
            np.where(batch.is_prefilling_np[: batch.num_reqs], batch.seq_lens_np[: batch.num_reqs], 0)
        ).to(torch.int32)
        local_rows = list(context.local_to_global_req_indices[self.num_decodes : self.num_decodes + self.num_prefills])
        gather_block_table = context.global_block_tables[self._pcp_cache_group_idx][: batch.num_reqs]
        return gather_lens, local_rows, gather_block_table

    def build_chunked_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        *,
        chunk_plan_lens_cpu: torch.Tensor | None = None,
        chunk_workspace_size: int | None = None,
    ):
        context_lens_cpu = (self.seq_lens - self.query_lens)[self.num_decodes : common_attn_metadata.num_reqs]
        gather_lens = context_lens_cpu if chunk_plan_lens_cpu is None else chunk_plan_lens_cpu
        use_default_chunk_workspace = chunk_workspace_size is None
        if chunk_workspace_size is None:
            chunk_workspace_size = self.chunked_prefill_workspace_size
        local_rows = None
        block_table = None
        if self.use_pcp and self.use_dcp:
            pcp_gather_lens, local_rows, block_table = self._get_pcp_prefill_kv_inputs()
            if pcp_gather_lens is not None:
                if chunk_plan_lens_cpu is None:
                    gather_lens = pcp_gather_lens
                # A global PCP request may expand into two local attention rows.
                if use_default_chunk_workspace:
                    chunk_workspace_size //= 2
        chunked_context_metadata = super().build_chunked_metadata(
            common_prefix_len,
            common_attn_metadata,
            chunk_plan_lens_cpu=gather_lens,
            chunk_workspace_size=chunk_workspace_size,
        )
        if chunked_context_metadata is None:
            return None

        # PCP sends use global request rows; attention retains local histories.
        num_context_rows = gather_lens.numel()
        local_context_lens_allranks = get_dcp_local_seq_lens(
            self.context_lens_cpu,
            dcp_size=self.dcp_size,
            cp_kv_cache_interleave_size=self.cp_local_block_size,
        )
        padded_local_context_lens_cpu = cdiv(gather_lens, self.cp_virtual_block_size) * self.cp_local_block_size
        padded_local_max_context_chunk_across_ranks = (
            cdiv(self.max_context_chunk, self.cp_virtual_block_size) * self.cp_local_block_size
        )
        local_chunk_starts = (
            torch.arange(self.num_chunks, dtype=torch.int32).unsqueeze(1).expand(-1, num_context_rows)
            * padded_local_max_context_chunk_across_ranks
        )
        local_chunk_ends = torch.min(
            padded_local_context_lens_cpu.unsqueeze(0),
            local_chunk_starts + padded_local_max_context_chunk_across_ranks,
        )
        padded_local_chunk_seq_lens = (local_chunk_ends - local_chunk_starts).clamp(min=0)
        padded_local_cu_chunk_seq_lens_cpu = torch.zeros(
            self.num_chunks,
            num_context_rows + 1,
            dtype=torch.int32,
            pin_memory=True,
        )
        torch.cumsum(
            padded_local_chunk_seq_lens,
            dim=1,
            out=padded_local_cu_chunk_seq_lens_cpu[:, 1:],
            dtype=torch.int32,
        )
        seq_tot = padded_local_chunk_seq_lens.sum(dim=1).tolist()
        return DCPChunkedContextMetadata(
            cu_seq_lens=chunked_context_metadata.cu_seq_lens,
            starts=local_chunk_starts.pin_memory().to(self.device, non_blocking=True),
            seq_tot=seq_tot,
            max_seq_lens=chunked_context_metadata.max_seq_lens,
            chunk_seq_lens=self.chunk_seq_lens,
            chunk_seq_lens_npu=chunked_context_metadata.chunk_seq_lens_npu,
            chunk_actual_seq_lengths_kv_list=chunked_context_metadata.chunk_actual_seq_lengths_kv_list,
            workspace=chunked_context_metadata.workspace,
            padded_chunk_seq_lens_npu=padded_local_chunk_seq_lens.to(self.device, non_blocking=True),
            padded_local_chunk_seq_lens=padded_local_chunk_seq_lens.tolist(),
            local_context_lens_allranks=local_context_lens_allranks.tolist(),
            padded_local_cu_seq_lens=padded_local_cu_chunk_seq_lens_cpu.pin_memory().to(
                self.device,
                non_blocking=True,
            ),
            cu_seq_lens_lst=self.cu_seq_lens_cpu.tolist(),
            chunk_size=padded_local_max_context_chunk_across_ranks,
            padded_local_cu_seq_lens_lst=padded_local_cu_chunk_seq_lens_cpu.tolist(),
            pcp_global_req_indices=local_rows,
            pcp_global_block_table=block_table,
            kv_rank_offsets=[[rank * toks for rank in self.dcp_collective_rank_order] for toks in seq_tot],
        )

    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendMLADecodeMetadata:
        decode_metadata = super().build_decode_metadata(common_prefix_len, common_attn_metadata)
        assert isinstance(decode_metadata, AscendMLADCPDecodeMetadata)
        dcp_local_seq_lens_cpu = common_attn_metadata.dcp_local_seq_lens_cpu
        assert dcp_local_seq_lens_cpu is not None
        seq_lens_cpu = self.seq_lens[: self.num_decodes]
        decode_metadata.update_dcp_seq_lens_cpu(
            seq_lens_cpu,
            dcp_local_seq_lens_cpu[: self.num_decodes],
            self.query_lens[: seq_lens_cpu.shape[0]],
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
            cp_kv_cache_interleave_size=self.cp_local_block_size,
        )
        decode_metadata.dcp_mtp_attn_mask = None
        return decode_metadata


class AscendMlaDCPImpl(DCPImplMixin, AscendMLAImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    can_return_lse_for_decode: bool = True
    supports_mtp_with_cp_non_trivial_interleave_size: bool = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dcp_current_kv_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._dcp_prefill_current_kv_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        # PCP cannot consume the RoPE PROLOG_V3 workspaces. The legacy no-rope
        # MLAPO decode buffer still has a consumer with PCP enabled.
        if self.use_dcp and (
            (self.supports_prolog_v3_writer() and not self.use_pcp and not self.is_pcp_decode_sharded)
            or (self.enable_mlapo and not self.fa_quant_layer and not self.use_mla_rope and not self.enable_kv_nz)
        ):
            self._prepare_dcp_current_kv_buffers()
        if self.use_dcp:
            # The shared writer contract checks these workspaces. Prepare the
            # projections after them, still inside the loader memory profile.
            self._prepare_prolog_v3_weights_after_loading()

    def _required_dcp_current_kv_capacity(self) -> int:
        """Upper bound on query rows in a DCP decode graph."""
        decode_query_len = 1
        if self.speculative_config is not None:
            decode_query_len += self.speculative_config.num_speculative_tokens or 0
        scheduler_config = self.vllm_config.scheduler_config
        return min(
            scheduler_config.max_num_batched_tokens,
            scheduler_config.max_num_seqs * decode_query_len,
        )

    def _required_dcp_prefill_current_kv_capacity(self) -> int:
        """Upper bound on current rows in a DCP prefill graph."""
        return self.vllm_config.scheduler_config.max_num_batched_tokens

    def _allocate_dcp_current_kv_buffers(
        self,
        max_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size = self.vllm_config.cache_config.block_size
        cache_shape = (cdiv(max_tokens, block_size), block_size, self.num_kv_heads)
        return (
            torch.empty((*cache_shape, self.kv_lora_rank), dtype=self.dtype, device=device),
            torch.empty((*cache_shape, self.qk_rope_head_dim), dtype=self.dtype, device=device),
            torch.arange(max_tokens, dtype=torch.int64, device=device),
        )

    def _prepare_dcp_current_kv_buffers(self) -> None:
        uses_prolog_v3_writer = self.supports_prolog_v3_writer()
        if self._dcp_current_kv_buffers is not None and (
            not uses_prolog_v3_writer or self._dcp_prefill_current_kv_buffers is not None
        ):
            # Graph capture stores these buffer addresses. A weight reload must
            # not replace the allocation out from under a captured graph.
            return
        assert self.fused_qkv_a_proj is not None
        device = self.fused_qkv_a_proj.weight.device
        if self._dcp_current_kv_buffers is None:
            decode_capacity = self._required_dcp_current_kv_capacity()
            if not uses_prolog_v3_writer:
                # The legacy MLAPO DCP contract retains its operator token cap.
                decode_capacity = min(decode_capacity, MLAPO_MAX_SUPPORTED_TOKENS)
            self._dcp_current_kv_buffers = self._allocate_dcp_current_kv_buffers(decode_capacity, device)
        if uses_prolog_v3_writer and self._dcp_prefill_current_kv_buffers is None:
            # Mixed batches keep decode and prefill current chunks alive until
            # their attention kernels run, so the two workspaces are disjoint.
            prefill_capacity = self._required_dcp_prefill_current_kv_capacity()
            self._dcp_prefill_current_kv_buffers = self._allocate_dcp_current_kv_buffers(prefill_capacity, device)

    def _dcp_current_kv_buffers_are_sufficient(self) -> bool:
        """Whether allocation-time buffers cover every admitted DCP graph."""
        decode_buffers = self._dcp_current_kv_buffers
        if decode_buffers is None or self._required_dcp_current_kv_capacity() > decode_buffers[2].numel():
            return False
        if not self.supports_prolog_v3_writer():
            return True
        prefill_buffers = self._dcp_prefill_current_kv_buffers
        return (
            prefill_buffers is not None
            and self._required_dcp_prefill_current_kv_capacity() <= prefill_buffers[2].numel()
        )

    def _prefill_requires_current_kv(self, attn_metadata: AscendMLAMetadata) -> bool:
        """Whether DCP prefill attention needs the complete current chunk."""
        return (
            self.use_dcp
            and attn_metadata.prefill is not None
            and attn_metadata.num_actual_tokens > attn_metadata.num_decode_tokens
        )

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config=None,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        pass

    def get_context_block_table(self, attn_metadata: AscendMLAMetadata) -> torch.Tensor:
        """Use the global PCP request table for DCP prefill KV loading."""
        if not self.use_pcp:
            return super().get_context_block_table(attn_metadata)
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None
        assert prefill_metadata.chunked_context is not None
        assert isinstance(prefill_metadata.chunked_context, DCPChunkedContextMetadata)
        assert prefill_metadata.chunked_context.pcp_global_block_table is not None
        return prefill_metadata.chunked_context.pcp_global_block_table

    def get_context_seq_len_npu(self, index: int, attn_metadata: AscendMLAMetadata):
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None
        assert prefill_metadata.chunked_context is not None
        assert isinstance(prefill_metadata.chunked_context, DCPChunkedContextMetadata)
        assert prefill_metadata.chunked_context.padded_chunk_seq_lens_npu is not None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        assert 0 <= index < iters
        return prefill_metadata.chunked_context.padded_chunk_seq_lens_npu[index]

    def reorg_decode_q(self, decode_q_nope, decode_q_pe):
        return self._dcp_all_gather_fragments(
            decode_q_nope,
            decode_q_pe,
            dim=1,
        )

    def _decode_requires_current_kv(self, attn_metadata: AscendMLAMetadata) -> bool:
        if attn_metadata.external_flashmla is not None:
            return attn_metadata.external_flashmla.current is not None
        return use_history_current_split_decode(
            attn_metadata,
            is_draft_model=_EXTRA_CTX.is_draft_model,
            is_draft_model_prefill=_EXTRA_CTX.is_draft_model_prefill,
            use_spec_decode=self.speculative_config is not None,
        )

    def _forward_external_flashmla(self, preprocessed, fused_cache, metadata):
        assert metadata.external_flashmla is not None
        if not isinstance(fused_cache, torch.Tensor):
            raise RuntimeError("DCP FlashMLA requires the original fused BBND cache")
        record_attention_compute_start()
        # The framework owns stream ordering, exchange, reduction and V-up.
        return self._forward_decode_split_attention(
            preprocessed.ql_nope,
            preprocessed.q_pe,
            fused_cache,
            None,
            preprocessed.current_k_nope,
            preprocessed.current_k_pe,
            fused_cache.shape[1],
            metadata,
        )

    def _prepare_dcp_flashmla_current(self, flash, q_nope, q_pe, k_nope, k_pe):
        """Stage current KV and Q on the main stream before history packing."""
        current = flash.current
        assert current is not None
        assert k_nope is not None and k_pe is not None
        torch_npu.npu_scatter_pa_kv_cache(
            key=k_nope.reshape(-1, 1, self.kv_lora_rank).contiguous(),
            value=k_pe.reshape(-1, 1, self.qk_rope_head_dim).contiguous(),
            key_cache=current.cache[..., : self.kv_lora_rank],
            value_cache=current.cache[..., self.kv_lora_rank :],
            slot_mapping=current.slots,
            cache_mode="Norm",
        )
        current.query[..., : self.kv_lora_rank].copy_(q_nope)
        current.query[..., self.kv_lora_rank :].copy_(q_pe)

    def _run_dcp_flashmla_attention(self, flash, q_nope, q_pe, k_nope, k_pe, *, attention_kind):
        """Adapt FlashMLA to TND/TN1; current KV/Q must already be staged."""
        if attention_kind is CPKVScope.CURRENT:
            current = flash.current
            assert current is not None
            query, cache = current.query, current.cache
            adapter, schedule = current.adapter, current.schedule
            block_table, cache_lens, attn_mask = current.block_table, flash.used_q, current.attn_mask
        else:
            query, cache = flash.query, k_nope
            adapter, schedule = flash.adapter, flash.schedule
            block_table, cache_lens, attn_mask = flash.block_table, flash.cache_lens, None
            query[..., : self.kv_lora_rank].copy_(q_nope)
            query[..., self.kv_lora_rank :].copy_(q_pe)
        output, lse = adapter.attention(
            query,
            cache,
            block_table=block_table,
            cache_seqlens=cache_lens,
            cu_seqlens_q=flash.cu,
            seqused_q=flash.used_q,
            metadata=schedule,
            attn_mask=attn_mask,
        )
        return output.transpose(0, 1), lse.transpose(0, 1).unsqueeze(-1)

    def _run_dcp_mtp_split_attention_op(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        attn_mask: torch.Tensor | None,
        sparse_mode: int,
        block_table: torch.Tensor | None,
        block_size: int,
        actual_seq_lengths: list[int],
        actual_seq_lengths_kv: list[int],
        attention_kind: CPKVScope,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def _forward_decode_split_attention(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_pe: torch.Tensor | None,
        current_k_nope: torch.Tensor | None,
        current_k_pe: torch.Tensor | None,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        pass

    def _forward_decode(
        self,
        decode_preprocess_res: DecodeMLAPreprocessResult,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        pass

    def _reorg_kvcache(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        chunked_context: ChunkedContextMetadata,
        chunk_idx: int,
        toks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        reorg and unpad kvcache after cp local gather to tp layout for attn kernel.
        e.g.
        kv_c_normed in rank0 = [T0_0, T0_1, T0_2, T0_3, T1_0, T1_1, ...]
        kv_c_normed in rank1 = [T0_4, T0_5, pad, pad, T1_2, pad, ...]
        allgatered_kv_c_normed = [T0_0, T0_1, T0_2, T0_3, T1_0, T1_1, ...,
                                T0_4, T0_5, pad, pad, T1_2, pad, ...]
        -> reorganized_kv_c_normed = [T0_0, T0_1, T0_2, T0_3, T0_4, T0_5,
                                    T1_0, T1_1, T1_2, ...]
        Args:
            padded_local_chunk_seq_lens_lst: local chunk context lengths
                under current CP rank.
            local_context_lens_allranks: local context lengths on each CP rank.
            sum_seq_len: the sum of cp_chunk_seq_lens_lst.
            max_seq_len: the max value of cp_chunk_seq_lens_lst.
            chunk_size: the local padded max context chunk from
                chunked_context_metadata building.
            chunk_idx: chunk idx of chunked_prefill.
            toks: the number of tokens for local gather cache.
        """
        assert isinstance(chunked_context, DCPChunkedContextMetadata)
        assert chunked_context.padded_local_chunk_seq_lens is not None
        assert chunked_context.local_context_lens_allranks is not None
        assert chunked_context.cu_seq_lens_lst is not None
        assert chunked_context.max_seq_lens is not None
        assert chunked_context.chunk_size is not None
        assert chunked_context.kv_rank_offsets is not None

        padded_local_chunk_seq_lens_lst = chunked_context.padded_local_chunk_seq_lens[chunk_idx]
        local_context_lens_allranks = chunked_context.local_context_lens_allranks
        sum_seq_len = chunked_context.cu_seq_lens_lst[chunk_idx][-1]
        max_seq_len = chunked_context.max_seq_lens[chunk_idx]
        chunk_size: int = chunked_context.chunk_size
        kv_rank_offsets = chunked_context.kv_rank_offsets[chunk_idx]
        cache_kv_c_k_pe = torch.cat([kv_c_normed, k_pe], dim=-1)
        cache_kv_c_k_pe = self._dcp_all_gather(cache_kv_c_k_pe, 0)

        allgatered_kv_c_normed, allgatered_k_pe = cache_kv_c_k_pe.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        kv_c_segments = []
        k_pe_segments = []
        src_token_idx = 0
        max_seq_len_check = 0
        for row, local_context_lens in enumerate(local_context_lens_allranks):
            source_row = (
                chunked_context.pcp_global_req_indices[row]
                if chunked_context.pcp_global_req_indices is not None
                else row
            )
            padded_local_chunk_seq_len = padded_local_chunk_seq_lens_lst[source_row]
            if chunked_context.pcp_global_req_indices is not None:
                assert chunked_context.padded_local_cu_seq_lens_lst is not None
                src_token_idx = chunked_context.padded_local_cu_seq_lens_lst[chunk_idx][source_row]
            cur_seq_len = 0
            for rank, local_context_len in enumerate(local_context_lens):
                # Note(qcs): We split the context into multiple chunks,
                # depending on the size of the workspace.
                # local_context in dcp0:   |-----------------|
                # local_context in dcp1:   |--------------|
                # n*padded_local_chunk:    |-----|-----|-----|
                # local_chunk_len in dcp1: |-----|-----|--|
                # so we need update the last chunk length in dcp1.
                local_chunk_len = min(
                    max(0, local_context_len - chunk_idx * chunk_size),
                    padded_local_chunk_seq_len,
                )
                if local_chunk_len != 0:
                    src_start = kv_rank_offsets[rank] + src_token_idx
                    kv_c_segment = allgatered_kv_c_normed[src_start : src_start + local_chunk_len]
                    k_pe_segment = allgatered_k_pe[src_start : src_start + local_chunk_len]
                    kv_c_segments.append(kv_c_segment)
                    k_pe_segments.append(k_pe_segment)
                    cur_seq_len += local_chunk_len
            max_seq_len_check = max(max_seq_len_check, cur_seq_len)
            src_token_idx += padded_local_chunk_seq_len
        if not kv_c_segments:
            return kv_c_normed[:0], k_pe[:0]
        reorganized_kv_c_normed = torch.cat(kv_c_segments, dim=0)
        reorganized_k_pe = torch.cat(k_pe_segments, dim=0)
        assert reorganized_kv_c_normed.shape[0] == sum_seq_len, (
            "DCP MLA KV reorg length mismatch: "
            f"actual={reorganized_kv_c_normed.shape[0]}, "
            f"expected={sum_seq_len}, chunk_idx={chunk_idx}, "
            f"toks={toks}, chunk_size={chunk_size}, "
            f"padded_local_chunk_seq_lens="
            f"{padded_local_chunk_seq_lens_lst}, "
            f"local_context_lens_allranks="
            f"{local_context_lens_allranks}, "
            f"cu_seq_lens={chunked_context.cu_seq_lens_lst[chunk_idx]}, "
            f"max_seq_len={max_seq_len}, "
            f"max_seq_len_check={max_seq_len_check}"
        )
        assert reorganized_k_pe.shape[0] == sum_seq_len
        assert max_seq_len_check == max_seq_len
        return reorganized_kv_c_normed, reorganized_k_pe
