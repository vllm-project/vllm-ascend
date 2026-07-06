from typing import Any, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.triton_utils import HAS_TRITON
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import AscendPCPMetadata
from vllm_ascend.attention.sfa_v1 import (
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    DCPContext,
    DCPQueryGatherContext,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, enabling_mlapo, split_decodes_and_prefills
from vllm_ascend.distributed.utils import (
    all_gather_async,
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
)
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso

M = TypeVar("M", bound=AscendSFAMetadata)


class AscendSFACPMetadataBuilder(AscendSFAMetadataBuilder):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen)

        # In sfa, pcp prefill does not support mlapo
        self.enable_mlapo = enabling_mlapo(self.vllm_config)

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.pcp_group = get_pcp_group().device_group if self.pcp_size > 1 else None

        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0
        self.dcp_group = get_dcp_group().device_group if self.dcp_size > 1 else None
        self.cp_local_block_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        self.cp_virtual_block_size = self.cp_local_block_size * self.dcp_size * self.pcp_size
        self.block_size = (self.block_size * self.cp_virtual_block_size) // np.gcd(
            self.block_size, self.cp_virtual_block_size
        )
        self.slot_mapping_buf = torch.empty(
            (
                vllm_config.scheduler_config.max_num_batched_tokens
                + 2 * self.pcp_size * vllm_config.scheduler_config.max_num_seqs,
            ),
            dtype=torch.int32,
            device=device,
        )
        self.block_arange_buffer = torch.arange(self.pcp_size * self.dcp_size, dtype=torch.int32, device=device)

    def _compact_varlen_decode_slot_mapping(
        self,
        decode_slot_mapping: torch.Tensor,
        decode_query_lens: torch.Tensor,
    ) -> None:
        device = decode_slot_mapping.device
        decode_query_lens_cpu = decode_query_lens.to(device="cpu", dtype=torch.int64, non_blocking=True)
        total_valid_tokens = int(decode_query_lens_cpu.sum().item())
        if total_valid_tokens == 0:
            return
        decode_query_lens = decode_query_lens_cpu.to(device=device, dtype=torch.int64, non_blocking=True)

        req_spans = decode_query_lens * self.pcp_size
        req_starts = torch.cumsum(req_spans, dim=0) - req_spans

        token_offsets = torch.arange(total_valid_tokens, device=device, dtype=torch.int64)
        token_base = torch.cumsum(decode_query_lens, dim=0) - decode_query_lens
        token_offsets = token_offsets - torch.repeat_interleave(token_base, decode_query_lens)

        expanded_req_starts = torch.repeat_interleave(req_starts, decode_query_lens)
        valid_in_idx = expanded_req_starts + token_offsets * self.pcp_size
        valid_out_idx = expanded_req_starts + token_offsets

        valid_slots = decode_slot_mapping[valid_in_idx]
        decode_slot_mapping.fill_(-1)
        decode_slot_mapping.index_copy_(0, valid_out_idx, valid_slots)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> AscendSFAMetadata:
        metadata_cls = super().build(common_prefix_len, common_attn_metadata, fast_build, **kwargs)
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            treat_short_extends_as_decodes=False,
        )
        num_reqs = common_attn_metadata.num_reqs
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == common_attn_metadata.num_actual_tokens

        sfa_cp_metadata = self.build_cp_metadata(self.block_arange_buffer, metadata_cls.seq_lens, common_attn_metadata)
        metadata_cls.num_decode_tokens = num_decode_tokens
        metadata_cls.num_decodes = num_decodes
        metadata_cls.num_prefills = num_prefills
        actual_seq_lengths_query = metadata_cls.cum_query_lens
        if num_prefills > 0:
            assert sfa_cp_metadata is not None
            # Prefill uses a compact block view so it can all-gather only the
            # real KV blocks it needs instead of the request-scoped decode view.
            valid_block_ids, block_table_cp = self.build_prefill_compact_block_metadata(
                metadata_cls.block_table, num_decodes
            )
            sfa_cp_metadata.valid_block_ids = valid_block_ids
            sfa_cp_metadata.block_table_cp = block_table_cp

            # Mixed batches store decode requests first, so prefill cumulative
            # query lengths must be rebased to the prefill-only token range.
            if num_decode_tokens > 0:
                prefill_q_cum_seqlens = (
                    actual_seq_lengths_query[num_decodes:] - actual_seq_lengths_query[num_decodes - 1]
                )
            else:
                prefill_q_cum_seqlens = actual_seq_lengths_query
            assert sfa_cp_metadata is not None
            sfa_cp_metadata.prefill_q_cum_seqlens = prefill_q_cum_seqlens

        if self.pcp_size > 1:
            long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
            assert long_seq_metadata is not None
            num_actual_tokens_pcp_padded = long_seq_metadata.num_actual_tokens_pcp_padded
            self.slot_mapping_buf[:num_actual_tokens_pcp_padded].copy_(
                common_attn_metadata.slot_mapping[:num_actual_tokens_pcp_padded], non_blocking=True
            )
            if self.enable_mlapo:
                self.slot_mapping_buf[:num_decode_tokens] = self.slot_mapping_buf[
                    : num_decode_tokens * self.pcp_size : self.pcp_size
                ]
                self.slot_mapping_buf[num_decode_tokens : num_decode_tokens * self.pcp_size].fill_(-1)
            elif self.speculative_config is not None and num_decodes > 0:
                # when mtp, pcp_allgather_restore_idx=[696,-1,697,-1,560,-1,561,-1,100,101,102],
                # slot_mapping should be [696,697,-1,-1,560,561,-1,-1,100,101,102]
                # corner case: decode requests in the same MTP batch can have
                # different query lengths when some drafts are clipped near
                # max_model_len, so compact slot_mapping by per-request length
                # instead of assuming each request has decode_threshold tokens.
                decode_query_lens = long_seq_metadata.query_lens_pcp_full_cpu[:num_decodes]
                decode_slot_mapping = self.slot_mapping_buf[: num_decode_tokens * self.pcp_size]
                self._compact_varlen_decode_slot_mapping(
                    decode_slot_mapping,
                    decode_query_lens,
                )
            metadata_cls.slot_mapping = self.slot_mapping_buf[:num_actual_tokens_pcp_padded]
        metadata_cls.sfa_cp_metadata = sfa_cp_metadata
        return metadata_cls

    def build_prefill_compact_block_metadata(
        self, block_table: torch.Tensor, num_decodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefill_block_table = block_table[num_decodes:]
        valid_block_ids, new_block_table = prefill_block_table.flatten().unique(return_inverse=True)
        num_blocks = valid_block_ids.shape[0]
        # Remap prefill block ids to the compact KV buffer after CP all-gather.
        block_table_cp = (
            new_block_table.unsqueeze(-1).to(prefill_block_table)
            + (self.block_arange_buffer * num_blocks).view(1, 1, -1).to(prefill_block_table)
        ).reshape(prefill_block_table.shape[0], -1)
        return valid_block_ids, block_table_cp

    def build_cp_metadata(
        self,
        block_arange: torch.Tensor,
        seq_lens: torch.Tensor,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendPCPMetadata | None:
        common_long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        assert common_long_seq_metadata is not None
        num_computed_tokens = common_attn_metadata.num_computed_tokens_cpu.to(seq_lens.device)
        q_head_kv_lens = (seq_lens // 2) * (self.pcp_rank + 1) + num_computed_tokens
        q_tail_kv_lens = seq_lens * self.pcp_size - (seq_lens // 2) * self.pcp_rank + num_computed_tokens
        return AscendPCPMetadata(
            q_head_idx=common_long_seq_metadata.q_head_idx_tensor,
            q_tail_idx=common_long_seq_metadata.q_tail_idx_tensor,
            q_full_idx=common_long_seq_metadata.q_full_idx,
            head_attn_nomask_seqlens=q_head_kv_lens,
            tail_attn_nomask_seqlens=q_tail_kv_lens,
            pcp_allgather_restore_idx=common_long_seq_metadata.pcp_allgather_restore_idx,
            block_arange=block_arange,
        )


class AscendSFACPImpl(AscendSFAImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **kwargs,
        )
        # In sfa, pcp prefill does not support mlapo
        self.enable_mlapo = enabling_mlapo(self.vllm_config)
        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.pcp_group = get_pcp_group().device_group if self.pcp_size > 1 else None

        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0
        self.dcp_group = get_dcp_group().device_group if self.dcp_size > 1 else None

    def _execute_sparse_flash_attention_process(
        self, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
    ):
        kv = kv_cache[0]
        key_rope = kv_cache[1]

        assert attn_metadata.sfa_cp_metadata is not None
        sfa_cp_metadata = attn_metadata.sfa_cp_metadata
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills
        decode_attn_out = None
        if num_decode_tokens > 0:
            decode_block_table_src = attn_metadata.block_table[:num_decodes]
            decode_kv, decode_block_num = self.gather_kv_cross_cp(kv, decode_block_table_src)
            decode_key_rope, _ = self.gather_kv_cross_cp(key_rope, decode_block_table_src)
            decode_block_table = self.gather_block_table(
                decode_block_num, decode_block_table_src, sfa_cp_metadata.block_arange
            )
            decode_attn_out = self._execute_sparse_flash_attention(
                ql_nope[:num_decode_tokens],
                q_pe[:num_decode_tokens],
                decode_kv,
                decode_key_rope,
                decode_block_table,
                topk_indices[:num_decode_tokens],
                actual_seq_lengths_query[:num_decodes],
                actual_seq_lengths_key[:num_decodes],
            )

        if num_prefills < 1:
            return self._align_to_graph_bucket_tokens(decode_attn_out, attn_metadata)

        prefill_valid_block_ids = sfa_cp_metadata.valid_block_ids
        prefill_block_table = sfa_cp_metadata.block_table_cp
        assert prefill_valid_block_ids is not None and prefill_block_table is not None
        prefill_kv = self.gather_kv_cross_cp_compact(kv, prefill_valid_block_ids)
        prefill_key_rope = self.gather_kv_cross_cp_compact(key_rope, prefill_valid_block_ids)
        prefill_ql_nope = ql_nope[num_decode_tokens:]
        prefill_q_pe = q_pe[num_decode_tokens:]
        prefill_topk_indices = topk_indices[num_decode_tokens:]
        prefill_actual_seq_lengths_key = actual_seq_lengths_key[num_decodes:]
        if self.pcp_size == 1:
            prefill_attn_out = self._execute_sparse_flash_attention(
                prefill_ql_nope,
                prefill_q_pe,
                prefill_kv,
                prefill_key_rope,
                prefill_block_table,
                prefill_topk_indices,
                sfa_cp_metadata.prefill_q_cum_seqlens,
                prefill_actual_seq_lengths_key,
            )
            if decode_attn_out is not None:
                prefill_attn_out = torch.cat([decode_attn_out, prefill_attn_out], dim=0)
            return self._align_to_graph_bucket_tokens(prefill_attn_out, attn_metadata)

        # q split for head and tail
        q_head_idx = sfa_cp_metadata.q_head_idx
        q_tail_idx = sfa_cp_metadata.q_tail_idx

        # q head compute
        q_head_actual_seq_lengths_key = sfa_cp_metadata.head_attn_nomask_seqlens[num_decodes:]
        q_head_output = self._execute_sparse_flash_attention(
            torch.index_select(prefill_ql_nope, 0, q_head_idx),
            torch.index_select(prefill_q_pe, 0, q_head_idx),
            prefill_kv,
            prefill_key_rope,
            prefill_block_table,
            torch.index_select(prefill_topk_indices, 0, q_head_idx),
            sfa_cp_metadata.prefill_q_cum_seqlens // 2,
            q_head_actual_seq_lengths_key,
        )

        # q tail compute
        q_tail_actual_seq_lengths_key = sfa_cp_metadata.tail_attn_nomask_seqlens[num_decodes:]
        q_tail_output = self._execute_sparse_flash_attention(
            torch.index_select(prefill_ql_nope, 0, q_tail_idx),
            torch.index_select(prefill_q_pe, 0, q_tail_idx),
            prefill_kv,
            prefill_key_rope,
            prefill_block_table,
            torch.index_select(prefill_topk_indices, 0, q_tail_idx),
            sfa_cp_metadata.prefill_q_cum_seqlens // 2,
            q_tail_actual_seq_lengths_key,
        )

        q_full_idx = sfa_cp_metadata.q_full_idx
        attn_output = torch.index_select(torch.cat([q_head_output, q_tail_output], dim=0), 0, q_full_idx)

        if decode_attn_out is not None:
            attn_output = torch.cat([decode_attn_out, attn_output], dim=0)
        return self._align_to_graph_bucket_tokens(attn_output, attn_metadata)

    def _align_to_graph_bucket_tokens(self, attn_output: torch.Tensor | None, attn_metadata: M) -> torch.Tensor | None:
        if attn_output is None or self.pcp_size == 1:
            return attn_output
        # In graph mode, output buffer uses graph bucket token size
        # (forward_context.num_tokens), while PCP path may compute only valid
        # tokens. Align to the larger one to avoid later write-back mismatch.
        forward_context = get_forward_context()
        target_tokens = max(
            attn_metadata.num_input_tokens,
            forward_context.num_tokens if forward_context is not None else 0,
        )

        if attn_output.shape[0] == target_tokens:
            return attn_output
        aligned = torch.zeros(
            (target_tokens, *attn_output.shape[1:]),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        valid_tokens = min(attn_output.shape[0], target_tokens)
        aligned[:valid_tokens] = attn_output[:valid_tokens]
        return aligned

    def _execute_sparse_flash_attention(
        self, ql_nope, q_pe, kv, key_rope, block_table, topk_indices, actual_seq_lengths_query, actual_seq_lengths_key
    ):
        attn_output, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv,
            value=kv,
            sparse_indices=topk_indices,
            scale_value=self.scale,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=key_rope,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
        )
        return attn_output

    def gather_kv_cross_cp(self, kv_cache: torch.Tensor, block_tables: torch.Tensor) -> tuple[torch.Tensor, int]:
        # Note(qcs): we need set kv_cache_interleave_size = block_size for sfa!!!
        # Decode path uses request-scoped KV: first select the blocks referenced
        # by its block table, then all-gather only that request-local view.
        req_kv_cache = torch.index_select(kv_cache, 0, block_tables.flatten())
        block_num = req_kv_cache.shape[0]
        if self.dcp_size > 1:
            req_kv_cache = get_dcp_group().all_gather(req_kv_cache, 0)
        if self.pcp_size > 1:
            req_kv_cache = get_pcp_group().all_gather(req_kv_cache, 0)
        return req_kv_cache, block_num

    def gather_kv_cross_cp_compact(self, kv_cache: torch.Tensor, valid_block_ids: torch.Tensor) -> torch.Tensor:
        # prefill path uses compact KV: valid_block_ids
        kv_cache = torch.index_select(kv_cache, 0, valid_block_ids)
        if self.dcp_size > 1:
            kv_cache = get_dcp_group().all_gather(kv_cache, 0)
        if self.pcp_size > 1:
            kv_cache = get_pcp_group().all_gather(kv_cache, 0)
        return kv_cache

    def gather_block_table(self, block_num: int, block_tables: torch.Tensor, block_arange: torch.Tensor):
        # Remap original block ids to positions in the request-scoped KV buffer
        # generated by gather_kv_cross_cp().
        new_block_tables = torch.arange(block_tables.numel(), device=block_tables.device).view(block_tables.shape)
        block_tables = (
            (new_block_tables.unsqueeze(-1) + (block_arange * block_num).view(1, 1, -1).to(block_tables))
            .reshape(block_tables.shape[0], -1)
            .to(block_tables.dtype)
        )
        return block_tables

    def indexer_select_post_process(
        self,
        x: torch.Tensor,
        q_c: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ):
        kw, _ = self.wk_weights_proj(x)
        weights = kw[:, self.head_dim :]
        q_li, _ = self.wq_b(q_c)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
        q_li = q_li.view(-1, self.n_head, self.head_dim)  # [n_toks,64,128]
        if HAS_TRITON:
            q_li = rope_forward_triton_siso(
                q_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            q_li_pe, q_li_nope = torch.split(
                q_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )  # [b,s,64,64+64]

            q_li_pe = q_li_pe.unsqueeze(2)
            q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
            q_li_pe = q_li_pe.squeeze(2)
            q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)  # [b*s,64,128]

        q = q_li

        key = kv_cache[2]
        assert attn_metadata.sfa_cp_metadata is not None
        sfa_cp_metadata = attn_metadata.sfa_cp_metadata
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills
        decode_topk_indices = None
        if num_decode_tokens > 0:
            decode_block_table_src = attn_metadata.block_table[:num_decodes]
            decode_key, decode_block_num = self.gather_kv_cross_cp(key, decode_block_table_src)
            decode_block_table = self.gather_block_table(
                decode_block_num, decode_block_table_src, sfa_cp_metadata.block_arange
            )
            decode_topk_indices = self._execute_indexer_select(
                q[:num_decode_tokens],
                decode_key,
                weights[:num_decode_tokens],
                actual_seq_lengths_query[:num_decodes],
                actual_seq_lengths_key[:num_decodes],
                decode_block_table,
            )
        # prefill compute
        if num_prefills == 0:
            return decode_topk_indices

        prefill_valid_block_ids = sfa_cp_metadata.valid_block_ids
        prefill_block_table = sfa_cp_metadata.block_table_cp
        assert prefill_valid_block_ids is not None and prefill_block_table is not None
        prefill_key = self.gather_kv_cross_cp_compact(key, prefill_valid_block_ids)
        prefill_q = q[num_decode_tokens:]
        prefill_weights = weights[num_decode_tokens:]
        prefill_actual_seq_lengths_key = actual_seq_lengths_key[num_decodes:]
        if self.pcp_size == 1:
            prefill_topk_indices = self._execute_indexer_select(
                prefill_q,
                prefill_key,
                prefill_weights,
                sfa_cp_metadata.prefill_q_cum_seqlens,
                prefill_actual_seq_lengths_key,
                prefill_block_table,
            )
            if decode_topk_indices is not None:
                prefill_topk_indices = torch.cat([decode_topk_indices, prefill_topk_indices], dim=0)
            return prefill_topk_indices

        # pcp split for head and tail
        q_head_idx = sfa_cp_metadata.q_head_idx
        q_tail_idx = sfa_cp_metadata.q_tail_idx

        # q head compute
        q_head_actual_seq_lengths_key = sfa_cp_metadata.head_attn_nomask_seqlens[num_decodes:]
        q_head_topk_indices = self._execute_indexer_select(
            q=torch.index_select(prefill_q, 0, q_head_idx),
            key=prefill_key,
            weights=torch.index_select(prefill_weights, 0, q_head_idx),
            actual_seq_lengths_query=sfa_cp_metadata.prefill_q_cum_seqlens // 2,
            actual_seq_lengths_key=q_head_actual_seq_lengths_key,
            block_table=prefill_block_table,
        )

        # q tail compute
        q_tail_actual_seq_lengths_key = sfa_cp_metadata.tail_attn_nomask_seqlens[num_decodes:]
        q_tail_topk_indices = self._execute_indexer_select(
            q=torch.index_select(prefill_q, 0, q_tail_idx),
            key=prefill_key,
            weights=torch.index_select(prefill_weights, 0, q_tail_idx),
            actual_seq_lengths_query=sfa_cp_metadata.prefill_q_cum_seqlens // 2,
            actual_seq_lengths_key=q_tail_actual_seq_lengths_key,
            block_table=prefill_block_table,
        )

        q_full_idx = sfa_cp_metadata.q_full_idx
        topk_indices = torch.index_select(torch.cat([q_head_topk_indices, q_tail_topk_indices], dim=0), 0, q_full_idx)
        if decode_topk_indices is not None:
            topk_indices = torch.cat([decode_topk_indices, topk_indices], dim=0)
        return topk_indices

    def _execute_indexer_select(self, q, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table):
        if self.use_torch_npu_lightning_indexer:
            topk_indices, _ = torch_npu.npu_lightning_indexer(
                query=q,
                key=key,
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        else:
            topk_indices, _ = torch.ops._C_ascend.npu_lightning_indexer(
                query=q,
                key=key,
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=2048,
                sparse_mode=3,
            )
        return topk_indices

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
    ):
        if self.pcp_size == 1:
            return super().exec_kv(kv_no_split, cos, sin, kv_cache, slots, attn_metadata)
        kv_c, k_pe = kv_no_split.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())  # type: ignore[misc]
        assert len(kv_cache) > 1, "the number of kv cache should be greater than 1, namely (nope_cache and rope_cache)"
        assert attn_metadata.sfa_cp_metadata is not None
        kv_c_normed = kv_c_normed.view([kv_c_normed.shape[0], self.num_kv_heads, -1])
        k_pe = k_pe.unsqueeze(1)
        k_pe = self.rope_single(k_pe, cos, sin)
        kv_c_k_pe = torch.cat([kv_c_normed, k_pe], dim=-1)
        kv_c_k_pe = get_pcp_group().all_gather(kv_c_k_pe, 0)
        kv_c_k_pe = torch.index_select(kv_c_k_pe, 0, attn_metadata.sfa_cp_metadata.pcp_allgather_restore_idx)
        kv_c_normed, k_pe = kv_c_k_pe.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        slot_mapping = attn_metadata.slot_mapping
        torch_npu._npu_reshape_and_cache(
            key=kv_c_normed, value=k_pe, key_cache=kv_cache[0], value_cache=kv_cache[1], slot_indices=slot_mapping
        )
        return None, None

    def _get_full_kv(self, k, attn_metadata: M):
        if self.pcp_size == 1 or self.enable_mlapo:
            return k
        else:
            assert attn_metadata.sfa_cp_metadata is not None
            k = get_pcp_group().all_gather(k.contiguous(), 0)
            k = torch.index_select(k, 0, attn_metadata.sfa_cp_metadata.pcp_allgather_restore_idx)
            return k


# SFA DCP replicated-indexer layout:
#
# - LightningIndexer cache is replicated on every DCP rank so index selection
#   can run against the full sequence and keep the same sparse topk semantics as
#   non-DCP SFA.
# - SFA KV cache remains DCP-local to preserve the KV memory saving. The sparse
#   topk indices produced from the replicated indexer view are remapped to local
#   KV indices before calling sparse flash attention.
# - BlockTable only owns the DCP-local physical layout. This builder derives the
#   replicated block table and slot mapping on demand, temporarily builds the
#   indexer-facing metadata with that replicated view, and then stores the
#   original DCP-local view in metadata.dcp_context for KV writes and SFA reads.
# - The replicated view uses the same logical/kernel block size as BlockTable,
#   including hybrid block splitting.
class AscendSFADCPMetadataBuilder(AscendSFAMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen)
        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0
        self.cp_kv_cache_interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        self._dcp_metadata_debug_print_count = 0
        self.dcp_local_seq_lens_buf = torch.empty(
            vllm_config.scheduler_config.max_num_seqs,
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
        self.block_table_replicated_view_buf: torch.Tensor | None = None
        self.slot_mapping_replicated_view_buf: torch.Tensor | None = None
        assert self.pcp_size == 1, "AscendSFADCPMetadataBuilder only supports DCP without PCP."
        assert self.dcp_size > 1, "AscendSFADCPMetadataBuilder requires DCP world size > 1."
        if self.cp_kv_cache_interleave_size <= 0:
            raise RuntimeError(f"Invalid cp_kv_cache_interleave_size: {self.cp_kv_cache_interleave_size}")

    def _get_dcp_local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        total_cp_size = self.pcp_size * self.dcp_size
        current_rank = self.pcp_rank * self.dcp_size + self.dcp_rank
        interleave_size = self.cp_kv_cache_interleave_size
        base = seq_lens // interleave_size // total_cp_size * interleave_size
        remainder = seq_lens - base * total_cp_size
        remainder = torch.clamp(
            remainder - current_rank * interleave_size,
            0,
            interleave_size,
        )
        return base + remainder

    def _ensure_replicated_view_buffers(
        self,
        num_reqs: int,
        num_input_tokens: int,
        local_block_table_cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_table_cols = local_block_table_cols * self.pcp_size * self.dcp_size
        if (
            self.block_table_replicated_view_buf is None
            or self.block_table_replicated_view_buf.shape[0] < num_reqs
            or self.block_table_replicated_view_buf.shape[1] < block_table_cols
        ):
            self.block_table_replicated_view_buf = torch.empty(
                (num_reqs, block_table_cols),
                dtype=torch.int32,
                device=self.device,
            )
        if (
            self.slot_mapping_replicated_view_buf is None
            or self.slot_mapping_replicated_view_buf.shape[0] < num_input_tokens
        ):
            self.slot_mapping_replicated_view_buf = torch.empty(
                (num_input_tokens,),
                dtype=torch.int32,
                device=self.device,
            )
        return (
            self.block_table_replicated_view_buf[:num_reqs, :block_table_cols],
            self.slot_mapping_replicated_view_buf[:num_input_tokens],
        )

    def _build_block_table_replicated_view(
        self,
        dcp_block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = dcp_block_table.shape[0]
        local_block_table_cols = dcp_block_table.shape[1]
        block_table_replicated_view, _ = self._ensure_replicated_view_buffers(
            num_reqs,
            0,
            local_block_table_cols,
        )

        total_cp_size = self.pcp_size * self.dcp_size
        blocks_per_phys_block = self.blocks_per_phys_block
        block_table_cols = block_table_replicated_view.shape[1]
        replicated_col_idx = torch.arange(
            block_table_cols,
            dtype=torch.long,
            device=dcp_block_table.device,
        )
        local_col_idx = (
            replicated_col_idx // (total_cp_size * blocks_per_phys_block) * blocks_per_phys_block
            + replicated_col_idx % blocks_per_phys_block
        )
        rank_in_replicated_view = (replicated_col_idx // blocks_per_phys_block) % total_cp_size

        local_logical_blocks = torch.index_select(dcp_block_table, 1, local_col_idx)
        block_dtype = local_logical_blocks.dtype
        rank_in_replicated_view = rank_in_replicated_view.to(block_dtype)
        if blocks_per_phys_block == 1:
            replicated_blocks = local_logical_blocks * total_cp_size + rank_in_replicated_view
        else:
            local_sub_blocks = local_logical_blocks % blocks_per_phys_block
            local_phys_blocks = local_logical_blocks // blocks_per_phys_block
            replicated_blocks = (
                local_phys_blocks * total_cp_size + rank_in_replicated_view
            ) * blocks_per_phys_block + local_sub_blocks

        valid_req_mask = seq_lens[:num_reqs].to(device=self.device) > 0
        replicated_blocks = torch.where(
            valid_req_mask[:, None],
            replicated_blocks,
            torch.zeros((), dtype=block_dtype, device=self.device),
        )
        block_table_replicated_view.copy_(replicated_blocks)
        return block_table_replicated_view

    def _build_slot_mapping_replicated_view(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        block_table_replicated_view: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        num_actual_tokens = min(common_attn_metadata.num_actual_tokens, num_input_tokens)
        _, slot_mapping_replicated_view = self._ensure_replicated_view_buffers(
            num_reqs,
            num_input_tokens,
            common_attn_metadata.block_table_tensor.shape[1],
        )
        slot_mapping_replicated_view.fill_(-1)
        if num_actual_tokens == 0:
            return slot_mapping_replicated_view

        query_lens = (
            common_attn_metadata.query_start_loc[1 : num_reqs + 1] - common_attn_metadata.query_start_loc[:num_reqs]
        )
        req_indices = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.long, device=self.device),
            query_lens.to(device=self.device, dtype=torch.long),
        )[:num_actual_tokens]
        if req_indices.numel() == 0:
            return slot_mapping_replicated_view

        num_actual_tokens = min(num_actual_tokens, req_indices.shape[0])
        req_indices = req_indices[:num_actual_tokens]
        positions = common_attn_metadata.positions[:num_actual_tokens].to(
            device=self.device,
            dtype=torch.long,
        )
        logical_block_idx = positions // self.replicated_view_block_size
        block_offsets = positions % self.replicated_view_block_size
        block_table_indices = req_indices * block_table_replicated_view.shape[1] + logical_block_idx
        block_numbers = block_table_replicated_view.flatten()[block_table_indices]
        slot_mapping_replicated_view[:num_actual_tokens] = (
            block_numbers * self.replicated_view_block_size + block_offsets
        ).to(slot_mapping_replicated_view.dtype)
        return slot_mapping_replicated_view

    def _update_dsa_cp_slot_mapping_for_dcp(
        self,
        metadata: AscendSFAMetadata,
        dcp_slot_mapping: torch.Tensor,
        num_input_tokens: int,
    ) -> None:
        if metadata.dsa_cp_context is None:
            return

        dsa_cp_context = metadata.dsa_cp_context
        slot_mapping = dcp_slot_mapping[:num_input_tokens]
        if dsa_cp_context.num_tokens_pad > slot_mapping.shape[0]:
            slot_mapping = torch.nn.functional.pad(
                slot_mapping,
                (0, dsa_cp_context.num_tokens_pad - slot_mapping.shape[0]),
                value=-1,
            )
        else:
            slot_mapping = slot_mapping[: dsa_cp_context.num_tokens_pad]
        dsa_cp_context.slot_mapping_cp = slot_mapping[dsa_cp_context.local_start : dsa_cp_context.local_end_with_pad]

    def _build_with_replicated_view_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        build_metadata,
        **kwargs,
    ) -> AscendSFAMetadata:
        dcp_slot_mapping = common_attn_metadata.slot_mapping
        dcp_block_table = common_attn_metadata.block_table_tensor
        dcp_slot_mapping_cpu = common_attn_metadata.slot_mapping_cpu
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        block_table_replicated_view = self._build_block_table_replicated_view(
            dcp_block_table[:num_reqs],
            common_attn_metadata.seq_lens,
        )
        slot_mapping_replicated_view = self._build_slot_mapping_replicated_view(
            common_attn_metadata,
            block_table_replicated_view,
        )
        if get_ascend_config().c8_enable_reshape_optim:
            slot_mapping_replicated_view_cpu = slot_mapping_replicated_view.to("cpu")
        else:
            slot_mapping_replicated_view_cpu = dcp_slot_mapping_cpu

        common_attn_metadata.slot_mapping = slot_mapping_replicated_view
        common_attn_metadata.block_table_tensor = block_table_replicated_view
        common_attn_metadata.slot_mapping_cpu = slot_mapping_replicated_view_cpu
        try:
            metadata = build_metadata()
        finally:
            common_attn_metadata.slot_mapping = dcp_slot_mapping
            common_attn_metadata.block_table_tensor = dcp_block_table
            common_attn_metadata.slot_mapping_cpu = dcp_slot_mapping_cpu

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

        metadata.dcp_context = DCPContext(
            slot_mapping=dcp_slot_mapping[:num_input_tokens],
            block_table=dcp_block_table[:num_reqs],
            seq_lens=local_seq_lens,
        )
        self._update_dsa_cp_slot_mapping_for_dcp(metadata, dcp_slot_mapping, num_input_tokens)
        return metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendSFAMetadata:
        return self._build_with_replicated_view_metadata(
            common_attn_metadata,
            lambda: super(AscendSFADCPMetadataBuilder, self).build(
                common_prefix_len,
                common_attn_metadata,
                fast_build,
                **kwargs,
            ),
            **kwargs,
        )

    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
        **kwargs,
    ) -> AscendSFAMetadata:
        return self._build_with_replicated_view_metadata(
            common_attn_metadata,
            lambda: super(AscendSFADCPMetadataBuilder, self).build_for_drafting(
                common_attn_metadata,
                draft_index,
                **kwargs,
            ),
            **kwargs,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        **kwargs,
    ):
        if attn_state not in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            raise NotImplementedError("Currently we only support building dummy metadata for DecodeOnly state")

        attn_metadata = self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )
        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendSFADCPImpl(AscendSFAImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **kwargs,
        )
        # DCP shards only the SFA KV cache. MLAPO writes the SFA KV cache
        # internally, so keep DCP on the native path where we pass the DCP
        # slot mapping explicitly.
        self.enable_mlapo = False
        dcp_group = get_dcp_group()
        self.dcp_size = dcp_group.world_size
        self.dcp_rank = dcp_group.rank_in_group if self.dcp_size > 1 else 0
        self.dcp_group = dcp_group if self.dcp_size > 1 else None
        self.cp_kv_cache_interleave_size = self.vllm_config.parallel_config.cp_kv_cache_interleave_size
        if self.cp_kv_cache_interleave_size <= 0:
            raise RuntimeError(f"Invalid cp_kv_cache_interleave_size: {self.cp_kv_cache_interleave_size}")
        self._dcp_remap_interleave_size = self.cp_kv_cache_interleave_size
        self._dcp_remap_interleave_size_is_one = self._dcp_remap_interleave_size == 1
        self._dcp_remap_scalar_values = {
            "zero": 0,
            "neg_one": -1,
            "dcp_rank": self.dcp_rank,
            "dcp_size": self.dcp_size,
            "interleave_size": self._dcp_remap_interleave_size,
            "combined_size": self.dcp_size * self._dcp_remap_interleave_size,
        }
        self._dcp_remap_scalar_tensors: dict[tuple[str, torch.dtype], dict[str, torch.Tensor]] = {}
        self._dcp_remap_order_buffers: dict[tuple[str, int, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
        self._dcp_remap_index_topk = None
        for config in (
            getattr(self.vllm_config.model_config, "hf_text_config", None),
            getattr(self.vllm_config.model_config, "hf_config", None),
        ):
            index_topk = getattr(config, "index_topk", None)
            if isinstance(index_topk, int) and index_topk > 0:
                self._dcp_remap_index_topk = index_topk
                break
        if self.dcp_size > 1 and torch.npu.is_available():
            device = torch.device("npu", torch.npu.current_device())
            self._get_dcp_remap_constants(device, torch.float32)
            if self._dcp_remap_index_topk is not None:
                self._get_dcp_remap_order_and_count(self._dcp_remap_index_topk, device, torch.float32)

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

    def _get_dcp_remap_constants(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        key = (str(device), dtype)
        constants = self._dcp_remap_scalar_tensors.get(key)
        if constants is not None:
            return constants

        constants = {
            name: torch.tensor(value, dtype=dtype, device=device)
            for name, value in self._dcp_remap_scalar_values.items()
        }
        self._dcp_remap_scalar_tensors[key] = constants
        return constants

    def _get_dcp_remap_order_and_count(
        self,
        topk_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), topk_count, dtype)
        order_and_count = self._dcp_remap_order_buffers.get(key)
        if order_and_count is None:
            order = torch.arange(topk_count, dtype=dtype, device=device)
            count = torch.tensor(topk_count, dtype=dtype, device=device)
            order_and_count = (order, count)
            self._dcp_remap_order_buffers[key] = order_and_count
        return order_and_count

    def _map_indices_to_local(self, topk_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        constants = self._get_dcp_remap_constants(topk_indices.device, torch.float32)
        topk_float = topk_indices.to(torch.float32)
        local_block_indices = torch.floor(topk_float / constants["interleave_size"])
        local_owner_base = torch.floor(local_block_indices / constants["dcp_size"]) * constants["dcp_size"]
        local_owner = local_block_indices - local_owner_base
        local_owner_mask = (topk_float >= constants["zero"]) & (local_owner == constants["dcp_rank"])
        if self._dcp_remap_interleave_size_is_one:
            remapped_float = torch.floor(topk_float / constants["dcp_size"])
        else:
            local_offsets = topk_float - local_block_indices * constants["interleave_size"]
            remapped_float = torch.floor(topk_float / constants["combined_size"])
            remapped_float = remapped_float * constants["interleave_size"] + local_offsets
        remapped_float = torch.where(local_owner_mask, remapped_float, constants["neg_one"])
        return remapped_float.to(topk_indices.dtype), local_owner_mask

    def _remap_sparse_indices(self, topk_indices: torch.Tensor) -> torch.Tensor:
        if self.dcp_size <= 1:
            return topk_indices

        # Remap the top-k indices to local KV indices and filter out non-local indices.
        remapped_indices, local_owner_mask = self._map_indices_to_local(topk_indices)

        # Compact local indices to the front without changing their top-k order.
        topk_count = topk_indices.shape[-1]
        pack_key_dtype = torch.float32
        original_order, topk_count_tensor = self._get_dcp_remap_order_and_count(
            topk_count,
            topk_indices.device,
            pack_key_dtype,
        )
        original_order = original_order.expand_as(topk_indices)
        pack_keys = original_order + (~local_owner_mask).to(pack_key_dtype) * topk_count_tensor
        _, pack_order = torch.sort(pack_keys, dim=-1)
        return torch.gather(remapped_indices, dim=-1, index=pack_order.to(torch.int32))

    def _merge_dsa_cp_dcp_outputs(
        self,
        sfa_output: torch.Tensor,
        softmax_lse: torch.Tensor,
    ) -> torch.Tensor:
        # DSA-CP keeps the head dimension replicated/full.  Only DCP shards KV,
        # so merge the per-DCP partial outputs explicitly instead of using the
        # common CP helper, which assumes DCP also shards heads.
        num_tokens, num_heads, _ = sfa_output.shape
        assert self.dcp_group is not None
        out_flat = self.dcp_group.all_gather(sfa_output.contiguous(), dim=0)
        lse_flat = self.dcp_group.all_gather(softmax_lse.contiguous(), dim=0)
        out_flat = out_flat.view(
            self.dcp_size,
            num_tokens,
            num_heads,
            self.kv_lora_rank,
        )
        lse_flat = lse_flat.view(
            self.dcp_size,
            num_tokens,
            num_heads,
            1,
        )
        out_flat = out_flat.to(torch.float32).flatten(1, 2)
        lse_flat = lse_flat.to(torch.float32).flatten(1, -1)
        output, _ = torch_npu.npu_attention_update(lse_flat.unbind(0), out_flat.unbind(0), 0)
        return output.view(num_tokens, num_heads, self.kv_lora_rank)

    @staticmethod
    def _merge_dcp_outputs_with_torch(
        output_recv: torch.Tensor,
        lse_recv: torch.Tensor,
    ) -> torch.Tensor:
        lse_recv = torch.where(
            torch.isnan(lse_recv) | torch.isinf(lse_recv),
            torch.full_like(lse_recv, float("-inf")),
            lse_recv,
        )
        lse_max = torch.amax(lse_recv, dim=0)
        lse_max = torch.where(
            lse_max == float("-inf"),
            torch.zeros_like(lse_max),
            lse_max,
        )
        weights = torch.exp(lse_recv - lse_max.unsqueeze(0))
        weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-10)

        output = (output_recv.to(torch.float32) * weights.unsqueeze(-1)).sum(dim=0)
        return output.permute(1, 0, 2).contiguous()

    def _merge_dcp_outputs(
        self,
        sfa_output: torch.Tensor,
        softmax_lse: torch.Tensor,
    ) -> torch.Tensor:
        assert self.dcp_group is not None
        num_tokens, num_heads, head_size = sfa_output.shape
        assert num_heads % self.dcp_size == 0
        local_num_heads = num_heads // self.dcp_size

        output_send = sfa_output.permute(1, 0, 2).contiguous()
        output_recv = torch.empty_like(output_send)
        dist.all_to_all_single(
            output_recv,
            output_send,
            group=self.dcp_group.device_group,
        )
        output_recv = output_recv.view(self.dcp_size, local_num_heads, num_tokens, head_size)

        lse_send = softmax_lse.to(torch.float32).permute(1, 0, 2).contiguous()
        lse_recv = torch.empty_like(lse_send)
        dist.all_to_all_single(
            lse_recv,
            lse_send,
            group=self.dcp_group.device_group,
        )
        lse_recv = lse_recv.view(self.dcp_size, local_num_heads, num_tokens)

        return self._merge_dcp_outputs_with_torch(output_recv, lse_recv)

    def _select_dsa_cp_local_tokens(self, output: torch.Tensor, attn_metadata: M) -> torch.Tensor:
        assert attn_metadata.dsa_cp_context is not None
        dsa_cp_context = attn_metadata.dsa_cp_context
        return output[dsa_cp_context.local_start : dsa_cp_context.local_end_with_pad]

    def _start_dcp_query_gather(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        attn_metadata: M,
    ) -> DCPQueryGatherContext:
        assert attn_metadata.dcp_context is not None
        query_gather_dim = 0 if self.enable_dsa_cp else 1
        assert self.dcp_group is not None
        if ql_nope.shape[:-1] != q_pe.shape[:-1] or ql_nope.dtype != q_pe.dtype:
            raise RuntimeError(
                "Cannot fuse DCP query gather for ql_nope/q_pe with "
                f"shapes {tuple(ql_nope.shape)} / {tuple(q_pe.shape)} "
                f"and dtypes {ql_nope.dtype} / {q_pe.dtype}."
            )

        ql_nope_dim = ql_nope.shape[-1]
        q_pe_dim = q_pe.shape[-1]
        # Avoid back-to-back DCP all_gather calls for the two SFA query
        # fragments. On Ascend the separate gathers can leave SFA with an
        # incomplete stream dependency on the first prefill. DSA-CP restores
        # token shards on dim 0; native DCP restores query shards on dim 1.
        fused_q = torch.cat([ql_nope, q_pe], dim=-1).contiguous()
        gathered, handle, restore_perm = self._all_gather_dim_async(fused_q, query_gather_dim)
        return gathered, handle, restore_perm, ql_nope_dim, q_pe_dim

    def _record_dcp_query_gather_context(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        attn_metadata: M,
    ) -> None:
        assert attn_metadata.dcp_context is not None
        attn_metadata.dcp_context.query_gather_context = self._start_dcp_query_gather(ql_nope, q_pe, attn_metadata)

    def _finish_all_gather_query_for_dcp(
        self,
        gathered: torch.Tensor,
        handle: torch.distributed.Work | None,
        restore_perm: tuple[int, ...] | None,
        ql_nope_dim: int,
        q_pe_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if handle is not None:
            handle.wait()
        if restore_perm is not None:
            gathered = gathered.permute(restore_perm).contiguous()
        fused_q = gathered
        ql_nope, q_pe = torch.split(fused_q, [ql_nope_dim, q_pe_dim], dim=-1)
        return ql_nope, q_pe

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
    ):
        assert attn_metadata.dcp_context is not None
        assert self.dcp_group is not None
        dcp_context = attn_metadata.dcp_context
        query_gather_context = dcp_context.query_gather_context
        dcp_context.query_gather_context = None
        if query_gather_context is None:
            query_gather_context = self._start_dcp_query_gather(ql_nope, q_pe, attn_metadata)
        q_gathered, q_handle, q_restore_perm, ql_nope_dim, q_pe_dim = query_gather_context
        if self.enable_dsa_cp:
            # DSA-CP shards the token sequence. Restore the flat token order for
            # SFA, and use the original full query lengths for varlen metadata.
            actual_seq_lengths_query = attn_metadata.cum_query_lens
            # topk_indices are in per-request global token coordinates. Gather
            # the DSA token shards first, then remap for this receiver rank's
            # DCP-local KV shard.
            topk_indices = self.dcp_group.all_gather(topk_indices.contiguous(), dim=0)
        topk_indices = self._remap_sparse_indices(topk_indices)
        ql_nope, q_pe = self._finish_all_gather_query_for_dcp(
            q_gathered,
            q_handle,
            q_restore_perm,
            ql_nope_dim,
            q_pe_dim,
        )
        kv = kv_cache[0]
        key_rope = kv_cache[1]
        sfa_output, softmax_max, softmax_sum = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv,
            value=kv,
            sparse_indices=topk_indices,
            scale_value=self.scale,
            sparse_block_size=1,
            block_table=attn_metadata.dcp_context.block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=attn_metadata.dcp_context.seq_lens,
            query_rope=q_pe,
            key_rope=key_rope,
            layout_query="TND",
            layout_kv="PA_BSND",
            # The replicated-view indexer already applies the causal visibility rule.
            # After DCP remaps topk indices to local KV positions, local KV
            # length no longer shares the same coordinate system as global
            # query length, so SFA must not apply its right-down causal crop.
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=True,
        )
        output_dtype = sfa_output.dtype
        # SFA returns softmax max/sum separately. Convert them to LSE so the
        # existing CP merge helper can combine local-KV partial outputs.
        softmax_lse = softmax_max.to(torch.float32) + torch.log(softmax_sum.to(torch.float32))
        softmax_lse = softmax_lse.permute(1, 0, 2).reshape(softmax_lse.shape[1], -1, 1)
        if self.enable_dsa_cp:
            output = self._merge_dsa_cp_dcp_outputs(sfa_output, softmax_lse)
            output = self._select_dsa_cp_local_tokens(output, attn_metadata)
            return output.to(output_dtype)
        output = self._merge_dcp_outputs(sfa_output, softmax_lse)
        return output.to(output_dtype)
