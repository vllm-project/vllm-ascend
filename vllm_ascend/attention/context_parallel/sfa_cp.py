from typing import TypeVar

import numpy as np
import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.attention.context_parallel.common_cp import AscendPCPMetadata
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, enabling_mlapo, split_decodes_and_prefills
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.utils import vllm_version_is

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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendSFAMetadata:
        metadata_cls = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
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
                num_tokens_per_request = num_decode_tokens // num_decodes
                decode_slot_mapping = self.slot_mapping_buf[: num_decode_tokens * self.pcp_size].reshape(
                    num_decodes, -1
                )
                decode_slot_mapping[:, :num_tokens_per_request] = decode_slot_mapping[
                    :, : num_tokens_per_request * self.pcp_size : self.pcp_size
                ]
                decode_slot_mapping[:, num_tokens_per_request : num_tokens_per_request * self.pcp_size].fill_(-1)
                self.slot_mapping_buf[: num_decode_tokens * self.pcp_size] = decode_slot_mapping.flatten()
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
        attn_output = torch.ops._C_ascend.npu_sparse_flash_attention(
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
        if vllm_version_is("0.19.0"):
            weights, _ = self.weights_proj(x)
        else:
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
            topk_indices = torch.ops._C_ascend.npu_lightning_indexer(
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
