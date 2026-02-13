from typing import TypeVar

import numpy as np
import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group, get_pcp_group

from vllm_ascend.attention.context_parallel.common_cp import AscendPCPMetadata
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, enabling_mlapo, split_decodes_and_prefills

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

        block_table = metadata_cls.block_table
        valid_block_ids, new_block_table = block_table.flatten().unique(return_inverse=True)
        num_blocks = valid_block_ids.shape[0]
        # Note(qcs): `block_table_cp` will have dirty values in the part beyond kv_lens.
        # We assume that we can always get the correct kv_lens or kv index,
        # so we omit the dirty value processing here.
        block_table_cp = (
            new_block_table.unsqueeze(-1).to(block_table)
            + (torch.arange(self.pcp_size * self.dcp_size) * num_blocks).view(1, 1, -1).to(block_table)
        ).reshape(block_table.shape[0], -1)

        sfa_cp_metadata = self.build_cp_metadata(
            block_table_cp, valid_block_ids, metadata_cls.seq_lens, common_attn_metadata
        )
        metadata_cls.num_decode_tokens = num_decode_tokens
        metadata_cls.num_decodes = num_decodes
        metadata_cls.num_prefills = num_prefills

        if self.pcp_size > 1:
            long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
            assert long_seq_metadata is not None
            num_actual_tokens_pcp_padded = long_seq_metadata.num_actual_tokens_pcp_padded
            slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens_pcp_padded]
            if self.enable_mlapo:
                slot_mapping[:num_decode_tokens] = slot_mapping[: num_decode_tokens * self.pcp_size : self.pcp_size]
                slot_mapping[num_decode_tokens : num_decode_tokens * self.pcp_size].fill_(-1)
            metadata_cls.slot_mapping = slot_mapping
            actual_seq_lengths_query = metadata_cls.cum_query_lens
            if num_prefills > 0 and num_decode_tokens > 0:
                prefill_q_cum_seqlens = (
                    actual_seq_lengths_query[num_decode_tokens:] - actual_seq_lengths_query[num_decode_tokens - 1]
                )
            else:
                prefill_q_cum_seqlens = actual_seq_lengths_query
            assert sfa_cp_metadata is not None
            sfa_cp_metadata.prefill_q_cum_seqlens = prefill_q_cum_seqlens
        metadata_cls.sfa_cp_metadata = sfa_cp_metadata
        return metadata_cls

    def build_cp_metadata(
        self,
        block_table_cp: torch.Tensor,
        valid_block_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendPCPMetadata | None:
        common_long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        assert common_long_seq_metadata is not None
        q_head_kv_lens = (seq_lens // 2) * (self.pcp_rank + 1)
        q_tail_kv_lens = seq_lens * self.pcp_size - (seq_lens // 2) * self.pcp_rank
        return AscendPCPMetadata(
            q_head_idx=common_long_seq_metadata.q_head_idx_tensor,
            q_tail_idx=common_long_seq_metadata.q_tail_idx_tensor,
            q_full_idx=common_long_seq_metadata.q_full_idx,
            head_attn_nomask_seqlens=q_head_kv_lens,
            tail_attn_nomask_seqlens=q_tail_kv_lens,
            pcp_allgather_restore_idx=common_long_seq_metadata.pcp_allgather_restore_idx,
            block_table_cp=block_table_cp,
            valid_block_ids=valid_block_ids,
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
        valid_block_ids = attn_metadata.sfa_cp_metadata.valid_block_ids
        kv = self.gather_kv_cross_cp(kv, valid_block_ids)
        key_rope = self.gather_kv_cross_cp(key_rope, valid_block_ids)
        block_table = attn_metadata.sfa_cp_metadata.block_table_cp

        if self.pcp_size == 1:
            return self._execute_sparse_flash_attention(
                ql_nope, q_pe, kv, key_rope, block_table, topk_indices, actual_seq_lengths_query, actual_seq_lengths_key
            )
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills
        decode_attn_out = None
        if num_decode_tokens > 0:
            decode_attn_out = self._execute_sparse_flash_attention(
                ql_nope[:num_decode_tokens],
                q_pe[:num_decode_tokens],
                kv,
                key_rope,
                block_table[:num_decode_tokens],
                topk_indices[:num_decode_tokens],
                actual_seq_lengths_query[:num_decode_tokens],
                actual_seq_lengths_key[:num_decode_tokens],
            )

        if num_prefills < 1:
            return decode_attn_out

        # q split for head and tail
        q_head_idx = attn_metadata.sfa_cp_metadata.q_head_idx
        q_tail_idx = attn_metadata.sfa_cp_metadata.q_tail_idx
        ql_nope = ql_nope[num_decode_tokens:]
        q_pe = q_pe[num_decode_tokens:]
        topk_indices = topk_indices[num_decode_tokens:]
        block_table = block_table[num_decode_tokens:]

        # q head compute
        q_head_actual_seq_lengths_key = attn_metadata.sfa_cp_metadata.head_attn_nomask_seqlens[num_decode_tokens:]
        q_head_output = self._execute_sparse_flash_attention(
            torch.index_select(ql_nope, 0, q_head_idx),
            torch.index_select(q_pe, 0, q_head_idx),
            kv,
            key_rope,
            block_table,
            torch.index_select(topk_indices, 0, q_head_idx),
            attn_metadata.sfa_cp_metadata.prefill_q_cum_seqlens // 2,
            q_head_actual_seq_lengths_key,
        )

        # q tail compute
        q_tail_actual_seq_lengths_key = attn_metadata.sfa_cp_metadata.tail_attn_nomask_seqlens[num_decode_tokens:]
        q_tail_output = self._execute_sparse_flash_attention(
            torch.index_select(ql_nope, 0, q_tail_idx),
            torch.index_select(q_pe, 0, q_tail_idx),
            kv,
            key_rope,
            block_table,
            torch.index_select(topk_indices, 0, q_tail_idx),
            attn_metadata.sfa_cp_metadata.prefill_q_cum_seqlens // 2,
            q_tail_actual_seq_lengths_key,
        )

        q_full_idx = attn_metadata.sfa_cp_metadata.q_full_idx
        attn_output = torch.index_select(torch.cat([q_head_output, q_tail_output], dim=0), 0, q_full_idx)

        if decode_attn_out is not None:
            attn_output = torch.cat([decode_attn_out, attn_output], dim=0)
        return attn_output

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

    def gather_kv_cross_cp(self, kv_cache: torch.Tensor, valid_block_ids: torch.Tensor) -> torch.Tensor:
        # Note(qcs): we need set kv_cache_interleave_size = block_size for sfa!!!
        kv_cache = torch.index_select(kv_cache, 0, valid_block_ids)
        if self.dcp_size > 1:
            kv_cache = get_dcp_group().all_gather(kv_cache, 0)
        if self.pcp_size > 1:
            kv_cache = get_pcp_group().all_gather(kv_cache, 0)
        return kv_cache

    def indexer_select_post_process(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        q: torch.Tensor | None,
        k: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        need_gather_q_kv: bool = False,
    ):
        if q is None:
            q, _ = self.wq_b(qr)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
            q = q.view(-1, self.n_head, self.head_dim)  # [n_toks,64,128]
            cos_q, sin_q = cos, sin

            q_pe, q_nope = torch.split(
                q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )  # [b,s,64,64+64]

            q_pe = q_pe.unsqueeze(2)
            q_pe = torch_npu.npu_rotary_mul(q_pe, cos_q, sin_q)
            q_pe = q_pe.squeeze(2)
            q = torch.cat([q_pe, q_nope], dim=-1)  # [b*s,64,128]

        if kv_cache is not None:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            torch_npu.npu_scatter_nd_update_(
                kv_cache[2].view(-1, k.shape[-1]), attn_metadata.slot_mapping.view(-1, 1), k.view(-1, k.shape[-1])
            )  # b, s, n, d
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()

        weights, _ = self.weights_proj(x)
        weights = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(weights, need_gather_q_kv)

        key = kv_cache[2]
        assert attn_metadata.sfa_cp_metadata is not None
        key = self.gather_kv_cross_cp(key, attn_metadata.sfa_cp_metadata.valid_block_ids)
        block_table = attn_metadata.sfa_cp_metadata.block_table_cp

        if self.pcp_size == 1:
            return self._execute_indexer_select(
                q, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table
            )

        # decode compute
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills
        decode_topk_indices = None
        if num_decode_tokens > 0:
            decode_topk_indices = self._execute_indexer_select(
                q[:num_decode_tokens],
                key,
                weights[:num_decode_tokens],
                actual_seq_lengths_query[:num_decode_tokens],
                actual_seq_lengths_key[:num_decode_tokens],
                block_table[:num_decode_tokens],
            )

        # prefill compute
        if num_prefills == 0:
            return decode_topk_indices
        q = q[num_decode_tokens:]
        weights = weights[num_decode_tokens:]
        actual_seq_lengths_key = actual_seq_lengths_key[num_decode_tokens:]
        block_table = block_table[num_decode_tokens:]
        # pcp split for head and tail
        q_head_idx = attn_metadata.sfa_cp_metadata.q_head_idx
        q_tail_idx = attn_metadata.sfa_cp_metadata.q_tail_idx

        # q head compute
        q_head_actual_seq_lengths_key = attn_metadata.sfa_cp_metadata.head_attn_nomask_seqlens[num_decode_tokens:]
        q_head_topk_indices = self._execute_indexer_select(
            q=torch.index_select(q, 0, q_head_idx),
            key=key,
            weights=torch.index_select(weights, 0, q_head_idx),
            actual_seq_lengths_query=attn_metadata.sfa_cp_metadata.prefill_q_cum_seqlens // 2,
            actual_seq_lengths_key=q_head_actual_seq_lengths_key,
            block_table=block_table,
        )

        # q tail compute
        q_tail_actual_seq_lengths_key = attn_metadata.sfa_cp_metadata.tail_attn_nomask_seqlens[num_decode_tokens:]
        q_tail_topk_indices = self._execute_indexer_select(
            q=torch.index_select(q, 0, q_tail_idx),
            key=key,
            weights=torch.index_select(weights, 0, q_tail_idx),
            actual_seq_lengths_query=attn_metadata.sfa_cp_metadata.prefill_q_cum_seqlens // 2,
            actual_seq_lengths_key=q_tail_actual_seq_lengths_key,
            block_table=block_table,
        )

        q_full_idx = attn_metadata.sfa_cp_metadata.q_full_idx
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
        kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
        assert len(kv_cache) > 1, "the number of kv cache should be greater than 1, namely (nope_cache and rope_cache)"
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
            k = get_pcp_group().all_gather(k.contiguous(), 0)
            k = torch.index_select(k, 0, attn_metadata.sfa_cp_metadata.pcp_allgather_restore_idx)
            return k
