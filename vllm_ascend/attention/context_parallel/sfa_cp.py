from typing import TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.attention.context_parallel.common_cp import (
    AscendPCPMetadata,
    _npu_attention_update,
    _process_attn_out_lse,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, enabling_mlapo, split_decodes_and_prefills
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
        self.decode_actual_seq_lengths_key_buf = torch.zeros(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=device,
        )

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

            restore_idx = sfa_cp_metadata.pcp_allgather_restore_idx
            assert torch.is_tensor(restore_idx), "pcp_allgather_restore_idx must be a device tensor."
            if num_decode_tokens == 0:
                sfa_cp_metadata.prefill_allgather_restore_idx = restore_idx
            else:
                # pcp_allgather_restore_idx is built for a mixed per-rank layout:
                #   rank0: [decode, prefill], rank1: [decode, prefill], ...
                # Here we all-gather only prefill tensors, whose layout is:
                #   rank0: [prefill], rank1: [prefill], ...
                # Example: decode=2, prefill=3, pcp=2. Mixed index 8 is
                # rank1/P1, but in prefill-only layout it becomes index 4.
                local_tokens_with_decode = num_decode_tokens + num_prefill_tokens
                rank_idx = torch.div(restore_idx, local_tokens_with_decode, rounding_mode="floor")
                local_idx = restore_idx.remainder(local_tokens_with_decode)
                prefill_mask = local_idx >= num_decode_tokens
                sfa_cp_metadata.prefill_allgather_restore_idx = rank_idx[prefill_mask] * num_prefill_tokens + (
                    local_idx[prefill_mask] - num_decode_tokens
                )

        if num_decode_tokens > 0:
            # Decode reads only the KV blocks stored on this CP rank. The kernel
            # therefore needs local per-request KV lengths, not global seq_lens.
            total_cp_size = self.pcp_size * self.dcp_size
            cp_rank = self.pcp_rank * self.dcp_size + self.dcp_rank
            decode_seq_lens = metadata_cls.seq_lens[:num_decodes].to(torch.int32)
            base = decode_seq_lens // self.cp_local_block_size // total_cp_size * self.cp_local_block_size
            remainder = decode_seq_lens - base * total_cp_size
            decode_actual_seq_lengths_key = base + torch.clamp(
                remainder - cp_rank * self.cp_local_block_size,
                0,
                self.cp_local_block_size,
            )
            self.decode_actual_seq_lengths_key_buf.zero_()
            self.decode_actual_seq_lengths_key_buf[:num_decodes].copy_(decode_actual_seq_lengths_key)
            sfa_cp_metadata.decode_actual_seq_lengths_key = self.decode_actual_seq_lengths_key_buf[:num_decodes]
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
            elif num_decode_tokens > 0:
                self.slot_mapping_buf[:num_decode_tokens] = self.slot_mapping_buf[
                    : num_decode_tokens * self.pcp_size : self.pcp_size
                ]
                self.slot_mapping_buf[num_decode_tokens : num_decode_tokens * self.pcp_size].fill_(-1)
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
        q_head_actual_kv_lens = torch.tensor(
            common_long_seq_metadata.head_actual_seq_lengths_kv,
            dtype=torch.int32,
            device=seq_lens.device,
        )
        q_tail_actual_kv_lens = torch.tensor(
            common_long_seq_metadata.tail_actual_seq_lengths_kv,
            dtype=torch.int32,
            device=seq_lens.device,
        )
        # pcp_utils stores cumulative TND lengths. SFA PA_BSND expects a
        # per-request length list, so convert [a, a+b, ...] to [a, b, ...].
        if q_head_actual_kv_lens.numel() > 1:
            q_head_actual_kv_lens[1:] -= q_head_actual_kv_lens[:-1].clone()
        if q_tail_actual_kv_lens.numel() > 1:
            q_tail_actual_kv_lens[1:] -= q_tail_actual_kv_lens[:-1].clone()
        return AscendPCPMetadata(
            q_head_idx=common_long_seq_metadata.q_head_idx_tensor,
            q_tail_idx=common_long_seq_metadata.q_tail_idx_tensor,
            q_full_idx=common_long_seq_metadata.q_full_idx,
            head_actual_seq_lengths_kv=q_head_actual_kv_lens,
            tail_actual_seq_lengths_kv=q_tail_actual_kv_lens,
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
            decode_block_table = attn_metadata.block_table[:num_decodes]
            decode_kv = kv
            decode_key_rope = key_rope
            decode_actual_seq_lengths_key = sfa_cp_metadata.decode_actual_seq_lengths_key
            assert decode_actual_seq_lengths_key is not None
            decode_actual_seq_lengths_key = decode_actual_seq_lengths_key[:num_decodes]
            decode_ql_nope, decode_q_pe = self._gather_decode_sfa_q_across_dcp(
                ql_nope[:num_decode_tokens], q_pe[:num_decode_tokens]
            )
            decode_partial, decode_softmax_max, decode_softmax_sum = self._execute_sparse_flash_attention(
                decode_ql_nope,
                decode_q_pe,
                decode_kv,
                decode_key_rope,
                decode_block_table,
                topk_indices[:num_decode_tokens],
                actual_seq_lengths_query[:num_decodes],
                decode_actual_seq_lengths_key,
                return_softmax_lse=True,
            )
            decode_attn_out = self._merge_decode_sfa_output(
                decode_partial, decode_softmax_max, decode_softmax_sum
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
            prefill_attn_out, _, _ = self._execute_sparse_flash_attention(
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
        q_head_actual_seq_lengths_key = sfa_cp_metadata.head_actual_seq_lengths_kv
        q_head_output, _, _ = self._execute_sparse_flash_attention(
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
        q_tail_actual_seq_lengths_key = sfa_cp_metadata.tail_actual_seq_lengths_kv
        q_tail_output, _, _ = self._execute_sparse_flash_attention(
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

    def _all_gather_rank_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        # Gather local topk scores into an explicit leading CP-rank dimension:
        # [cp_rank, token, kv_head, topk].
        rank_tensor = tensor.unsqueeze(0)
        if self.dcp_size > 1:
            dcp_out = torch.empty(
                (self.dcp_size, *tensor.shape),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            dist.all_gather_into_tensor(dcp_out, tensor.contiguous(), group=self.dcp_group)
            rank_tensor = dcp_out

        if self.pcp_size > 1:
            cp_world_size = self.pcp_size * self.dcp_size
            pcp_out = torch.empty(
                (self.pcp_size, *rank_tensor.shape),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            dist.all_gather_into_tensor(pcp_out, rank_tensor.contiguous(), group=self.pcp_group)
            rank_tensor = pcp_out.reshape(cp_world_size, *tensor.shape)

        return rank_tensor

    def _global_topk_to_local_indices(self, local_indices: torch.Tensor, local_scores: torch.Tensor) -> torch.Tensor:
        valid_score = local_scores.to(torch.float32)
        valid_score = torch.where(local_indices >= 0, valid_score, torch.full_like(valid_score, -float("inf")))

        gathered_scores = self._all_gather_rank_dim(valid_score)
        rank_count = gathered_scores.shape[0]
        topk_count = gathered_scores.shape[-1]
        flat_scores = gathered_scores.permute(1, 2, 0, 3).reshape(
            local_scores.shape[0],
            local_scores.shape[1],
            rank_count * topk_count,
        )
        global_scores, global_positions = torch.topk(flat_scores, topk_count, dim=-1)
        # global_positions encodes both owner rank and that rank's local topk
        # slot. Keep only winners owned by this rank; other ranks will keep
        # their own winners for the same global topk set.
        owner_rank = torch.div(global_positions, topk_count, rounding_mode="floor")
        local_slot = global_positions.remainder(topk_count)

        cp_rank = self.pcp_rank * self.dcp_size + self.dcp_rank
        current_mask = owner_rank == cp_rank
        current_scores = torch.where(current_mask, global_scores, torch.full_like(global_scores, -float("inf")))
        _, current_order = torch.topk(current_scores, topk_count, dim=-1)
        current_slot = torch.gather(local_slot, -1, current_order)
        current_valid = torch.gather(current_mask, -1, current_order)
        current_indices = torch.gather(local_indices, -1, current_slot)
        return torch.where(current_valid, current_indices, torch.full_like(current_indices, -1))

    def _merge_decode_sfa_output(
        self,
        partial_output: torch.Tensor,
        softmax_max: torch.Tensor,
        softmax_sum: torch.Tensor,
    ) -> torch.Tensor:
        output_dtype = partial_output.dtype
        # SFA returns softmax max/sum separately. Convert them to LSE so the
        # existing CP merge helper can combine local-KV partial outputs.
        softmax_lse = softmax_max.to(torch.float32) + torch.log(softmax_sum.to(torch.float32))
        softmax_lse = softmax_lse.permute(1, 0, 2).reshape(softmax_lse.shape[1], -1, 1)
        attn_out_lse = _process_attn_out_lse(partial_output.to(torch.float32), softmax_lse)
        output = _npu_attention_update(self.kv_lora_rank, attn_out_lse)
        return output.to(output_dtype)

    def _gather_and_restore_prefill_kv_cross_pcp(
        self, prefill_tensor: torch.Tensor, attn_metadata: M
    ) -> torch.Tensor:
        prefill_tokens = prefill_tensor.shape[0]
        if prefill_tokens == 0:
            return prefill_tensor

        # Only prefill participates in this PCP all-gather. Use the prefill-only
        # restore index built from the mixed [decode, prefill] restore index.
        gathered = get_pcp_group().all_gather(prefill_tensor.contiguous(), 0)
        assert attn_metadata.sfa_cp_metadata is not None
        restore_idx = attn_metadata.sfa_cp_metadata.prefill_allgather_restore_idx
        return torch.index_select(gathered, 0, restore_idx)

    def _pad_decode_tensor_for_pcp_slot_mapping(
        self,
        decode_tensor: torch.Tensor,
        num_decode_tokens: int,
    ) -> torch.Tensor:
        if num_decode_tokens == 0 or self.pcp_size == 1:
            return decode_tensor
        # slot_mapping still reserves decode_tokens * pcp_size entries. Pad
        # decode k_li so its rows stay aligned with that PCP-padded mapping.
        pad_tokens = num_decode_tokens * (self.pcp_size - 1)
        pad_shape = (pad_tokens, *decode_tensor.shape[1:])
        pad = torch.zeros(pad_shape, dtype=decode_tensor.dtype, device=decode_tensor.device)
        return torch.cat([decode_tensor, pad], dim=0)

    def _gather_decode_sfa_q_across_dcp(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dcp_size == 1:
            return ql_nope, q_pe
        q = torch.cat([ql_nope, q_pe], dim=-1)
        q = get_dcp_group().all_gather(q.contiguous(), 1)
        return q.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

    def _execute_sparse_flash_attention(
        self,
        ql_nope,
        q_pe,
        kv,
        key_rope,
        block_table,
        topk_indices,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        return_softmax_lse: bool = False,
    ):
        sfa_output, softmax_max, softmax_sum = torch.ops._C_ascend.npu_sparse_flash_attention(
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
            return_softmax_lse=return_softmax_lse,
        )
        return sfa_output, softmax_max, softmax_sum

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
            decode_block_table = attn_metadata.block_table[:num_decodes]
            decode_key = key
            decode_actual_seq_lengths_key = sfa_cp_metadata.decode_actual_seq_lengths_key
            assert decode_actual_seq_lengths_key is not None
            decode_actual_seq_lengths_key = decode_actual_seq_lengths_key[:num_decodes]
            decode_topk_indices, decode_scores = self._execute_indexer_select(
                q[:num_decode_tokens],
                decode_key,
                weights[:num_decode_tokens],
                actual_seq_lengths_query[:num_decodes],
                decode_actual_seq_lengths_key,
                decode_block_table,
                return_value=True,
            )
            decode_topk_indices = self._global_topk_to_local_indices(decode_topk_indices, decode_scores)
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
            prefill_topk_indices, _ = self._execute_indexer_select(
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
        q_head_actual_seq_lengths_key = sfa_cp_metadata.head_actual_seq_lengths_kv
        q_head_topk_indices, _ = self._execute_indexer_select(
            q=torch.index_select(prefill_q, 0, q_head_idx),
            key=prefill_key,
            weights=torch.index_select(prefill_weights, 0, q_head_idx),
            actual_seq_lengths_query=sfa_cp_metadata.prefill_q_cum_seqlens // 2,
            actual_seq_lengths_key=q_head_actual_seq_lengths_key,
            block_table=prefill_block_table,
        )

        # q tail compute
        q_tail_actual_seq_lengths_key = sfa_cp_metadata.tail_actual_seq_lengths_kv
        q_tail_topk_indices, _ = self._execute_indexer_select(
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

    def _execute_indexer_select(
        self,
        q,
        key,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        return_value: bool = False,
    ):
        topk_indices, topk_scores = torch.ops._C_ascend.npu_lightning_indexer(
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
            return_value=return_value,
        )
        return topk_indices, topk_scores

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
        slot_mapping = attn_metadata.slot_mapping
        num_decode_tokens = attn_metadata.num_decode_tokens

        if num_decode_tokens > 0:
            torch_npu._npu_reshape_and_cache(
                key=kv_c_normed[:num_decode_tokens],
                value=k_pe[:num_decode_tokens],
                key_cache=kv_cache[0],
                value_cache=kv_cache[1],
                slot_indices=slot_mapping[:num_decode_tokens],
            )

        if kv_c_normed.shape[0] > num_decode_tokens:
            prefill_kv_c_k_pe = torch.cat(
                [kv_c_normed[num_decode_tokens:], k_pe[num_decode_tokens:]], dim=-1
            )
            prefill_kv_c_k_pe = self._gather_and_restore_prefill_kv_cross_pcp(prefill_kv_c_k_pe, attn_metadata)
            prefill_kv_c_normed, prefill_k_pe = prefill_kv_c_k_pe.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            prefill_slot_mapping = slot_mapping[num_decode_tokens * self.pcp_size :]
            torch_npu._npu_reshape_and_cache(
                key=prefill_kv_c_normed,
                value=prefill_k_pe,
                key_cache=kv_cache[0],
                value_cache=kv_cache[1],
                slot_indices=prefill_slot_mapping,
            )
        return None, None

    def _get_full_kv(self, k, attn_metadata: M):
        if self.pcp_size == 1 or self.enable_mlapo:
            return k

        assert attn_metadata.sfa_cp_metadata is not None
        num_decode_tokens = attn_metadata.num_decode_tokens
        decode_k = self._pad_decode_tensor_for_pcp_slot_mapping(k[:num_decode_tokens], num_decode_tokens)

        if k.shape[0] <= num_decode_tokens:
            return decode_k

        prefill_k = self._gather_and_restore_prefill_kv_cross_pcp(
            k[num_decode_tokens:],
            attn_metadata,
        )
        if num_decode_tokens == 0:
            return prefill_k
        return torch.cat([decode_k, prefill_k], dim=0)
