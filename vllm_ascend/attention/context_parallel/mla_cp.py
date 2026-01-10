from typing import Optional, Tuple, TypeVar

import numpy as np
import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import (get_dcp_group,
                              get_decode_context_model_parallel_rank,
                              get_decode_context_model_parallel_world_size,
                              get_pcp_group)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

# isort: off
from vllm_ascend.attention.mla_v1 import (
    AscendMLADecodeMetadata, AscendMLAImpl, AscendMLAMetadata,
    AscendMLAMetadataBuilder, AscendMLAPrefillMetadata,
    DecodeMLAPreprocessResult, PrefillMLAPreprocessResult,
    BUILD_METADATA_STEP_PREFILL)
#isort: on

from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata)
from vllm_ascend.attention.context_parallel.common_cp import (
    AscendPCPMetadata, CPChunkedContextMetadata, _process_attn_out_lse,
    _npu_attention_update)
from vllm_ascend.compilation.acl_graph import (get_draft_graph_params,
                                               get_graph_params,
                                               update_graph_params_workspaces)
from vllm_ascend.utils import weak_ref_tensors

MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024

M = TypeVar("M", bound=AscendMLAMetadata)


class AscendMlaCPMetadataBuilder(AscendMLAMetadataBuilder):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec: MLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendMLAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         metadata_cls, supports_dcp_with_varlen)

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group(
        ).rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank(
        ) if self.dcp_size > 1 else 0
        self.cp_local_block_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        self.cp_virtual_block_size = self.cp_local_block_size * self.dcp_size * self.pcp_size
        scheduler_config = vllm_config.scheduler_config
        decode_max_num_seqs = getattr(scheduler_config, 'decode_max_num_seqs',
                                      0)
        max_num_seqs = max(scheduler_config.max_num_seqs, decode_max_num_seqs)
        self.batch_seq_mask_buf = torch.empty(max_num_seqs *
                                              self.decode_threshold,
                                              dtype=torch.uint8,
                                              device=device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMLAMetadata:
        metadata_cls = super().build(common_prefix_len, common_attn_metadata)
        if self.num_prefills == 0 and self.pcp_size > 1:
            self.slot_mapping[:self.
                              num_decode_tokens] = self.slot_mapping[:self.
                                                                     num_decode_tokens
                                                                     * self.
                                                                     pcp_size:
                                                                     self.
                                                                     pcp_size]
            self.slot_mapping[self.num_decode_tokens:self.num_decode_tokens *
                              self.pcp_size].fill_(-1)
        metadata_cls.slot_mapping = self.slot_mapping
        return metadata_cls

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendMlaCPMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def set_num_actual_tokens(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        if long_seq_metadata is None:
            raise AssertionError("long_seq_metadata should not be None.")

        # In dcp only spec decode graph padding case,
        # num_actual_tokens_pcp_padded may be less than num_actual_tokens
        self.num_actual_tokens = max(
            long_seq_metadata.num_actual_tokens_pcp_padded,
            common_attn_metadata.num_actual_tokens)

    def build_cp_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendPCPMetadata | None:
        common_long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        assert common_long_seq_metadata is not None
        return AscendPCPMetadata(
            q_head_idx=common_long_seq_metadata.q_head_idx_tensor,
            q_tail_idx=common_long_seq_metadata.q_tail_idx_tensor,
            kv_with_q_head_nomask_idx=common_long_seq_metadata.
            kv_with_q_head_nomask_idx_tensor,
            kv_with_q_head_mask_idx=common_long_seq_metadata.
            kv_with_q_head_mask_idx_tensor,
            kv_with_q_tail_nomask_idx=common_long_seq_metadata.
            kv_with_q_tail_nomask_idx_tensor,
            kv_with_q_tail_mask_idx=common_long_seq_metadata.
            kv_with_q_tail_mask_idx_tensor,
            attn_mask_seqlens=common_long_seq_metadata.attn_mask_seqlens,
            head_attn_nomask_seqlens=common_long_seq_metadata.
            head_attn_nomask_seqlens,
            tail_attn_nomask_seqlens=common_long_seq_metadata.
            tail_attn_nomask_seqlens,
            q_full_idx=common_long_seq_metadata.q_full_idx,
            pcp_allgather_restore_idx=common_long_seq_metadata.
            pcp_allgather_restore_idx)

    def build_chunked_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        chunked_context_metadata = super().build_chunked_metadata(
            common_prefix_len, common_attn_metadata)
        if chunked_context_metadata is None:
            return None

        long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        assert long_seq_metadata is not None
        num_computed_tokens_of_pcp_dcp = long_seq_metadata.num_computed_tokens_of_pcp_dcp
        assert num_computed_tokens_of_pcp_dcp is not None
        local_context_lens_allranks = torch.tensor(
            num_computed_tokens_of_pcp_dcp[self.num_decodes_flatten:]).reshape(
                -1, self.dcp_size * self.pcp_size)
        # Note(qcs): The max local context lengths
        # padded to `cp_local_block_size`.
        padded_local_context_lens_cpu = (cdiv(
            self.context_lens_cpu,
            self.cp_virtual_block_size,
        ) * self.cp_local_block_size)
        padded_local_max_context_chunk_across_ranks = (cdiv(
            self.max_context_chunk,
            self.cp_virtual_block_size,
        ) * self.cp_local_block_size)
        local_chunk_starts = (torch.arange(
            self.num_chunks, dtype=torch.int32).unsqueeze(1).expand(
                -1, self.num_prefills) *
                              padded_local_max_context_chunk_across_ranks)
        local_chunk_ends = torch.min(
            padded_local_context_lens_cpu.unsqueeze(0),
            local_chunk_starts + padded_local_max_context_chunk_across_ranks,
        )
        padded_local_chunk_seq_lens = (local_chunk_ends -
                                       local_chunk_starts).clamp(min=0)
        padded_local_cu_chunk_seq_lens_cpu = torch.zeros(self.num_chunks,
                                                         self.num_prefills + 1,
                                                         dtype=torch.int32,
                                                         pin_memory=True)
        torch.cumsum(
            padded_local_chunk_seq_lens,
            dim=1,
            out=padded_local_cu_chunk_seq_lens_cpu[:, 1:],
            dtype=torch.int32,
        )
        chunked_metadata = CPChunkedContextMetadata(
            cu_seq_lens=chunked_context_metadata.cu_seq_lens,
            starts=local_chunk_starts.pin_memory().to(self.device,
                                                      non_blocking=True),
            seq_tot=padded_local_chunk_seq_lens.sum(dim=1).tolist(),
            max_seq_lens=chunked_context_metadata.max_seq_lens,
            chunk_seq_lens=self.chunk_seq_lens,
            chunk_seq_lens_npu=chunked_context_metadata.chunk_seq_lens_npu,
            workspace=chunked_context_metadata.workspace,
            padded_chunk_seq_lens_npu=padded_local_chunk_seq_lens.npu(),
            padded_local_chunk_seq_lens=padded_local_chunk_seq_lens.tolist(),
            local_context_lens_allranks=local_context_lens_allranks.tolist(),
            padded_local_cu_seq_lens=padded_local_cu_chunk_seq_lens_cpu.
            pin_memory().to(self.device, non_blocking=True),
            cu_seq_lens_lst=self.cu_seq_lens_cpu.tolist(),
            chunk_size=padded_local_max_context_chunk_across_ranks,
        )
        return chunked_metadata

    def get_block_table_size(
            self, common_attn_metadata: AscendCommonAttentionMetadata,
            build_metadata_step: int):
        self.num_decodes_flatten = self.query_lens[:self.num_decodes].sum(
        ).item()
        if build_metadata_step == BUILD_METADATA_STEP_PREFILL:
            # For pcp + spec decode, we flatten seq_lens and block_table
            # to avoid irregular attn_mask shape
            return self.num_decodes_flatten + self.num_prefills
        else:
            return self.num_decodes_flatten

    def build_prefill_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendMLAPrefillMetadata:
        prefill_metadata = super().build_prefill_metadata(
            common_prefix_len, common_attn_metadata)
        prefill_metadata.pcp_metadata = self.build_cp_metadata(
            common_prefix_len, common_attn_metadata)
        prefill_metadata.block_table = self.block_table[
            self.num_decodes_flatten:, ...]
        return prefill_metadata

    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendMLADecodeMetadata:
        decode_metadata = super().build_decode_metadata(
            common_prefix_len, common_attn_metadata)

        long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        assert long_seq_metadata is not None
        num_computed_tokens_of_pcp_dcp = long_seq_metadata.num_computed_tokens_of_pcp_dcp
        assert num_computed_tokens_of_pcp_dcp is not None
        # [bs, pcp_size, dcp_size]
        num_computed_tokens_of_cp_dcp_array = np.array(
            num_computed_tokens_of_pcp_dcp)[:self.num_decodes_flatten]

        cp_seq_len = num_computed_tokens_of_cp_dcp_array[:, self.pcp_rank,
                                                         self.dcp_rank]
        cp_seq_len = torch.tensor(cp_seq_len, dtype=torch.int32)
        batch_seq_mask = (cp_seq_len == 0)
        self.batch_seq_mask_buf[:batch_seq_mask.shape[0]].copy_(
            batch_seq_mask, non_blocking=True)
        batch_seq_mask = self.batch_seq_mask_buf[:batch_seq_mask.shape[0]]
        cp_seq_len = torch.where(cp_seq_len == 0, 1, cp_seq_len)
        decode_metadata.cp_seq_len = cp_seq_len
        decode_metadata.batch_seq_mask = batch_seq_mask
        return decode_metadata


class AscendMlaCPImpl(AscendMLAImpl):
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
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        **kwargs,
    ):
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **kwargs)

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group(
        ).rank_in_group if self.pcp_size > 1 else 0
        self.pcp_group = get_pcp_group(
        ).device_group if self.pcp_size > 1 else None

        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank(
        ) if self.dcp_size > 1 else 0
        self.dcp_group = get_dcp_group(
        ).device_group if self.dcp_size > 1 else None

    def get_num_actual_tokens(self, attn_metadata: M):
        if self.pcp_size > 1:
            return attn_metadata.num_actual_tokens_pcp_padded // self.pcp_size
        else:
            return attn_metadata.num_actual_tokens

    def _v_up_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.W_UV)
        # # Convert from (N, B, V) to (B, N * V)
        x = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        return x

    def mla_preprocess_prefill(self, q_c, kv_no_split, kv_cache,
                               attn_metadata):
        if not self.pcp_size > 1:
            return super().mla_preprocess_prefill(q_c, kv_no_split, kv_cache,
                                                  attn_metadata)
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_actual_tokens = (attn_metadata.num_actual_tokens_pcp_padded -
                             self.pcp_size * num_decode_tokens
                             ) // self.pcp_size + num_decode_tokens
        prefill_q_c = q_c[num_decode_tokens:num_actual_tokens]
        prefill_q = self.q_proj(prefill_q_c)[0] \
            .view(-1, self.num_heads, self.qk_head_dim)
        prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
        prefill_q_nope = prefill_q[..., :self.qk_nope_head_dim]
        cos = attn_metadata.prefill.cos[:num_actual_tokens - num_decode_tokens]
        sin = attn_metadata.prefill.sin[:num_actual_tokens - num_decode_tokens]
        prefill_q_pe = self.rope_single(prefill_q_pe, cos, sin)
        prefill_kv_no_split = kv_no_split[:num_actual_tokens]
        kv_c, k_pe = prefill_kv_no_split.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
        assert len(
            kv_cache
        ) > 1, "the number of kv cache should be greater than 1, namely (nope_cache and rope_cache)"
        kv_c_normed = kv_c_normed.view(
            [num_actual_tokens, self.num_kv_heads, -1])
        k_pe = k_pe.unsqueeze(1)
        prefill_k_pe = k_pe
        prefill_k_pe[num_decode_tokens:num_actual_tokens] = self.rope_single(
            prefill_k_pe[num_decode_tokens:num_actual_tokens], cos, sin)
        prefill_k_c_normed = kv_c_normed[:num_actual_tokens]
        prefill_kv_c_k_pe = torch.cat([prefill_k_c_normed, prefill_k_pe],
                                      dim=-1)
        prefill_kv_c_k_pe = get_pcp_group().all_gather(prefill_kv_c_k_pe, 0)
        prefill_kv_c_k_pe = torch.index_select(
            prefill_kv_c_k_pe, 0,
            attn_metadata.prefill.pcp_metadata.pcp_allgather_restore_idx)
        prefill_kv_c_k_pe = prefill_kv_c_k_pe[num_decode_tokens *
                                              self.pcp_size:]
        prefill_k_c_normed, prefill_k_pe = prefill_kv_c_k_pe.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed, k_pe = prefill_k_c_normed, prefill_k_pe
        prefill_k_c_normed = prefill_k_c_normed.squeeze()
        slot_mapping = attn_metadata.slot_mapping[self.pcp_size *
                                                  num_decode_tokens:]
        torch_npu._npu_reshape_and_cache(key=kv_c_normed,
                                         value=k_pe,
                                         key_cache=kv_cache[0],
                                         value_cache=kv_cache[1],
                                         slot_indices=slot_mapping)
        prefill_k_nope, prefill_value = self.kv_b_proj(
            prefill_k_c_normed)[0].view(
                -1, self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim).split(
                    [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        prefill_k_pe = prefill_k_pe.expand((*prefill_k_nope.shape[:-1], -1))
        return PrefillMLAPreprocessResult(prefill_q_nope, prefill_q_pe,
                                          prefill_k_nope, prefill_k_pe,
                                          prefill_value)

    def mla_preprocess_decode(self, q_c, kv_no_split, kv_cache, attn_metadata):
        num_decode_tokens = attn_metadata.num_decode_tokens
        decode_q_c = q_c[:num_decode_tokens]
        cos = attn_metadata.decode.cos
        sin = attn_metadata.decode.sin
        decode_ql_nope, decode_q_pe = \
            self._q_proj_and_k_up_proj(decode_q_c)
        decode_ql_nope, decode_q_pe = self.reorg_decode_q(
            decode_ql_nope, decode_q_pe)
        decode_q_pe = self.rope_single(decode_q_pe, cos, sin)
        decode_slots = attn_metadata.slot_mapping[:num_decode_tokens]
        decode_kv_no_split = kv_no_split[:num_decode_tokens]
        decode_k_pe, decode_k_nope = self.exec_kv_decode(
            decode_kv_no_split, cos, sin, kv_cache, decode_slots)
        return DecodeMLAPreprocessResult(decode_ql_nope, decode_q_pe,
                                         decode_k_nope, decode_k_pe)

    def get_context_seq_len_npu(self, index: int,
                                attn_metadata: AscendMLAMetadata):
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None
        assert prefill_metadata.chunked_context is not None
        assert isinstance(prefill_metadata.chunked_context,
                          CPChunkedContextMetadata)
        assert prefill_metadata.chunked_context.padded_chunk_seq_lens_npu is not None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        assert 0 <= index < iters
        return prefill_metadata.chunked_context.padded_chunk_seq_lens_npu[
            index]

    def reorg_decode_q(self, decode_q_nope, decode_q_pe):
        if self.dcp_size > 1:
            decode_q_no_split = torch.cat([decode_q_nope, decode_q_pe], dim=-1)
            decode_q_no_split = get_dcp_group().all_gather(
                decode_q_no_split, 1)
            decode_q_nope, decode_q_pe = decode_q_no_split.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return decode_q_nope, decode_q_pe

    def _forward_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        value: torch.Tensor,
        kv_c_and_k_pe_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        if not self.pcp_size > 1:
            return super()._forward_prefill(q_nope, q_pe, k_nope, k_pe, value,
                                            kv_c_and_k_pe_cache, attn_metadata)
        assert attn_metadata.prefill is not None
        assert attn_metadata.prefill.pcp_metadata is not None
        num_tokens = q_nope.size(0)
        # Use precomputed indices from the metadata (already converted to tensors and on device)
        q_head_idx = attn_metadata.prefill.pcp_metadata.q_head_idx
        q_tail_idx = attn_metadata.prefill.pcp_metadata.q_tail_idx
        kv_with_q_head_nomask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_head_nomask_idx
        kv_with_q_head_mask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_head_mask_idx
        kv_with_q_tail_nomask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_tail_nomask_idx
        kv_with_q_tail_mask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_tail_mask_idx
        attn_mask_seqlens = attn_metadata.prefill.pcp_metadata.attn_mask_seqlens
        head_attn_nomask_seqlens = attn_metadata.prefill.pcp_metadata.head_attn_nomask_seqlens
        tail_attn_nomask_seqlens = attn_metadata.prefill.pcp_metadata.tail_attn_nomask_seqlens
        output_head, lse_head = self._attention_with_mask_and_nomask(
            q_nope=torch.index_select(q_nope, 0, q_head_idx),
            q_pe=torch.index_select(q_pe, 0, q_head_idx),
            k_nope=k_nope,
            k_pe=k_pe,
            value=value,
            kv_mask_idx=kv_with_q_head_mask_idx,
            kv_nomask_idx=kv_with_q_head_nomask_idx,
            attn_mask_seqlens=attn_mask_seqlens,
            attn_nomask_seqlens=head_attn_nomask_seqlens,
            mask=attn_metadata.attn_mask)

        output_tail, lse_tail = self._attention_with_mask_and_nomask(
            q_nope=torch.index_select(q_nope, 0, q_tail_idx),
            q_pe=torch.index_select(q_pe, 0, q_tail_idx),
            k_nope=k_nope,
            k_pe=k_pe,
            value=value,
            kv_mask_idx=kv_with_q_tail_mask_idx,
            kv_nomask_idx=kv_with_q_tail_nomask_idx,
            attn_mask_seqlens=attn_mask_seqlens,
            attn_nomask_seqlens=tail_attn_nomask_seqlens,
            mask=attn_metadata.attn_mask)

        q_full_idx = attn_metadata.prefill.pcp_metadata.q_full_idx
        attn_output = torch.index_select(
            torch.cat([output_head, output_tail], dim=0), 0, q_full_idx)
        attn_lse = torch.index_select(torch.cat([lse_head, lse_tail], dim=1),
                                      1, q_full_idx)

        output, _ = self._compute_prefill_context(q_nope, q_pe,
                                                  kv_c_and_k_pe_cache,
                                                  self.qk_rope_head_dim,
                                                  attn_metadata, attn_output,
                                                  attn_lse)

        output = output.reshape([num_tokens, self.num_heads * self.v_head_dim])

        return output

    def _attention_with_mask_and_nomask(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        value: torch.Tensor,
        kv_mask_idx: torch.Tensor,
        kv_nomask_idx: list[torch.Tensor],
        attn_mask_seqlens: torch.Tensor,
        attn_nomask_seqlens: list[torch.Tensor],
        mask: torch.Tensor,
    ):
        attn_output = torch.empty(q_nope.shape[0],
                                  self.num_heads,
                                  self.v_head_dim,
                                  dtype=k_pe.dtype,
                                  device=k_pe.device)
        attn_lse = torch.empty(self.num_heads,
                               q_pe.shape[0],
                               dtype=torch.float32,
                               device=k_pe.device)
        # mask
        k_nope_mask = torch.index_select(k_nope, 0, kv_mask_idx)
        value_mask = torch.index_select(value, 0, kv_mask_idx)
        k_pe_mask = torch.index_select(k_pe, 0, kv_mask_idx)
        torch_npu.atb.npu_ring_mla(q_nope=q_nope,
                                   q_rope=q_pe,
                                   k_nope=k_nope_mask,
                                   k_rope=k_pe_mask,
                                   value=value_mask,
                                   mask=mask,
                                   seqlen=attn_mask_seqlens,
                                   head_num=self.num_heads,
                                   kv_head_num=self.num_heads,
                                   pre_out=None,
                                   prev_lse=None,
                                   qk_scale=self.scale,
                                   kernel_type="kernel_type_high_precision",
                                   mask_type="mask_type_triu",
                                   input_layout="type_bsnd",
                                   calc_type="calc_type_first_ring",
                                   output=attn_output,
                                   softmax_lse=attn_lse)

        # nomask
        if not kv_nomask_idx or len(kv_nomask_idx[0]) == 0:
            return attn_output, attn_lse

        for kv_nomask_idx_split, attn_nomask_seqlens_split in zip(
                kv_nomask_idx, attn_nomask_seqlens):
            k_nope_nomask = torch.index_select(k_nope, 0, kv_nomask_idx_split)
            value_nomask = torch.index_select(value, 0, kv_nomask_idx_split)
            k_pe_nomask = torch.index_select(k_pe, 0, kv_nomask_idx_split)
            torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,
                q_rope=q_pe,
                k_nope=k_nope_nomask,
                k_rope=k_pe_nomask,
                value=value_nomask,
                mask=mask,
                seqlen=attn_nomask_seqlens_split,
                head_num=self.num_heads,
                kv_head_num=self.num_heads,
                pre_out=attn_output,
                prev_lse=attn_lse,
                qk_scale=self.scale,
                kernel_type="kernel_type_high_precision",
                mask_type="no_mask",
                input_layout="type_bsnd",
                calc_type="calc_type_default",
                output=attn_output,
                softmax_lse=attn_lse)
        return attn_output, attn_lse

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        decode_meta = attn_metadata.decode
        assert decode_meta is not None
        num_tokens = q_nope.size(0)
        # shape of knope/k_pe for npu graph mode should be:
        # [num_blocks, num_kv_heads, block_size, self.kv_lora_rank/self.qk_rope_head_dim]
        if self.dcp_size > 1:
            num_heads = self.num_heads * self.dcp_size
        else:
            num_heads = self.num_heads

        k_nope = k_nope.view(-1, block_size, self.num_kv_heads,
                             self.kv_lora_rank)
        k_pe = k_pe.view(-1, block_size, self.num_kv_heads,
                         self.qk_rope_head_dim)
        q_nope = q_nope.view(num_tokens, num_heads, -1)
        q_pe = q_pe.view(num_tokens, num_heads, -1)
        # use pcp & dcp split computed token nums from scheduler to compute actual seq_len and seq_mask
        seq_len = decode_meta.cp_seq_len

        common_kwargs = {
            "return_lse": True,
            "calc_type": "calc_type_ring",
        }
        forward_context: ForwardContext = get_forward_context()
        if forward_context.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        if forward_context.capturing:
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu.atb._npu_multi_head_latent_attention_get_workspace(
                    q_nope, q_pe, k_nope, k_pe, decode_meta.block_table,
                    seq_len, num_heads, self.scale, self.num_kv_heads,
                    **common_kwargs)
                update_graph_params_workspaces(num_tokens, workspace)
            attn_output = torch.empty_like(q_nope)
            softmax_lse = torch.empty((num_tokens, num_heads, 1),
                                      dtype=q_nope.dtype,
                                      device=q_nope.device)
            graph_params.attn_params[num_tokens].append(
                (weak_ref_tensors(q_nope), weak_ref_tensors(q_pe),
                 weak_ref_tensors(k_nope), weak_ref_tensors(k_pe),
                 decode_meta.block_table, seq_len, num_heads, self.scale,
                 self.num_kv_heads, weak_ref_tensors(attn_output),
                 weak_ref_tensors(softmax_lse)))
            torch.npu.graph_task_group_begin(stream)
            torch_npu.atb.npu_multi_head_latent_attention(
                q_nope,
                q_pe,
                k_nope,
                k_pe,
                decode_meta.block_table,
                seq_len,
                num_heads,
                self.scale,
                self.num_kv_heads,
                **common_kwargs,
                workspace=workspace,
                output=attn_output,
                lse=softmax_lse)
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
        else:
            attn_output = torch.empty_like(q_nope)
            softmax_lse = torch.empty((num_tokens, num_heads, 1),
                                      dtype=q_nope.dtype,
                                      device=q_nope.device)
            torch_npu.atb.npu_multi_head_latent_attention(
                q_nope,
                q_pe,
                k_nope,
                k_pe,
                decode_meta.block_table,
                seq_len,
                num_heads,
                self.scale,
                self.num_kv_heads,
                return_lse=True,
                calc_type="calc_type_ring",
                output=attn_output,
                lse=softmax_lse)

        # Update out&lse
        attn_out_lse = _process_attn_out_lse(attn_output, softmax_lse,
                                             decode_meta.batch_seq_mask)
        attn_output = _npu_attention_update(self.kv_lora_rank, attn_out_lse)
        return self._v_up_proj(attn_output)

    def _out_lse_reshape(self, attn_out: torch.Tensor,
                         attn_lse: torch.Tensor) -> torch.Tensor:
        attn_out = attn_out.contiguous().view(
            attn_out.shape[0] * attn_out.shape[1], attn_out.shape[2])
        attn_lse = attn_lse.contiguous().view(
            attn_lse.shape[0] * attn_lse.shape[1] * attn_lse.shape[2])
        return attn_out, attn_lse

    def _reorg_kvcache(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        chunked_context: CPChunkedContextMetadata,
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
        assert chunked_context is not None
        assert chunked_context.padded_local_chunk_seq_lens is not None
        assert chunked_context.local_context_lens_allranks is not None
        assert chunked_context.cu_seq_lens_lst is not None
        assert chunked_context.max_seq_lens is not None
        assert chunked_context.chunk_size is not None

        padded_local_chunk_seq_lens_lst = chunked_context.padded_local_chunk_seq_lens[
            chunk_idx]
        local_context_lens_allranks = chunked_context.local_context_lens_allranks
        sum_seq_len = chunked_context.cu_seq_lens_lst[chunk_idx][-1]
        max_seq_len = chunked_context.max_seq_lens[chunk_idx]
        chunk_size: int = chunked_context.chunk_size
        cache_kv_c_k_pe = torch.cat([kv_c_normed, k_pe], dim=-1)
        if self.dcp_size > 1:
            cache_kv_c_k_pe = get_dcp_group().all_gather(cache_kv_c_k_pe, 0)

        if self.pcp_size > 1:
            cache_kv_c_k_pe = get_pcp_group().all_gather(cache_kv_c_k_pe, 0)

        allgatered_kv_c_normed, allgatered_k_pe = cache_kv_c_k_pe.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        kv_c_segments = []
        k_pe_segments = []
        src_token_idx = 0
        max_seq_len_check = 0
        for padded_local_chunk_seq_len, local_context_lens in zip(
                padded_local_chunk_seq_lens_lst, local_context_lens_allranks):
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
                    kv_c_segment = allgatered_kv_c_normed[rank * toks +
                                                          src_token_idx:rank *
                                                          toks +
                                                          src_token_idx +
                                                          local_chunk_len]
                    k_pe_segment = allgatered_k_pe[rank * toks +
                                                   src_token_idx:rank * toks +
                                                   src_token_idx +
                                                   local_chunk_len]
                    kv_c_segments.append(kv_c_segment)
                    k_pe_segments.append(k_pe_segment)
                    cur_seq_len += local_chunk_len
            max_seq_len_check = max(max_seq_len_check, cur_seq_len)
            src_token_idx += padded_local_chunk_seq_len
        reorganized_kv_c_normed = torch.cat(kv_c_segments, dim=0)
        reorganized_k_pe = torch.cat(k_pe_segments, dim=0)
        assert reorganized_kv_c_normed.shape[0] == sum_seq_len
        assert reorganized_k_pe.shape[0] == sum_seq_len
        assert max_seq_len_check == max_seq_len
        return reorganized_kv_c_normed, reorganized_k_pe
