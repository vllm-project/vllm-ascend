from typing import ClassVar, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (get_dcp_group,
                              get_decode_context_model_parallel_rank,
                              get_decode_context_model_parallel_world_size,
                              get_pcp_group)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.kv_cache_interface import MLAAttentionSpec

# isort: off
from vllm_ascend.attention.mla_v1 import (AscendMLADecodeMetadata,
                                          AscendMLAImpl, AscendMLAMetadata,
                                          AscendMLAMetadataBuilder,
                                          AscendMLAPrefillMetadata,
                                          DecodeMLAPreprocessResult,
                                          PrefillMLAPreprocessResult)
#isort: on

from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         maybe_save_kv_layer_to_connector,
                                         wait_for_kv_layer_from_connector)
from vllm_ascend.attention.common_cp import AscendPCPMetadata, CPChunkedContextMetadata
from vllm_ascend.compilation.acl_graph import (get_graph_params,
                                               get_mtp_graph_params,
                                               update_graph_params_workspaces)
from vllm_ascend.ops.shared_weight_layer import (
    is_hidden_layer, reach_layer_for_shared_weight_series)
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.utils import weak_ref_tensors

MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024

M = TypeVar("M", bound=AscendMLAMetadata)


class AscendMlaCPMetadataBuilder(AscendMLAMetadataBuilder):
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
    AttentionCGSupport.UNIFORM_BATCH
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(self,
                 kv_cache_spec: MLAAttentionSpec,
                 layer_names: list[str],
                 vllm_config: VllmConfig,
                 device: torch.device,
                 metadata_cls: Optional[AscendMLAMetadata] = None):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device,
                         metadata_cls)

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

    def set_num_actual_tokens(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        if long_seq_metadata is None:
            raise AssertionError("long_seq_metadata should not be None.")

        self.num_actual_tokens = max(
            long_seq_metadata.num_actual_tokens_pcp_padded,
            common_attn_metadata.num_actual_tokens)

    def build_cp_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: nn.Module,
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
            pcp_prefill_mask=common_long_seq_metadata.pcp_prefill_mask,
            pcp_allgather_restore_idx=common_long_seq_metadata.
            pcp_allgather_restore_idx)

    def build_chunked_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: nn.Module,
    ):
        chunked_context_metadata = super().build_chunked_metadata(
            common_prefix_len, common_attn_metadata, model)
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

    def set_prefill_block_table(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        # For pcp + spec decode, we flatten seq_lens and block_table
        # to avoid irregular spec_attn_mask shape
        self.num_decodes_flatten = self.query_lens[:self.num_decodes].sum(
        ).item()
        self.block_table = common_attn_metadata.block_table_tensor[:self.
                                                                   num_decodes_flatten
                                                                   + self.
                                                                   num_prefills]

    def set_decode_block_table(
            self, common_attn_metadata: AscendCommonAttentionMetadata):
        self.block_table = self.block_table[:self.num_decodes_flatten, ...]

    def build_prefill_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: nn.Module,
    ) -> AscendMLAPrefillMetadata:
        prefill_metadata = super().build_prefill_metadata(
            common_prefix_len, common_attn_metadata, model)
        prefill_metadata.pcp_metadata = self.build_cp_metadata(
            common_prefix_len, common_attn_metadata, model)
        prefill_metadata.block_table = self.block_table[
            self.num_decodes_flatten:, ...]
        return prefill_metadata

    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: nn.Module,
    ) -> AscendMLADecodeMetadata:
        decode_metadata = super().build_decode_metadata(
            common_prefix_len, common_attn_metadata, model)

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

    def _v_up_proj(self, x):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        x = torch.bmm(x, self.W_UV)
        # # Convert from (N, B, V) to (B, N * V)
        x = x.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)
        return x

    def _compute_prefill_context(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: Tuple[torch.Tensor],
        rope_dim: int,
        attn_metadata: AscendMLAMetadata,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
    ):
        assert len(kv_c_and_k_pe_cache) > 1
        prefill_metadata = attn_metadata.prefill
        if prefill_metadata is None or prefill_metadata.chunked_context is None:
            return prefix_output, prefix_lse

        iters = len(prefill_metadata.chunked_context.seq_tot)

        current_seq_len = torch.tensor(prefill_metadata.query_lens,
                                       dtype=torch.int32)
        cache_kv_c = kv_c_and_k_pe_cache[0]
        cache_k_pe = kv_c_and_k_pe_cache[1]
        num_heads = cache_k_pe.size(2)
        latent_kv_dim = kv_c_and_k_pe_cache[0].size(-1)
        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            # chunk_seq_lens will be padded when pcp&dcp
            context_seq_len = prefill_metadata.chunked_context.chunk_seq_lens[
                i]
            context_seq_len_npu = prefill_metadata.chunked_context.padded_chunk_seq_lens_npu[
                i]
            seq_len = torch.stack([current_seq_len, context_seq_len])
            kv_c_normed = torch.empty(toks,
                                      num_heads,
                                      latent_kv_dim,
                                      dtype=q_nope.dtype,
                                      device=q_nope.device)
            k_pe = torch.empty(toks,
                               num_heads,
                               rope_dim,
                               dtype=q_nope.dtype,
                               device=q_nope.device)

            torch_npu.atb.npu_paged_cache_load(
                cache_kv_c,
                cache_k_pe,
                prefill_metadata.block_table,
                context_seq_len_npu,
                seq_starts=prefill_metadata.chunked_context.starts[i],
                key=kv_c_normed,
                value=k_pe,
            )

            cache_kv_c_k_pe = torch.cat([kv_c_normed, k_pe], dim=-1)
            if self.dcp_size > 1:
                cache_kv_c_k_pe = get_dcp_group().all_gather(
                    cache_kv_c_k_pe, 0)

            if self.pcp_size > 1:
                cache_kv_c_k_pe = get_pcp_group().all_gather(
                    cache_kv_c_k_pe, 0)

            allgatered_kv_c_normed, allgatered_k_pe = cache_kv_c_k_pe.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_c_normed, k_pe = self._reorg_kvcache(
                allgatered_kv_c_normed,
                allgatered_k_pe,
                padded_local_chunk_seq_lens_lst=prefill_metadata.
                chunked_context.padded_local_chunk_seq_lens[i],
                local_context_lens_allranks=prefill_metadata.chunked_context.
                local_context_lens_allranks,
                sum_seq_len=prefill_metadata.chunked_context.cu_seq_lens_lst[i]
                [-1],
                max_seq_len=prefill_metadata.chunked_context.max_seq_lens[i],
                chunk_size=prefill_metadata.chunked_context.chunk_size,
                chunk_idx=i,
                toks=toks,
            )

            kv_c_normed = kv_c_normed.squeeze()
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope \
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k_pe = k_pe.expand((*k_nope.shape[:-1], -1))

            mask = attn_metadata.attn_mask
            torch_npu.atb.npu_ring_mla(
                q_nope=q_nope,
                q_rope=q_pe,
                k_nope=k_nope,
                k_rope=k_pe,
                value=v,
                mask=mask,
                seqlen=seq_len,
                head_num=self.num_heads,
                kv_head_num=self.num_heads,
                pre_out=prefix_output,
                prev_lse=prefix_lse,
                qk_scale=self.scale,
                kernel_type="kernel_type_high_precision",
                mask_type="no_mask",
                input_layout="type_bsnd",
                calc_type="calc_type_default",
                output=prefix_output,
                softmax_lse=prefix_lse)
        return prefix_output, prefix_lse

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            if self.fc2_o_shared_enable and is_hidden_layer(
                    self.vllm_config, self.o_proj):
                reach_layer_for_shared_weight_series(self.o_proj)
            return output.fill_(0)

        forward_context = get_forward_context()

        if self.pcp_size > 1:
            num_actual_tokens = attn_metadata.num_actual_tokens_pcp_padded // self.pcp_size
        else:
            num_actual_tokens = attn_metadata.num_actual_tokens
        assert attn_metadata.num_decodes is not None and \
               attn_metadata.num_prefills is not None and \
               attn_metadata.num_decode_tokens is not None

        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens
        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        o_proj_input_shape = (forward_context.num_tokens,
                              self.num_heads * self.v_head_dim)
        o_proj_input = torch.empty(o_proj_input_shape,
                                   dtype=hidden_states.dtype,
                                   device=hidden_states.device)

        # MLA Preprocess
        if self.enable_mlapo and not has_prefill:
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                hidden_states.contiguous(), need_gather_q_kv)
            decode_preprocess_res, prefill_preprocess_res = self._mla_decode_preprocess(
                hidden_states, kv_cache, attn_metadata)
        else:
            decode_preprocess_res, prefill_preprocess_res = self._mla_preprocess(
                layer_name, hidden_states, kv_cache, attn_metadata,
                need_gather_q_kv)

        if decode_preprocess_res is not None:
            # MLA Preprocess for decoding
            if self.pcp_size * self.dcp_size > 1:
                output_decode = self._forward_decode_pcp_dcp(
                    decode_preprocess_res.ql_nope,
                    decode_preprocess_res.q_pe,
                    decode_preprocess_res.k_nope,
                    decode_preprocess_res.k_pe,
                    kv_cache[0].shape[1],
                    attn_metadata,
                )
            else:
                output_decode = self._forward_decode(
                    decode_preprocess_res.ql_nope, decode_preprocess_res.q_pe,
                    decode_preprocess_res.k_nope, decode_preprocess_res.k_pe,
                    kv_cache[0].shape[1], attn_metadata)

            o_proj_input[:num_decode_tokens] = output_decode

        if prefill_preprocess_res is not None:
            # FIX: aicore move should be also placed on the comm stream in dbo,
            # otherwise it may affect the accuracy
            # TODO: use an elegant way to overlap
            if self.pcp_size > 1:
                output_prefill = self._forward_prefill_cp(
                    prefill_preprocess_res.q_nope, prefill_preprocess_res.q_pe,
                    prefill_preprocess_res.k_nope, prefill_preprocess_res.k_pe,
                    prefill_preprocess_res.value, kv_cache, attn_metadata)
            else:
                output_prefill = self._forward_prefill(
                    prefill_preprocess_res.q_nope, prefill_preprocess_res.q_pe,
                    prefill_preprocess_res.k_nope, prefill_preprocess_res.k_pe,
                    prefill_preprocess_res.value, kv_cache, attn_metadata)

            o_proj_input[num_decode_tokens:num_actual_tokens] = output_prefill
        # O proj
        MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024
        maybe_npu_prefetch(inputs=self.o_proj.weight,
                           dependency=o_proj_input,
                           max_size=MAX_O_PROJ_PREFETCH_SIZE,
                           enabled=self.enable_prefetch)

        output[...] = self.o_proj(o_proj_input,
                                  is_prefill=(prefill_preprocess_res
                                              is not None))[0]

        del o_proj_input

        if has_prefill:
            maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
        return output_padded

    def _mla_preprocess(self, layer_name, hidden_states, kv_cache,
                        attn_metadata, need_gather_q_kv):
        # MLA Preprocess:
        # 1. Perform fused_qkv_a_proj and q_a_layernorm to obtain q_c and kv_no_split
        # or
        #    Perform kv_a_proj_with_mqa to obtain kv_no_split
        # 2. If need_gather_q_kv, perform all_gather.
        # 3. Preprocess decode tokens, write kv cache and get:
        # decode_ql_nope, decode_q_pe, decode_k_pe, decode_k_nope
        # 4. Preprocess prefill tokens, write kv cache and get:
        # prefill_q_nope, prefill_q_pe, prefill_k_nope, prefill_k_pe, prefill_value
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_actual_tokens = attn_metadata.num_actual_tokens
        if self.fused_qkv_a_proj is not None:
            maybe_npu_prefetch(inputs=self.fused_qkv_a_proj.weight,
                               dependency=hidden_states,
                               enabled=self.enable_prefetch)
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            # allgather need contiguous data
            kv_no_split = kv_no_split.contiguous()
        else:
            q_c = hidden_states
            kv_no_split = self.kv_a_proj_with_mqa(hidden_states)[0]

        # Process for Flash Comm V1
        q_c = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            q_c.contiguous(), need_gather_q_kv)
        kv_no_split = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            kv_no_split.contiguous(), need_gather_q_kv)

        if self.fc2_o_shared_enable and is_hidden_layer(
                self.vllm_config, self.o_proj):
            reach_layer_for_shared_weight_series(self.o_proj)

        decode_preprocess_res = None
        prefill_preprocess_res = None
        if has_prefill:
            wait_for_kv_layer_from_connector(layer_name)
        # Preprocess for decode tokens
        if has_decode:
            decode_q_c = q_c[:num_decode_tokens]
            cos = attn_metadata.decode.cos
            sin = attn_metadata.decode.sin
            decode_ql_nope, decode_q_pe = \
                self._q_proj_and_k_up_proj(decode_q_c)
            if self.dcp_size > 1:
                decode_q_no_split = torch.cat([decode_ql_nope, decode_q_pe],
                                              dim=-1)
                decode_q_no_split = get_dcp_group().all_gather(
                    decode_q_no_split, 1)
                decode_ql_nope, decode_q_pe = decode_q_no_split.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            decode_q_pe = self.rope_single(decode_q_pe, cos, sin)
            decode_slots = attn_metadata.slot_mapping[:num_decode_tokens *
                                                      self.pcp_size:self.
                                                      pcp_size]
            decode_kv_no_split = kv_no_split[:num_decode_tokens]
            decode_k_pe, decode_k_nope = self.exec_kv_decode(
                decode_kv_no_split, cos, sin, kv_cache, decode_slots)
            decode_preprocess_res = DecodeMLAPreprocessResult(
                decode_ql_nope, decode_q_pe, decode_k_nope, decode_k_pe)
        # Preprocess for prefill tokens
        if has_prefill:
            if self.pcp_size > 1:
                num_actual_tokens = (attn_metadata.num_actual_tokens_pcp_padded
                                     - self.pcp_size * num_decode_tokens
                                     ) // self.pcp_size + num_decode_tokens
            prefill_kv_no_split = kv_no_split[
                num_decode_tokens:num_actual_tokens]
            prefill_q_c = q_c[num_decode_tokens:num_actual_tokens]
            prefill_q = self.q_proj(prefill_q_c)[0] \
                .view(-1, self.num_heads, self.qk_head_dim)
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
            prefill_q_nope = prefill_q[..., :self.qk_nope_head_dim]
            if self.pcp_size > 1:
                cos = attn_metadata.prefill.cos[:num_actual_tokens -
                                                num_decode_tokens]
                sin = attn_metadata.prefill.sin[:num_actual_tokens -
                                                num_decode_tokens]
            else:
                cos = attn_metadata.prefill.cos
                sin = attn_metadata.prefill.sin
            prefill_slots = attn_metadata.slot_mapping[
                num_decode_tokens:num_actual_tokens]
            prefill_q_pe = self.rope_single(prefill_q_pe, cos, sin)
            if self.pcp_size > 1:
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
                prefill_k_pe[
                    num_decode_tokens:num_actual_tokens] = self.rope_single(
                        prefill_k_pe[num_decode_tokens:num_actual_tokens], cos,
                        sin)
                prefill_k_c_normed = kv_c_normed[:num_actual_tokens]
                prefill_kv_c_k_pe = torch.cat(
                    [prefill_k_c_normed, prefill_k_pe], dim=-1)
                prefill_kv_c_k_pe = get_pcp_group().all_gather(
                    prefill_kv_c_k_pe, 0)
                prefill_kv_c_k_pe = torch.index_select(
                    prefill_kv_c_k_pe, 0, attn_metadata.prefill.pcp_metadata.
                    pcp_allgather_restore_idx)
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
            else:
                prefill_k_pe, prefill_k_c_normed = self.exec_kv_prefill(
                    prefill_kv_no_split, cos, sin, kv_cache, prefill_slots)
            prefill_k_nope, prefill_value = self.kv_b_proj(
                prefill_k_c_normed)[0].view(
                    -1, self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim).split(
                        [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            if not self.pcp_size > 1:
                prefill_k_pe = prefill_k_pe.view(prefill_q_c.shape[0],
                                                 self.num_kv_heads, -1)
            prefill_k_pe = prefill_k_pe.expand(
                (*prefill_k_nope.shape[:-1], -1))
            prefill_preprocess_res = PrefillMLAPreprocessResult(
                prefill_q_nope, prefill_q_pe, prefill_k_nope, prefill_k_pe,
                prefill_value)
        return decode_preprocess_res, prefill_preprocess_res

    def _mla_decode_preprocess(self, hidden_states, kv_cache, attn_metadata):
        bsz = attn_metadata.num_decode_tokens
        hidden_states = hidden_states[:bsz]

        cos_shape = attn_metadata.decode.cos.shape
        cos = attn_metadata.decode.cos.view(cos_shape[0], cos_shape[-1])
        sin = attn_metadata.decode.sin.view(cos_shape[0], cos_shape[-1])

        decode_k_nope, decode_k_pe = kv_cache[0], kv_cache[1]
        decode_q_nope = torch.empty(
            (hidden_states.shape[0], self.W_UK_T.shape[0],
             decode_k_nope.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        decode_q_pe = torch.empty(
            (hidden_states.shape[0], self.W_UK_T.shape[0],
             decode_k_pe.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops._C_ascend.mla_preprocess(
            hidden_states,
            self.wd_qkv,
            self.deq_scale_qkv,
            self.gamma1,
            self.beta1,
            self.wu_q,
            self.qb_deq_scl,
            self.gamma2,
            cos,
            sin,
            self.W_UK_T,
            decode_k_nope,
            decode_k_pe,
            attn_metadata.slot_mapping[:bsz].flatten(),
            quant_scale0=self.quant_scale0,
            quant_offset0=self.quant_offset0,
            bias0=self.quant_bias_qkv,
            quant_scale1=self.quant_scale1,
            quant_offset1=self.quant_offset1,
            bias1=self.qb_qt_bias,
            ctkv_scale=self.ctkv_scale,
            q_nope_scale=self.q_nope_scale,
            cache_mode="krope_ctkv",
            quant_mode="per_tensor_quant_asymm",
            q_out0=decode_q_nope,
            kv_cache_out0=decode_k_nope,
            q_out1=decode_q_pe,
            kv_cache_out1=decode_k_pe,
            enable_inner_out=False,
            inner_out=torch.tensor([], device=hidden_states.device))
        decode_q_nope = decode_q_nope.view(bsz, self.num_heads,
                                           self.kv_lora_rank)
        decode_q_pe = decode_q_pe.view(bsz, self.num_heads, -1)

        if self.dcp_size > 1:
            decode_q_no_split = torch.cat([decode_q_nope, decode_q_pe], dim=-1)
            decode_q_no_split = get_dcp_group().all_gather(
                decode_q_no_split, 1)
            decode_q_nope, decode_q_pe = decode_q_no_split.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        decode_preprocess_res = DecodeMLAPreprocessResult(
            decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe)
        return decode_preprocess_res, None

    def _forward_prefill_cp(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        value: torch.Tensor,
        kv_c_and_k_pe_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
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
        mask = attn_metadata.prefill.pcp_metadata.pcp_prefill_mask
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
            mask=mask)

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
            mask=mask)

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
            self, q_nope: torch.Tensor, q_pe: torch.Tensor,
            k_nope: torch.Tensor, k_pe: torch.Tensor, value: torch.Tensor,
            kv_mask_idx: torch.Tensor, kv_nomask_idx: torch.Tensor,
            attn_mask_seqlens: torch.Tensor, attn_nomask_seqlens: torch.Tensor,
            mask: torch.Tensor):
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
        if kv_nomask_idx.shape[0] == 0:
            return attn_output, attn_lse

        k_nope_nomask = torch.index_select(k_nope, 0, kv_nomask_idx)
        value_nomask = torch.index_select(value, 0, kv_nomask_idx)
        k_pe_nomask = torch.index_select(k_pe, 0, kv_nomask_idx)
        torch_npu.atb.npu_ring_mla(q_nope=q_nope,
                                   q_rope=q_pe,
                                   k_nope=k_nope_nomask,
                                   k_rope=k_pe_nomask,
                                   value=value_nomask,
                                   mask=mask,
                                   seqlen=attn_nomask_seqlens,
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

    def _forward_decode_pcp_dcp(
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
        if forward_context.is_mtp_model:
            graph_params = get_mtp_graph_params()
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
        attn_out_lse = self._process_attn_out_lse(attn_output, softmax_lse,
                                                  decode_meta)
        attn_output = self._npu_attention_update(attn_out_lse)
        return self._v_up_proj(attn_output)

    def _npu_attention_update(self,
                              attn_out_lse: torch.Tensor) -> torch.Tensor:
        # [PCP * S, DCP * H, D+1]
        B_total, H_total, D_plus_1 = attn_out_lse.shape
        S = B_total // self.pcp_size
        H = H_total // self.dcp_size
        D = self.kv_lora_rank
        assert D_plus_1 == D + 1
        # [PCP, S, DCP, H, D+1]
        x = attn_out_lse.view(self.pcp_size, S, self.dcp_size, H, D_plus_1)
        # [PCP, DCP, S, H, D+1]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # Flatten [N, S, H, D+1], N = pcp_size * dcp_size
        x = x.view(-1, S, H, D_plus_1)
        # Split out lse
        out_flat, lse_flat = torch.split(x, [D, 1],
                                         dim=-1)  # [N, S, H, D], [N, S, H, 1]
        #    out: [N, S, H, D] -> [N, S*H, D]
        #    lse: [N, S, H, 1] -> [N, S*H]
        out_flat = out_flat.flatten(1, 2)  # [N, S*H, D]
        lse_flat = lse_flat.flatten(1, -1)  # [N, S*H]
        #  unbind to list
        out_list = out_flat.unbind(0)  # [S*H, D]
        lse_list = lse_flat.unbind(0)  # [S*H]
        attn_out, _ = torch_npu.npu_attention_update(lse_list, out_list, 0)
        attn_out = attn_out.view(-1, H, D)
        return attn_out

    def _out_lse_reshape(self, attn_out: torch.Tensor,
                         attn_lse: torch.Tensor) -> torch.Tensor:
        attn_out = attn_out.contiguous().view(
            attn_out.shape[0] * attn_out.shape[1], attn_out.shape[2])
        attn_lse = attn_lse.contiguous().view(
            attn_lse.shape[0] * attn_lse.shape[1] * attn_lse.shape[2])
        return attn_out, attn_lse

    def _process_attn_out_lse(
        self,
        attn_output: torch.Tensor,
        softmax_lse: torch.Tensor,
        decode_meta: AscendMLADecodeMetadata,
    ) -> torch.Tensor:
        out_mask = decode_meta.batch_seq_mask[:, None,
                                              None].expand_as(attn_output)
        attn_output = torch.where(out_mask, 0, attn_output)
        lse_mask = decode_meta.batch_seq_mask[:, None,
                                              None].expand_as(softmax_lse)
        softmax_lse = torch.where(lse_mask, -torch.inf, softmax_lse)

        softmax_lse = softmax_lse.to(torch.float32)
        attn_output = attn_output.to(torch.float32)
        # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
        attn_out_lse = torch.cat([attn_output, softmax_lse], dim=-1)
        if self.dcp_size > 1:
            # permute: [bs, num_heads, v_head_dim+1] -> [num_heads, v_head_dim+1, bs]
            attn_out_lse = attn_out_lse.permute([1, 2, 0]).contiguous()
            attn_out_lse_all2all = torch.empty_like(attn_out_lse)
            dist.all_to_all_single(attn_out_lse_all2all,
                                   attn_out_lse,
                                   group=self.dcp_group)
            attn_out_lse = attn_out_lse_all2all.permute([2, 0, 1])

        if self.pcp_size > 1:
            # AllGather out&lse within CP group
            attn_out_lse = get_pcp_group().all_gather(
                attn_out_lse.contiguous(), dim=0)

        return attn_out_lse

    def _reorg_kvcache(
        self,
        allgatered_kv_c_normed: torch.Tensor,
        allgatered_k_pe: torch.Tensor,
        padded_local_chunk_seq_lens_lst: list[int],
        local_context_lens_allranks: list[list[int]],
        sum_seq_len: int,
        max_seq_len: int,
        chunk_size: int,
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
