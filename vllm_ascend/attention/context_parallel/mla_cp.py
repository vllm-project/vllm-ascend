from dataclasses import dataclass

import numpy as np
import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

# isort: off
from vllm_ascend.attention.mla_v1 import (
    ChunkedContextMetadata,
    AscendMLADecodeMetadata,
    AscendMLAImpl,
    AscendMLAMetadata,
    AscendMLAMetadataBuilder,
)
# isort: on

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.context_parallel.common_cp import (
    DCPImplMixin,
    DCPMetadataBuilderMixin,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    update_graph_params_workspaces,
)
from vllm_ascend.utils import weak_ref_tensors


@dataclass
class DCPChunkedContextMetadata(ChunkedContextMetadata):
    """MLA chunk metadata for DCP-local context shards."""

    padded_chunk_seq_lens_npu: torch.Tensor = None
    padded_local_chunk_seq_lens: list[list[int]] | None = None
    local_context_lens_allranks: list[list[int]] | None = None
    padded_local_cu_seq_lens: torch.Tensor = None
    cu_seq_lens_lst: list[list[int]] | None = None
    chunk_size: int | None = None


@dataclass
class AscendMLADCPDecodeMetadata(AscendMLADecodeMetadata):
    """MLA decode metadata fields used only by DCP."""

    cp_seq_len: torch.Tensor = None
    dcp_mtp_attn_mask: torch.Tensor = None


class AscendMlaDCPMetadataBuilder(
    DCPMetadataBuilderMixin,
    AscendMLAMetadataBuilder,
):
    """Build MLA metadata for decode context parallelism."""

    decode_metadata_cls = AscendMLADCPDecodeMetadata

    def __init__(
        self,
        kv_cache_spec: AscendMLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendMLAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen)
        self.cp_local_block_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        self.cp_virtual_block_size = self.cp_local_block_size * self.dcp_size
        self.block_size = (self.block_size * self.cp_virtual_block_size) // np.gcd(
            self.block_size,
            self.cp_virtual_block_size,
        )

    def build_chunked_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        chunked_context_metadata = super().build_chunked_metadata(common_prefix_len, common_attn_metadata)
        if chunked_context_metadata is None:
            return None

        local_context_lens_allranks = self._get_dcp_context_lens(
            common_attn_metadata,
            start=self.num_decodes,
        )
        padded_local_context_lens_cpu = (
            cdiv(self.context_lens_cpu, self.cp_virtual_block_size) * self.cp_local_block_size
        )
        padded_local_max_context_chunk_across_ranks = (
            cdiv(self.max_context_chunk, self.cp_virtual_block_size) * self.cp_local_block_size
        )
        local_chunk_starts = (
            torch.arange(self.num_chunks, dtype=torch.int32).unsqueeze(1).expand(-1, self.num_prefills)
            * padded_local_max_context_chunk_across_ranks
        )
        local_chunk_ends = torch.min(
            padded_local_context_lens_cpu.unsqueeze(0),
            local_chunk_starts + padded_local_max_context_chunk_across_ranks,
        )
        padded_local_chunk_seq_lens = (local_chunk_ends - local_chunk_starts).clamp(min=0)
        padded_local_cu_chunk_seq_lens_cpu = torch.zeros(
            self.num_chunks,
            self.num_prefills + 1,
            dtype=torch.int32,
            pin_memory=True,
        )
        torch.cumsum(
            padded_local_chunk_seq_lens,
            dim=1,
            out=padded_local_cu_chunk_seq_lens_cpu[:, 1:],
            dtype=torch.int32,
        )
        return DCPChunkedContextMetadata(
            cu_seq_lens=chunked_context_metadata.cu_seq_lens,
            starts=local_chunk_starts.pin_memory().to(self.device, non_blocking=True),
            seq_tot=padded_local_chunk_seq_lens.sum(dim=1).tolist(),
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
        )

    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendMLADecodeMetadata:
        decode_metadata = super().build_decode_metadata(common_prefix_len, common_attn_metadata)
        assert isinstance(decode_metadata, AscendMLADCPDecodeMetadata)
        dcp_metadata = self._require_dcp_metadata(common_attn_metadata)
        if dcp_metadata.draft_cp_seq_len is not None:
            decode_metadata.cp_seq_len = dcp_metadata.draft_cp_seq_len[: self.num_decodes]
        else:
            decode_metadata.cp_seq_len = self._get_dcp_rank_context_lens(
                common_attn_metadata,
                end=self.num_decodes,
            ).tolist()
        decode_metadata.actual_seq_lengths_q = torch.arange(self.num_decodes) + 1
        decode_metadata.dcp_mtp_attn_mask = dcp_metadata.dcp_mtp_attn_mask
        return decode_metadata


class AscendMlaDCPImpl(DCPImplMixin, AscendMLAImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config=None,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
            attn_metadata = draft_attn_metadatas
            attn_keys = list(attn_metadata[0].keys())
        else:
            graph_params = get_graph_params()
            attn_metadata = forward_context.attn_metadata
            attn_keys = list(attn_metadata.keys())
        # FIXME: Behold! We are using a temporary hack here to update the args
        # for each layer's attention op in the graph.
        num_layers = len(attn_keys)
        if num_layers == 0:
            return
        if _EXTRA_CTX.is_draft_model:
            attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
        attn_count = 0
        with torch.npu.stream(update_stream):
            for key, param, handle, event in zip(
                attn_keys,
                graph_params.attn_params[num_tokens],
                graph_params.handles[num_tokens],
                graph_params.events[num_tokens],
            ):
                (
                    q_nope,
                    k_nope,
                    q_pe,
                    k_pe,
                    num_heads,
                    num_kv_heads,
                    input_layout,
                    spec_attn_mask,
                    sparse_mode,
                    scale,
                    block_table,
                    block_size,
                    actual_seq_lengths,
                    actual_seq_lengths_kv,
                    attn_output,
                    softmax_lse,
                ) = param

                if _EXTRA_CTX.is_draft_model:
                    draft_step = attn_count // num_layers
                    decode_meta = attn_metadata[draft_step][key].decode
                    attn_count = attn_count + 1
                else:
                    decode_meta = attn_metadata[key].decode

                seq_len = decode_meta.cp_seq_len
                if isinstance(seq_len, torch.Tensor):
                    seq_len = seq_len.tolist()
                actual_seq_lengths_kv = seq_len

                pad_length = num_tokens - len(actual_seq_lengths_kv)
                if pad_length > 0:
                    actual_seq_lengths_kv = actual_seq_lengths_kv + [0] * (num_tokens - len(actual_seq_lengths_kv))

                torch.npu.graph_task_update_begin(update_stream, handle)

                torch_npu.npu_fused_infer_attention_score.out(
                    q_nope,
                    k_nope,
                    k_nope,
                    query_rope=q_pe,
                    key_rope=k_pe,
                    num_heads=num_heads,
                    num_key_value_heads=num_kv_heads,
                    input_layout=input_layout,
                    atten_mask=spec_attn_mask,
                    sparse_mode=sparse_mode,
                    scale=scale,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    softmax_lse_flag=True,
                    block_table=block_table,
                    block_size=block_size,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    actual_seq_lengths=actual_seq_lengths,
                    workspace=graph_params.workspaces.get(num_tokens),
                    out=[attn_output, softmax_lse],
                )
                torch.npu.graph_task_update_end(update_stream)

                event.record(update_stream)

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

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
        dequant_scale_q_nope=None,
    ) -> torch.Tensor:
        decode_meta = attn_metadata.decode
        assert decode_meta is not None
        assert isinstance(decode_meta, AscendMLADCPDecodeMetadata)
        num_tokens = q_nope.size(0)
        # shape of knope/k_pe for npu graph mode should be:
        # [num_blocks, num_kv_heads, block_size, self.kv_lora_rank/self.qk_rope_head_dim]
        if self.dcp_size > 1:
            num_heads = self.num_heads * self.dcp_size
        else:
            num_heads = self.num_heads
        # Use DCP-local computed token counts to build sequence lengths and masks.
        k_nope = k_nope.view(-1, self.num_kv_heads, block_size, self.kv_lora_rank)
        k_pe = k_pe.view(-1, self.num_kv_heads, block_size, self.qk_rope_head_dim)

        actual_seq_lengths = None
        input_layout = "BNSD"

        if (
            attn_metadata.attn_state
            in [
                AscendAttentionState.SpecDecoding,
                AscendAttentionState.ChunkedPrefill,
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.PrefillCacheHit,
            ]
            and self.speculative_config is not None
        ):
            input_layout = "BSND"
            num_decodes = attn_metadata.num_decodes
            # TODO: If the driver is upgraded later, the contiguous function can be deleted.
            q_nope = q_nope.view(num_decodes, -1, q_nope.shape[1], q_nope.shape[-1]).contiguous()
            q_pe = q_pe.view(num_decodes, -1, q_pe.shape[1], q_pe.shape[-1])
            sparse_mode = 0
            spec_attn_mask = attn_metadata.decode.dcp_mtp_attn_mask  # type:ignore
            decode_query_lens = attn_metadata.query_lens[:num_decodes]
            assert sum(decode_query_lens) == num_tokens
            # This function only runs the decode sub-batch. A mixed
            # decode/prefill batch still carries query lengths for every
            # request in the common metadata, but FIA requires the query-length
            # list to match the decode batch dimension and block table.
            actual_seq_lengths = attn_metadata.query_lens[:num_decodes]
        else:
            q_nope = q_nope.view(num_tokens, num_heads, 1, -1).contiguous()
            q_pe = q_pe.view(num_tokens, num_heads, 1, -1)
            sparse_mode = 0
            spec_attn_mask = None

        common_kwargs = {
            "query_rope": q_pe,
            "key_rope": k_pe,
            "num_heads": num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "input_layout": input_layout,
            "atten_mask": spec_attn_mask,
            "sparse_mode": sparse_mode,
            "scale": self.scale,
            "antiquant_mode": 0,
            "antiquant_scale": None,
            "block_table": decode_meta.block_table,
            "block_size": block_size,
            "actual_seq_lengths": actual_seq_lengths,
            "actual_seq_lengths_kv": decode_meta.cp_seq_len,
            "softmax_lse_flag": True,
        }

        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        if _EXTRA_CTX.capturing:
            stream = torch_npu.npu.current_stream()
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    q_nope,
                    k_nope,
                    k_nope,
                    **common_kwargs,
                )
                update_graph_params_workspaces(num_tokens, workspace)
            attn_output = torch.empty_like(q_nope)
            if input_layout == "BSND":
                num_decodes = attn_metadata.num_decodes
                softmax_lse = torch.empty(
                    (num_decodes, num_heads, q_nope.shape[1], 1), dtype=torch.float, device=q_nope.device
                )
            elif input_layout == "BNSD":
                softmax_lse = torch.empty((num_tokens, num_heads, 1, 1), dtype=torch.float, device=q_nope.device)
            else:
                softmax_lse = torch.empty((num_tokens, num_heads, 1), dtype=torch.float, device=q_nope.device)

            graph_params.attn_params[num_tokens].append(
                (
                    weak_ref_tensors(q_nope),
                    weak_ref_tensors(k_nope),
                    weak_ref_tensors(q_pe),
                    weak_ref_tensors(k_pe),
                    num_heads,
                    self.num_kv_heads,
                    input_layout,
                    weak_ref_tensors(spec_attn_mask) if spec_attn_mask is not None else None,
                    sparse_mode,
                    self.scale,
                    weak_ref_tensors(decode_meta.block_table),
                    block_size,
                    actual_seq_lengths,
                    decode_meta.cp_seq_len,
                    weak_ref_tensors(attn_output),
                    weak_ref_tensors(softmax_lse),
                )
            )
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                q_nope, k_nope, k_nope, **common_kwargs, workspace=workspace, out=[attn_output, softmax_lse]
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
        else:
            attn_output, softmax_lse = torch_npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                k_nope,
                **common_kwargs,
            )
        if input_layout == "BSND":
            attn_output = attn_output.view(-1, attn_output.shape[2], attn_output.shape[3])
            softmax_lse = softmax_lse.transpose(1, 2).reshape(-1, softmax_lse.shape[1], 1)

        if input_layout == "BNSD":
            B_attn, N_attn, S, D = attn_output.shape
            B_lse, N_lse, Q_S, _ = softmax_lse.shape

            attn_output = attn_output.permute(0, 2, 1, 3).reshape(B_attn * S, N_attn, D)
            softmax_lse = softmax_lse.permute(0, 2, 1, 3).reshape(B_lse * Q_S, N_lse, 1)

        # Update out&lse
        attn_output = self._merge_dcp_attention_output(
            attn_output,
            softmax_lse,
            self.kv_lora_rank,
        )
        return self._v_up_proj_batch_major(attn_output)

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

        padded_local_chunk_seq_lens_lst = chunked_context.padded_local_chunk_seq_lens[chunk_idx]
        local_context_lens_allranks = chunked_context.local_context_lens_allranks
        sum_seq_len = chunked_context.cu_seq_lens_lst[chunk_idx][-1]
        max_seq_len = chunked_context.max_seq_lens[chunk_idx]
        chunk_size: int = chunked_context.chunk_size
        cache_kv_c_k_pe = torch.cat([kv_c_normed, k_pe], dim=-1)
        cache_kv_c_k_pe = self._dcp_all_gather(cache_kv_c_k_pe, 0)

        allgatered_kv_c_normed, allgatered_k_pe = cache_kv_c_k_pe.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        kv_c_segments = []
        k_pe_segments = []
        src_token_idx = 0
        max_seq_len_check = 0
        for padded_local_chunk_seq_len, local_context_lens in zip(
            padded_local_chunk_seq_lens_lst, local_context_lens_allranks
        ):
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
                    kv_c_segment = allgatered_kv_c_normed[
                        rank * toks + src_token_idx : rank * toks + src_token_idx + local_chunk_len
                    ]
                    k_pe_segment = allgatered_k_pe[
                        rank * toks + src_token_idx : rank * toks + src_token_idx + local_chunk_len
                    ]
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
