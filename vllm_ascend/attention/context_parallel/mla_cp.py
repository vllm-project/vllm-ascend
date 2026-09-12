from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np
import torch
import torch_npu
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.utils.math_utils import cdiv

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
    DCPImplMixin,
    DCPMetadataBuilderMixin,
    get_dcp_local_seq_lens,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    update_graph_params_workspaces,
)
from vllm_ascend.ops.triton.sfa_cp import fused_sfa_dcp_lse_combine
from vllm_ascend.utils import weak_ref_tensors


class MLASplitAttentionKind(Enum):
    HISTORY = "split_history"
    CURRENT = "split_current"


class MLASplitAttentionGraphParams(NamedTuple):
    """One split attention task and its layer metadata identity."""

    attention_params: tuple
    attention_kind: MLASplitAttentionKind
    layer_name: str


_DCP_MTP_COMM_STREAM: torch.npu.Stream | None = None


def _dcp_mtp_comm_stream() -> torch.npu.Stream:
    global _DCP_MTP_COMM_STREAM
    if _DCP_MTP_COMM_STREAM is None:
        _DCP_MTP_COMM_STREAM = torch_npu.npu.Stream()
    return _DCP_MTP_COMM_STREAM


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
    cp_history_seq_len: list[int] | None = None


class AscendMlaDCPMetadataBuilder(
    DCPMetadataBuilderMixin,
    AscendMLAMetadataBuilder,
):
    """Build MLA metadata for decode context parallelism."""

    decode_metadata_cls = AscendMLADCPDecodeMetadata
    # Non-causal block drafters (e.g. DSpark) attend to all DCP-local KV.
    # Causal multi-token queries still need history/current split attention.
    supports_non_causal_multi_token_dcp = True

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
        # Use the DCP CPU mirror: it includes corrected verifier lengths
        # and the draft-step extension. Do not synchronize GPU lengths here.
        local_lengths = self._get_dcp_context_lens(common_attn_metadata, end=self.num_decodes)
        # DCP lengths contain real requests; FULL graph query lengths also
        # include padded requests. Compute real histories before padding.
        query_lens = self.query_lens[: local_lengths.shape[0]]
        history_lens = (local_lengths.sum(dim=-1) - query_lens).clamp(min=0)
        decode_metadata.cp_history_seq_len = get_dcp_local_seq_lens(
            history_lens,
            self.dcp_size,
            self.cp_local_block_size,
        )[:, self.dcp_rank].tolist()
        # Preserve the base builder's cumulative TND query boundaries,
        # including graph padding; the old BSND path used per-request lengths.
        num_padded = len(decode_metadata.actual_seq_lengths_q) - len(decode_metadata.cp_history_seq_len)
        decode_metadata.cp_history_seq_len += [0] * num_padded
        decode_metadata.dcp_mtp_attn_mask = None
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
        attn_count = 0
        with torch.npu.stream(update_stream):
            for param, handle, event in zip(
                graph_params.attn_params[num_tokens],
                graph_params.handles[num_tokens],
                graph_params.events[num_tokens],
            ):
                if isinstance(param, MLASplitAttentionGraphParams):
                    split_kind = param.attention_kind
                    key = param.layer_name
                    attention_params = param.attention_params
                else:
                    split_kind = None
                    key = attn_keys[attn_count % num_layers]
                    attention_params = param
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
                ) = attention_params

                if _EXTRA_CTX.is_draft_model:
                    draft_step = attn_count // num_layers
                    decode_meta = attn_metadata[draft_step][key].decode
                else:
                    decode_meta = attn_metadata[key].decode

                # History/current share one layer invocation. Advance after
                # selecting its metadata, once current or an unsplit task is reached.
                if split_kind is not MLASplitAttentionKind.HISTORY:
                    attn_count += 1

                if split_kind is not None:
                    actual_seq_lengths = decode_meta.actual_seq_lengths_q
                    seq_len = (
                        decode_meta.cp_history_seq_len
                        if split_kind is MLASplitAttentionKind.HISTORY
                        else actual_seq_lengths
                    )
                    block_table = decode_meta.block_table if split_kind is MLASplitAttentionKind.HISTORY else None
                else:
                    seq_len = decode_meta.cp_seq_len
                if isinstance(seq_len, torch.Tensor):
                    seq_len = seq_len.tolist()
                actual_seq_lengths_kv = seq_len

                pad_length = num_tokens - len(actual_seq_lengths_kv)
                if split_kind is None and pad_length > 0:
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

    def _use_history_current_split_decode(self, attn_metadata: AscendMLAMetadata) -> bool:
        if not attn_metadata.causal:
            return False
        if not _EXTRA_CTX.is_draft_model:
            return True
        # Keep the captured task layout: the autoregressive draft dummy run can
        # describe multiple queries even for later single-token draft steps.
        if _EXTRA_CTX.capturing or get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
            return True
        # The first draft pass may process several tokens from the target pass;
        # only later autoregressive steps are necessarily single-token decodes.
        query_lens = attn_metadata.query_lens
        assert query_lens is not None
        return any(query_len > 1 for query_len in query_lens[: attn_metadata.num_decodes])

    def _decode_requires_current_kv(self, attn_metadata: AscendMLAMetadata) -> bool:
        return self._use_history_current_split_decode(attn_metadata)

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
        attention_kind: MLASplitAttentionKind,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = q_nope.size(0)
        num_heads = q_nope.size(1)
        common_kwargs = {
            "query_rope": q_pe,
            "key_rope": k_pe,
            "num_heads": num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "input_layout": "TND",
            "atten_mask": attn_mask,
            "sparse_mode": sparse_mode,
            "scale": self.scale,
            "antiquant_mode": 0,
            "antiquant_scale": None,
            "actual_seq_lengths": actual_seq_lengths,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "softmax_lse_flag": True,
        }
        if block_table is not None:
            common_kwargs["block_table"] = block_table
            common_kwargs["block_size"] = block_size

        if not _EXTRA_CTX.capturing:
            return torch_npu.npu_fused_infer_attention_score(
                q_nope,
                k_nope,
                k_nope,
                **common_kwargs,
            )

        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        assert graph_params is not None
        assert self.layer_name is not None

        stream = torch_npu.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)

        workspace = graph_params.workspaces.get(num_tokens)
        assert workspace is not None

        attn_output = torch.empty_like(q_nope)
        softmax_lse = torch.empty((num_tokens, num_heads, 1), dtype=torch.float, device=q_nope.device)
        graph_params.attn_params[num_tokens].append(
            MLASplitAttentionGraphParams(
                attention_params=(
                    weak_ref_tensors(q_nope),
                    weak_ref_tensors(k_nope),
                    weak_ref_tensors(q_pe),
                    weak_ref_tensors(k_pe),
                    num_heads,
                    self.num_kv_heads,
                    "TND",
                    weak_ref_tensors(attn_mask) if attn_mask is not None else None,
                    sparse_mode,
                    self.scale,
                    weak_ref_tensors(block_table) if block_table is not None else None,
                    block_size,
                    actual_seq_lengths,
                    actual_seq_lengths_kv,
                    weak_ref_tensors(attn_output),
                    weak_ref_tensors(softmax_lse),
                ),
                attention_kind=attention_kind,
                layer_name=self.layer_name,
            )
        )
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            q_nope,
            k_nope,
            k_nope,
            **common_kwargs,
            workspace=workspace,
            out=[attn_output, softmax_lse],
        )
        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return attn_output, softmax_lse

    def _forward_decode_split_attention(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_pe: torch.Tensor,
        current_k_nope: torch.Tensor,
        current_k_pe: torch.Tensor,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        decode_meta = attn_metadata.decode
        assert decode_meta is not None
        assert isinstance(decode_meta, AscendMLADCPDecodeMetadata)
        assert decode_meta.cp_history_seq_len is not None
        assert current_k_nope is not None and current_k_pe is not None

        num_tokens = q_nope.size(0)
        num_heads = self.num_heads * self.dcp_size if self.dcp_size > 1 else self.num_heads
        q_nope = q_nope.view(num_tokens, num_heads, -1).contiguous()
        q_pe = q_pe.view(num_tokens, num_heads, -1)
        history_k_nope = cache_k_nope.view(-1, self.num_kv_heads, block_size, self.kv_lora_rank)
        history_k_pe = cache_k_pe.view(-1, self.num_kv_heads, block_size, self.qk_rope_head_dim)

        head_start = self.dcp_rank * self.num_heads
        head_end = head_start + self.num_heads
        current_q_nope = q_nope[:, head_start:head_end].contiguous()
        current_q_pe = q_pe[:, head_start:head_end].contiguous()
        current_k_nope = current_k_nope.view(num_tokens, self.num_kv_heads, self.kv_lora_rank).contiguous()
        current_k_pe = current_k_pe.view(num_tokens, self.num_kv_heads, self.qk_rope_head_dim).contiguous()

        if _EXTRA_CTX.capturing:
            if _EXTRA_CTX.is_draft_model:
                graph_params = (
                    get_draft_graph_prefill_params() if _EXTRA_CTX.is_draft_model_prefill else get_draft_graph_params()
                )
            else:
                graph_params = get_graph_params()
            assert graph_params is not None
            if graph_params.workspaces.get(num_tokens) is None:
                # Both FIA calls execute serially on the main stream. Size the
                # shared workspace before either call is captured so its address
                # stays fixed; history all-to-all only reads the FIA outputs.
                workspace_kwargs = {
                    "num_key_value_heads": self.num_kv_heads,
                    "input_layout": "TND",
                    "scale": self.scale,
                    "antiquant_mode": 0,
                    "antiquant_scale": None,
                    "actual_seq_lengths": decode_meta.actual_seq_lengths_q,
                    "softmax_lse_flag": True,
                }
                workspaces = [
                    torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        q_nope,
                        history_k_nope,
                        history_k_nope,
                        query_rope=q_pe,
                        key_rope=history_k_pe,
                        num_heads=num_heads,
                        atten_mask=None,
                        sparse_mode=0,
                        block_table=decode_meta.block_table,
                        block_size=block_size,
                        actual_seq_lengths_kv=decode_meta.cp_history_seq_len,
                        **workspace_kwargs,
                    ),
                    torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        current_q_nope,
                        current_k_nope,
                        current_k_nope,
                        query_rope=current_q_pe,
                        key_rope=current_k_pe,
                        num_heads=self.num_heads,
                        atten_mask=decode_meta.attn_mask,
                        sparse_mode=3,
                        actual_seq_lengths_kv=decode_meta.actual_seq_lengths_q,
                        **workspace_kwargs,
                    ),
                ]
                graph_params.workspaces[num_tokens] = max(
                    workspaces, key=lambda workspace: workspace.numel() * workspace.element_size()
                )
                del workspaces

        history_output, history_lse = self._run_dcp_mtp_split_attention_op(
            q_nope,
            q_pe,
            history_k_nope,
            history_k_pe,
            attn_mask=None,
            sparse_mode=0,
            block_table=decode_meta.block_table,
            block_size=block_size,
            actual_seq_lengths=decode_meta.actual_seq_lengths_q,
            actual_seq_lengths_kv=decode_meta.cp_history_seq_len,
            attention_kind=MLASplitAttentionKind.HISTORY,
        )

        # Overlap history all-to-all with current-token attention.
        main_stream = torch.npu.current_stream()
        comm_stream = _dcp_mtp_comm_stream()
        history_ready = main_stream.record_event()
        history_output.record_stream(comm_stream)
        history_lse.record_stream(comm_stream)
        with torch.npu.stream(comm_stream):
            comm_stream.wait_event(history_ready)
            history_attn_out_lse = torch.ops.vllm.sfa_dcp_a2a_fused(
                history_output.float(),
                history_lse.float(),
                self.dcp_size,
                1,
                self.dcp_group.unique_name if self.dcp_size > 1 else "",
                return_lse=True,
            )
            history_comm_done = comm_stream.record_event()
        # The result is allocated on the communication stream and consumed
        # on the main stream; keep its storage alive through the merge.
        history_attn_out_lse.record_stream(main_stream)

        # Current K/V is replicated on every CP rank. Each DCP rank computes
        # only the Q heads it owns after history all-to-all. Merge this chunk
        # locally after the collective so it is counted exactly once.
        current_output, current_lse = self._run_dcp_mtp_split_attention_op(
            current_q_nope,
            current_q_pe,
            current_k_nope.contiguous(),
            current_k_pe.contiguous(),
            attn_mask=decode_meta.attn_mask,
            sparse_mode=3,
            block_table=None,
            block_size=0,
            actual_seq_lengths=decode_meta.actual_seq_lengths_q,
            actual_seq_lengths_kv=decode_meta.actual_seq_lengths_q,
            attention_kind=MLASplitAttentionKind.CURRENT,
        )

        # Join the history communication only when both branches are ready.
        main_stream.wait_event(history_comm_done)
        # Merge two local contributions; this helper performs no collective.
        # Its FP32 kernel ignores invalid LSE, including FIA's +inf sentinel.
        current_attn_out_lse = torch.cat((current_output.float(), current_lse.float()), dim=-1)
        attn_output = fused_sfa_dcp_lse_combine(
            torch.stack((history_attn_out_lse, current_attn_out_lse)),
            self.kv_lora_rank,
            scatter_dim=0,
        )
        return self._v_up_proj_batch_major(attn_output)

    def _forward_decode(
        self,
        decode_preprocess_res: DecodeMLAPreprocessResult,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        if self._use_history_current_split_decode(attn_metadata):
            return self._forward_decode_split_attention(
                decode_preprocess_res.ql_nope,
                decode_preprocess_res.q_pe,
                decode_preprocess_res.k_nope,
                decode_preprocess_res.k_pe,
                decode_preprocess_res.current_k_nope,
                decode_preprocess_res.current_k_pe,
                block_size,
                attn_metadata,
            )
        q_nope = decode_preprocess_res.ql_nope
        q_pe = decode_preprocess_res.q_pe
        k_nope = decode_preprocess_res.k_nope
        k_pe = decode_preprocess_res.k_pe
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
            spec_attn_mask = decode_meta.dcp_mtp_attn_mask if attn_metadata.causal else None
            query_lens = attn_metadata.query_lens
            assert query_lens is not None
            decode_query_lens = query_lens[:num_decodes]
            assert sum(decode_query_lens) == num_tokens
            # This function only runs the decode sub-batch. A mixed
            # decode/prefill batch still carries query lengths for every
            # request in the common metadata, but FIA requires the query-length
            # list to match the decode batch dimension and block table.
            actual_seq_lengths = decode_query_lens
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
