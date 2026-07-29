#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.context_parallel.common_cp import (
    DCPImplMixin,
    DCPMetadataBuilderMixin,
    _npu_attn_out_lse_update,
    _update_out_and_lse,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    filter_chunked_req_indices,
    split_decodes_and_prefills,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.memcache_comm_fence import record_attention_compute_start
from vllm_ascend.utils import cp_chunkedprefill_comm_stream, weak_ref_tensors


@dataclass
class AscendMetadataForPrefill:
    """GQA prefill metadata used only by DCP."""

    @dataclass
    class ChunkedContextMetadata:
        actual_chunk_seq_lengths: torch.Tensor
        actual_seq_lengths_kv: torch.Tensor
        starts: torch.Tensor
        chunk_seq_mask_filtered_indices: torch.Tensor
        chunked_req_mask: list[bool] | None = None
        local_context_lens_allranks: list[list[int]] | None = None
        local_total_toks: int | None = None

    chunked_context: ChunkedContextMetadata | None = None
    block_tables: torch.Tensor = None
    actual_seq_lengths_q: torch.Tensor = None


@dataclass
class AscendMetadataForDecode:
    """GQA decode metadata used only by DCP."""

    num_computed_tokens_of_dcp: list[list[int]] | None = None
    block_tables: torch.Tensor = None
    dcp_mtp_attn_mask: torch.Tensor = None


@dataclass
class AscendAttentionDCPMetadata(AscendMetadata):
    """GQA metadata fields used only by the DCP execution path."""

    prefill: AscendMetadataForPrefill | None = None
    decode_meta: AscendMetadataForDecode | None = None


class AscendAttentionDCPMetadataBuilder(
    DCPMetadataBuilderMixin,
    AscendAttentionMetadataBuilder,
):
    """Build attention metadata for decode context parallelism."""

    metadata_cls = AscendAttentionDCPMetadata

    def _split_decodes_and_prefills(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> tuple[int, int, int, int]:
        return split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            treat_short_extends_as_decodes=False,
        )

    @staticmethod
    def _get_chunked_req_mask(local_context_lens_allranks) -> list[bool]:
        if len(local_context_lens_allranks) == 0:
            return []
        return [(req.sum() > 0).item() for req in local_context_lens_allranks if req is not None]

    def _build_backend_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        *,
        block_table: torch.Tensor,
        query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        num_decodes: int,
        num_prefills: int,
    ) -> dict[str, object]:
        dcp_metadata = self._require_dcp_metadata(common_attn_metadata)
        prefill_metadata = None
        if num_prefills > 0:
            prefill_query_lens = query_lens[num_decodes:]
            context_lens_cpu = (seq_lens - query_lens)[num_decodes:]
            chunked_context_metadata = None
            if self.chunked_prefill_enabled and context_lens_cpu.numel() > 0 and context_lens_cpu.max().item() > 0:
                local_context_lens_allranks = self._get_dcp_context_lens(
                    common_attn_metadata,
                    start=num_decodes,
                    device=self.device,
                )
                local_chunked_kv_lens = local_context_lens_allranks[:, self.dcp_rank]
                chunked_req_mask = self._get_chunked_req_mask(local_context_lens_allranks)
                chunked_context_metadata = AscendMetadataForPrefill.ChunkedContextMetadata(
                    actual_chunk_seq_lengths=torch.cumsum(prefill_query_lens, dim=0),
                    actual_seq_lengths_kv=torch.cumsum(local_chunked_kv_lens, dim=0).tolist(),
                    chunked_req_mask=chunked_req_mask,
                    starts=torch.zeros(
                        len(local_context_lens_allranks),
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    local_context_lens_allranks=local_context_lens_allranks,
                    chunk_seq_mask_filtered_indices=filter_chunked_req_indices(
                        prefill_query_lens,
                        chunked_req_mask,
                    ).to(self.device),
                    local_total_toks=local_chunked_kv_lens.sum().item(),
                )
            prefill_metadata = AscendMetadataForPrefill(
                chunked_context=chunked_context_metadata,
                block_tables=block_table[num_decodes:],
                actual_seq_lengths_q=torch.cumsum(prefill_query_lens, dim=0),
            )

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = AscendMetadataForDecode(
                num_computed_tokens_of_dcp=np.asarray(dcp_metadata.num_computed_tokens_of_dcp)[:num_decodes],
                block_tables=block_table[:num_decodes],
                dcp_mtp_attn_mask=dcp_metadata.dcp_mtp_attn_mask,
            )

        return {
            "prefill": prefill_metadata,
            "decode_meta": decode_metadata,
        }


class AscendAttentionDCPImpl(DCPImplMixin, AscendAttentionBackendImpl):
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
                    value,
                    num_heads,
                    num_kv_heads,
                    scale,
                    block_table,
                    block_size,
                    actual_seq_lengths_kv,
                    actual_seq_lengths_q,
                    attn_output,
                    softmax_lse,
                    dcp_size,
                    dcp_rank,
                    attn_mask,
                ) = param

                if _EXTRA_CTX.is_draft_model:
                    draft_step = attn_count // num_layers
                    actual_seq_lengths_kv = attn_metadata[draft_step][key].decode_meta.num_computed_tokens_of_dcp[
                        :, dcp_rank
                    ]
                    pad_length = num_tokens - len(actual_seq_lengths_kv)
                    if pad_length > 0:
                        pad_tensor = np.zeros(pad_length, dtype=actual_seq_lengths_kv.dtype)
                        actual_seq_lengths_kv = np.concatenate([actual_seq_lengths_kv, pad_tensor])

                    actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                    attn_count = attn_count + 1
                else:
                    actual_seq_lengths_kv = attn_metadata[key].decode_meta.num_computed_tokens_of_dcp[:, dcp_rank]
                    pad_length = num_tokens - len(actual_seq_lengths_kv)
                    if pad_length > 0:
                        pad_tensor = np.zeros(pad_length, dtype=actual_seq_lengths_kv.dtype)
                        actual_seq_lengths_kv = np.concatenate([actual_seq_lengths_kv, pad_tensor])

                    actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q

                if dcp_size > 1:
                    num_heads = num_heads * dcp_size

                torch.npu.graph_task_update_begin(update_stream, handle)

                input_layout = "TND"
                if speculative_config is not None:
                    input_layout = "BSND"
                    actual_seq_lengths_q = [actual_seq_lengths_q[0] for _ in range(len(actual_seq_lengths_q))]

                torch_npu.npu_fused_infer_attention_score.out(
                    q_nope,
                    k_nope,
                    value,
                    num_heads=num_heads,
                    num_key_value_heads=num_kv_heads,
                    input_layout=input_layout,
                    atten_mask=attn_mask,
                    scale=scale,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    softmax_lse_flag=True,
                    block_table=block_table,
                    block_size=block_size,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    actual_seq_lengths=actual_seq_lengths_q,
                    workspace=graph_params.workspaces.get(num_tokens),
                    out=[attn_output, softmax_lse],
                )
                torch.npu.graph_task_update_end(update_stream)

                event.record(update_stream)

    def _forward_decode_dcp(
        self,
        query: torch.Tensor,
        attn_metadata: AscendAttentionDCPMetadata,
    ) -> torch.Tensor:
        assert self.key_cache is not None
        assert self.value_cache is not None

        if self.dcp_size > 1:
            query = self._dcp_all_gather(query, 1)
            num_heads = self.num_heads * self.dcp_size
        else:
            num_heads = self.num_heads

        k_nope = self.key_cache.view(self.key_cache.shape[0], self.key_cache.shape[1], -1)
        value = self.value_cache.view(self.key_cache.shape[0], self.key_cache.shape[1], -1)

        attn_mask = None
        input_layerout = "TND"
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q[: attn_metadata.num_decodes]
        if self.vllm_config.speculative_config is not None:
            input_layerout = "BSND"
            num_decodes = attn_metadata.num_decodes
            if attn_metadata.decode_meta.dcp_mtp_attn_mask is not None:
                attn_mask = attn_metadata.decode_meta.dcp_mtp_attn_mask
            else:
                attn_mask = None
            query = query.view(num_decodes, -1, query.shape[1], query.shape[-1])
            actual_seq_lengths_q = [actual_seq_lengths_q[0] for _ in range(len(actual_seq_lengths_q))]

        common_kwargs = {
            "num_heads": num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "input_layout": input_layerout,
            "atten_mask": attn_mask,
            "scale": self.scale,
            "antiquant_mode": 0,
            "antiquant_scale": None,
            "softmax_lse_flag": True,
            "block_table": attn_metadata.decode_meta.block_tables,
            "block_size": self.key_cache.shape[1],
            "actual_seq_lengths_kv": attn_metadata.decode_meta.num_computed_tokens_of_dcp[
                : attn_metadata.num_decodes, self.dcp_rank
            ],
            "actual_seq_lengths": actual_seq_lengths_q,
        }

        if _EXTRA_CTX.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()

        if input_layerout == "TND":
            num_tokens = query.shape[0]
        else:
            num_tokens = query.shape[0] * query.shape[1]

        if _EXTRA_CTX.capturing:
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)

            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    query, k_nope, value, **common_kwargs
                )
                if _EXTRA_CTX.is_draft_model:
                    update_draft_graph_params_workspaces(num_tokens, workspace)
                else:
                    update_graph_params_workspaces(num_tokens, workspace)
            attn_out = torch.empty_like(query)
            if input_layerout == "TND":
                attn_lse = torch.empty((num_tokens, num_heads, 1), dtype=torch.float, device=query.device)
            else:
                attn_lse = torch.empty(
                    (query.shape[0], num_heads, query.shape[1], 1), dtype=torch.float, device=query.device
                )
            graph_params.attn_params[num_tokens].append(
                (
                    weak_ref_tensors(query),
                    weak_ref_tensors(k_nope),
                    weak_ref_tensors(value),
                    self.num_heads,
                    self.num_kv_heads,
                    self.scale,
                    attn_metadata.block_tables,
                    self.key_cache.shape[1],
                    attn_metadata.decode_meta.num_computed_tokens_of_dcp[: attn_metadata.num_decodes, self.dcp_rank],
                    actual_seq_lengths_q,
                    weak_ref_tensors(attn_out),
                    weak_ref_tensors(attn_lse),
                    self.dcp_size,
                    self.dcp_rank,
                    attn_mask,
                )
            )
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                query, k_nope, value, **common_kwargs, workspace=workspace, out=[attn_out, attn_lse]
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
        else:
            attn_out, attn_lse = torch_npu.npu_fused_infer_attention_score(query, k_nope, value, **common_kwargs)
        if input_layerout == "BSND":
            attn_out = attn_out.view(-1, attn_out.shape[2], attn_out.shape[3])
            attn_lse = attn_lse.transpose(1, 2).reshape(-1, attn_lse.shape[1], 1)
        return self._merge_dcp_attention_output(
            attn_out,
            attn_lse,
            self.head_size,
        )

    def _update_chunk_attn_out_lse_with_current_attn_out_lse(
        self,
        current_attn_output_prefill,
        current_attn_lse_prefill,
        attn_output_full_chunk,
        attn_lse_full_chunk,
        prefill_query,
        attn_metadata,
    ):
        num_tokens = prefill_query.size(0)
        attn_output_full_chunk = attn_output_full_chunk[:num_tokens]
        attn_lse_full_chunk = attn_lse_full_chunk[:num_tokens]

        assert (
            attn_output_full_chunk.shape == current_attn_output_prefill.shape
            and attn_lse_full_chunk.shape == current_attn_lse_prefill.shape
        )
        filtered_indices = attn_metadata.prefill.chunked_context.chunk_seq_mask_filtered_indices

        attn_output_prefill_filtered = current_attn_output_prefill[filtered_indices, :, :]
        attn_lse_prefill_filtered = current_attn_lse_prefill[filtered_indices, :, :]
        attn_output_full_chunk = attn_output_full_chunk[filtered_indices, :, :]
        attn_lse_full_chunk = attn_lse_full_chunk[filtered_indices, :, :]

        attn_output_filtered = _npu_attn_out_lse_update(
            attn_lse_prefill_filtered, attn_lse_full_chunk, attn_output_prefill_filtered, attn_output_full_chunk
        )

        current_attn_output_prefill[filtered_indices, :, :] = attn_output_filtered.to(current_attn_output_prefill.dtype)

    def _prefill_query_all_gather(self, attn_metadata, prefill_query):
        return self._dcp_all_gather(prefill_query, 1)

    def _compute_prefill_context(
        self,
        query: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendAttentionDCPMetadata,
    ):
        assert len(kv_cache) > 1
        assert attn_metadata is not None
        assert attn_metadata.prefill is not None
        assert attn_metadata.prefill.chunked_context is not None
        prefill_metadata = attn_metadata.prefill
        local_chunked_kv_lens = prefill_metadata.chunked_context.local_context_lens_allranks
        assert local_chunked_kv_lens is not None

        local_chunked_kv_lens_rank = local_chunked_kv_lens[:, self.dcp_rank]
        total_toks = prefill_metadata.chunked_context.local_total_toks
        key, value = self._load_kv_for_chunk(attn_metadata, kv_cache, local_chunked_kv_lens_rank, query, total_toks)
        if self.dcp_size > 1:
            num_heads = self.num_heads * self.dcp_size
        else:
            num_heads = self.num_heads

        if total_toks == 0:
            return (
                torch.full(
                    (query.size(0), num_heads, self.head_size), fill_value=0, dtype=query.dtype, device=query.device
                ),
                torch.full(
                    (query.size(0), num_heads, 1), fill_value=-torch.inf, dtype=torch.float32, device=query.device
                ),
            )

        prefix_chunk_output, prefix_chunk_lse = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            key.contiguous(),
            value.contiguous(),
            num_heads=num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            atten_mask=None,
            scale=self.scale,
            sparse_mode=0,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=True,
            actual_seq_lengths_kv=prefill_metadata.chunked_context.actual_seq_lengths_kv,
            actual_seq_lengths=attn_metadata.prefill.chunked_context.actual_chunk_seq_lengths,
        )

        return prefix_chunk_output, prefix_chunk_lse

    def _load_kv_for_chunk(self, attn_metadata, kv_cache, local_chunked_kv_lens_rank, query, total_toks):
        cache_key = kv_cache[0]
        cache_value = kv_cache[1]
        num_heads = cache_key.size(2)
        head_size = kv_cache[0].size(-1)

        key = torch.empty(total_toks, num_heads, head_size, dtype=query.dtype, device=query.device)
        value = torch.empty(total_toks, num_heads, head_size, dtype=query.dtype, device=query.device)
        if total_toks > 0:
            DeviceOperator.kv_cache_load(
                cache_key,
                cache_value,
                attn_metadata.prefill.block_tables,
                local_chunked_kv_lens_rank,
                # slot offsets of current chunk in current iteration
                attn_metadata.prefill.chunked_context.starts,
                key=key,
                value=value,
            )
        return key, value

    def _gather_global_context_output(self, local_context_attn_output):
        if self.dcp_size > 1:
            dcp_context_attn_output = torch.empty_like(local_context_attn_output)
            dist.all_to_all_single(
                dcp_context_attn_output,
                local_context_attn_output,
                group=self.dcp_device_group,
            )
        else:
            dcp_context_attn_output = local_context_attn_output

        return dcp_context_attn_output

    def _update_global_context_output(self, global_context_output):
        B_total, H_total, D_plus_1 = global_context_output.shape
        S = B_total
        H = H_total // self.dcp_size
        D = self.head_size
        assert D_plus_1 == D + 1
        x = global_context_output.view(S, self.dcp_size, H, D_plus_1)
        x = x.permute(1, 0, 2, 3).contiguous()
        # Split out lse
        attn_out_allgather, attn_lse_allgather = torch.split(x, [D, 1], dim=-1)  # [N, S, H, D], [N, S, H, 1]
        context_output, context_lse = _update_out_and_lse(attn_out_allgather, attn_lse_allgather)
        return context_output, context_lse

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, AscendAttentionDCPMetadata)
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens
        if has_decode:
            decode_query = query[:num_decode_tokens].contiguous()
            output_decode = self._forward_decode_dcp(decode_query, attn_metadata)
            output[:num_decode_tokens] = output_decode
        if has_prefill:
            assert attn_metadata.prefill is not None
            # chunked prefill vars init
            has_chunked_context = attn_metadata.prefill.chunked_context is not None
            # Note(qcs): we use multi-stream for computation-communication overlap
            # when enabling chunked prefill.
            # current part
            # current_stream: init -- pre -- head attn ------------------ tail attn -- post -- update
            # context part                                                                     -/
            # current_stream: -----                    -- context attn --                     -/
            # COMM_STREAM:         \-- all_gather Q --/                  \-- a2a ag output --/

            # qkv init
            prefill_query = query[num_decode_tokens : attn_metadata.num_actual_tokens].contiguous()
            key = key[num_decode_tokens : attn_metadata.num_actual_tokens].contiguous()
            value = value[num_decode_tokens : attn_metadata.num_actual_tokens].contiguous()

            if has_chunked_context:
                # all_gather q for chunked prefill // overlap the computation inner current chunk
                cp_chunkedprefill_comm_stream().wait_stream(torch.npu.current_stream())
                with torch_npu.npu.stream(cp_chunkedprefill_comm_stream()):
                    prefill_query_all = self._prefill_query_all_gather(attn_metadata, prefill_query.clone())

            # Record the compute-stream gate once before any attention phase
            # starts, so the layerwise transfer thread can overlap H2D copies
            # with the prefill computation.
            record_attention_compute_start()

            attn_output_prefill, attn_lse_prefill = torch.ops.npu.npu_fused_infer_attention_score(
                prefill_query,
                key.contiguous(),
                value.contiguous(),
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                atten_mask=attn_metadata.attn_mask,
                scale=self.scale,
                sparse_mode=3,
                antiquant_mode=0,
                antiquant_scale=None,
                softmax_lse_flag=True,
                actual_seq_lengths_kv=attn_metadata.prefill.actual_seq_lengths_q,
                actual_seq_lengths=attn_metadata.prefill.actual_seq_lengths_q,
            )

            if has_chunked_context:
                torch.npu.current_stream().wait_stream(cp_chunkedprefill_comm_stream())
                # computation of context
                context_output = self._compute_prefill_context(prefill_query_all, kv_cache, attn_metadata)
                # Note(qcs): (output, lse) -> [Seq, Head_num, Head_dim+1] -> [Head_num, Head_dim+1, Seq]
                local_context_output = torch.cat(context_output, dim=-1).permute([1, 2, 0]).contiguous()

                # all2all and all_gather output&lse // overlap the computation inner current chunk
                cp_chunkedprefill_comm_stream().wait_stream(torch.npu.current_stream())
                with torch_npu.npu.stream(cp_chunkedprefill_comm_stream()):
                    global_context_output = self._gather_global_context_output(local_context_output)

            if has_chunked_context:
                # update the output of current chunk with context part
                torch.npu.current_stream().wait_stream(cp_chunkedprefill_comm_stream())
                global_context_output = global_context_output.permute([2, 0, 1]).contiguous()
                context_output, context_lse = self._update_global_context_output(global_context_output)
                self._update_chunk_attn_out_lse_with_current_attn_out_lse(
                    attn_output_prefill, attn_lse_prefill, context_output, context_lse, prefill_query, attn_metadata
                )

            output[num_decode_tokens : attn_output_prefill.shape[0] + num_decode_tokens] = attn_output_prefill
        return output
