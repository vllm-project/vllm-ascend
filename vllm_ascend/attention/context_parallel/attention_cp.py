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
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

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
    AscendDCPMetadata,
    enable_dcp,
    filter_chunked_req_indices,
    split_decodes_and_prefills,
)
from vllm_ascend.compilation.updatable_graph import get_capture_resource, register_task
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.utils import (
    cp_chunkedprefill_comm_stream,
    is_pd_decode_recompute_scheduler_enabled,
)


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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dcp_enabled = enable_dcp()

    def _split_decodes_and_prefills(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> tuple[int, int, int, int]:
        return split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            require_uniform=self.speculative_config is not None,
            # Full decode graphs replay verification batches with up to
            # decode_threshold queries even when the input batch retains a
            # prefill flag. Their GQA DCP metadata must use the decode path.
            treat_short_extends_as_decodes=(
                self.speculative_config is not None
                or (self.dcp_enabled and is_pd_decode_recompute_scheduler_enabled(self.vllm_config))
            ),
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
        dcp_metadata = common_attn_metadata.dcp_context
        if dcp_metadata is None or dcp_metadata.num_computed_tokens_of_dcp is None:
            # V2 only attaches the verification mask to the common context.
            # Populate the GQA decode view from the available CPU lengths.
            local_seq_lens = get_dcp_local_seq_lens(
                seq_lens[: common_attn_metadata.num_reqs],
                dcp_size=self.dcp_size,
                cp_kv_cache_interleave_size=self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
            )
            dcp_metadata = AscendDCPMetadata(
                num_computed_tokens_of_dcp=local_seq_lens.numpy(),
                query_lens_cpu=query_lens,
                max_query_len=common_attn_metadata.max_query_len,
                dcp_mtp_attn_mask=(dcp_metadata.dcp_mtp_attn_mask if dcp_metadata is not None else None),
            )
        prefill_metadata = None
        if num_prefills > 0:
            prefill_query_lens = query_lens[num_decodes:]
            context_lens_cpu = (seq_lens - query_lens)[num_decodes:]
            chunked_context_metadata = None
            if self.chunked_prefill_enabled and context_lens_cpu.numel() > 0 and context_lens_cpu.max().item() > 0:
                # Use the cached history, not total lengths, for prefill
                # context. V2 does not publish the legacy DCP payload here.
                local_context_lens_allranks = get_dcp_local_seq_lens(
                    context_lens_cpu,
                    dcp_size=self.dcp_size,
                    cp_kv_cache_interleave_size=self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
                ).to(self.device)
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


@dataclass(frozen=True, slots=True)
class DCPFIAParamProvider:
    layer_name: str | None
    dcp_rank: int
    use_bsnd: bool

    def resolve(self, attn_metadata) -> dict[str, object]:
        metadata = attn_metadata[self.layer_name]
        decode_metadata = metadata.decode_meta
        assert decode_metadata is not None
        query_lens = metadata.actual_seq_lengths_q[: metadata.num_decodes]
        if self.use_bsnd:
            query_lens = [query_lens[0]] * len(query_lens)
        local_seq_lens = decode_metadata.num_computed_tokens_of_dcp[:, self.dcp_rank].tolist()
        local_seq_lens.extend([0] * (len(query_lens) - len(local_seq_lens)))
        return {
            "actual_seq_lengths": query_lens,
            "actual_seq_lengths_kv": local_seq_lens,
            "block_table": decode_metadata.block_tables,
            "atten_mask": decode_metadata.dcp_mtp_attn_mask,
        }


class AscendAttentionDCPImpl(DCPImplMixin, AscendAttentionBackendImpl):
    can_return_lse_for_decode: bool = True
    supports_mtp_with_cp_non_trivial_interleave_size: bool = True

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
        num_decodes = attn_metadata.num_decodes
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q[:num_decodes]
        if self.vllm_config.speculative_config is not None:
            input_layerout = "BSND"
            # Padded metadata may contain zero-query requests. The sliced
            # decode query contains only the active, uniform query prefix.
            query_len = actual_seq_lengths_q[0]
            query = query.view(-1, query_len, query.shape[1], query.shape[-1])
            num_decodes = query.shape[0]
            if attn_metadata.decode_meta.dcp_mtp_attn_mask is not None:
                attn_mask = attn_metadata.decode_meta.dcp_mtp_attn_mask[:num_decodes]
            actual_seq_lengths_q = [query_len] * num_decodes

        common_kwargs = {
            "num_heads": num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "input_layout": input_layerout,
            "atten_mask": attn_mask,
            "scale": self.scale,
            "antiquant_mode": 0,
            "antiquant_scale": None,
            "softmax_lse_flag": True,
            "block_table": attn_metadata.decode_meta.block_tables[:num_decodes],
            "block_size": self.key_cache.shape[1],
            "actual_seq_lengths_kv": attn_metadata.decode_meta.num_computed_tokens_of_dcp[:num_decodes, self.dcp_rank],
            "actual_seq_lengths": actual_seq_lengths_q,
        }

        if _EXTRA_CTX.capturing:
            workspace = get_capture_resource(
                (DCPFIAParamProvider, num_heads, self.num_kv_heads, input_layerout),
                lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    query, k_nope, value, **common_kwargs
                ),
                self._use_max_workspace_for_fia_graph,
            )
            attn_out = torch.empty_like(query)
            if input_layerout == "TND":
                attn_lse = torch.empty((query.shape[0], num_heads, 1), dtype=torch.float, device=query.device)
            else:
                attn_lse = torch.empty(
                    (query.shape[0], num_heads, query.shape[1], 1), dtype=torch.float, device=query.device
                )
            register_task(
                torch_npu.npu_fused_infer_attention_score.out,
                {
                    "query": query,
                    "key": k_nope,
                    "value": value,
                    **common_kwargs,
                    "workspace": workspace,
                    "out": [attn_out, attn_lse],
                },
                DCPFIAParamProvider(self._graph_metadata_layer_name(), self.dcp_rank, input_layerout == "BSND"),
            )
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
