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
import torch_npu
from vllm.distributed import get_pcp_group
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.context_parallel.common_cp import (
    CPKVScope,
    DCPImplMixin,
    DCPMetadataBuilderMixin,
    use_history_current_split_decode,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_dcp,
    filter_chunked_req_indices,
    split_decodes_and_prefills,
)
from vllm_ascend.compilation.updatable_graph import get_capture_resource, register_task
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.ops.triton.dcp.dcp_a2a import fused_dcp_lse_combine
from vllm_ascend.utils import (
    cp_chunkedprefill_comm_stream,
    cp_decode_comm_stream,
    get_gqa_full_dcp_kv_heads,
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
        local_context_lens: torch.Tensor | None = None
        local_total_toks: int | None = None
        empty_context_query_mask: torch.Tensor | None = None

    chunked_context: ChunkedContextMetadata | None = None
    block_tables: torch.Tensor = None
    actual_seq_lengths_q: torch.Tensor = None
    pcp_actual_seq_lengths_q: list[int] | None = None
    pcp_prefill_restore_idx: torch.Tensor | None = None
    pcp_local_prefill_indices: torch.Tensor | None = None
    pcp_local_num_input_tokens: int | None = None
    pcp_local_num_decode_tokens: int = 0
    pcp_global_num_decode_tokens: int = 0


@dataclass
class AscendMetadataForDecode:
    """GQA decode metadata used only by DCP."""

    num_computed_tokens_of_dcp: list[list[int]] | None = None
    block_tables: torch.Tensor = None
    cp_history_seq_len: list[int] | None = None
    actual_seq_lengths_q: list[int] | None = None
    seq_lens_list: list[int] | None = None

    def update_dcp_seq_lens_cpu(
        self,
        seq_lens_cpu: torch.Tensor,
        dcp_local_seq_lens_cpu: torch.Tensor,
        query_lens_cpu: torch.Tensor,
        *,
        dcp_size: int,
        dcp_rank: int,
        cp_kv_cache_interleave_size: int,
    ) -> None:
        """Partition history after removing the global current-token chunk."""
        self.num_computed_tokens_of_dcp = get_dcp_local_seq_lens(
            seq_lens_cpu,
            dcp_size=dcp_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        ).numpy()
        self.num_computed_tokens_of_dcp[:, dcp_rank] = 0
        self.num_computed_tokens_of_dcp[: dcp_local_seq_lens_cpu.numel(), dcp_rank] = dcp_local_seq_lens_cpu.numpy()
        self.cp_history_seq_len = get_dcp_local_seq_lens(
            (seq_lens_cpu - query_lens_cpu).clamp(min=0),
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        ).tolist()


@dataclass
class AscendAttentionDCPMetadata(AscendMetadata):
    """GQA metadata fields used only by the DCP execution path."""

    prefill: AscendMetadataForPrefill | None = None
    decode: AscendMetadataForDecode | None = None


class AscendAttentionDCPMetadataBuilder(
    DCPMetadataBuilderMixin,
    AscendAttentionMetadataBuilder,
):
    """Build attention metadata for decode context parallelism."""

    metadata_cls = AscendAttentionDCPMetadata
    consumes_pcp_context = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dcp_enabled = enable_dcp()
        self.pcp_group = get_pcp_group() if getattr(self, "pcp_enabled", False) else None
        self._pcp_context = None
        self._pcp_cache_group_idx = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        *,
        pcp_context=None,
        pcp_cache_group_idx: int | None = None,
    ) -> AscendAttentionDCPMetadata:
        self._pcp_context = pcp_context
        self._pcp_cache_group_idx = pcp_cache_group_idx
        return super().build(common_prefix_len, common_attn_metadata, fast_build)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        **kwargs,
    ) -> AscendAttentionDCPMetadata:
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata, **kwargs
        )

    def _split_decodes_and_prefills(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> tuple[int, int, int, int]:
        return split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            treat_short_extends_as_decodes=(
                self.speculative_config is not None
                or (self.dcp_enabled and is_pd_decode_recompute_scheduler_enabled(self.vllm_config))
            ),
        )

    @staticmethod
    def _get_chunked_req_mask(context_lens_cpu: torch.Tensor) -> list[bool]:
        return (context_lens_cpu > 0).tolist()

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
        prefill_metadata = None
        if num_prefills > 0:
            prefill_query_lens = query_lens[num_decodes:]
            prefill_query_ends = torch.cumsum(prefill_query_lens, dim=0)
            context_lens_cpu = (seq_lens - query_lens)[num_decodes:]
            prefill_block_table = block_table[num_decodes:]
            pcp_actual_seq_lengths_q = None
            pcp_prefill_restore_idx = None
            pcp_local_prefill_indices = None
            pcp_local_num_input_tokens = None
            pcp_local_num_decode_tokens = 0
            pcp_global_num_decode_tokens = 0

            pcp_context = getattr(self, "_pcp_context", None)
            if getattr(self, "pcp_enabled", False) and pcp_context is not None and bool(
                pcp_context.global_batch.is_prefilling_np.any()
            ):
                assert self.pcp_group is not None
                if self._pcp_cache_group_idx is None:
                    raise RuntimeError("GQA PCP+DCP prefill requires the PCP cache-group index.")
                if pcp_context.padded_gather_idx is None:
                    raise RuntimeError("GQA PCP+DCP prefill requires PCP's padded gather layout.")

                global_batch = pcp_context.global_batch
                global_num_reqs = global_batch.num_reqs
                global_is_prefilling = global_batch.is_prefilling_np[:global_num_reqs]
                global_num_decodes = int((~global_is_prefilling).sum())
                if global_is_prefilling[:global_num_decodes].any() or (
                    ~global_is_prefilling[global_num_decodes:]
                ).any():
                    raise RuntimeError("GQA PCP+DCP expects decode requests before prefill requests.")

                global_query_lens = torch.from_numpy(
                    np.diff(global_batch.query_start_loc_np[: global_num_reqs + 1]).copy()
                ).to(torch.int32)
                global_seq_lens = torch.from_numpy(
                    global_batch.seq_lens_np[:global_num_reqs].copy()
                ).to(torch.int32)
                prefill_query_lens = global_query_lens[global_num_decodes:]
                prefill_query_ends = torch.cumsum(prefill_query_lens, dim=0)
                context_lens_cpu = (
                    global_seq_lens[global_num_decodes:] - prefill_query_lens
                ).clamp(min=0)
                prefill_block_table = pcp_context.global_block_tables[
                    self._pcp_cache_group_idx
                ][global_num_decodes:global_num_reqs]
                pcp_actual_seq_lengths_q = prefill_query_ends.tolist()
                pcp_local_num_input_tokens = (
                    pcp_context.padded_gather_idx.numel() // self.pcp_group.world_size
                )
                local_start = self.pcp_group.rank_in_group * pcp_local_num_input_tokens
                local_num_actual_tokens = common_attn_metadata.num_actual_tokens
                pcp_local_num_decode_tokens = int(query_lens[:num_decodes].sum().item())
                pcp_global_num_decode_tokens = int(
                    global_batch.query_start_loc_np[global_num_decodes]
                )
                pcp_local_prefill_indices = pcp_context.padded_gather_idx[
                    local_start + pcp_local_num_decode_tokens : local_start + local_num_actual_tokens
                ] - pcp_global_num_decode_tokens
                global_prefill_restore_idx = pcp_context.hidden_restore_idx[
                    pcp_global_num_decode_tokens:
                ]
                restore_rank = torch.div(
                    global_prefill_restore_idx,
                    pcp_local_num_input_tokens,
                    rounding_mode="floor",
                )
                restore_offset = (
                    global_prefill_restore_idx % pcp_local_num_input_tokens
                ) - pcp_local_num_decode_tokens
                local_prefill_capacity = pcp_local_num_input_tokens - pcp_local_num_decode_tokens
                pcp_prefill_restore_idx = restore_rank * local_prefill_capacity + restore_offset

            chunked_context_metadata = None
            if (
                (self.chunked_prefill_enabled or pcp_actual_seq_lengths_q is not None)
                and context_lens_cpu.numel() > 0
                and context_lens_cpu.max().item() > 0
            ):
                local_chunked_kv_lens_cpu = get_dcp_local_seq_lens(
                    context_lens_cpu,
                    dcp_size=self.dcp_size,
                    dcp_rank=self.dcp_rank,
                    cp_kv_cache_interleave_size=self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
                )
                chunked_req_mask = self._get_chunked_req_mask(context_lens_cpu)
                # KV cache load uses device-local history; host FIA parameters stay on CPU.
                if pcp_actual_seq_lengths_q is not None:
                    local_context_lens = local_chunked_kv_lens_cpu.to(self.device)
                else:
                    prefill_end = num_decodes + num_prefills
                    query_start_loc = common_attn_metadata.query_start_loc[num_decodes : prefill_end + 1]
                    context_lens = common_attn_metadata.seq_lens[num_decodes:prefill_end] - torch.diff(
                        query_start_loc
                    )
                    local_context_lens = get_dcp_local_seq_lens(
                        context_lens,
                        dcp_size=self.dcp_size,
                        dcp_rank=self.dcp_rank,
                        cp_kv_cache_interleave_size=self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
                    )
                chunked_context_metadata = AscendMetadataForPrefill.ChunkedContextMetadata(
                    actual_chunk_seq_lengths=prefill_query_ends,
                    actual_seq_lengths_kv=torch.cumsum(local_chunked_kv_lens_cpu, dim=0).tolist(),
                    chunked_req_mask=chunked_req_mask,
                    starts=torch.zeros(
                        context_lens_cpu.numel(),
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    local_context_lens=local_context_lens,
                    chunk_seq_mask_filtered_indices=filter_chunked_req_indices(
                        prefill_query_lens,
                        chunked_req_mask,
                    ).to(self.device),
                    local_total_toks=local_chunked_kv_lens_cpu.sum().item(),
                    empty_context_query_mask=torch.repeat_interleave(
                        local_chunked_kv_lens_cpu == 0,
                        prefill_query_lens,
                    ).to(self.device),
                )
            prefill_metadata = AscendMetadataForPrefill(
                chunked_context=chunked_context_metadata,
                block_tables=prefill_block_table,
                actual_seq_lengths_q=torch.cumsum(query_lens[num_decodes:], dim=0),
                pcp_actual_seq_lengths_q=pcp_actual_seq_lengths_q,
                pcp_prefill_restore_idx=pcp_prefill_restore_idx,
                pcp_local_prefill_indices=pcp_local_prefill_indices,
                pcp_local_num_input_tokens=pcp_local_num_input_tokens,
                pcp_local_num_decode_tokens=pcp_local_num_decode_tokens,
                pcp_global_num_decode_tokens=pcp_global_num_decode_tokens,
            )

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = AscendMetadataForDecode(
                block_tables=block_table[:num_decodes],
                actual_seq_lengths_q=query_lens[:num_decodes].cumsum(0).tolist(),
                seq_lens_list=seq_lens[:num_decodes].tolist(),
            )
            dcp_local_seq_lens_cpu = common_attn_metadata.dcp_local_seq_lens_cpu
            assert dcp_local_seq_lens_cpu is not None
            decode_metadata.update_dcp_seq_lens_cpu(
                seq_lens[:num_decodes],
                dcp_local_seq_lens_cpu[:num_decodes],
                query_lens[:num_decodes],
                dcp_size=self.dcp_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
            )

        return {
            "prefill": prefill_metadata,
            "decode": decode_metadata,
        }


@dataclass(frozen=True, slots=True)
class DCPFIAParamProvider:
    metadata_layer_name: str | None
    dcp_rank: int
    attention_kind: CPKVScope

    @property
    def layer_name(self) -> tuple[str | None, CPKVScope]:
        # Draft SharedSource matches each captured task independently.
        return self.metadata_layer_name, self.attention_kind

    def resolve(self, attn_metadata) -> dict[str, object]:
        metadata = attn_metadata[self.metadata_layer_name]
        decode = metadata.decode
        assert decode is not None
        query_lens = decode.actual_seq_lengths_q
        assert query_lens is not None
        if self.attention_kind == CPKVScope.CURRENT:
            kv_lens = query_lens
            block_table = None
        elif self.attention_kind == CPKVScope.HISTORY:
            assert decode.cp_history_seq_len is not None
            kv_lens = decode.cp_history_seq_len
            block_table = decode.block_tables
        else:
            kv_lens = decode.num_computed_tokens_of_dcp[:, self.dcp_rank].tolist()
            block_table = decode.block_tables
        return {
            "actual_seq_lengths": query_lens,
            "actual_seq_lengths_kv": kv_lens,
            "block_table": block_table,
        }


def build_dcp_fia_params(
    layer_name: str,
    metadata,
    dcp_rank: int,
    *,
    is_draft_model: bool = False,
    is_draft_model_prefill: bool = False,
    use_spec_decode: bool = False,
) -> list[dict[str, object]]:
    """Publish parameters for the captured split or ordinary cache task."""
    use_split = use_history_current_split_decode(
        metadata,
        is_draft_model=is_draft_model,
        is_draft_model_prefill=is_draft_model_prefill,
        use_spec_decode=use_spec_decode,
    )
    kinds = (CPKVScope.HISTORY, CPKVScope.CURRENT) if use_split else (CPKVScope.FULL,)
    params = []
    for kind in kinds:
        provider = DCPFIAParamProvider(layer_name, dcp_rank, kind)
        params.append({"layer_name": provider.layer_name, **provider.resolve({layer_name: metadata})})
    return params


class AscendAttentionDCPImpl(DCPImplMixin, AscendAttentionBackendImpl):
    can_return_lse_for_decode: bool = True
    supports_mtp_with_cp_non_trivial_interleave_size: bool = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pcp_enabled = self.vllm_config.parallel_config.prefill_context_parallel_size > 1
        self._pcp_gathered_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        full_dcp_kv_heads = get_gqa_full_dcp_kv_heads(
            self.vllm_config.model_config,
            self.vllm_config.parallel_config,
        )
        self.dcp_cache_num_kv_heads = full_dcp_kv_heads or self.num_kv_heads
        self.replicate_full_dcp_kv_heads = self.dcp_cache_num_kv_heads > self.num_kv_heads
        self.kv_head_replication_factor = (
            self.tp_group.world_size // self.dcp_cache_num_kv_heads
            if self.replicate_full_dcp_kv_heads
            else 1
        )

    def _record_pcp_gathered_kv(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self._pcp_gathered_kv = key, value

    def _gather_full_dcp_kv_heads(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replicate distinct model KV heads on every full-domain DCP rank."""
        if not self.replicate_full_dcp_kv_heads:
            return key, value
        key = self.tp_group.all_gather(key.contiguous(), dim=1)
        value = self.tp_group.all_gather(value.contiguous(), dim=1)
        replica_stride = self.kv_head_replication_factor
        return (
            key[:, ::replica_stride].contiguous(),
            value[:, ::replica_stride].contiguous(),
        )

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        key, value = self._gather_full_dcp_kv_heads(key, value)
        return super().reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        key, value = self._gather_full_dcp_kv_heads(key, value)
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    def _run_dcp_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendAttentionDCPMetadata,
        attention_kind: CPKVScope,
        num_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        provider = DCPFIAParamProvider(self._graph_metadata_layer_name(), self.dcp_rank, attention_kind)
        params = provider.resolve({provider.metadata_layer_name: attn_metadata})
        is_current = attention_kind == CPKVScope.CURRENT
        cache_num_kv_heads = getattr(self, "dcp_cache_num_kv_heads", self.num_kv_heads)
        num_key_value_heads = self.num_kv_heads if is_current else cache_num_kv_heads
        kwargs = {
            "num_heads": num_heads,
            "num_key_value_heads": num_key_value_heads,
            "input_layout": "TND",
            "atten_mask": attn_metadata.attn_mask if is_current and attn_metadata.causal else None,
            "sparse_mode": 3 if is_current and attn_metadata.causal else 0,
            "scale": self.scale,
            "antiquant_mode": 0,
            "antiquant_scale": None,
            "softmax_lse_flag": True,
            **params,
        }
        if not is_current:
            kwargs["block_size"] = self.key_cache.shape[1]
            kwargs["inner_precise"] = 1
        if not _EXTRA_CTX.capturing:
            return torch_npu.npu_fused_infer_attention_score(query, key, value, **kwargs)

        # Match attention_v1: keep addresses stable and update only runtime
        # lengths/block tables through UpdatableGraph parameter providers.
        workspace = get_capture_resource(
            (DCPFIAParamProvider, attention_kind, num_heads, num_key_value_heads),
            lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(query, key, value, **kwargs),
            self._use_max_workspace_for_fia_graph,
        )
        output = torch.empty_like(query)
        lse = torch.empty((query.shape[0], num_heads, 1), dtype=torch.float32, device=query.device)
        register_task(
            torch_npu.npu_fused_infer_attention_score.out,
            {
                "query": query,
                "key": key,
                "value": value,
                **kwargs,
                "workspace": workspace,
                "out": [output, lse],
            },
            provider,
        )
        return output, lse

    def _forward_decode_dcp(
        self,
        query: torch.Tensor,
        attn_metadata: AscendAttentionDCPMetadata,
        current_key: torch.Tensor | None = None,
        current_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.key_cache is not None and self.value_cache is not None
        (history_query,) = self._dcp_all_gather_fragments(query, dim=1)
        num_heads = history_query.shape[1]
        key = self.key_cache.view(self.key_cache.shape[0], self.key_cache.shape[1], -1)
        value = self.value_cache.view(self.value_cache.shape[0], self.value_cache.shape[1], -1)
        use_split = use_history_current_split_decode(
            attn_metadata,
            is_draft_model=_EXTRA_CTX.is_draft_model,
            is_draft_model_prefill=_EXTRA_CTX.is_draft_model_prefill,
            use_spec_decode=self.vllm_config.speculative_config is not None,
        )
        kind = CPKVScope.HISTORY if use_split else CPKVScope.FULL
        history_output, history_lse = self._run_dcp_attention(history_query, key, value, attn_metadata, kind, num_heads)
        if not use_split:
            return self._merge_dcp_attention_output(history_output, history_lse)

        assert current_key is not None and current_value is not None
        main_stream = torch.npu.current_stream()
        attn_stream = cp_decode_comm_stream()
        history_ready = main_stream.record_event()
        for tensor in (query, current_key, current_value, attn_metadata.attn_mask):
            if tensor is not None:
                tensor.record_stream(attn_stream)
        # Match MLA: current attention overlaps history packing and A2A.
        with torch.npu.stream(attn_stream):
            attn_stream.wait_event(history_ready)
            current_output, current_lse = self._run_dcp_attention(
                query,
                current_key.contiguous(),
                current_value.contiguous(),
                attn_metadata,
                CPKVScope.CURRENT,
                self.num_heads,
            )
            current_attn_done = attn_stream.record_event()
        current_output.record_stream(main_stream)
        current_lse.record_stream(main_stream)
        history_recv = self._merge_dcp_attention_output(
            history_output,
            history_lse,
            defer_combine=True,
        )
        main_stream.wait_event(current_attn_done)
        # Only historical shards participate in A2A. Current K/V are
        # replicated across DCP ranks and must contribute exactly once.
        return fused_dcp_lse_combine(
            history_recv,
            self.head_size,
            scatter_dim=1,
            local_output=current_output,
            local_lse=current_lse,
        )

    def _prefill_query_all_gather(self, attn_metadata, prefill_query):
        (prefill_query,) = self._dcp_all_gather_fragments(prefill_query, dim=1)
        return prefill_query

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
        local_chunked_kv_lens_rank = prefill_metadata.chunked_context.local_context_lens
        assert local_chunked_kv_lens_rank is not None
        total_toks = prefill_metadata.chunked_context.local_total_toks
        key, value = self._load_kv_for_chunk(attn_metadata, kv_cache, local_chunked_kv_lens_rank, query, total_toks)
        num_heads = query.shape[1]

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
            key,
            value,
            num_heads=num_heads,
            num_key_value_heads=getattr(self, "dcp_cache_num_kv_heads", self.num_kv_heads),
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

        empty_context_query_mask = prefill_metadata.chunked_context.empty_context_query_mask
        if empty_context_query_mask is not None:
            prefix_chunk_lse.masked_fill_(empty_context_query_mask[:, None, None], -torch.inf)

        return prefix_chunk_output, prefix_chunk_lse

    def _gather_pcp_prefill_query(
        self,
        query: torch.Tensor,
        prefill_metadata: AscendMetadataForPrefill,
    ) -> torch.Tensor:
        local_num_input_tokens = prefill_metadata.pcp_local_num_input_tokens
        restore_idx = prefill_metadata.pcp_prefill_restore_idx
        local_num_decode_tokens = prefill_metadata.pcp_local_num_decode_tokens
        assert local_num_input_tokens is not None and restore_idx is not None
        gathered_query = self.pcp_group.all_gather(
            query[local_num_decode_tokens:local_num_input_tokens].contiguous(),
            dim=0,
        )
        return torch.index_select(gathered_query, 0, restore_idx)

    def _get_pcp_prefill_current_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        prefill_metadata: AscendMetadataForPrefill,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_num_input_tokens = prefill_metadata.pcp_local_num_input_tokens
        restore_idx = prefill_metadata.pcp_prefill_restore_idx
        local_num_decode_tokens = prefill_metadata.pcp_local_num_decode_tokens
        assert local_num_input_tokens is not None and restore_idx is not None
        if self._pcp_gathered_kv is None:
            gathered_key = self.pcp_group.all_gather(
                key[local_num_decode_tokens:local_num_input_tokens].contiguous(), dim=0
            )
            gathered_value = self.pcp_group.all_gather(
                value[local_num_decode_tokens:local_num_input_tokens].contiguous(), dim=0
            )
        else:
            gathered_key, gathered_value = self._pcp_gathered_kv
            gathered_key = gathered_key[local_num_decode_tokens:]
            gathered_value = gathered_value[local_num_decode_tokens:]
        return (
            torch.index_select(gathered_key, 0, restore_idx),
            torch.index_select(gathered_value, 0, restore_idx),
        )

    def _forward_prefill_pcp_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendAttentionDCPMetadata,
    ) -> torch.Tensor:
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata is not None
        actual_seq_lengths = prefill_metadata.pcp_actual_seq_lengths_q
        local_indices = prefill_metadata.pcp_local_prefill_indices
        assert actual_seq_lengths is not None and local_indices is not None

        global_query = self._gather_pcp_prefill_query(query, prefill_metadata)
        global_key, global_value = self._get_pcp_prefill_current_kv(
            key,
            value,
            prefill_metadata,
        )
        current_output, current_lse = torch.ops.npu.npu_fused_infer_attention_score(
            global_query,
            global_key,
            global_value,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            atten_mask=attn_metadata.attn_mask if attn_metadata.causal else None,
            scale=self.scale,
            sparse_mode=3 if attn_metadata.causal else 0,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=True,
            actual_seq_lengths_kv=actual_seq_lengths,
            actual_seq_lengths=actual_seq_lengths,
        )

        if prefill_metadata.chunked_context is not None:
            (history_query,) = self._dcp_all_gather_fragments(global_query, dim=1)
            history_output, history_lse = self._compute_prefill_context(
                history_query,
                kv_cache,
                attn_metadata,
            )
            history_recv = self._merge_dcp_attention_output(
                history_output,
                history_lse,
                defer_combine=True,
            )
            current_output = fused_dcp_lse_combine(
                history_recv,
                self.head_size,
                scatter_dim=1,
                local_output=current_output,
                local_lse=current_lse,
            )

        return torch.index_select(current_output, 0, local_indices.to(torch.int64))

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
            output_decode = self._forward_decode_dcp(
                decode_query,
                attn_metadata,
                key[:num_decode_tokens] if key is not None else None,
                value[:num_decode_tokens] if value is not None else None,
            )
            output[:num_decode_tokens] = output_decode
        if has_prefill:
            assert attn_metadata.prefill is not None
            if self.pcp_enabled:
                try:
                    prefill_output = self._forward_prefill_pcp_dcp(
                        query,
                        key,
                        value,
                        kv_cache,
                        attn_metadata,
                    )
                finally:
                    self._pcp_gathered_kv = None
                output[
                    num_decode_tokens : num_decode_tokens + prefill_output.shape[0]
                ] = prefill_output
                return output
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
                key,
                value,
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
                history_output, history_lse = self._compute_prefill_context(
                    prefill_query_all,
                    kv_cache,
                    attn_metadata,
                )

                # Exchange and merge the DCP history while current attention
                # runs on the main stream. The shared helper also handles DCP
                # groups that overlap PCP ranks without splitting query heads.
                cp_chunkedprefill_comm_stream().wait_stream(torch.npu.current_stream())
                with torch_npu.npu.stream(cp_chunkedprefill_comm_stream()):
                    history_recv = self._merge_dcp_attention_output(
                        history_output,
                        history_lse,
                        defer_combine=True,
                    )

            if has_chunked_context:
                # Merge the history shards with the current chunk exactly once.
                torch.npu.current_stream().wait_stream(cp_chunkedprefill_comm_stream())
                attn_output_prefill = fused_dcp_lse_combine(
                    history_recv,
                    self.head_size,
                    scatter_dim=1,
                    local_output=attn_output_prefill,
                    local_lse=attn_lse_prefill,
                )

            output[num_decode_tokens : attn_output_prefill.shape[0] + num_decode_tokens] = attn_output_prefill
        return output
