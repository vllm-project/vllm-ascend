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

from typing import ClassVar

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dcp_group,
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
    get_pcp_group,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.context_parallel.common_cp import (
    AscendMetadataForDecode,
    AscendMetadataForPrefill,
    AscendPCPMetadata,
    _npu_attention_update,
    _process_attn_out_lse,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    filter_chunked_req_indices,
    split_decodes_and_prefills,
)
from vllm_ascend.compilation.acl_graph import get_graph_params, update_graph_params_workspaces
from vllm_ascend.utils import cp_chunkedprefill_comm_stream, weak_ref_tensors


class AscendAttentionCPMetadataBuilder(AscendAttentionMetadataBuilder):
    """
    Builder for constructing AscendMetadata with Context Parallelism support.

    Extends AscendAttentionMetadataBuilder with PCP/DCP metadata handling.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionCPMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.ALWAYS

    def _get_chunked_req_mask(self, local_context_lens_allranks) -> list[bool]:
        """
        given 4-d list [req][pcp][dcp], return:
        1. if each req has any chunk (list[bool])
        """
        assert local_context_lens_allranks is not None
        if len(local_context_lens_allranks) == 0:
            return []
        chunked_req_mask = [(req.sum() > 0).item() for req in local_context_lens_allranks if req is not None]
        return chunked_req_mask

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ):
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_actual_tokens

        block_table = common_attn_metadata.block_table_tensor
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        self.num_decodes_flatten = query_lens[:num_decodes].sum().item()
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        num_actual_tokens_pcp_padded = long_seq_metadata.num_actual_tokens_pcp_padded if long_seq_metadata else None
        if num_actual_tokens_pcp_padded is None:
            num_actual_tokens_pcp_padded = num_actual_tokens

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens_pcp_padded]
        attn_mask = self.attn_mask_builder.get_attention_mask(self.model_config)
        attn_state = common_attn_metadata.attn_state
        num_computed_tokens_cpu = seq_lens - query_lens

        query_start_loc = query_start_loc_cpu.to(self.device, non_blocking=True)

        common_long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        prefill_metadata = None
        decode_metadata = None
        if common_long_seq_metadata is None:
            raise AssertionError("common_long_seq_metadata should not be None.")
        num_computed_tokens_of_pcp_dcp = common_long_seq_metadata.num_computed_tokens_of_pcp_dcp
        assert num_computed_tokens_of_pcp_dcp is not None
        chunked_context_metadata = None
        if num_prefills > 0:
            query_lens = query_lens[num_decode_tokens:]
            context_lens_cpu = num_computed_tokens_cpu[num_decodes:num_reqs]
            max_context_len_cpu = context_lens_cpu.max().item()
            pcp_size = get_pcp_group().world_size
            if self.chunked_prefill_enabled and max_context_len_cpu > 0:
                local_context_lens_allranks = (
                    torch.tensor(num_computed_tokens_of_pcp_dcp)[self.num_decodes_flatten :]
                    .to(self.device)
                    .to(dtype=torch.int32)
                )
                local_chunked_kv_lens_rank = local_context_lens_allranks[:, self.pcp_rank, self.dcp_rank]
                actual_seq_lengths_kv = torch.cumsum(local_chunked_kv_lens_rank, dim=0).tolist()
                local_total_toks = local_chunked_kv_lens_rank.sum()
                chunked_req_mask = self._get_chunked_req_mask(local_context_lens_allranks)
                local_chunk_starts = torch.zeros(
                    (len(local_context_lens_allranks),), dtype=torch.int32, device=self.device
                )
                # Note(qcs): we only do restore and recover for pcp, and set these vars to None
                # when only using dcp.
                if self.pcp_size > 1:
                    kv_inverse_idx_for_chunk = torch.argsort(
                        common_long_seq_metadata.pcp_allgather_restore_idx[pcp_size * num_decode_tokens :].to(
                            torch.float32
                        )
                    )
                    cp_kv_recover_idx_for_chunk = torch.argsort(kv_inverse_idx_for_chunk)
                else:
                    kv_inverse_idx_for_chunk = None
                    cp_kv_recover_idx_for_chunk = None

                batch_chunk_seq_mask = local_context_lens_allranks[:, self.pcp_rank, self.dcp_rank] == 0
                batch_chunk_seq_mask = torch.repeat_interleave(
                    batch_chunk_seq_mask, repeats=(query_lens * self.pcp_size).to(self.device)
                )
                chunk_seq_mask_filtered_indices = filter_chunked_req_indices(query_lens, chunked_req_mask).to(
                    self.device
                )
                chunked_context_metadata = AscendMetadataForPrefill.ChunkedContextMetadata(
                    actual_chunk_seq_lengths=torch.cumsum(query_lens * pcp_size, dim=0),
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    chunked_req_mask=chunked_req_mask,
                    starts=local_chunk_starts,
                    local_context_lens_allranks=local_context_lens_allranks,
                    cp_kv_recover_idx_for_chunk=cp_kv_recover_idx_for_chunk,
                    kv_inverse_idx_for_chunk=kv_inverse_idx_for_chunk,
                    batch_chunk_seq_mask=batch_chunk_seq_mask,
                    chunk_seq_mask_filtered_indices=chunk_seq_mask_filtered_indices,
                    local_total_toks=local_total_toks.item(),
                )
            attn_mask_seqlens = common_long_seq_metadata.attn_mask_seqlens
            head_attn_nomask_seqlens = common_long_seq_metadata.head_attn_nomask_seqlens
            tail_attn_nomask_seqlens = common_long_seq_metadata.tail_attn_nomask_seqlens
            if pcp_size > 1:
                attn_mask_seqlens = torch.cumsum(attn_mask_seqlens[0], dim=0).tolist()
                head_attn_nomask_seqlens = torch.cumsum(head_attn_nomask_seqlens[1], dim=0).tolist()
                tail_attn_nomask_seqlens = torch.cumsum(tail_attn_nomask_seqlens[1], dim=0).tolist()

            pcp_metadata = AscendPCPMetadata(
                q_head_idx=common_long_seq_metadata.q_head_idx_tensor,
                q_tail_idx=common_long_seq_metadata.q_tail_idx_tensor,
                kv_with_q_head_nomask_idx=common_long_seq_metadata.kv_with_q_head_nomask_idx_tensor,
                kv_with_q_head_mask_idx=common_long_seq_metadata.kv_with_q_head_mask_idx_tensor,
                kv_with_q_tail_nomask_idx=common_long_seq_metadata.kv_with_q_tail_nomask_idx_tensor,
                kv_with_q_tail_mask_idx=common_long_seq_metadata.kv_with_q_tail_mask_idx_tensor,
                attn_mask_seqlens=attn_mask_seqlens,
                head_attn_nomask_seqlens=head_attn_nomask_seqlens,
                tail_attn_nomask_seqlens=tail_attn_nomask_seqlens,
                q_full_idx=common_long_seq_metadata.q_full_idx,
                pcp_allgather_restore_idx=common_long_seq_metadata.pcp_allgather_restore_idx,
            )

            prefill_metadata = AscendMetadataForPrefill(
                pcp_metadata=pcp_metadata,
                chunked_context=chunked_context_metadata,
                block_tables=block_table[self.num_decodes_flatten :, ...],
                actual_seq_lengths_q=torch.cumsum(query_lens, dim=0),
            )

        if num_decodes > 0:
            num_computed_tokens_array = np.array(num_computed_tokens_of_pcp_dcp)
            num_computed_tokens_array = num_computed_tokens_array[: self.num_decodes_flatten]
            # TODO: numpy array mode of the shared memory is used to improve performance
            decode_metadata = AscendMetadataForDecode(
                num_computed_tokens_of_pcp_dcp=num_computed_tokens_array,
                block_tables=block_table[: self.num_decodes_flatten],
            )

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
            num_decodes_flatten=self.num_decodes_flatten,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            prefill=prefill_metadata,
            decode_meta=decode_metadata,
        )
        return attn_metadata


class AscendAttentionCPImpl(AscendAttentionBackendImpl):
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
    ) -> None:
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
        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.pcp_group = get_pcp_group().device_group if self.pcp_size > 1 else None

        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0
        self.dcp_group = get_dcp_group().device_group if self.dcp_size > 1 else None

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config=None,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        graph_params = get_graph_params()
        # FIXME: Behold! We are using a temporary hack here to update the args
        # for each layer's attention op in the graph.
        with torch.npu.stream(update_stream):
            for key, param, handle, event in zip(
                forward_context.attn_metadata,
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
                    pcp_rank,
                    dcp_rank,
                ) = param
                attn_metadata = forward_context.attn_metadata[key]
                actual_seq_lengths_kv = attn_metadata.decode_meta.num_computed_tokens_of_pcp_dcp[:, pcp_rank, dcp_rank]
                pad_length = num_tokens - len(actual_seq_lengths_kv)
                if pad_length > 0:
                    pad_tensor = np.zeros(pad_length, dtype=actual_seq_lengths_kv.dtype)
                    actual_seq_lengths_kv = np.concatenate([actual_seq_lengths_kv, pad_tensor])

                actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q

                if dcp_size > 1:
                    num_heads = num_heads * dcp_size

                torch.npu.graph_task_update_begin(update_stream, handle)

                torch_npu.npu_fused_infer_attention_score.out(
                    q_nope,
                    k_nope,
                    value,
                    num_heads=num_heads,
                    num_key_value_heads=num_kv_heads,
                    input_layout="TND",
                    atten_mask=None,
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

    def _attention_with_nomask_and_mask(
        self,
        q: torch.Tensor,
        q_seqlens: list[int],
        k_nomask: torch.Tensor,
        v_nomask: torch.Tensor,
        kv_seqlens_nomask: list[int],
        k_mask: torch.Tensor,
        v_mask: torch.Tensor,
        kv_seqlens_mask: list[int],
        mask: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        # nomask Attention
        if k_nomask is not None:
            attn_out_nomask, attn_lse_nomask = torch.ops.npu.npu_fused_infer_attention_score(
                q,
                k_nomask,
                v_nomask,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                atten_mask=None,
                scale=self.scale,
                sparse_mode=0,
                antiquant_mode=0,
                antiquant_scale=None,
                softmax_lse_flag=True,
                actual_seq_lengths_kv=kv_seqlens_nomask,
                actual_seq_lengths=q_seqlens,
            )

        # mask Attention
        attn_out_mask, attn_lse_mask = torch.ops.npu.npu_fused_infer_attention_score(
            q,
            k_mask,
            v_mask,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            atten_mask=mask,
            scale=self.scale,
            sparse_mode=3,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=True,
            actual_seq_lengths_kv=kv_seqlens_mask,
            actual_seq_lengths=q_seqlens,
        )
        # update
        output = attn_out_mask
        attn_lse = attn_lse_mask
        if k_nomask is not None:
            if attn_metadata.prefill is not None and attn_metadata.prefill.chunked_context is None:
                output = self._npu_attn_out_lse_update(attn_lse_mask, attn_lse_nomask, attn_out_mask, attn_out_nomask)
                attn_lse = None
            else:
                output, attn_lse = self._update_out_and_lse(
                    torch.stack([attn_out_nomask, attn_out_mask], dim=0),
                    torch.stack([attn_lse_nomask, attn_lse_mask], dim=0),
                )

        return output, attn_lse

    def _npu_attn_out_lse_update(self, attn_lse_mask, attn_lse_nomask, attn_out_mask, attn_out_nomask):
        T = attn_out_mask.shape[0]
        N = attn_out_mask.shape[1]
        D = attn_out_mask.shape[2]
        attn_out_mask, attn_lse_mask = self._out_lse_reshape(attn_out_mask, attn_lse_mask)
        attn_out_nomask, attn_lse_nomask = self._out_lse_reshape(attn_out_nomask, attn_lse_nomask)
        attn_out_mask = attn_out_mask.to(torch.float32)
        attn_out_nomask = attn_out_nomask.to(torch.float32)
        attn_lse_mask = attn_lse_mask.to(torch.float32)
        attn_lse_nomask = attn_lse_nomask.to(torch.float32)
        attn_output = [attn_out_nomask, attn_out_mask]
        attn_lse = [attn_lse_nomask, attn_lse_mask]
        update_type = 0
        output, _ = torch_npu.npu_attention_update(attn_lse, attn_output, update_type)
        output = output.view(T, N, D)
        return output

    def _forward_prefill_cp(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata
    ) -> torch.Tensor:
        data_head, data_tail = self._forward_prefill_cp_pre(query, key, value, attn_metadata)

        output_head, lse_head = self._forward_prefill_cp_attn(data_head, True, attn_metadata)
        output_tail, lse_tail = self._forward_prefill_cp_attn(data_tail, False, attn_metadata)

        output, attn_lse = self._forward_prefill_cp_post(
            [output_head, output_tail],
            [lse_head, lse_tail],
            attn_metadata,
        )
        return output, attn_lse

    def _forward_prefill_cp_pre(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata
    ) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.prefill is not None
        assert attn_metadata.prefill.pcp_metadata is not None
        # Use precomputed indices from the metadata (already converted to tensors and on device)
        q_head_idx = attn_metadata.prefill.pcp_metadata.q_head_idx
        q_tail_idx = attn_metadata.prefill.pcp_metadata.q_tail_idx
        kv_with_q_head_nomask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_head_nomask_idx
        kv_with_q_head_mask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_head_mask_idx
        kv_with_q_tail_nomask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_tail_nomask_idx
        kv_with_q_tail_mask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_tail_mask_idx
        q_head = torch.index_select(query, 0, q_head_idx)
        q_tail = torch.index_select(query, 0, q_tail_idx)
        k_head_nomask = torch.index_select(key, 0, kv_with_q_head_nomask_idx) if self.pcp_rank > 0 else None
        v_head_nomask = torch.index_select(value, 0, kv_with_q_head_nomask_idx) if self.pcp_rank > 0 else None
        k_head_mask = torch.index_select(key, 0, kv_with_q_head_mask_idx)
        v_head_mask = torch.index_select(value, 0, kv_with_q_head_mask_idx)
        k_tail_nomask = torch.index_select(key, 0, kv_with_q_tail_nomask_idx)
        v_tail_nomask = torch.index_select(value, 0, kv_with_q_tail_nomask_idx)
        k_tail_mask = torch.index_select(key, 0, kv_with_q_tail_mask_idx)
        v_tail_mask = torch.index_select(value, 0, kv_with_q_tail_mask_idx)
        return (
            {
                "q": q_head,
                "k_nomask": k_head_nomask,
                "v_nomask": v_head_nomask,
                "k_mask": k_head_mask,
                "v_mask": v_head_mask,
            },
            {
                "q": q_tail,
                "k_nomask": k_tail_nomask,
                "v_nomask": v_tail_nomask,
                "k_mask": k_tail_mask,
                "v_mask": v_tail_mask,
            },
        )

    def _forward_prefill_cp_attn(self, data, is_head, attn_metadata):
        attn_mask_seqlens = attn_metadata.prefill.pcp_metadata.attn_mask_seqlens
        nomask_seqlens = (
            attn_metadata.prefill.pcp_metadata.head_attn_nomask_seqlens
            if is_head
            else attn_metadata.prefill.pcp_metadata.tail_attn_nomask_seqlens
        )
        output, lse = self._attention_with_nomask_and_mask(
            **data,
            q_seqlens=attn_mask_seqlens,
            kv_seqlens_nomask=nomask_seqlens,
            kv_seqlens_mask=attn_mask_seqlens,
            mask=attn_metadata.attn_mask,
            attn_metadata=attn_metadata,
        )
        return output, lse

    def _forward_prefill_cp_post(self, outputs, lses, attn_metadata):
        q_full_idx = attn_metadata.prefill.pcp_metadata.q_full_idx
        output = torch.index_select(torch.cat(outputs, dim=0), 0, q_full_idx)
        attn_lse = None
        if attn_metadata.prefill is not None and attn_metadata.prefill.chunked_context is not None:
            attn_lse = torch.index_select(torch.cat(lses, dim=0), 0, q_full_idx)
        return output, attn_lse

    def _out_lse_reshape(self, attn_out: torch.Tensor, attn_lse: torch.Tensor) -> torch.Tensor:
        attn_out = attn_out.contiguous().view(attn_out.shape[0] * attn_out.shape[1], attn_out.shape[2])
        attn_lse = attn_lse.contiguous().view(attn_lse.shape[0] * attn_lse.shape[1] * attn_lse.shape[2])
        return attn_out, attn_lse

    def _forward_decode_pcp_dcp(self, query: torch.Tensor, attn_metadata: AscendMetadata) -> torch.Tensor:
        assert self.key_cache is not None
        assert self.value_cache is not None

        if self.dcp_size > 1:
            query = get_dcp_group().all_gather(query, 1)
            num_heads = self.num_heads * self.dcp_size
        else:
            num_heads = self.num_heads

        k_nope = self.key_cache.view(self.key_cache.shape[0], self.key_cache.shape[1], -1)
        value = self.value_cache.view(self.key_cache.shape[0], self.key_cache.shape[1], -1)
        common_kwargs = {
            "num_heads": num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "input_layout": "TND",
            "atten_mask": None,
            "scale": self.scale,
            "antiquant_mode": 0,
            "antiquant_scale": None,
            "softmax_lse_flag": True,
            "block_table": attn_metadata.decode_meta.block_tables,
            "block_size": self.key_cache.shape[1],
            "actual_seq_lengths_kv": attn_metadata.decode_meta.num_computed_tokens_of_pcp_dcp[
                :, self.pcp_rank, self.dcp_rank
            ],
            "actual_seq_lengths": torch.arange(attn_metadata.num_decodes_flatten) + 1,
        }
        graph_params = get_graph_params()
        forward_context: ForwardContext = get_forward_context()
        num_tokens = query.shape[0]
        if forward_context.capturing:
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
                update_graph_params_workspaces(num_tokens, weak_ref_tensors(workspace))
            attn_out = torch.empty_like(query)
            attn_lse = torch.empty((num_tokens, num_heads, 1), dtype=torch.float, device=query.device)

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
                    attn_metadata.decode_meta.num_computed_tokens_of_pcp_dcp[:, self.pcp_rank, self.dcp_rank],
                    attn_metadata.actual_seq_lengths_q[: attn_metadata.num_decodes],
                    weak_ref_tensors(attn_out),
                    weak_ref_tensors(attn_lse),
                    self.dcp_size,
                    self.pcp_rank,
                    self.dcp_rank,
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
        attn_out_lse = _process_attn_out_lse(attn_out, attn_lse)
        attn_out = _npu_attention_update(self.head_size, attn_out_lse)
        return attn_out

    def _update_out_and_lse(self, out_list: torch.Tensor, lse_list: torch.Tensor) -> torch.Tensor:
        """LSE_final = log(sum(exp(LSE_i))), O_final = sum(exp(LSE_i - LSE_final) * O_i)
        Args:
            out_list: shape = [N, batch_size, num_heads, head_size]
            lse_list: shape = [N, batch_size, num_heads, 1]
        Returns:
            out_final: shape = [batch_size, num_heads, head_size]
            lse_final: shape = [batch_size, num_heads, 1]
        """
        lse_final = torch.logsumexp(lse_list, dim=0, keepdim=False)
        out_final = torch.sum(torch.exp(lse_list - lse_final) * out_list, dim=0)
        return out_final, lse_final

    def _update_chunk_attn_out_lse_with_current_attn_out_lse(
        self,
        current_attn_output_prefill,
        current_attn_lse_prefill,
        attn_output_full_chunk,
        attn_lse_full_chunk,
        prefill_query,
        attn_metadata,
    ):
        if self.pcp_size > 1:
            inverse_idx = attn_metadata.prefill.chunked_context.kv_inverse_idx_for_chunk
            attn_output_full_chunk = torch.index_select(attn_output_full_chunk, 0, inverse_idx)
            attn_lse_full_chunk = torch.index_select(attn_lse_full_chunk, 0, inverse_idx)
        num_tokens = prefill_query.size(0)
        attn_output_full_chunk = attn_output_full_chunk[
            self.pcp_rank * num_tokens : (self.pcp_rank + 1) * num_tokens, :, :
        ]
        attn_lse_full_chunk = attn_lse_full_chunk[self.pcp_rank * num_tokens : (self.pcp_rank + 1) * num_tokens, :, :]

        assert (
            attn_output_full_chunk.shape == current_attn_output_prefill.shape
            and attn_lse_full_chunk.shape == current_attn_lse_prefill.shape
        )
        filtered_indices = attn_metadata.prefill.chunked_context.chunk_seq_mask_filtered_indices

        attn_output_prefill_filtered = current_attn_output_prefill[filtered_indices, :, :]
        attn_lse_prefill_filtered = current_attn_lse_prefill[filtered_indices, :, :]
        attn_output_full_chunk = attn_output_full_chunk[filtered_indices, :, :]
        attn_lse_full_chunk = attn_lse_full_chunk[filtered_indices, :, :]

        attn_output_filtered = self._npu_attn_out_lse_update(
            attn_lse_prefill_filtered, attn_lse_full_chunk, attn_output_prefill_filtered, attn_output_full_chunk
        )

        current_attn_output_prefill[filtered_indices, :, :] = attn_output_filtered.to(current_attn_output_prefill.dtype)

    def _prefill_query_all_gather(self, attn_metadata, prefill_query):
        if self.pcp_size > 1:
            prefill_query = get_pcp_group().all_gather(prefill_query, 0)
            prefill_query = torch.index_select(
                prefill_query, 0, attn_metadata.prefill.chunked_context.cp_kv_recover_idx_for_chunk
            )
        if self.dcp_size > 1:
            prefill_query = get_dcp_group().all_gather(prefill_query, 1)
        return prefill_query

    def _compute_prefill_context(
        self, query: torch.Tensor, kv_cache: tuple[torch.Tensor], attn_metadata: AscendMetadata
    ):
        assert len(kv_cache) > 1
        assert attn_metadata is not None
        assert attn_metadata.prefill is not None
        assert attn_metadata.prefill.chunked_context is not None
        prefill_metadata = attn_metadata.prefill
        local_chunked_kv_lens = prefill_metadata.chunked_context.local_context_lens_allranks
        assert local_chunked_kv_lens is not None

        local_chunked_kv_lens_rank = local_chunked_kv_lens[:, self.pcp_rank, self.dcp_rank]
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
            key,
            value,
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
            torch_npu.atb.npu_paged_cache_load(
                cache_key,
                cache_value,
                attn_metadata.prefill.block_tables,
                local_chunked_kv_lens_rank,
                # slot offsets of current chunk in current iteration
                seq_starts=attn_metadata.prefill.chunked_context.starts,
                key=key,
                value=value,
            )
        return key, value

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
    ):
        num_decode_tokens = attn_metadata.num_decode_tokens
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0

        if len(kv_cache) > 1:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            if has_decode:
                slot_mapping = attn_metadata.slot_mapping[: num_decode_tokens * self.pcp_size : self.pcp_size]
                torch_npu._npu_reshape_and_cache(
                    key=key[:num_decode_tokens],
                    value=value[:num_decode_tokens],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=slot_mapping,
                )

            if has_prefill:
                if self.pcp_size > 1:
                    kv = torch.cat([key, value], dim=-1)
                    num_actual_tokens_pcp_padded = attn_metadata.num_actual_tokens_pcp_padded // self.pcp_size
                    all_kv = get_pcp_group().all_gather(kv[:num_actual_tokens_pcp_padded].contiguous(), dim=0)
                    assert attn_metadata.prefill is not None
                    assert attn_metadata.prefill.pcp_metadata is not None
                    pcp_allgather_restore_idx = attn_metadata.prefill.pcp_metadata.pcp_allgather_restore_idx
                    all_kv = torch.index_select(all_kv, 0, pcp_allgather_restore_idx)
                    key, value = all_kv.split([self.head_size, self.head_size], dim=-1)
                prefill_key = key[self.pcp_size * num_decode_tokens : attn_metadata.num_actual_tokens_pcp_padded]
                prefill_value = value[self.pcp_size * num_decode_tokens : attn_metadata.num_actual_tokens_pcp_padded]
                slot_mapping = attn_metadata.slot_mapping[
                    self.pcp_size * num_decode_tokens : attn_metadata.num_actual_tokens_pcp_padded
                ]
                torch_npu._npu_reshape_and_cache(
                    key=prefill_key,
                    value=prefill_value,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=slot_mapping,
                )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
        return key, value

    def _gather_global_context_output(self, local_context_attn_output):
        if self.dcp_size > 1:
            dcp_context_attn_output = torch.empty_like(local_context_attn_output)
            dist.all_to_all_single(dcp_context_attn_output, local_context_attn_output, group=self.dcp_group)
        else:
            dcp_context_attn_output = local_context_attn_output

        if self.pcp_size > 1:
            # AllGather out&lse within CP group
            global_context_attn_output = get_pcp_group().all_gather(dcp_context_attn_output, dim=-1)
        else:
            global_context_attn_output = dcp_context_attn_output

        return global_context_attn_output

    def _update_global_context_output(self, global_context_output):
        B_total, H_total, D_plus_1 = global_context_output.shape
        S = B_total // self.pcp_size
        H = H_total // self.dcp_size
        D = self.head_size
        assert D_plus_1 == D + 1
        # [PCP, S, DCP, H, D+1]
        x = global_context_output.view(self.pcp_size, S, self.dcp_size, H, D_plus_1)
        # [PCP, DCP, S, H, D+1]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # Flatten [N, S, H, D+1], N = pcp_size * dcp_size
        x = x.view(-1, S, H, D_plus_1)
        # Split out lse
        attn_out_allgather, attn_lse_allgather = torch.split(x, [D, 1], dim=-1)  # [N, S, H, D], [N, S, H, 1]
        context_output, context_lse = self._update_out_and_lse(attn_out_allgather, attn_lse_allgather)
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
        assert attn_metadata is not None
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens
        if has_decode:
            decode_query = query[:num_decode_tokens]
            output_decode = self._forward_decode_pcp_dcp(decode_query, attn_metadata)
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
            num_actual_tokens_pcp_padded = attn_metadata.num_actual_tokens_pcp_padded // self.pcp_size
            prefill_query = query[num_decode_tokens:num_actual_tokens_pcp_padded].contiguous()
            key = key[self.pcp_size * num_decode_tokens :].contiguous()
            value = value[self.pcp_size * num_decode_tokens :].contiguous()

            if has_chunked_context:
                # all_gather q for chunked prefill // overlap the computation inner current chunk
                cp_chunkedprefill_comm_stream().wait_stream(torch.npu.current_stream())
                with torch_npu.npu.stream(cp_chunkedprefill_comm_stream()):
                    prefill_query_all = self._prefill_query_all_gather(attn_metadata, prefill_query.clone())

            if self.pcp_size > 1:
                # Scenario of Enabling PCP or PCP&DCP
                # prepare qkv and compute the head part // overlap the communication of all gather q
                data_head, data_tail = self._forward_prefill_cp_pre(prefill_query, key, value, attn_metadata)
                output_head, lse_head = self._forward_prefill_cp_attn(data_head, True, attn_metadata)
            else:
                # Scenario of Enabling DCP Individually
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
                context_output = self._compute_prefill_context(prefill_query_all, kv_cache, attn_metadata)
                # Note(qcs): (output, lse) -> [Seq, Head_num, Head_dim+1] -> [Head_num, Head_dim+1, Seq]
                local_context_output = torch.cat(context_output, dim=-1).permute([1, 2, 0]).contiguous()

                # all2all and all_gather output&lse // overlap the computation inner current chunk
                cp_chunkedprefill_comm_stream().wait_stream(torch.npu.current_stream())
                with torch_npu.npu.stream(cp_chunkedprefill_comm_stream()):
                    global_context_output = self._gather_global_context_output(local_context_output)

            if self.pcp_size > 1:
                # compute the tail part and reorg output&lse // overlap the communication of output
                output_tail, lse_tail = self._forward_prefill_cp_attn(data_tail, False, attn_metadata)

                attn_output_prefill, attn_lse_prefill = self._forward_prefill_cp_post(
                    [output_head, output_tail],
                    [lse_head, lse_tail],
                    attn_metadata,
                )

            if attn_metadata.prefill is not None and attn_metadata.prefill.chunked_context is not None:
                # update the output of current chunk with context part
                torch.npu.current_stream().wait_stream(cp_chunkedprefill_comm_stream())
                global_context_output = global_context_output.permute([2, 0, 1]).contiguous()
                context_output, context_lse = self._update_global_context_output(global_context_output)
                self._update_chunk_attn_out_lse_with_current_attn_out_lse(
                    attn_output_prefill, attn_lse_prefill, context_output, context_lse, prefill_query, attn_metadata
                )

            output[num_decode_tokens : attn_output_prefill.shape[0] + num_decode_tokens] = attn_output_prefill
        return output
