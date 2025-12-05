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
from enum import Enum
from typing import ClassVar, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.config import VllmConfig
from vllm.distributed import (get_dcp_group,
                              get_decode_context_model_parallel_rank,
                              get_decode_context_model_parallel_world_size)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         filter_chunked_req_indices,
                                         split_decodes_and_prefills)
from vllm_ascend.compilation.acl_graph import (get_graph_params,
                                               update_graph_params_workspaces)
from vllm_ascend.ops.attention import vanilla_chunked_prefill
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, AscendDeviceType,
                               aligned_16, get_ascend_device_type, nd_to_nz_2d,
                               nd_to_nz_spec, prefill_context_parallel_enable,
                               weak_ref_tensors)

# isort: off
if prefill_context_parallel_enable():
    from vllm.distributed import (get_pcp_group,
                                  get_prefill_context_model_parallel_rank,
                                  get_prefill_context_model_parallel_world_size
                                  )

# isort: on

from vllm.attention.backends.registry import (AttentionBackendEnum,
                                              register_backend)


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if get_ascend_device_type() == AscendDeviceType._310P:
            return (2, num_blocks, num_kv_heads * head_size // 16, block_size,
                    16)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_bsh_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_block_size() -> list[int]:
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendPCPMetadata:
    q_head_idx: torch.Tensor = None
    q_tail_idx: torch.Tensor = None
    kv_with_q_head_nomask_idx: torch.Tensor = None
    kv_with_q_head_mask_idx: torch.Tensor = None
    kv_with_q_tail_nomask_idx: torch.Tensor = None
    kv_with_q_tail_mask_idx: torch.Tensor = None
    attn_mask_seqlens: torch.Tensor = None
    head_attn_nomask_seqlens: torch.Tensor = None
    tail_attn_nomask_seqlens: torch.Tensor = None
    q_full_idx: torch.Tensor = None
    pcp_prefill_mask: torch.Tensor = None


@dataclass
class AscendMetadataForPrefill:

    @dataclass
    class ChunkedContextMetadata:
        actual_chunk_seq_lengths: list[int]
        actual_seq_lengths_kv: list[int]
        starts: torch.Tensor
        chunk_seq_mask_filtered_indices: torch.Tensor
        chunked_req_mask: Optional[list[bool]] = None
        local_context_lens_allranks: Optional[list[list[int]]] = None
        cp_kv_recover_idx_for_chunk: Optional[list[int]] = None
        kv_inverse_idx_for_chunk: Optional[list[int]] = None
        batch_chunk_seq_mask: Optional[list[bool]] = None

    """ Prefill Specific Metadata for Ascend"""
    pcp_metadata: Optional[AscendPCPMetadata] = None
    pcp_allgather_restore_idx: Optional[List[int]] = None
    chunked_context: Optional[ChunkedContextMetadata] = None
    block_tables: torch.Tensor = None
    actual_seq_lengths_q: torch.Tensor = None


@dataclass
class AscendMetadataForDecode:
    """ Decode Specific Metadata for Ascend"""
    num_computed_tokens_of_pcp_dcp: Optional[list[list[list[int]]]] = None
    batch_seq_mask: torch.Tensor = None
    block_tables: torch.Tensor = None


@dataclass
class AscendMetadata:
    # **************************** Basic Properties ************************** #
    attn_mask: Optional[torch.Tensor] = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_list: List[int] = None  # type: ignore
    actual_seq_lengths_q: List[int] = None  # type: ignore
    query_start_loc_list: List[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    query_lens: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: Optional[int] = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None

    prefill: Optional[AscendMetadataForPrefill] = None

    decode_meta: Optional[AscendMetadataForDecode] = None


class AscendAttentionMetadataBuilder:
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.ALWAYS
    # AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
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
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len,
            AscendAttentionBackend.get_supported_block_size()[0])
        self.batch_seq_mask_buf = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.uint8,
            device=device)
        self.pcp_size = get_prefill_context_model_parallel_world_size(
        ) if prefill_context_parallel_enable() else 1
        self.pcp_rank = get_prefill_context_model_parallel_rank(
        ) if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank(
        ) if self.dcp_size > 1 else 0

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"

        AscendAttentionMetadataBuilder.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill

    def reorder_batch(self, input_batch,
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: Optional[nn.Module] = None,
    ):
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_actual_tokens

        block_table = common_attn_metadata.block_table_tensor
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        num_actual_tokens_pcp_padded = long_seq_metadata.num_actual_tokens_pcp_padded if long_seq_metadata else None
        if num_actual_tokens_pcp_padded is None:
            num_actual_tokens_pcp_padded = num_actual_tokens

        slot_mapping = common_attn_metadata.slot_mapping[:
                                                         num_actual_tokens_pcp_padded]
        # slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        attn_mask = common_attn_metadata.attn_mask
        attn_state = common_attn_metadata.attn_state
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]
        num_computed_tokens_cpu = (seq_lens - query_lens)

        if common_attn_metadata.num_input_tokens > num_actual_tokens:
            padded_num_tokens = common_attn_metadata.num_input_tokens - num_actual_tokens
            seq_lens = torch.cat([
                seq_lens,
                torch.tensor([padded_num_tokens
                              ]).to(seq_lens.device).to(seq_lens.dtype)
            ])
            block_table_padding = torch.zeros(
                (padded_num_tokens, ) + block_table.shape[1:],
                dtype=block_table.dtype,
                device=block_table.device)
            block_table = torch.cat([block_table, block_table_padding], dim=0)
            query_start_loc_cpu = torch.cat([
                query_start_loc_cpu,
                torch.tensor([query_start_loc_cpu[-1] + padded_num_tokens]).to(
                    query_start_loc_cpu.device).to(query_start_loc_cpu.dtype)
            ])

        query_start_loc = query_start_loc_cpu.to(self.device,
                                                 non_blocking=True)

        if get_ascend_device_type() == AscendDeviceType._310P:
            if attn_state == AscendAttentionState.PrefillNoCache:
                mask_nz = nd_to_nz_2d(attn_mask)
                attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(),
                                                      ACL_FORMAT_FRACTAL_NZ)
            elif attn_state == AscendAttentionState.ChunkedPrefill:
                mask_nz = nd_to_nz_spec(attn_mask)
                attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(),
                                                      ACL_FORMAT_FRACTAL_NZ)

        common_long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        prefill_metadata = None
        decode_metadata = None
        if common_long_seq_metadata is not None:
            num_computed_tokens_of_pcp_dcp = common_long_seq_metadata.num_computed_tokens_of_pcp_dcp
            assert num_computed_tokens_of_pcp_dcp is not None
            chunked_context_metadata = None
            if num_prefills > 0:
                query_lens = query_lens[num_decode_tokens:]
                context_lens_cpu = num_computed_tokens_cpu[
                    num_decodes:num_reqs]
                max_context_len_cpu = context_lens_cpu.max().item()
                pcp_size = get_prefill_context_model_parallel_world_size(
                ) if prefill_context_parallel_enable() else 1
                if self.chunked_prefill_enabled and max_context_len_cpu > 0:
                    local_context_lens_allranks = torch.tensor(
                        num_computed_tokens_of_pcp_dcp
                    )[num_decodes:num_reqs].to(
                        self.device).to(dtype=torch.int32)
                    local_chunked_kv_lens_rank = local_context_lens_allranks[:,
                                                                             self
                                                                             .
                                                                             pcp_rank,
                                                                             self
                                                                             .
                                                                             dcp_rank]
                    actual_seq_lengths_kv = torch.cumsum(
                        local_chunked_kv_lens_rank, dim=0).tolist()
                    chunked_req_mask = self._get_chunked_req_mask(
                        local_context_lens_allranks)
                    local_chunk_starts = torch.zeros(
                        (len(local_context_lens_allranks)),
                        dtype=torch.int32,
                        device=self.device)
                    cp_kv_recover_idx_for_chunk = common_long_seq_metadata.cp_kv_recover_idx_for_chunk
                    kv_inverse_idx_for_chunk = torch.argsort(
                        cp_kv_recover_idx_for_chunk.to(torch.float32)
                    ) if cp_kv_recover_idx_for_chunk is not None else None

                    batch_chunk_seq_mask = (
                        local_context_lens_allranks[:, self.pcp_rank,
                                                    self.dcp_rank] == 0)
                    batch_chunk_seq_mask = torch.repeat_interleave(
                        batch_chunk_seq_mask,
                        repeats=(query_lens * self.pcp_size).to(self.device))
                    chunk_seq_mask_filtered_indices = filter_chunked_req_indices(
                        query_lens, chunked_req_mask).to(self.device)
                    chunked_context_metadata = \
                        AscendMetadataForPrefill.ChunkedContextMetadata(
                            actual_chunk_seq_lengths=torch.cumsum(query_lens * pcp_size, dim=0),
                            actual_seq_lengths_kv=actual_seq_lengths_kv,
                            chunked_req_mask=chunked_req_mask,
                            starts=local_chunk_starts,
                            local_context_lens_allranks=local_context_lens_allranks,
                            cp_kv_recover_idx_for_chunk=cp_kv_recover_idx_for_chunk,
                            kv_inverse_idx_for_chunk=kv_inverse_idx_for_chunk,
                            batch_chunk_seq_mask=batch_chunk_seq_mask,
                            chunk_seq_mask_filtered_indices=chunk_seq_mask_filtered_indices
                        )
                attn_mask_seqlens = common_long_seq_metadata.attn_mask_seqlens
                head_attn_nomask_seqlens = common_long_seq_metadata.head_attn_nomask_seqlens
                tail_attn_nomask_seqlens = common_long_seq_metadata.tail_attn_nomask_seqlens
                if pcp_size > 1:
                    attn_mask_seqlens = torch.cumsum(attn_mask_seqlens[0],
                                                     dim=0).tolist()
                    head_attn_nomask_seqlens = torch.cumsum(
                        head_attn_nomask_seqlens[1], dim=0).tolist()
                    tail_attn_nomask_seqlens = torch.cumsum(
                        tail_attn_nomask_seqlens[1], dim=0).tolist()

                pcp_metadata = AscendPCPMetadata(
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
                    attn_mask_seqlens=attn_mask_seqlens,
                    head_attn_nomask_seqlens=head_attn_nomask_seqlens,
                    tail_attn_nomask_seqlens=tail_attn_nomask_seqlens,
                    q_full_idx=common_long_seq_metadata.q_full_idx,
                    pcp_prefill_mask=common_long_seq_metadata.pcp_prefill_mask)

                prefill_metadata = AscendMetadataForPrefill(
                    pcp_metadata=pcp_metadata,
                    pcp_allgather_restore_idx=common_long_seq_metadata.
                    pcp_allgather_restore_idx
                    if common_long_seq_metadata is not None else None,
                    chunked_context=chunked_context_metadata,
                    block_tables=block_table[num_decodes:],
                    actual_seq_lengths_q=torch.cumsum(query_lens, dim=0))

            if num_decodes > 0:
                num_computed_tokens_array = np.array(
                    num_computed_tokens_of_pcp_dcp)
                num_computed_tokens_array = num_computed_tokens_array[:
                                                                      num_decodes]
                batch_seq_mask = (
                    num_computed_tokens_array[:, self.pcp_rank,
                                              self.dcp_rank] == 0)
                # TODO: numpy array mode of the shared memory is used to improve performance
                self.batch_seq_mask_buf[:batch_seq_mask.shape[0]].copy_(
                    torch.from_numpy(batch_seq_mask), non_blocking=True)
                decode_metadata = AscendMetadataForDecode(
                    num_computed_tokens_of_pcp_dcp=num_computed_tokens_array,
                    batch_seq_mask=self.batch_seq_mask_buf[:batch_seq_mask.
                                                           shape[0]],
                    block_tables=block_table[:num_decodes])

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            num_actual_tokens_pcp_padded=num_actual_tokens_pcp_padded,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_start_loc_list=query_start_loc_cpu[1:].tolist(),
            query_lens=query_lens,
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
            decode_meta=decode_metadata)
        return attn_metadata

    def _get_chunked_req_mask(self, local_context_lens_allranks) -> List[bool]:
        """
        given 4-d list [req][pcp][dcp], return:
        1. if each req has any chunk (list[bool])
        """
        assert local_context_lens_allranks is not None
        if len(local_context_lens_allranks) == 0:
            return []
        chunked_req_mask = [(req.sum() > 0).item()
                            for req in local_context_lens_allranks
                            if req is not None]
        return chunked_req_mask

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        model: Optional[nn.Module] = None,
    ):
        if attn_state == AscendAttentionState.DecodeOnly:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.pcp_size = get_prefill_context_model_parallel_world_size(
        ) if prefill_context_parallel_enable() else 1
        self.pcp_rank = get_prefill_context_model_parallel_rank(
        ) if self.pcp_size > 1 else 0
        self.pcp_group = get_pcp_group(
        ).device_group if self.pcp_size > 1 else None

        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank(
        ) if self.dcp_size > 1 else 0
        self.dcp_group = get_dcp_group(
        ).device_group if self.dcp_size > 1 else None

    def full_graph_attention(self,
                             query: torch.Tensor,
                             key: torch.Tensor,
                             value: torch.Tensor,
                             kv_cache: Tuple[torch.Tensor],
                             attn_metadata: AscendMetadata,
                             output: torch.Tensor,
                             num_tokens=0):
        if self.pcp_size * self.dcp_size > 1:
            intermediate_output = self._forward_pcp_dcp(
                query, key, value, kv_cache, attn_metadata, output)
            return intermediate_output, query.shape[0]
        elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.query_start_loc_list
        elif attn_metadata.attn_state == \
            AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.query_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # Normal V1 situation.
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list

        num_tokens = attn_metadata.query_start_loc_list[-1]
        graph_params = get_graph_params()
        query_start_loc = attn_metadata.query_start_loc_list
        # Prepare tensors for attention output
        # TODO: Refactor this to step-level instead of layer-level

        # Get workspace from cache or calculate it if not present.
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=query_start_loc,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=3,
                scale=self.scale,
            )
            update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (weak_ref_tensors(query), weak_ref_tensors(key),
             weak_ref_tensors(value), weak_ref_tensors(block_table),
             weak_ref_tensors(attn_metadata.attn_mask), block_size,
             actual_seq_lengths_kv, query_start_loc, self.num_kv_heads,
             self.num_heads, self.scale, weak_ref_tensors(output),
             weak_ref_tensors(softmax_lse)))

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=query_start_loc,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
            workspace=workspace,
            out=[output, softmax_lse],
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def _forward_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        num_tokens=0,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        mask = attn_metadata.attn_mask

        if get_ascend_device_type() == AscendDeviceType._310P:
            # align q k v output tensors
            query = aligned_16(query)
            key = aligned_16(key)
            value = aligned_16(value)
            output = aligned_16(output)
            # do reformat in case of broadcasted tensors
            mask = mask.repeat(attn_metadata.seq_lens.size(0), 1, 1, 1)
            mask = torch_npu.npu_format_cast(mask.contiguous(),
                                             ACL_FORMAT_FRACTAL_NZ)

        torch_npu._npu_flash_attention(query=query,
                                       key=key,
                                       value=value,
                                       mask=mask,
                                       seq_len=attn_metadata.seq_lens,
                                       scale_value=self.scale,
                                       num_heads=self.num_heads,
                                       num_kv_heads=self.num_kv_heads,
                                       out=output)
        assert output is not None
        return output[:num_tokens]

    def _forward_prefill_cache_hit(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        compress_mask = attn_metadata.attn_mask
        batch_size = attn_metadata.query_lens.shape[0]
        block_table = attn_metadata.block_tables[:batch_size, :]
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore

        if block_size == 128:
            # TODO:The npu_fused_infer_attention_score op is planned to
            # be utilized in a wider range in upcoming versions.
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)

            output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=compress_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
        else:
            torch_npu._npu_flash_attention_qlens(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                block_table=block_table,
                mask=compress_mask,
                seq_len=attn_metadata.query_lens,
                context_lens=attn_metadata.seq_lens,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                out=output)
        return output

    def _forward_decode_only(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_ascend_device_type() == AscendDeviceType._310P:
            # seq_lens_tensor needs to be transferred to the device for 310P.
            attn_metadata.seq_lens = \
                attn_metadata.seq_lens.to(device=query.device)
        if self.sliding_window is not None and attn_metadata.seq_lens.shape[
                0] == query.size(0):
            batch_size = attn_metadata.seq_lens.shape[0]
            block_size = 128
            query = query.view(batch_size, 1, self.num_heads * self.head_size)
            key = self.key_cache
            value = self.value_cache
            if self.key_cache is not None and self.value_cache is not None:
                block_size = self.key_cache.shape[1]
                key = self.key_cache.flatten(2, 3).contiguous()
                value = self.value_cache.flatten(2, 3).contiguous()

            output, _ = torch_npu.npu_fused_infer_attention_score(
                query,
                key,
                value,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BSH",
                block_size=block_size,
                pre_tokens=self.sliding_window,
                scale=self.scale,
                block_table=attn_metadata.block_tables,
                actual_seq_lengths=[1] * len(attn_metadata.seq_lens),
                actual_seq_lengths_kv=attn_metadata.seq_lens)

            output = output.view(batch_size, self.num_heads, self.head_size)
        else:
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output)
        return output

    def _forward_v1_style(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use chunked prefill for head size 192 scenario, like deepseek
        # paged_attention_splitfuse maybe crash at such scenario.
        # TODO: vanilla path will be removed after the kernel support
        # head_size 192 scenario.
        if self.head_size == 192:
            cu_seqlen_q = [0] + attn_metadata.query_lens.tolist()
            cu_seqlen_k = [0] + attn_metadata.seq_lens.tolist()
            cu_seqlen_q = torch.tensor(cu_seqlen_q, device=query.device)
            cu_seqlen_k = torch.tensor(cu_seqlen_k, device=query.device)
            cu_seqlen_q = torch.cumsum(cu_seqlen_q, dim=0)
            cu_seqlen_k = torch.cumsum(cu_seqlen_k, dim=0)
            max_seqlen_q = torch.max(attn_metadata.query_lens)
            max_seqlen_k = torch.max(attn_metadata.seq_lens)
            vanilla_chunked_prefill(output, query, self.key_cache,
                                    self.value_cache,
                                    attn_metadata.block_tables, cu_seqlen_q,
                                    cu_seqlen_k, max_seqlen_q, max_seqlen_k,
                                    self.scale, None, True)
            return output

        # Use paged attention.
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        if get_ascend_device_type() == AscendDeviceType._310P:
            # Do reformat in case of broadcasted tensors.
            attn_metadata.attn_mask = \
                torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(),
                                          ACL_FORMAT_FRACTAL_NZ)
            attn_metadata.seq_lens = \
                attn_metadata.seq_lens.to(device=query.device)

        # TODO:The npu_fused_infer_attention_score op is planned to
        # be utilized in a wider range in upcoming versions.
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
        key = self.key_cache.view(  # type: ignore
            num_block, block_size, -1)
        value = self.value_cache.view(  # type: ignore
            num_block, block_size, -1)

        output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=attn_metadata.block_tables,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        return output

    def _attention_with_nomask_and_mask(self, q: torch.Tensor,
                                        q_seqlens: List[int],
                                        k_nomask: torch.Tensor,
                                        v_nomask: torch.Tensor,
                                        kv_seqlens_nomask: List[int],
                                        k_mask: torch.Tensor,
                                        v_mask: torch.Tensor,
                                        kv_seqlens_mask: List[int],
                                        mask: torch.Tensor,
                                        attn_metadata) -> torch.Tensor:
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
                actual_seq_lengths=q_seqlens)

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
            actual_seq_lengths=q_seqlens)
        # update
        output = attn_out_mask
        attn_lse = attn_lse_mask
        if k_nomask is not None:
            if attn_metadata.prefill is not None and attn_metadata.prefill.chunked_context is None:
                output = self._npu_attn_out_lse_update(attn_lse_mask,
                                                       attn_lse_nomask,
                                                       attn_out_mask,
                                                       attn_out_nomask)
                attn_lse = None
            else:
                output, attn_lse = self._update_out_and_lse(
                    torch.stack([attn_out_nomask, attn_out_mask], dim=0),
                    torch.stack([attn_lse_nomask, attn_lse_mask], dim=0))

        return output, attn_lse

    def _npu_attn_out_lse_update(self, attn_lse_mask, attn_lse_nomask,
                                 attn_out_mask, attn_out_nomask):
        T = attn_out_mask.shape[0]
        N = attn_out_mask.shape[1]
        D = attn_out_mask.shape[2]
        attn_out_mask, attn_lse_mask = self._out_lse_reshape(
            attn_out_mask, attn_lse_mask)
        attn_out_nomask, attn_lse_nomask = self._out_lse_reshape(
            attn_out_nomask, attn_lse_nomask)
        attn_out_mask = attn_out_mask.to(torch.float32)
        attn_out_nomask = attn_out_nomask.to(torch.float32)
        attn_lse_mask = attn_lse_mask.to(torch.float32)
        attn_lse_nomask = attn_lse_nomask.to(torch.float32)
        attn_output = [attn_out_nomask, attn_out_mask]
        attn_lse = [attn_lse_nomask, attn_lse_mask]
        update_type = 0
        output, _ = torch_npu.npu_attention_update(attn_lse, attn_output,
                                                   update_type)
        output = output.view(T, N, D)
        return output

    def _forward_prefill_cp(self, query: torch.Tensor, key: torch.Tensor,
                            value: torch.Tensor,
                            attn_metadata: AscendMetadata) -> torch.Tensor:
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
        attn_mask_seqlens = attn_metadata.prefill.pcp_metadata.attn_mask_seqlens
        head_attn_nomask_seqlens = attn_metadata.prefill.pcp_metadata.head_attn_nomask_seqlens
        tail_attn_nomask_seqlens = attn_metadata.prefill.pcp_metadata.tail_attn_nomask_seqlens
        mask = attn_metadata.prefill.pcp_metadata.pcp_prefill_mask

        # 1. Attention calculation in the first half of Q in load balancing
        output_heads, lse_heads = self._attention_with_nomask_and_mask(
            q=torch.index_select(query, 0, q_head_idx),
            q_seqlens=attn_mask_seqlens,
            k_nomask=torch.index_select(key, 0, kv_with_q_head_nomask_idx)
            if self.pcp_rank > 0 else None,
            v_nomask=torch.index_select(value, 0, kv_with_q_head_nomask_idx)
            if self.pcp_rank > 0 else None,
            kv_seqlens_nomask=head_attn_nomask_seqlens,
            k_mask=torch.index_select(key, 0, kv_with_q_head_mask_idx),
            v_mask=torch.index_select(value, 0, kv_with_q_head_mask_idx),
            kv_seqlens_mask=attn_mask_seqlens,
            mask=mask,
            attn_metadata=attn_metadata)

        # 2. the Attention calculation in the latter half of Q in load balancing
        # pcp_rank0: Q3*KV0~KV2 + Q3*KV3
        # pcp_rank1: Q2*KV0~KV1 + Q2*KV2
        output_tails, lse_tails = self._attention_with_nomask_and_mask(
            q=torch.index_select(query, 0, q_tail_idx),
            q_seqlens=attn_mask_seqlens,
            k_nomask=torch.index_select(key, 0, kv_with_q_tail_nomask_idx),
            v_nomask=torch.index_select(value, 0, kv_with_q_tail_nomask_idx),
            kv_seqlens_nomask=tail_attn_nomask_seqlens,
            k_mask=torch.index_select(key, 0, kv_with_q_tail_mask_idx),
            v_mask=torch.index_select(value, 0, kv_with_q_tail_mask_idx),
            kv_seqlens_mask=attn_mask_seqlens,
            mask=mask,
            attn_metadata=attn_metadata)

        q_full_idx = attn_metadata.prefill.pcp_metadata.q_full_idx
        output = torch.index_select(
            torch.cat([output_heads, output_tails], dim=0), 0, q_full_idx)
        attn_lse = None
        if attn_metadata.prefill is not None and attn_metadata.prefill.chunked_context is not None:
            attn_lse = torch.index_select(
                torch.cat([lse_heads, lse_tails], dim=0), 0, q_full_idx)
        return output, attn_lse

    def _out_lse_reshape(self, attn_out: torch.Tensor,
                         attn_lse: torch.Tensor) -> torch.Tensor:
        attn_out = attn_out.contiguous().view(
            attn_out.shape[0] * attn_out.shape[1], attn_out.shape[2])
        attn_lse = attn_lse.contiguous().view(
            attn_lse.shape[0] * attn_lse.shape[1] * attn_lse.shape[2])
        return attn_out, attn_lse

    def _npu_attention_update(
            self, attn_out_lse_list: List[torch.Tensor]) -> torch.Tensor:
        update_type = 0

        batch = attn_out_lse_list[0].shape[0]
        num_heads = attn_out_lse_list[0].shape[1]
        head_dim = attn_out_lse_list[0].shape[2] - 1

        attn_out_split_cp = []
        attn_lse_split_cp = []

        for i in attn_out_lse_list:
            attn_out_allgather, attn_lse_allgather = self._out_lse_reshape(
                *torch.split(i, [self.head_size, 1], dim=-1))
            attn_out_split_cp.append(attn_out_allgather)
            attn_lse_split_cp.append(attn_lse_allgather)

        attn_out, attn_lse = torch_npu.npu_attention_update(
            attn_lse_split_cp, attn_out_split_cp, update_type)
        attn_out = attn_out.view(batch, num_heads, head_dim)

        return attn_out

    def _forward_decode_pcp_dcp(self, query: torch.Tensor,
                                attn_metadata: AscendMetadata) -> torch.Tensor:
        assert self.key_cache is not None
        assert self.value_cache is not None

        if self.dcp_size > 1:
            query = get_dcp_group().all_gather(query, 1)
            num_heads = self.num_heads * self.dcp_size
        else:
            num_heads = self.num_heads

        k_nope = self.key_cache.view(self.key_cache.shape[0],
                                     self.key_cache.shape[1], -1)
        value = self.value_cache.view(self.key_cache.shape[0],
                                      self.key_cache.shape[1], -1)
        common_kwargs = {
            'num_heads':
            num_heads,
            'num_key_value_heads':
            self.num_kv_heads,
            'input_layout':
            'TND',
            'atten_mask':
            None,
            'scale':
            self.scale,
            'antiquant_mode':
            0,
            'antiquant_scale':
            None,
            'softmax_lse_flag':
            True,
            'block_table':
            attn_metadata.decode_meta.block_tables,
            'block_size':
            self.key_cache.shape[1],
            'actual_seq_lengths_kv':
            attn_metadata.decode_meta.
            num_computed_tokens_of_pcp_dcp[:, self.pcp_rank, self.dcp_rank],
            'actual_seq_lengths':
            attn_metadata.actual_seq_lengths_q[:attn_metadata.num_decodes],
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
                    query, k_nope, value, **common_kwargs)
                update_graph_params_workspaces(num_tokens,
                                               weak_ref_tensors(workspace))
            attn_out = torch.empty_like(query)
            attn_lse = torch.empty((num_tokens, num_heads, 1),
                                   dtype=torch.float,
                                   device=query.device)

            graph_params.attn_params[num_tokens].append((
                weak_ref_tensors(query), weak_ref_tensors(k_nope),
                weak_ref_tensors(value), self.num_heads, self.num_kv_heads,
                self.scale, attn_metadata.block_tables,
                self.key_cache.shape[1], attn_metadata.decode_meta.
                num_computed_tokens_of_pcp_dcp[:, self.pcp_rank,
                                               self.dcp_rank],
                attn_metadata.actual_seq_lengths_q[:attn_metadata.num_decodes],
                weak_ref_tensors(attn_out), weak_ref_tensors(attn_lse),
                self.dcp_size, self.pcp_rank, self.dcp_rank))
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                query,
                k_nope,
                value,
                **common_kwargs,
                workspace=workspace,
                out=[attn_out, attn_lse])
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
        else:
            attn_out, attn_lse = torch_npu.npu_fused_infer_attention_score(
                query, k_nope, value, **common_kwargs)

        out_mask = attn_metadata.decode_meta.batch_seq_mask[:, None,
                                                            None].expand_as(
                                                                attn_out)
        attn_out = torch.where(out_mask, 0, attn_out)

        lse_mask = attn_metadata.decode_meta.batch_seq_mask[:, None,
                                                            None].expand_as(
                                                                attn_lse)
        attn_lse = torch.where(lse_mask, -torch.inf, attn_lse)

        attn_out_lse_list = []
        # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
        attn_out_lse = torch.cat([attn_out, attn_lse], dim=-1)
        if self.dcp_size > 1:
            # permute: [bs, num_heads, v_head_dim+1] -> [num_heads, v_head_dim+1, bs]
            attn_out_lse = attn_out_lse.permute([1, 2, 0]).contiguous()
            attn_out_lse_all2all = torch.empty_like(attn_out_lse)
            dist.all_to_all_single(attn_out_lse_all2all,
                                   attn_out_lse,
                                   group=self.dcp_group)
            # permute: [num_heads, v_head_dim+1, bs] -> [bs, num_heads, v_head_dim+1]
            attn_out_lse_all2all = attn_out_lse_all2all.permute([2, 0, 1])
            if self.pcp_size > 1:
                attn_out_lse = attn_out_lse_all2all.contiguous()
            attn_out_lse_list = list(
                torch.chunk(attn_out_lse_all2all, self.dcp_size, dim=1))

        if self.pcp_size > 1:
            # AllGather out&lse within CP group
            attn_out_lse_list = [
                torch.empty_like(attn_out_lse) for _ in range(self.pcp_size)
            ]
            dist.all_gather(attn_out_lse_list,
                            attn_out_lse,
                            group=self.pcp_group)
        if self.dcp_size > 1 and self.pcp_size > 1:
            attn_out_lse_list_pcp_dcp = []
            for s in attn_out_lse_list:
                attn_out_lse_list_split = list(
                    torch.chunk(s, self.dcp_size, dim=1))
                attn_out_lse_list_pcp_dcp += attn_out_lse_list_split
            attn_out_lse_list = attn_out_lse_list_pcp_dcp
        # Update out&lse
        attn_out = self._npu_attention_update(attn_out_lse_list)
        return attn_out

    def _update_out_and_lse(self, out_list: torch.Tensor,
                            lse_list: torch.Tensor) -> torch.Tensor:
        """LSE_final = log(sum(exp(LSE_i))), O_final = sum(exp(LSE_i - LSE_final) * O_i)
        Args:
            out_list: shape = [N, batch_size, num_heads, head_size]
            lse_list: shape = [N, batch_size, num_heads, 1]
        Returns:
            out_final: shape = [batch_size, num_heads, head_size]
            lse_final: shape = [batch_size, num_heads, 1]
        """
        lse_final = torch.logsumexp(lse_list, dim=0, keepdim=False)
        out_final = torch.sum(torch.exp(lse_list - lse_final) * out_list,
                              dim=0)
        return out_final, lse_final

    def _forward_pcp_dcp(self, query: torch.Tensor, key: torch.Tensor,
                         value: torch.Tensor, kv_cache: Tuple[torch.Tensor],
                         attn_metadata: AscendMetadata,
                         output: torch.Tensor) -> torch.Tensor:
        assert attn_metadata is not None
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        if has_decode:
            decode_query = query[:num_decode_tokens]
            output_decode = self._forward_decode_pcp_dcp(
                decode_query, attn_metadata)
            output[:num_decode_tokens] = output_decode
        if has_prefill:
            assert attn_metadata.prefill is not None
            num_actual_tokens_pcp_padded = attn_metadata.num_actual_tokens_pcp_padded // self.pcp_size
            prefill_query = query[
                num_decode_tokens:num_actual_tokens_pcp_padded]
            key = key[self.pcp_size * num_decode_tokens:]
            value = value[self.pcp_size * num_decode_tokens:]
            if self.pcp_size > 1:
                # Scenario of Enabling PCP or PCP&DCP
                attn_output_prefill, attn_lse_prefill = self._forward_prefill_cp(
                    prefill_query, key, value, attn_metadata)
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
                    actual_seq_lengths_kv=attn_metadata.prefill.
                    actual_seq_lengths_q,
                    actual_seq_lengths=attn_metadata.prefill.
                    actual_seq_lengths_q)

            self._process_chunk_prefill(attn_output_prefill, attn_lse_prefill,
                                        kv_cache, prefill_query, attn_metadata)
            output[num_decode_tokens:attn_output_prefill.shape[0] +
                   num_decode_tokens] = attn_output_prefill
        return output

    def _process_chunk_prefill(self, current_attn_output_prefill,
                               current_attn_lse_prefill, kv_cache,
                               prefill_query, attn_metadata):
        if attn_metadata.prefill is not None and attn_metadata.prefill.chunked_context is not None:
            prefill_query_all = self._prefill_query_all_gather(
                attn_metadata, prefill_query)
            attn_output_full_chunk, attn_lse_full_chunk = self._compute_prefill_context(
                prefill_query_all, kv_cache, attn_metadata)
            self._update_chunk_attn_out_lse_with_current_attn_out_lse(
                current_attn_output_prefill, current_attn_lse_prefill,
                attn_output_full_chunk, attn_lse_full_chunk, prefill_query,
                attn_metadata)

    def _update_chunk_attn_out_lse_with_current_attn_out_lse(
            self, current_attn_output_prefill, current_attn_lse_prefill,
            attn_output_full_chunk, attn_lse_full_chunk, prefill_query,
            attn_metadata):
        if self.pcp_size > 1:
            inverse_idx = attn_metadata.prefill.chunked_context.kv_inverse_idx_for_chunk
            attn_output_full_chunk = torch.index_select(
                attn_output_full_chunk, 0, inverse_idx)
            attn_lse_full_chunk = torch.index_select(attn_lse_full_chunk, 0,
                                                     inverse_idx)
        num_tokens = prefill_query.size(0)
        attn_output_full_chunk = attn_output_full_chunk[
            self.pcp_rank * num_tokens:(self.pcp_rank + 1) * num_tokens, :, :]
        attn_lse_full_chunk = attn_lse_full_chunk[
            self.pcp_rank * num_tokens:(self.pcp_rank + 1) * num_tokens, :, :]

        assert attn_output_full_chunk.shape == current_attn_output_prefill.shape and attn_lse_full_chunk.shape == current_attn_lse_prefill.shape
        filtered_indices = attn_metadata.prefill.chunked_context.chunk_seq_mask_filtered_indices

        attn_output_prefill_filtered = current_attn_output_prefill[
            filtered_indices, :, :]
        attn_lse_prefill_filtered = current_attn_lse_prefill[
            filtered_indices, :, :]
        attn_output_full_chunk = attn_output_full_chunk[filtered_indices, :, :]
        attn_lse_full_chunk = attn_lse_full_chunk[filtered_indices, :, :]

        attn_output_filtered = self._npu_attn_out_lse_update(
            attn_lse_prefill_filtered, attn_lse_full_chunk,
            attn_output_prefill_filtered, attn_output_full_chunk)

        current_attn_output_prefill[
            filtered_indices, :, :] = attn_output_filtered.to(
                current_attn_output_prefill.dtype)

    def _prefill_query_all_gather(self, attn_metadata, prefill_query):
        if self.dcp_size > 1:
            prefill_query = get_dcp_group().all_gather(prefill_query, 1)

        if self.pcp_size > 1:
            prefill_query = get_pcp_group().all_gather(prefill_query, 0)

        prefill_query_all = torch.index_select(prefill_query,
                                               0,
                                               attn_metadata.prefill.chunked_context.cp_kv_recover_idx_for_chunk) \
            if self.pcp_size > 1 else prefill_query

        return prefill_query_all

    def _compute_prefill_context(self, query: torch.Tensor,
                                 kv_cache: Tuple[torch.Tensor],
                                 attn_metadata: AscendMetadata):
        assert len(kv_cache) > 1
        assert attn_metadata is not None
        assert attn_metadata.prefill is not None
        assert attn_metadata.prefill.chunked_context is not None
        prefill_metadata = attn_metadata.prefill
        local_chunked_kv_lens = prefill_metadata.chunked_context.local_context_lens_allranks
        assert local_chunked_kv_lens is not None

        local_chunked_kv_lens_rank = local_chunked_kv_lens[:, self.pcp_rank,
                                                           self.dcp_rank]
        total_toks = local_chunked_kv_lens_rank.sum()

        key, value = self._load_kv_for_chunk(attn_metadata, kv_cache,
                                             local_chunked_kv_lens_rank, query,
                                             total_toks)
        if self.dcp_size > 1:
            num_heads = self.num_heads * self.dcp_size
        else:
            num_heads = self.num_heads

        prefix_chunk_output = torch.full(
            (query.size(0), num_heads, self.head_size),
            fill_value=0,
            dtype=query.dtype,
            device=query.device)
        prefix_chunk_lse = torch.full((query.size(0), num_heads, 1),
                                      fill_value=-torch.inf,
                                      dtype=torch.float32,
                                      device=query.device)

        if total_toks > 0:
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
                actual_seq_lengths_kv=prefill_metadata.chunked_context.
                actual_seq_lengths_kv,
                actual_seq_lengths=attn_metadata.prefill.chunked_context.
                actual_chunk_seq_lengths)
            batch_chunk_seq_mask = attn_metadata.prefill.chunked_context.batch_chunk_seq_mask
            out_mask = batch_chunk_seq_mask[:, None, None].expand_as(
                prefix_chunk_output)
            prefix_chunk_output = torch.where(out_mask, 0, prefix_chunk_output)
            lse_mask = batch_chunk_seq_mask[:, None,
                                            None].expand_as(prefix_chunk_lse)
            prefix_chunk_lse = torch.where(lse_mask, -torch.inf,
                                           prefix_chunk_lse)

        prefix_output, prefix_lse = self._update_chunk_attn_out_lse(
            prefix_chunk_output, prefix_chunk_lse)

        return prefix_output, prefix_lse

    def _update_chunk_attn_out_lse(self, prefix_chunk_output,
                                   prefix_chunk_lse):
        # CP dimension all_gather and fusion
        chunk_attn_out_lse = torch.cat([prefix_chunk_output, prefix_chunk_lse],
                                       dim=-1)

        if self.dcp_size > 1:
            chunk_attn_out_lse = chunk_attn_out_lse.permute([1, 2,
                                                             0]).contiguous()
            attn_out_lse_all2all = torch.empty_like(chunk_attn_out_lse)
            dist.all_to_all_single(attn_out_lse_all2all,
                                   chunk_attn_out_lse,
                                   group=self.dcp_group)
            attn_out_lse_all2all = attn_out_lse_all2all.permute([2, 0, 1])
            if self.pcp_size > 1:
                chunk_attn_out_lse = attn_out_lse_all2all.contiguous()

            attn_out_lse_list = list(
                torch.chunk(attn_out_lse_all2all, self.dcp_size, dim=1))

        if self.pcp_size > 1:
            attn_out_lse_list = [
                torch.empty_like(chunk_attn_out_lse)
                for _ in range(self.pcp_size)
            ]
            dist.all_gather(attn_out_lse_list,
                            chunk_attn_out_lse,
                            group=self.pcp_group)

        if self.dcp_size > 1 and self.pcp_size > 1:
            attn_out_lse_list_pcp_dcp = []
            for s in attn_out_lse_list:
                attn_out_lse_list_split = list(
                    torch.chunk(s, self.dcp_size, dim=1))
                attn_out_lse_list_pcp_dcp += attn_out_lse_list_split
            attn_out_lse_list = attn_out_lse_list_pcp_dcp

        attn_out_lse_allgather = torch.stack(
            attn_out_lse_list,
            dim=0)  # [pcp, batch_size, num_heads, head_size+1]
        attn_out_allgather, attn_lse_allgather = torch.split(
            attn_out_lse_allgather, [self.head_size, 1], dim=-1)

        prefix_output, prefix_lse = self._update_out_and_lse(
            attn_out_allgather, attn_lse_allgather)

        return prefix_output, prefix_lse

    def _load_kv_for_chunk(self, attn_metadata, kv_cache,
                           local_chunked_kv_lens_rank, query, total_toks):
        cache_key = kv_cache[0]
        cache_value = kv_cache[1]
        num_heads = cache_key.size(2)
        head_size = kv_cache[0].size(-1)

        key = torch.empty(total_toks,
                          num_heads,
                          head_size,
                          dtype=query.dtype,
                          device=query.device)
        value = torch.empty(total_toks,
                            num_heads,
                            head_size,
                            dtype=query.dtype,
                            device=query.device)
        if total_toks > 0:
            torch_npu.atb.npu_paged_cache_load(
                cache_key,
                cache_value,
                attn_metadata.prefill.block_tables,
                local_chunked_kv_lens_rank,
                seq_starts=attn_metadata.prefill.chunked_context.
                starts,  # slot offsets of current chunk in current iteration
                key=key,
                value=value,
            )
        return key, value

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for AscendAttentionBackendImpl")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output

        # NOTE: Currently, we have various attention paths for different
        # scenarios, and not all of them are in-place operations. Therefore,
        # we need to create a separate tensor to hold the attention result.
        # In the future, we may consolidate them into fewer paths, which will
        # hopefully allow us to use in-place operation by default.
        intermediate_output: torch.Tensor

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER and attn_type != AttentionType.ENCODER_ONLY:
            raise NotImplementedError("Encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        num_decode_tokens = attn_metadata.num_decode_tokens
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0

        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            if has_decode:
                slot_mapping = attn_metadata.slot_mapping[:num_decode_tokens * self.pcp_size: self.pcp_size] \
                    if self.pcp_size * self.dcp_size > 1 else attn_metadata.slot_mapping[:num_decode_tokens]
                torch_npu._npu_reshape_and_cache(
                    key=key[:num_decode_tokens],
                    value=value[:num_decode_tokens],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=slot_mapping)

            if has_prefill:
                if self.pcp_size > 1:
                    kv = torch.cat([key, value], dim=-1)
                    num_actual_tokens_pcp_padded = attn_metadata.num_actual_tokens_pcp_padded // self.pcp_size
                    all_kv = get_pcp_group().all_gather(
                        kv[:num_actual_tokens_pcp_padded].contiguous(), dim=0)
                    pcp_allgather_restore_idx = attn_metadata.prefill.pcp_allgather_restore_idx if attn_metadata.prefill else None
                    all_kv = torch.index_select(all_kv, 0,
                                                pcp_allgather_restore_idx)
                    key, value = all_kv.split([self.head_size, self.head_size],
                                              dim=-1)

                torch_npu._npu_reshape_and_cache(
                    key=key[self.pcp_size * num_decode_tokens:attn_metadata.
                            num_actual_tokens_pcp_padded],
                    value=value[self.pcp_size *
                                num_decode_tokens:attn_metadata.
                                num_actual_tokens_pcp_padded],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=attn_metadata.
                    slot_mapping[self.pcp_size *
                                 num_decode_tokens:attn_metadata.
                                 num_actual_tokens_pcp_padded])

        forward_context: ForwardContext = get_forward_context()
        if not forward_context.capturing:
            if self.pcp_size * self.dcp_size > 1:
                intermediate_output = self._forward_pcp_dcp(
                    query, key, value, kv_cache, attn_metadata, output)
            elif attn_type == AttentionType.ENCODER_ONLY:
                # TODO(zzzwwjj): Deal with this `cum_seq_len` more elegantly.
                cum_seq_len = attn_metadata.query_start_loc[1:].tolist()
                intermediate_output = torch_npu.npu_fusion_attention(
                    query,
                    key,
                    value,
                    head_num=self.num_heads,
                    input_layout="TND",
                    scale=self.scale,
                    sparse_mode=4,
                    atten_mask=attn_metadata.attn_mask,
                    pre_tockens=attn_metadata.max_query_len,
                    next_tockens=attn_metadata.max_query_len,
                    actual_seq_qlen=cum_seq_len,
                    actual_seq_kvlen=cum_seq_len,
                )[0]
            # V0-Style scheduler situation.
            elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                intermediate_output = self._forward_prefill_no_cache(
                    query, key, value, attn_metadata, output, num_tokens)
            elif attn_metadata.attn_state == \
                    AscendAttentionState.PrefillCacheHit:
                intermediate_output = self._forward_prefill_cache_hit(
                    query, attn_metadata, output)
            elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                intermediate_output = self._forward_decode_only(
                    query, attn_metadata, output)
                # Normal V1 situation.
            else:
                # npu_fused_infer_attention_score does not support cases
                # where query.shape[0] != attn_metadata.query_start_loc[-1].
                # Thus we need unpad it here.
                num_tokens = attn_metadata.query_start_loc[-1]
                query = query[:num_tokens]
                intermediate_output = self._forward_v1_style(
                    query, attn_metadata, output)
        else:
            intermediate_output, num_tokens = self.full_graph_attention(
                query, key, value, kv_cache, attn_metadata, output)
        output[:num_tokens] = intermediate_output[:num_tokens]

        return output
