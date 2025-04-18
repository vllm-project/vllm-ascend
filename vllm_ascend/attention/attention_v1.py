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
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_ascend.ops.attention import vanilla_chunked_prefill

class AscendAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendMetadata"]:
        return AscendMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

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
        return (2, num_blocks, block_size, num_kv_heads, head_size)

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


class AscendAttentionState(Enum):
    PrefillOnly = 0
    DecodeOnly = 1
    ChunkedPrefill = 2


@dataclass
class AscendMetadata:
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    block_tables: Optional[torch.Tensor]
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor = None
    # TODO: Indicates whether there are only prefill requests.
    # FlashAttention can be used when there are only prefill requests.
    # FlashAttention has better performance than PageAtttention,
    # but it does not support decode requests.
    is_only_prefill: bool = False
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    attn_mask: Optional[torch.Tensor] = None


class AscendAttentionMetadataBuilder:
    def __init__(self, runner):
        self.runner = runner

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(self, num_reqs, num_actual_tokens, max_query_len, common_prefix_len):
        block_table = (
            self.runner.input_batch.block_table.get_device_tensor()[:num_reqs])
        query_seq_lens = self.runner.query_start_loc_cpu[1:num_reqs + 1] - self.runner.query_start_loc_cpu[:num_reqs]
        context_lens = self.runner.seq_lens_cpu[:num_reqs]
        slot_mapping = self.runner.slot_mapping_cpu[:num_reqs].to(self.runner.device, non_blocking=True)
        attn_mask = self.runner.attn_mask
        attn_state = self.runner.attn_state

        attn_metadata = AscendMetadata(
            block_tables=block_table,
            seq_lens=query_seq_lens,
            context_lens=context_lens,
            max_query_len=max_query_len,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state
        )
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
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
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

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache: shape = [2, num_blocks, block_size,
                               num_kv_heads * head_size]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads * head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * seq_len, num_heads, head_size]
        """
        num_tokens = query.shape[0]
        output = torch.empty(num_tokens,
                             self.num_heads,
                             self.head_size,
                             dtype=query.dtype,
                             device=query.device)

        if attn_metadata is None:
            # Profiling run.
            return output.view(num_tokens, self.hidden_size)
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")
        # View q k v to BSH.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        # TODO: Remove this contiguous in the future.
        value = value.contiguous()

        if kv_cache.numel() > 0:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping
            torch_npu._npu_reshape_and_cache(key=key,
                                             value=value,
                                             key_cache=self.key_cache,
                                             value_cache=self.value_cache,
                                             slot_indices=slots)

        if hasattr(layer, 'quant_method'):
            # TODO: Add attr (num_prefills, prefill_metadata, decode_metadata) to AscendMetadata
            pass
        # V0-Style scheduler situation.
        elif attn_metadata.attn_state == AscendAttentionState.PrefillOnly:
            assert attn_metadata is not None
            assert attn_metadata.attn_mask is not None
            mask = attn_metadata.attn_mask
            torch_npu._npu_flash_attention(query=query,
                                           key=key,
                                           value=value,
                                           mask=mask,
                                           seq_len=attn_metadata.seq_lens,
                                           scale_value=self.scale,
                                           num_heads=self.num_heads,
                                           num_kv_heads=self.num_kv_heads,
                                           out=output)
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            block_tables = attn_metadata.block_tables
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=block_tables,
                context_lens=attn_metadata.context_lens,
                out=output)
        # Normal V1 situation.
        else:
            # use chunked prefill for head size 192 scenario, like deepseek
            # paged_attention_splitfuse maybe crash at such scenario
            if self.head_size == 192:
                cu_seqlen_q = [0] + attn_metadata.seq_lens.tolist()
                cu_seqlen_k = [0] + attn_metadata.context_lens.tolist()
                cu_seqlen_q = torch.tensor(cu_seqlen_q, device="npu")
                cu_seqlen_k = torch.tensor(cu_seqlen_k, device="npu")
                cu_seqlen_q = torch.cumsum(cu_seqlen_q, dim=0)
                cu_seqlen_k = torch.cumsum(cu_seqlen_k, dim=0)
                max_seqlen_q = torch.max(attn_metadata.seq_lens)
                max_seqlen_k = torch.max(attn_metadata.context_lens)
                num_queries_per_kv = self.num_heads / self.num_kv_heads
                vanilla_chunked_prefill(
                    output,
                    query,
                    num_queries_per_kv,
                    self.key_cache,
                    self.value_cache,
                    attn_metadata.block_tables,
                    cu_seqlen_q,
                    cu_seqlen_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    self.scale,
                    None,
                    True)
            else:
                torch_npu._npu_paged_attention_splitfuse(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    mask=attn_metadata.attn_mask,
                    block_table=attn_metadata.block_tables,
                    seq_len=attn_metadata.seq_lens,
                    context_lens=attn_metadata.context_lens,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    out=output)
        return output.view(num_tokens, self.hidden_size)
