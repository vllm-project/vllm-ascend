#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState


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
    def get_kv_cache_shape(
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


@dataclass
class AscendMetadata(AttentionMetadata):
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    block_tables: Optional[torch.Tensor]

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None

    context_lens: Optional[List[int]] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["AscendMetadata"] = None
    _cached_decode_metadata: Optional["AscendMetadata"] = None

    # TODO: Indicates whether there are only prefill requests.
    # FlashAttention can be used when there are only prefill requests.
    # FlashAttention has better performance than PageAtttention,
    # but it does not support decode requests.
    is_only_prefill: bool = False
    attn_mask: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional["AscendMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure.
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))

        # Compute some attn_metadata fields which default to None.
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])

        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])

        # Construct & cache prefill-phase attention metadata structure.
        self._cached_prefill_metadata = AscendMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            block_tables=block_tables,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            enable_kv_scales_calculation=False)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["AscendMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure.
            return self._cached_decode_metadata

        # Compute some attn_metadata fields which default to None.
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        # Construct & cache decode-phase attention metadata structure.
        self._cached_decode_metadata = AscendMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            enable_kv_scales_calculation=False)
        return self._cached_decode_metadata


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
        self.seq_len_cpu_tensor = None

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

        if hasattr(layer, 'quant_method'):
            isPrefill = True if attn_metadata.num_prefills > 0 else False
            if isPrefill:
                assert attn_metadata.prefill_metadata is not None
                self.seq_lens_tensor_cpu = torch.from_numpy(
                    np.array(attn_metadata.prefill_metadata.seq_lens).astype(
                        np.int32))
            else:
                assert attn_metadata.decode_metadata is not None
                self.seq_lens_tensor_cpu = torch.from_numpy(
                    np.array(attn_metadata.decode_metadata.seq_lens).astype(
                        np.int32))
            block_tables = attn_metadata.decode_metadata.block_tables if attn_metadata.decode_metadata else None
            # Details of kv_cache arrangement in attention quantization
            # are implemented by quant_method.
            layer.quant_method.apply(layer, query, key, value, kv_cache,
                                     self.scale, self.seq_lens_tensor_cpu,
                                     block_tables, isPrefill, attn_metadata,
                                     output)
        else:
            if kv_cache.numel() > 0:
                key_cache, value_cache = kv_cache[0], kv_cache[1]
                num_blocks, block_size, _ = key_cache.shape
                key_cache = key_cache.view(num_blocks, block_size,
                                           self.num_kv_heads, self.head_size)
                value_cache = value_cache.view(num_blocks, block_size,
                                               self.num_kv_heads,
                                               self.head_size)
                slots = attn_metadata.slot_mapping
                torch_npu._npu_reshape_and_cache(key=key,
                                                 value=value,
                                                 key_cache=key_cache,
                                                 value_cache=value_cache,
                                                 slot_indices=slots)

            # use paged attention
            torch_npu._npu_paged_attention_splitfuse(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                mask=attn_metadata.attn_mask,
                block_table=attn_metadata.block_tables,
                seq_len=attn_metadata.seq_lens,
                context_lens=attn_metadata.context_lens,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                out=output)
        return output.view(num_tokens, self.hidden_size)
