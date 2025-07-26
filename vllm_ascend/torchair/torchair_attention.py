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
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch_npu
from vllm.attention.backends.abstract import AttentionLayer, AttentionType
from vllm.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.attention.attention_v1 import (AscendAttentionBackend,
                                                AscendAttentionBackendImpl,
                                                AscendAttentionMetadataBuilder,
                                                AscendAttentionState,
                                                AscendMetadata)
from vllm_ascend.utils import vllm_version_is


class AscendAttentionTorchairBackend(AscendAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "ASCEND_TORCHAIR"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionTorchairBackendImpl"]:
        if vllm_version_is("0.9.2"):
            return AscendAttentionTorchairBackendImpl092
        return AscendAttentionTorchairBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendTorchairMetadata"]:
        return AscendTorchairMetadata

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionTorchairMetadataBuilder"]:
        return AscendAttentionTorchairMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def get_bsh_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads * head_size)


@dataclass
class AscendDecodeMetadata:
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend.
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_lens: int
    seq_lens_list: list[int]
    attn_mask: Optional[torch.Tensor] = None


@dataclass
class AscendTorchairMetadata(AscendMetadata):
    decode: Optional[AscendDecodeMetadata] = None


class AscendAttentionTorchairMetadataBuilder(AscendAttentionMetadataBuilder):

    def __init__(self, runner):
        super().__init__(runner)

    def _get_graph_runner_block_tables(
        self,
        num_seqs: int,
        block_tables: torch.Tensor,
    ) -> torch.Tensor:
        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs

        if isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((max_batch_size, max_blocks),
                                             dtype=block_tables.dtype,
                                             device=block_tables.device)
        else:
            graph_block_tables = self.runner.graph_block_tables.to(
                device=block_tables.device, dtype=block_tables.dtype)

        num_blocks = block_tables.size(1)
        if num_blocks <= max_blocks:
            graph_block_tables[:num_seqs, :
                               num_blocks] = block_tables[:num_seqs, :
                                                          num_blocks]
        else:
            graph_block_tables[:num_seqs, :
                               max_blocks] = block_tables[:num_seqs, :
                                                          max_blocks]

        return graph_block_tables[:num_seqs, :max_blocks]

    def build_dummy(
        self,
        num_reqs: int,
        num_actual_tokens: int,
    ) -> AscendTorchairMetadata:
        device = self.runner.device
        _, max_blocks = self.runner.graph_block_tables.shape
        block_table = torch.zeros((num_reqs, max_blocks),
                                  dtype=torch.int32,
                                  device=device)
        block_table = self._get_graph_runner_block_tables(
            num_reqs, block_table)
        seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=device)
        input_positions = torch.zeros(num_reqs,
                                      dtype=torch.int32,
                                      device=device).long()
        slot_mapping = torch.full((num_reqs, ),
                                  PAD_SLOT_ID,
                                  dtype=torch.int32,
                                  device=device)
        query_start_loc = torch.full((num_reqs, ),
                                     -1,
                                     dtype=torch.int32,
                                     device=device)

        decode_metadata = AscendDecodeMetadata(input_positions=input_positions,
                                               block_table=block_table,
                                               seq_lens=seq_lens,
                                               seq_lens_list=seq_lens.tolist(),
                                               max_seq_lens=1)

        attn_metadata = AscendTorchairMetadata(
            num_actual_tokens=num_actual_tokens,
            block_tables=block_table,
            query_lens=0,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            attn_state=AscendAttentionState.DecodeOnly,
            max_num_tokens_across_dp=num_reqs,
            decode=decode_metadata)
        return attn_metadata

    def build(self,
              num_reqs,
              num_actual_tokens,
              max_query_len,
              graph_pad_size: int = -1,
              max_num_tokens_across_dp: int = 0,
              with_prefill_across_dp: bool = False):
        block_table, \
        query_start_loc, \
        query_lens, \
        seq_lens, \
        slot_mapping, \
        attn_mask, \
        attn_state \
        = self._prepare_build_info(num_reqs, max_query_len, num_actual_tokens)

        decode_metadata = None
        input_positions = self.runner.positions_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True).long()
        use_torchair_graph = graph_pad_size > -1

        if self.runner.attn_state == AscendAttentionState.DecodeOnly:
            max_seq_lens = seq_lens.max().item()
            num_seqs = len(seq_lens)
            if use_torchair_graph:
                pad_value = 1
                padded_seq_lens = seq_lens.tolist() + [pad_value
                                                       ] * graph_pad_size
                max_num_tokens_across_dp = len(padded_seq_lens)

                seq_lens = torch.from_numpy(
                    np.array(padded_seq_lens).astype(np.int32))
                padding = torch.full((graph_pad_size, ),
                                     PAD_SLOT_ID,
                                     dtype=slot_mapping.dtype,
                                     device=slot_mapping.device)
                slot_mapping = torch.cat([slot_mapping, padding])
                block_table_padding = torch.zeros(
                    (graph_pad_size, ) + block_table.shape[1:],
                    dtype=block_table.dtype,
                    device=block_table.device)
                block_table = torch.cat([block_table, block_table_padding],
                                        dim=0)
                block_table = self._get_graph_runner_block_tables(
                    num_seqs + graph_pad_size, block_table)
                padding_0 = torch.zeros(graph_pad_size,
                                        dtype=input_positions.dtype,
                                        device=input_positions.device)
                input_positions = torch.cat([input_positions, padding_0])

            decode_metadata = AscendDecodeMetadata(
                input_positions=input_positions,
                block_table=block_table,
                seq_lens=seq_lens,
                seq_lens_list=seq_lens.tolist(),
                max_seq_lens=max_seq_lens,
                attn_mask=None)

        return AscendTorchairMetadata(
            decode=decode_metadata,
            num_actual_tokens=num_actual_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_lens=query_lens,
            seq_lens=seq_lens,
            max_query_len=max_query_len,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            max_num_tokens_across_dp=max_num_tokens_across_dp,
            with_prefill_across_dp=with_prefill_across_dp)


class AscendAttentionTorchairBackendImpl(AscendAttentionBackendImpl):

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

    def _forward_decode_only(
        self,
        attn_metadata,
        query,
        output,
        key_cache,
        value_cache,
        num_tokens,
    ) -> torch.Tensor:
        decode_meta = attn_metadata.decode
        assert decode_meta is not None

        seq_lens = decode_meta.seq_lens_list
        block_table = decode_meta.block_table
        block_size = key_cache.shape[1]
        query = query.view(num_tokens, 1,
                           self.num_heads * self.head_size).contiguous()

        output = torch_npu.npu_incre_flash_attention(
            query,
            key_cache,
            value_cache,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            actual_seq_lengths=seq_lens,
            scale_value=self.scale,
            block_table=block_table,
            input_layout='BSH',
            block_size=block_size)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendTorchairMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = False,
    ) -> torch.Tensor:
        # `trace_flag` is not supported when using torchair
        trace_flag = False

        num_tokens, \
        output, \
        hit_check \
        = self._check_before_forward(layer, query, key, value, kv_cache,
                                     attn_metadata, output, trace_flag)
        if hit_check:
            return output

        output = output.view(-1, self.num_heads, self.head_size)

        if kv_cache is not None and kv_cache[0].numel() > 0:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping

            block_size = key_cache.shape[1]
            slots_indices = slots.reshape(-1, 1)
            block_indices = slots_indices // block_size
            slots_indices = slots_indices % block_size
            indices = torch.cat((block_indices, slots_indices), dim=1)
            torch_npu.npu_scatter_nd_update_(key_cache, indices, key)
            torch_npu.npu_scatter_nd_update_(value_cache, indices, value)

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            # View q k v to BSH.
            query = query.view(-1, self.num_heads, self.head_size)
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
            output = self._forward_prefill_no_cache(attn_metadata, query, key,
                                                    value, output, num_tokens)
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            output = self._forward_prefill_cache_hit(attn_metadata, query,
                                                     output)
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            output = self._forward_decode_only(attn_metadata, query, output,
                                               key_cache, value_cache,
                                               num_tokens)
        else:
            raise NotImplementedError(
                "Torchair graph mode with non-MLA attention backend is still experimental."
                "v1 scheduler(chunked prefill) is not supported at this moment. Please"
                "setting 'ascend_scheduler_config':{'enabled':true} in additional_config"
                "to use ascend scheduler.")

        return output.view(num_tokens, self.hidden_size)


class AscendAttentionTorchairBackendImpl092(AscendAttentionTorchairBackendImpl
                                            ):

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
        kv_sharing_target_layer_name: Optional[str] = None,
        use_irope: bool = False,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            use_irope=use_irope,
        )
