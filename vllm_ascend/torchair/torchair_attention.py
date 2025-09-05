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
from typing import List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch_npu
from vllm.attention.backends.abstract import (AttentionImpl, AttentionLayer,
                                              AttentionType)
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.utils import cdiv

from vllm_ascend.attention.attention_v1 import (AscendAttentionBackend,
                                                AscendAttentionMetadataBuilder,
                                                AscendAttentionState,
                                                AscendMetadata)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.torchair.utils import TorchairCommonAttentionMetadata
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, aligned_16, is_310p,
                               nd_to_nz_2d)


class AscendAttentionTorchairBackend(AscendAttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_TORCHAIR"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionTorchairBackendImpl"]:
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
        return (2, num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def get_bsh_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads * head_size)


@dataclass
class AscendDecodeMetadata:
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
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

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len,
            self.vllm_config.cache_config.block_size)
        self.max_blocks = (self.model_config.max_model_len +
                           self.vllm_config.cache_config.block_size -
                           1) // self.vllm_config.cache_config.block_size

    def _get_graph_runner_block_tables(
            self, num_seqs: int, block_tables: torch.Tensor) -> torch.Tensor:
        max_blocks = self.max_blocks

        graph_block_tables = torch.zeros((num_seqs, max_blocks),
                                         dtype=block_tables.dtype,
                                         device=block_tables.device)

        num_blocks = block_tables.size(1)
        if num_blocks <= max_blocks:
            graph_block_tables[:num_seqs, :
                               num_blocks] = block_tables[:num_seqs, :
                                                          num_blocks]
        else:
            graph_block_tables[:num_seqs, :
                               max_blocks] = block_tables[:num_seqs, :
                                                          max_blocks]

        return graph_block_tables[:, :max_blocks]

    def build_torchair_graph_dummy(
        self, common_attn_metadata: TorchairCommonAttentionMetadata
    ) -> AscendTorchairMetadata:
        device = self.device
        num_reqs = common_attn_metadata.num_reqs
        block_table = torch.zeros((num_reqs, self.max_blocks),
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
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            block_tables=block_table,
            query_lens=0,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            attn_state=AscendAttentionState.DecodeOnly,
            decode=decode_metadata)
        return attn_metadata

    def build(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: nn.Module,
    ):
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens

        block_table = common_attn_metadata.block_table_tensor
        block_table[:num_reqs, :self.max_num_blocks_per_req] = (
            block_table[:num_reqs])

        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping_cpu[:
                                                             num_actual_tokens].to(
                                                                 self.device,
                                                                 non_blocking=
                                                                 True)
        attn_mask = common_attn_metadata.attn_mask

        attn_state = common_attn_metadata.attn_state
        if is_310p() and attn_state == AscendAttentionState.PrefillNoCache:
            mask_nz = nd_to_nz_2d(attn_mask)
            attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(), 29)

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]
        query_start_loc = query_start_loc_cpu.to(self.device,
                                                 non_blocking=True)
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        input_positions = common_attn_metadata.positions[:
                                                         num_actual_tokens].long(
                                                         )

        decode_metadata = None
        graph_pad_size = common_attn_metadata.graph_pad_size
        use_torchair_graph = graph_pad_size > -1
        if common_attn_metadata.attn_state in [
                AscendAttentionState.DecodeOnly,
        ]:
            max_seq_lens = seq_lens.max().item()
            num_seqs = len(seq_lens)
            if use_torchair_graph and common_attn_metadata.attn_state in [
                    AscendAttentionState.DecodeOnly,
            ]:
                num_reqs_pad_size = 0
                num_token_pad_size = 0
                if graph_pad_size != 0:
                    pad_value = 0
                    num_token_pad_size = graph_pad_size - num_actual_tokens
                    num_reqs_pad_size = (
                        graph_pad_size //
                        common_attn_metadata.decode_token_per_req - num_reqs)
                pad_value = 1
                padded_seq_lens = seq_lens.tolist() + [pad_value
                                                       ] * num_reqs_pad_size

                seq_lens = torch.from_numpy(
                    np.array(padded_seq_lens).astype(np.int32))
                padding = torch.full((num_token_pad_size, ),
                                     PAD_SLOT_ID,
                                     dtype=slot_mapping.dtype,
                                     device=slot_mapping.device)
                slot_mapping = torch.cat([slot_mapping, padding])
                block_table_padding = torch.zeros(
                    (num_reqs_pad_size, ) + block_table.shape[1:],
                    dtype=block_table.dtype,
                    device=block_table.device)
                block_table = torch.cat([block_table, block_table_padding],
                                        dim=0)
                block_table = self._get_graph_runner_block_tables(
                    num_seqs + num_reqs_pad_size, block_table)
                padding_0 = torch.zeros(num_token_pad_size,
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

        attn_metadata = AscendTorchairMetadata(
            decode=decode_metadata,
            num_actual_tokens=num_actual_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_lens=query_lens,
            seq_lens=seq_lens,
            max_query_len=common_attn_metadata.max_query_len,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            enable_dbo_across_dp=common_attn_metadata.enable_dbo_across_dp)
        return attn_metadata


class AscendAttentionTorchairBackendImpl(AttentionImpl):

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
        self.scale_tensor = torch.zeros((), device='npu', dtype=torch.int32)

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
        """Forward pass with Ascend attention.
        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache: shape = [2, num_blocks, block_size,
                               num_kv_heads, head_size]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads, head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * seq_len, num_heads, head_size]
        """
        num_tokens = query.shape[0]
        use_kv_cache_quant = (kv_cache is not None and len(kv_cache) > 0
                              and kv_cache[0].numel() > 0
                              and kv_cache[0].dtype == torch.int8)
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)

        if hasattr(layer, 'quant_method') and use_kv_cache_quant:
            output = layer.quant_method.apply(layer, query, key, value,
                                              kv_cache, attn_metadata,
                                              self.attn_type, self.scale,
                                              output)
            return output.view(num_tokens, self.hidden_size)

        if attn_metadata is None:
            return output.view(num_tokens, self.hidden_size)

        output = output.view(-1, self.num_heads, self.head_size)

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "AscendAttentionTorchairBackendImpl")

        if kv_cache is not None and kv_cache[0].numel() > 0:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping

            block_size = self.scale_tensor + key_cache.shape[1]
            slots_indices = slots.reshape(-1, 1)
            block_indices = slots_indices // block_size
            slots_indices = slots_indices % block_size
            indices = torch.cat((block_indices, slots_indices), dim=1)
            torch_npu.npu_scatter_nd_update_(key_cache, indices, key)
            torch_npu.npu_scatter_nd_update_(value_cache, indices, value)
            if attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
                self.key_cache = key_cache
                self.value_cache = value_cache

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            assert attn_metadata is not None
            assert attn_metadata.attn_mask is not None
            mask = attn_metadata.attn_mask

            # View q k v to BSH.
            query = query.view(-1, self.num_heads, self.head_size)
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)

            if is_310p():
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
            output = output[:num_tokens, :, :]
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            assert attn_metadata is not None
            assert attn_metadata.attn_mask is not None
            compress_mask = attn_metadata.attn_mask
            batch_size = attn_metadata.query_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
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
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
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
        else:
            raise NotImplementedError(
                "Torchair graph mode with non-MLA attention backend is still experimental."
                "v1 scheduler(chunked prefill) is not supported at this moment. Please"
                "setting 'ascend_scheduler_config':{'enabled':true} in additional_config"
                "to use ascend scheduler.")

        return output.view(num_tokens, self.hidden_size)
