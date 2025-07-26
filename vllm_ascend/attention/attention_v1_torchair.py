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
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.utils import PAD_SLOT_ID, CommonAttentionState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, aligned_16, is_310p,
                               nd_to_nz_2d)


class AscendAttentionTorchairBackend(AttentionBackend):
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
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

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
class AscendTorchairMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    block_tables: torch.Tensor
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    query_start_loc: torch.Tensor
    query_lens: torch.Tensor
    seq_lens: torch.Tensor
    # max value of number of tokens across dp group
    max_num_tokens_across_dp: int = 0
    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    attn_mask: Optional[torch.Tensor] = None
    with_prefill_across_dp: bool = False
    decode: Optional[AscendDecodeMetadata] = None


class AscendAttentionTorchairMetadataBuilder:

    def __init__(self, runner):
        self.runner = runner

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def _get_graph_runner_block_tables(
            self, num_seqs: int, block_tables: torch.Tensor) -> torch.Tensor:

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

    def build_dummy(self, num_reqs: int,
                    num_actual_tokens: int) -> AscendTorchairMetadata:
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

        device = self.runner.device

        block_table = self.runner.input_batch.block_table[0].get_device_tensor(
        )
        block_table[:num_reqs, :self.runner.max_num_blocks_per_req] = (
            block_table[:num_reqs])

        query_lens = self.runner.query_lens
        seq_lens = self.runner.seq_lens_cpu[:num_reqs]
        slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True)
        attn_mask = self.runner.attn_mask

        attn_state = self.runner.attn_state
        if is_310p() and attn_state == AscendAttentionState.PrefillNoCache:
            mask_nz = nd_to_nz_2d(attn_mask)
            attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(), 29)

        query_start_loc_cpu = self.runner.query_start_loc_cpu[:num_reqs + 1]
        query_start_loc = query_start_loc_cpu.to(self.runner.device,
                                                 non_blocking=True)
        input_positions = self.runner.positions_cpu[:num_actual_tokens].to(
            device, non_blocking=True).long()

        decode_metadata = None
        use_torchair_graph = graph_pad_size > -1
        if self.runner.attn_state in [
                AscendAttentionState.DecodeOnly,
        ]:
            max_seq_lens = seq_lens.max().item()
            num_seqs = len(seq_lens)
            if use_torchair_graph and self.runner.attn_state in [
                    AscendAttentionState.DecodeOnly,
            ]:
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

        attn_metadata = AscendTorchairMetadata(
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
        use_kv_cache_quant = kv_cache is not None and kv_cache[0].numel(
        ) > 0 and kv_cache[0].dtype == torch.int8
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

            block_size = key_cache.shape[1]
            slots_indices = slots.reshape(-1, 1)
            block_indices = slots_indices // block_size
            slots_indices = slots_indices % block_size
            indices = torch.cat((block_indices, slots_indices), dim=1)
            torch_npu.npu_scatter_nd_update_(key_cache, indices, key)
            torch_npu.npu_scatter_nd_update_(value_cache, indices, value)

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
            torch_npu._npu_flash_attention_qlens(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                block_table=attn_metadata.block_tables,
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
