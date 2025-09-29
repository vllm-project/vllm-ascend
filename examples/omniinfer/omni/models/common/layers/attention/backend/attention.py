# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

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

import os
import numpy as np
import torch
import torch_npu
import torchair as tng
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils import (direct_register_custom_op, supports_dynamo)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable
from vllm.platforms import current_platform
from vllm.config import (get_current_vllm_config, CompilationLevel)
from omni.models.common.layers.attention.backend.attention_mask import AttentionMaskBuilder
from omni.models.common.layers.attention.backend.attention_dummy_builder import DummyAttentionMetadataBuilder

from vllm.logger import logger

class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3


def unified_ascend_attention_with_output(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]

    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    self.impl.forward(self,
                      query,
                      key,
                      value,
                      kv_cache,
                      attn_metadata,
                      output,
                      trace_flag=False)
    return


def unified_attention_with_output_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_ascend_attention_with_output",
    op_func=unified_ascend_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
    dispatch_key="PrivateUse1",
)


@dataclass
class AscendMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    block_tables: torch.Tensor
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    query_lens: torch.Tensor
    query_lens_list: List
    seq_lens: torch.Tensor
    seq_lens_list: List
    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor = None
    slot_indices: torch.Tensor = None
    is_only_prefill: bool = False
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None
    is_pd_seperate_d: bool = False
    kv_index: Optional[torch.Tensor] = None

class AscendAttentionMetadataBuilder(DummyAttentionMetadataBuilder):

    def __init__(self, runner, kv_cache_spec: AttentionSpec = None,
                 block_table: BlockTable = None):
        self.runner = runner
        self.dtype = runner.dtype
        self.device = runner.device
        self.kv_cache_spec = kv_cache_spec
        self.block_size = runner.block_size
        self.block_table = block_table

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the TritonMLA._forward_decode only supports
            # num_tokens = 1
            # Only in decode the spec tokens are scheduled
            if req_id in scheduler_output.scheduled_spec_decode_tokens or num_tokens == 1:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        # Save for next `build` call
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

        return modified_batch

    def _get_graph_runner_block_tables(
            self, num_decode_tokens: int, block_tables: torch.Tensor) -> torch.Tensor:

        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        if max_batch_size < num_decode_tokens:
            raise RuntimeError("max_batch_size must be greater than or equal to num_decode_tokens")

        if isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((max_batch_size, max_blocks),
                                             dtype=block_tables.dtype,
                                             device=block_tables.device)
        else:
            graph_block_tables = self.runner.graph_block_tables.to(
                device=block_tables.device, dtype=block_tables.dtype, non_blocking=True)

        graph_block_tables = graph_block_tables[:block_tables.shape[0]]

        num_blocks = block_tables.size(1)
        if num_blocks <= max_blocks:
            graph_block_tables[:num_decode_tokens, :
                               num_blocks] = block_tables[:num_decode_tokens, :
                                                          num_blocks]
        else:
            graph_block_tables[:num_decode_tokens, :
                               max_blocks] = block_tables[:num_decode_tokens, :
                                                          max_blocks]

        return graph_block_tables
    
    def get_kv_index(self, seq_lens: torch.Tensor, block_tables: torch.Tensor, block_size: int):
        num_reqs = seq_lens.shape[0]
        if block_tables.shape[0] != num_reqs:
            raise RuntimeError(f"block_tables and seq_lens do not match. {seq_lens.shape=}, {block_tables.shape=}")
        slots = block_tables[..., None] * block_size + torch.arange(block_size)
        slots = slots.reshape(num_reqs, -1)
        kv_index = torch.empty(seq_lens.sum(), dtype=slots.dtype)
        start = 0
        for i, n in enumerate(seq_lens):
            kv_index[start:start+n] = slots[i, :n]
            start += n
        return kv_index
    
    def build(self,
              num_reqs,
              num_actual_tokens,
              max_query_len,
              common_prefix_len,
              graph_pad_size=-1):

        block_table = self.block_table.get_device_tensor()[:num_reqs]

        seq_lens = self.runner.seq_lens_cpu[:num_reqs]
        query_lens = seq_lens - self.runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]

        slot_mapping = self.block_table.slot_mapping_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True)
        input_positions = self.runner.positions_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True)

        attn_state = self.runner.attn_state

        if graph_pad_size > 0:
            padding = torch.full((graph_pad_size, ),
                                    0,
                                    dtype=slot_mapping.dtype,
                                    device=self.runner.device)
            slot_mapping = torch.cat([slot_mapping, padding])
            padding_0 = torch.zeros(graph_pad_size,
                                    dtype=input_positions.dtype,
                                    device=self.runner.device)
            input_positions = torch.cat([input_positions, padding_0])

        if self.runner.attn_state == AscendAttentionState.DecodeOnly:
            if self._num_decode_tokens % self._num_decodes != 0:
                raise RuntimeError("self._num_decode_tokens must be divisible by self._num_decodes")
            num_tokens_per_req = self._num_decode_tokens // self._num_decodes
            seq_lens = (input_positions + 1).to(seq_lens.dtype)
            block_table = block_table[:self._num_decodes, ...]
            # has speculative tokens
            if num_tokens_per_req > 1:
                block_table = block_table.unsqueeze(1).repeat(1, num_tokens_per_req, 1).view(-1, block_table.shape[-1])
            block_table_padding = torch.zeros(
                (graph_pad_size, ) + block_table.shape[1:],
                dtype=block_table.dtype,
                device=block_table.device)
            block_table = torch.cat([block_table, block_table_padding],
                                    dim=0)
            block_table = self._get_graph_runner_block_tables(
                self._num_decode_tokens, block_table)
            kv_index = None
        else:
            kv_index = self.get_kv_index(
                seq_lens=seq_lens,
                block_tables=self.block_table.get_cpu_tensor()[:num_reqs],
                block_size=self.block_size,
            )
            kv_index = kv_index.to(self.runner.device)
            
        slot_indices = torch.stack([slot_mapping // self.block_size, slot_mapping % self.block_size], dim=1)

        cos, sin = self.runner.model.model.layers[0].self_attn.rotary_emb.get_cos_sin(input_positions)

        is_pd_seperate_d = self.runner.vllm_config.kv_transfer_config is not None and \
                           self.runner.vllm_config.kv_transfer_config.kv_role == 'kv_consumer'

        attn_metadata = AscendMetadata(num_actual_tokens=num_actual_tokens,
                                       block_tables=block_table,
                                       query_lens=query_lens,
                                       query_lens_list=query_lens.tolist(),
                                       seq_lens=seq_lens,
                                       seq_lens_list=seq_lens.tolist(),
                                       max_query_len=max_query_len,
                                       slot_mapping=slot_mapping,
                                       slot_indices=slot_indices,
                                       attn_state=attn_state,
                                       cos=cos,
                                       sin=sin,
                                       is_pd_seperate_d=is_pd_seperate_d,
                                       kv_index=kv_index)
        return attn_metadata

    def build_dummy(self, num_tokens: int, max_pad_size: int = -1) -> AscendMetadata:
        if max_pad_size == -1:
            max_pad_size = self.runner.max_batch_size
        slot_mapping = torch.zeros(max_pad_size,
                                   dtype=self.runner.slot_mapping_cpu.dtype,
                                   device=self.runner.device)
        if isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((max_pad_size, self.runner.graph_block_tables.shape[1]))
        block_table = graph_block_tables.to(
            device=self.runner.device,
            dtype=self.runner.input_batch.block_table[0].get_device_tensor().dtype
        )

        seq_lens = torch.ones(max_pad_size, dtype=torch.long, device=self.runner.device, pin_memory=True) * 2

        slot_indices = torch.stack([slot_mapping // self.block_size, slot_mapping % self.block_size], dim=1)

        fake_positions = torch.zeros(max_pad_size, dtype=torch.int64, device=self.device)

        cos, sin = self.runner.model.model.layers[0].self_attn.rotary_emb.get_cos_sin(fake_positions)

        is_pd_seperate_d = self.runner.vllm_config.kv_transfer_config is not None and \
                           self.runner.vllm_config.kv_transfer_config.kv_role == 'kv_consumer'

        return AscendMetadata(
            num_actual_tokens=num_tokens,
            block_tables=block_table,
            query_lens=seq_lens,
            query_lens_list=seq_lens.tolist(),
            seq_lens=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            slot_mapping=slot_mapping,
            slot_indices=slot_indices,
            is_only_prefill=False,
            attn_state=self.runner.attn_state,
            cos=cos,
            sin=sin,
            is_pd_seperate_d=is_pd_seperate_d
        )

    def mark_static_for_attn_metadata(self, attn_metadata):
        if attn_metadata is not None:
            torch._dynamo.mark_static(attn_metadata.block_tables)
            torch._dynamo.mark_static(attn_metadata.seq_lens)
            if attn_metadata.slot_mapping is not None:
                torch._dynamo.mark_static(attn_metadata.slot_mapping)
            if attn_metadata.slot_indices is not None:
                torch._dynamo.mark_static(attn_metadata.slot_indices)
            if attn_metadata.cos is not None:
                torch._dynamo.mark_static(attn_metadata.cos)
            if attn_metadata.sin is not None:
                torch._dynamo.mark_static(attn_metadata.sin)


class AscendAttentionBackendImpl(AttentionImpl):

    SHARE_MASK_TRIL_SPARSE = None

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
            use_irope: bool = False,
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
                                        device=current_platform.device_type)
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        if self.num_heads % self.num_kv_heads != 0:
            raise RuntimeError("self.num_heads must be divisible by self.num_kv_heads")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None

        cur_vllm_config = get_current_vllm_config()
        self.enable_graph_mode = (cur_vllm_config.npu_compilation_config.level >  \
                                  CompilationLevel.NO_COMPILATION and supports_dynamo())

        if AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE is None:
            AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE = ~torch.tril(
                torch.ones((2048, 2048), dtype=torch.bool, device="npu")
            )

    def forward(
            self,
            layer: AttentionLayer,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: Tuple,
            attn_metadata: AscendMetadata,
            output: Optional[torch.Tensor] = None,
            trace_flag: bool = True,
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
        # logger.warning(f"<<<key: {key}, key's shape:{key.shape}")
        # logger.warning(f"<<<value: {value}, value's shape:{value.shape}")
        num_tokens = query.shape[0]
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)

        if attn_metadata is None:
            return output.view(num_tokens, self.hidden_size)

        if not (layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0):
            raise RuntimeError("layer._k_scale_float and layer._v_scale_float must both be 1.0")
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
        value = value.contiguous()

        # update kv cache
        if kv_cache[0].numel() > 0 or kv_cache[1].numel():
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            block_size = self.key_cache.shape[1]

            cast_key = key.reshape(-1, 1, self.num_kv_heads * self.head_size)
            cast_value = value.reshape(-1, 1, self.num_kv_heads * self.head_size)

            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                ## npu logic
                if self.key_cache.device.type == "npu":
                    torch_npu._npu_reshape_and_cache(
                        key,
                        value,
                        self.key_cache.view(self.key_cache.shape[0], block_size, self.num_kv_heads, self.head_size),
                        self.value_cache.view(self.value_cache.shape[0], block_size, self.num_kv_heads, self.head_size),
                        attn_metadata.slot_mapping.int()
                    )
                elif self.key_cache.device.type == "cpu": 
                    # if k/v cache is on host RAM
                    slot_mapping = attn_metadata.slot_mapping.int()
                    key_cache_view = self.key_cache.view(
                        self.key_cache.shape[0], block_size, self.num_kv_heads, self.head_size
                    )
                    value_cache_view = self.value_cache.view(
                        self.value_cache.shape[0], block_size, self.num_kv_heads, self.head_size
                    )
                    key_expanded = key.to("cpu").unsqueeze(1)
                    value_expanded = value.to("cpu").unsqueeze(1)

                    key_cache_view[slot_mapping, :1, :, :] = key_expanded
                    value_cache_view[slot_mapping, :1, :, :] = value_expanded
                else:
                    raise RuntimeError("Device invalid.")
            else:
                torch_npu.scatter_update_(self.key_cache, attn_metadata.slot_indices, cast_key, -2)
                torch_npu.scatter_update_(self.value_cache, attn_metadata.slot_indices, cast_value, -2)

        if hasattr(layer, 'quant_method'):
            pass
        # V0-Style scheduler situation.
        elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            if attn_metadata is None:
                raise RuntimeError("attn_metadata must not be None")

            if len(attn_metadata.query_lens_list) == 1:
                attn_output = torch_npu.npu_fused_infer_attention_score(
                    query.unsqueeze(0),
                    key.unsqueeze(0),
                    value.unsqueeze(0),
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="BSND",
                    scale=self.scale,
                    sparse_mode=3,
                    actual_seq_lengths=attn_metadata.query_lens_list,
                    actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                    atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                )[0].view(-1, self.num_heads, self.head_size)

                output = output.view_as(attn_output)
                output.copy_(attn_output)
            else:
                actual_seq_qlen = np.array(attn_metadata.query_lens).cumsum().tolist()
                actual_seq_kvlen = np.array(attn_metadata.seq_lens).cumsum().tolist()

                attn_output = torch_npu.npu_fusion_attention(
                    query[:actual_seq_qlen[-1], :],
                    key[:actual_seq_qlen[-1], :],
                    value[:actual_seq_qlen[-1], :],
                    head_num=self.num_heads,
                    input_layout="TND",
                    scale=self.scale,
                    atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                    sparse_mode=3,
                    actual_seq_qlen=actual_seq_qlen,
                    actual_seq_kvlen=actual_seq_kvlen)[0]

                output[:actual_seq_qlen[-1], :].copy_(attn_output)

        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:

            block_num, block_size = self.key_cache.shape[0], self.key_cache.shape[1]

            num_batch = attn_metadata.seq_lens.shape[0]
            query = query.view(num_batch, -1, self.num_heads * self.head_size)
            block_tables = attn_metadata.block_tables
            attn_output = None
            if self.enable_graph_mode:
                attn_output, _ = tng.ops.npu_fused_infer_attention_score(
                    torch.transpose(query.view(num_batch, -1, self.num_heads, self.head_size), 1, 2),
                    self.key_cache,
                    self.value_cache,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="BNSD",
                    scale=self.scale,
                    actual_seq_lengths_kv=attn_metadata.seq_lens,
                    block_table=block_tables,
                    block_size=block_size,
                    inner_precise=1
                )
            else:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query,
                    self.key_cache,
                    self.value_cache,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="BSH",
                    scale=self.scale,
                    actual_seq_lengths_kv=attn_metadata.seq_lens,
                    block_table=block_tables,
                    block_size=block_size,
                )

            output = output.view_as(attn_output)
            output.copy_(attn_output)

        # Normal V1 situation.
        else:
            # use chunked prefill for head size 192 scenario, like deepseek
            # paged_attention_splitfuse maybe crash at such scenario
            #logger.warning(f"<<<kv_index's devices: {attn_metadata.kv_index.device}")
            #logger.warning(f"<<<key_cache's devices: {self.key_cache.device}")
            if self.key_cache.device.type == "npu":
                all_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)[attn_metadata.kv_index].contiguous()
                all_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)[attn_metadata.kv_index].contiguous()
            elif self.key_cache.device.type == "cpu":
                all_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)[attn_metadata.kv_index.cpu()].contiguous().npu()
                all_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)[attn_metadata.kv_index.cpu()].contiguous().npu()
            else: 
                raise RuntimeError(f"Invalid Device {self.key_cache.device}, {type(self.key_cache.device)}")
            actual_seq_qlen = np.array(attn_metadata.query_lens).cumsum().tolist()
            actual_seq_kvlen = np.array(attn_metadata.seq_lens).cumsum().tolist()
            attn_output = torch_npu.npu_fusion_attention(
                query,
                all_key,
                all_value,
                head_num=self.num_heads,
                input_layout="TND",
                scale=self.scale,
                atten_mask=AscendAttentionBackendImpl.SHARE_MASK_TRIL_SPARSE,
                sparse_mode=3,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
            )[0]

            output = output.view_as(attn_output)
            output.copy_(attn_output)
            
        return output.view(num_tokens, self.hidden_size)


class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

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
    def init_kv_cache_each_layer(kv_cache_shape, dtype, device, model_config: "ModelConfig", enable_graph_mode) -> \
    tuple[torch.Tensor, ...]:
        # KVCache needs to store the shape of the reduced dimension [num_blocks, block_size, 1, kv_lora_rank] [num_blocks, block_size, 1, rope_dim]
        # The shape of the augmented dimension is [num_blocks, block_size, head_num, head_dim]
        layer_kv_caches = torch.zeros(kv_cache_shape,
                                      dtype=dtype,
                                      device=device)
        if not int(os.getenv("NO_NPU_MOCK", "0")) and device != "cpu":
            torch_npu.npu_format_cast(layer_kv_caches, 2)
        return (layer_kv_caches[0], layer_kv_caches[1])
