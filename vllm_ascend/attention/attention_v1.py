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
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils import direct_register_custom_op
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_ascend.ops.attention import vanilla_chunked_prefill
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, aligned_16, is_310p,
                               nd_to_nz_2d, nd_to_nz_spec, vllm_version_is)


class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        if vllm_version_is("0.9.2"):
            return AscendAttentionBackendImpl092
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
        if is_310p():
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


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
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
    # TODO: Indicates whether there are only prefill requests.
    # FlashAttention can be used when there are only prefill requests.
    # FlashAttention has better performance than PageAtttention,
    # but it does not support decode requests.
    is_only_prefill: bool = False
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    attn_mask: Optional[torch.Tensor] = None

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    with_prefill_across_dp: bool = False


class AscendAttentionMetadataBuilder:

    def __init__(self, runner):
        self.runner = runner

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(self,
              num_reqs,
              num_actual_tokens,
              max_query_len,
              common_prefix_len,
              max_num_tokens_across_dp: int = 0,
              with_prefill_across_dp: bool = False):

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
        query_start_loc_cpu = self.runner.query_start_loc_cpu[:num_reqs + 1]
        query_start_loc = query_start_loc_cpu.to(self.runner.device,
                                                 non_blocking=True)

        if is_310p():
            if attn_state == AscendAttentionState.PrefillNoCache:
                mask_nz = nd_to_nz_2d(attn_mask)
                attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(),
                                                      ACL_FORMAT_FRACTAL_NZ)
            elif attn_state == AscendAttentionState.ChunkedPrefill:
                mask_nz = nd_to_nz_spec(attn_mask)
                attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(),
                                                      ACL_FORMAT_FRACTAL_NZ)

        attn_metadata = AscendMetadata(
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
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
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
        trace_flag: bool = True,
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
        use_kv_cache_int8 = kv_cache.numel(
        ) > 0 and kv_cache[0].dtype == torch.int8
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)
        ori_output = output
        if trace_flag:
            torch.ops.vllm.unified_ascend_attention_with_output(
                query=query,
                key=key,
                value=value,
                output=output,
                layer_name=layer.layer_name)

        elif hasattr(layer, 'quant_method') and use_kv_cache_int8:
            output = layer.quant_method.apply(layer, query, key, value,
                                              kv_cache, attn_metadata,
                                              self.attn_type, self.scale,
                                              output)

        else:
            if attn_metadata is None:
                return output.view(num_tokens, self.hidden_size)
            num_actual_tokens = attn_metadata.num_actual_tokens
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
                torch_npu._npu_reshape_and_cache(
                    key=key[:num_actual_tokens],
                    value=value[:num_actual_tokens],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=slots)

            # V0-Style scheduler situation.
            if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                assert attn_metadata is not None
                assert attn_metadata.attn_mask is not None
                mask = attn_metadata.attn_mask
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
                if is_310p():
                    # # seq_lens_tensor needs to be transferred to the device for 310P
                    attn_metadata.seq_lens = \
                        attn_metadata.seq_lens.to(device=query.device)
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
            # Normal V1 situation.
            else:
                # use chunked prefill for head size 192 scenario, like deepseek
                # paged_attention_splitfuse maybe crash at such scenario
                # TODO: vanilla path will be removed after the kernel support
                # head_size 192 scenario
                if self.head_size == 192:
                    cu_seqlen_q = [0] + attn_metadata.query_lens.tolist()
                    cu_seqlen_k = [0] + attn_metadata.seq_lens.tolist()
                    cu_seqlen_q = torch.tensor(cu_seqlen_q, device="npu")
                    cu_seqlen_k = torch.tensor(cu_seqlen_k, device="npu")
                    cu_seqlen_q = torch.cumsum(cu_seqlen_q, dim=0)
                    cu_seqlen_k = torch.cumsum(cu_seqlen_k, dim=0)
                    max_seqlen_q = torch.max(attn_metadata.query_lens)
                    max_seqlen_k = torch.max(attn_metadata.seq_lens)
                    vanilla_chunked_prefill(output, query, self.key_cache,
                                            self.value_cache,
                                            attn_metadata.block_tables,
                                            cu_seqlen_q, cu_seqlen_k,
                                            max_seqlen_q, max_seqlen_k,
                                            self.scale, None, True)
                else:
                    # use paged attention
                    assert attn_metadata is not None
                    assert attn_metadata.attn_mask is not None
                    if is_310p():
                        # do reformat in case of broadcasted tensors
                        attn_metadata.attn_mask = \
                            torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(), ACL_FORMAT_FRACTAL_NZ)
                        attn_metadata.seq_lens = \
                            attn_metadata.seq_lens.to(device=query.device)
                    torch_npu._npu_paged_attention_splitfuse(
                        query=query,
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        mask=attn_metadata.attn_mask,
                        block_table=attn_metadata.block_tables,
                        seq_len=attn_metadata.query_lens,
                        context_lens=attn_metadata.seq_lens,
                        num_kv_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        scale_value=self.scale,
                        out=output)

        # to make in-place change to the output tensor
        if hasattr(layer, 'quant_method') and use_kv_cache_int8:
            output = output.view(num_tokens, self.num_heads, self.head_size)
        ori_output[:, :, :] = output[:num_tokens, :, :]
        return output.view(num_tokens, self.hidden_size)


class AscendAttentionBackendImpl092(AscendAttentionBackendImpl):

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


def unified_ascend_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
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
