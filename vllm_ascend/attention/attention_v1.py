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
from typing import List, Optional, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.config import VllmConfig
from vllm.distributed import (get_context_model_parallel_rank,
                              get_context_model_parallel_world_size,
                              get_cp_group)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          is_v1_kv_transfer_group)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils import cdiv, direct_register_custom_op
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         split_decodes_and_prefills)
from vllm_ascend.ops.attention import vanilla_chunked_prefill
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, aligned_16, is_310p,
                               nd_to_nz_2d, nd_to_nz_spec)
from vllm_ascend.worker.npu_input_batch import InputBatch


def wait_for_kv_layer_from_connector(layer_name: str):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert ascendMetadata
    connector.wait_for_layer_load(layer_name)


def maybe_save_kv_layer_to_connector(
    layer_name: str,
    kv_cache_layer: List[torch.Tensor],
):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert ascendMetadata
    connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)


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
class AscendCpMetadata:
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
    cp_prefill_mask: torch.Tensor = None


@dataclass
class AscendPrefillMetadata:
    """ Prefill Specific Metadata for Ascend"""
    cp_metadata: Optional[AscendCpMetadata] = None
    cp_kv_recover_idx: Optional[List[int]] = None


@dataclass
class AscendMetadata:
    # **************************** Basic Properties ************************** #
    attn_mask: Optional[torch.Tensor] = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens: int = 0
    num_prefills: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    seq_lens: torch.Tensor = None

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

    # *************************** Other Properties *************************** #
    enable_dbo_across_dp: bool = False
    is_only_prefill: bool = False

    prefill: Optional[AscendPrefillMetadata] = None


class AscendAttentionMetadataBuilder:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(self.model_config.max_model_len,
                                           vllm_config.cache_config.block_size)

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: nn.Module,
    ):
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        decode_threshold = 1
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=decode_threshold)
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_actual_tokens

        block_table = common_attn_metadata.block_table_tensor
        block_table[:num_reqs, :self.max_num_blocks_per_req] = (
            block_table[:num_reqs])

        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        long_seq_metadata = common_attn_metadata.common_long_seq_metadata
        num_actual_tokens_cp_full = long_seq_metadata.num_actual_tokens_cp_full if long_seq_metadata else None
        if num_actual_tokens_cp_full is None:
            num_actual_tokens_cp_full = num_actual_tokens
        slot_mapping = common_attn_metadata.slot_mapping_cpu[:
                                                             num_actual_tokens_cp_full].to(
                                                                 self.device,
                                                                 non_blocking=
                                                                 True)

        attn_mask = common_attn_metadata.attn_mask
        attn_state = common_attn_metadata.attn_state

        query_start_loc = query_start_loc_cpu.to(self.device,
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

        prefill_metadata = None
        if num_prefills > 0:
            cp_metadata = None
            common_long_seq_metadata = common_attn_metadata.common_long_seq_metadata
            if common_long_seq_metadata is not None:
                cp_metadata = AscendCpMetadata(
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
                    attn_mask_seqlens=common_long_seq_metadata.
                    attn_mask_seqlens,
                    head_attn_nomask_seqlens=common_long_seq_metadata.
                    head_attn_nomask_seqlens,
                    tail_attn_nomask_seqlens=common_long_seq_metadata.
                    tail_attn_nomask_seqlens,
                    q_full_idx=common_long_seq_metadata.q_full_idx,
                    cp_prefill_mask=common_long_seq_metadata.cp_prefill_mask)
            prefill_metadata = AscendPrefillMetadata(
                cp_metadata=cp_metadata,
                cp_kv_recover_idx=common_long_seq_metadata.cp_kv_recover_idx
                if common_long_seq_metadata is not None else None)

        if num_decodes > 0:
            seq_lens = seq_lens[:num_decode_tokens]
            block_table = block_table[:num_decode_tokens, ...]
        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_lens=query_lens,
            seq_lens=seq_lens,
            max_query_len=common_attn_metadata.max_query_len,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            enable_dbo_across_dp=common_attn_metadata.enable_dbo_across_dp,
            is_only_prefill=common_attn_metadata.is_only_prefill,
            prefill=prefill_metadata)
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

        self.cp_size = get_context_model_parallel_world_size()
        self.cp_rank = get_context_model_parallel_rank() if self.cp_size > 1 else 0
        self.cp_group = get_cp_group().device_group if self.cp_size > 1 else None

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
        assert output is not None
        return output[:num_tokens, :, :]

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
        if is_310p():
            # seq_lens_tensor needs to be transferred to the device for 310P.
            attn_metadata.seq_lens = \
                attn_metadata.seq_lens.to(device=query.device)
        if self.sliding_window is not None:
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

        if is_310p():
            # Do reformat in case of broadcasted tensors.
            attn_metadata.attn_mask = \
                torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(),
                                          ACL_FORMAT_FRACTAL_NZ)
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
        return output

    def _pack_tnd_2_bsnd(self, tensor_tnd: torch.Tensor,
                         lengths: List[int]) -> torch.Tensor:
        max_len = max(lengths)
        splits = torch.split(tensor_tnd, lengths, dim=0)

        padded = []
        for s in splits:
            pad_len = max_len - s.shape[0]
            s_pad = F.pad(s, (0, 0, 0, 0, 0, pad_len))
            padded.append(s_pad)

        tensor_bsnd = torch.stack(padded, dim=0)
        return tensor_bsnd

    def _unpack_bsnd_2_tnd(self, tensor_bsnd: torch.Tensor,
                           lengths: List[int]) -> torch.Tensor:
        slices = []
        for i, length in enumerate(lengths):
            slices.append(tensor_bsnd[i, :length])
        tensor_tnd = torch.cat(slices, dim=0)
        return tensor_tnd

    def _attention_with_nomask_and_mask(self, q: torch.Tensor,
                                        q_seqlens: List[int],
                                        k_nomask: torch.Tensor,
                                        v_nomask: torch.Tensor,
                                        kv_seqlens_nomask: List[int],
                                        k_mask: torch.Tensor,
                                        v_mask: torch.Tensor,
                                        kv_seqlens_mask: List[int],
                                        mask: torch.Tensor) -> torch.Tensor:
        q = self._pack_tnd_2_bsnd(q, q_seqlens)

        # nomask Attention
        if k_nomask is not None:
            attn_out_nomask, attn_lse_nomask = torch.ops.npu.npu_fused_infer_attention_score(
                q,
                self._pack_tnd_2_bsnd(k_nomask, kv_seqlens_nomask),
                self._pack_tnd_2_bsnd(v_nomask, kv_seqlens_nomask),
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BSND",
                atten_mask=None,
                scale=self.scale,
                sparse_mode=0,
                antiquant_mode=0,
                antiquant_scale=None,
                softmax_lse_flag=True,
                actual_seq_lengths_kv=kv_seqlens_nomask,
                actual_seq_lengths=q_seqlens)
            attn_out_nomask = self._unpack_bsnd_2_tnd(attn_out_nomask,
                                                      q_seqlens)
            # (B, N, Q_S, 1) -> (B, Q_S, N, 1) -> (T, N, 1)
            attn_lse_nomask = self._unpack_bsnd_2_tnd(
                attn_lse_nomask.permute([0, 2, 1, 3]), q_seqlens)

        # mask Attention
        attn_out_mask, attn_lse_mask = torch.ops.npu.npu_fused_infer_attention_score(
            q,
            self._pack_tnd_2_bsnd(k_mask, kv_seqlens_mask),
            self._pack_tnd_2_bsnd(v_mask, kv_seqlens_mask),
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSND",
            atten_mask=mask,
            scale=self.scale,
            sparse_mode=0,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=True,
            actual_seq_lengths_kv=kv_seqlens_mask,
            actual_seq_lengths=q_seqlens)
        attn_out_mask = self._unpack_bsnd_2_tnd(attn_out_mask, q_seqlens)
        attn_lse_mask = self._unpack_bsnd_2_tnd(
            attn_lse_mask.permute([0, 2, 1, 3]), q_seqlens)

        # update
        output = attn_out_mask
        if k_nomask is not None:
            output, _ = self._update_out_and_lse(
                torch.stack([attn_out_nomask, attn_out_mask], dim=0),
                torch.stack([attn_lse_nomask, attn_lse_mask], dim=0))

        return output

    def _forward_prefill_cp(self, query: torch.Tensor, key: torch.Tensor,
                            value: torch.Tensor,
                            attn_metadata: AscendMetadata) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.prefill is not None
        assert attn_metadata.prefill.cp_metadata is not None
        # Use precomputed indices from the metadata (already converted to tensors and on device)
        q_head_idx = attn_metadata.prefill.cp_metadata.q_head_idx
        q_tail_idx = attn_metadata.prefill.cp_metadata.q_tail_idx
        kv_with_q_head_nomask_idx = attn_metadata.prefill.cp_metadata.kv_with_q_head_nomask_idx
        kv_with_q_head_mask_idx = attn_metadata.prefill.cp_metadata.kv_with_q_head_mask_idx
        kv_with_q_tail_nomask_idx = attn_metadata.prefill.cp_metadata.kv_with_q_tail_nomask_idx
        kv_with_q_tail_mask_idx = attn_metadata.prefill.cp_metadata.kv_with_q_tail_mask_idx
        attn_mask_seqlens = attn_metadata.prefill.cp_metadata.attn_mask_seqlens
        head_attn_nomask_seqlens = attn_metadata.prefill.cp_metadata.head_attn_nomask_seqlens
        tail_attn_nomask_seqlens = attn_metadata.prefill.cp_metadata.tail_attn_nomask_seqlens
        mask = attn_metadata.prefill.cp_metadata.cp_prefill_mask

        # 1. Attention calculation in the first half of Q in load balancing
        output_head = self._attention_with_nomask_and_mask(
            q=torch.index_select(query, 0, q_head_idx),
            q_seqlens=attn_mask_seqlens[0].tolist(),
            k_nomask=torch.index_select(key, 0, kv_with_q_head_nomask_idx)
            if self.cp_rank > 0 else None,
            v_nomask=torch.index_select(value, 0, kv_with_q_head_nomask_idx)
            if self.cp_rank > 0 else None,
            kv_seqlens_nomask=head_attn_nomask_seqlens[1].tolist(),
            k_mask=torch.index_select(key, 0, kv_with_q_head_mask_idx),
            v_mask=torch.index_select(value, 0, kv_with_q_head_mask_idx),
            kv_seqlens_mask=attn_mask_seqlens[0].tolist(),
            mask=mask)

        # 2. the Attention calculation in the latter half of Q in load balancing
        # cp_rank0: Q3*KV0~KV2 + Q3*KV3
        # cp_rank1: Q2*KV0~KV1 + Q2*KV2
        output_tail = self._attention_with_nomask_and_mask(
            q=torch.index_select(query, 0, q_tail_idx),
            q_seqlens=attn_mask_seqlens[0].tolist(),
            k_nomask=torch.index_select(key, 0, kv_with_q_tail_nomask_idx),
            v_nomask=torch.index_select(value, 0, kv_with_q_tail_nomask_idx),
            kv_seqlens_nomask=tail_attn_nomask_seqlens[1].tolist(),
            k_mask=torch.index_select(key, 0, kv_with_q_tail_mask_idx),
            v_mask=torch.index_select(value, 0, kv_with_q_tail_mask_idx),
            kv_seqlens_mask=attn_mask_seqlens[0].tolist(),
            mask=mask)

        # 3. Combine the output of the first half and second half.
        q_full_idx = attn_metadata.prefill.cp_metadata.q_full_idx
        output = torch.index_select(
            torch.cat([output_head, output_tail], dim=0), 0, q_full_idx)
        return output

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

    def _forward_decode_cp(self, query: torch.Tensor,
                           attn_metadata: AscendMetadata) -> torch.Tensor:
        assert self.key_cache is not None
        assert self.value_cache is not None

        # 1. Compute out&lse by "npu_fused_infer_attention_score"
        attn_out, attn_lse = torch.ops.npu.npu_fused_infer_attention_score(
            query.view(query.shape[0], 1, query.shape[1], query.shape[2]),
            # [b,num_heads,head_size] -> [b,1,num_heads,head_size]
            self.key_cache.view(self.key_cache.shape[0],
                                self.key_cache.shape[1], -1),
            self.value_cache.view(self.key_cache.shape[0],
                                  self.key_cache.shape[1], -1),
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSND",
            atten_mask=None,
            scale=self.scale,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=True,
            block_table=attn_metadata.block_tables,
            block_size=self.key_cache.shape[1],
            actual_seq_lengths_kv=attn_metadata.seq_lens,
        )
        attn_out = attn_out.view(attn_out.shape[0], attn_out.shape[2],
                                 attn_out.shape[3])
        attn_lse = attn_lse.view(attn_lse.shape[0], attn_lse.shape[1], 1)
        # 2. Concat out&lse: [bs,num_heads,head_size] + [bs,num_heads,1] -> [bs,num_heads,head_size+1]
        attn_out_lse = torch.cat([attn_out, attn_lse], dim=-1)
        # 3. AllGather out&lse within CP group
        attn_out_lse_list = [
            torch.empty_like(attn_out_lse) for _ in range(self.cp_size)
        ]
        dist.all_gather(attn_out_lse_list, attn_out_lse, group=self.cp_group)
        # 4. Update out&lse
        attn_out_lse_allgather = torch.stack(
            attn_out_lse_list,
            dim=0)  # [cp, batch_size, num_heads, head_size+1]
        attn_out_allgather, attn_lse_allgather = torch.split(
            attn_out_lse_allgather, [self.head_size, 1], dim=-1)
        output, _ = self._update_out_and_lse(attn_out_allgather,
                                             attn_lse_allgather)
        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache: shape = [key_cache, value_cache]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads, head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * seq_len, num_heads, head_size]
        """
        num_tokens = query.shape[0]
        use_kv_cache_int8 = len(
            kv_cache) > 0 and kv_cache[0].dtype == torch.int8
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

            if self.cp_size > 1 and attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                kv = torch.cat([key, value], dim=-1)  # []
                kv_list = [torch.empty_like(kv) for _ in range(self.cp_size)]
                dist.all_gather(kv_list, kv, self.cp_group)
                all_kv = torch.cat(kv_list, dim=0)
                cp_kv_recover_idx = attn_metadata.prefill.cp_kv_recover_idx if attn_metadata.prefill else None
                all_kv = torch.index_select(all_kv, 0, cp_kv_recover_idx)
                key, value = all_kv.split([self.head_size, self.head_size],
                                          dim=-1)

            if len(kv_cache) > 1:
                if self.key_cache is None:
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
                slots = attn_metadata.slot_mapping
                torch_npu._npu_reshape_and_cache(
                    key=key if self.cp_size > 1 else key[:num_actual_tokens],
                    value=value
                    if self.cp_size > 1 else value[:num_actual_tokens],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=slots)

            # V0-Style scheduler situation.
            if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                if self.cp_size > 1:
                    output = self._forward_prefill_cp(query, key, value,
                                                      attn_metadata)
                else:
                    output = self._forward_prefill_no_cache(
                        query, key, value, attn_metadata, output, num_tokens)
            elif attn_metadata.attn_state == \
                AscendAttentionState.PrefillCacheHit:
                output = self._forward_prefill_cache_hit(
                    query, attn_metadata, output)
            elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                if self.cp_size > 1:
                    output = self._forward_decode_cp(query, attn_metadata)
                else:
                    output = self._forward_decode_only(query, attn_metadata,
                                                       output)
            # Normal V1 situation.
            else:
                output = self._forward_v1_style(query, attn_metadata, output)

        # to make in-place change to the output tensor
        if hasattr(layer, 'quant_method') and use_kv_cache_int8:
            output = output.view(num_tokens, self.num_heads, self.head_size)
        ori_output[:, :, :] = output[:num_tokens, :, :]
        return output.view(num_tokens, self.hidden_size)


def unified_ascend_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)
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
    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
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
