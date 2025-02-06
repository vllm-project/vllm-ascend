#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/attention/backends
# Copyright 2023 The vLLM team.
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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (PAD_SLOT_ID, CommonAttentionState,
                                           CommonMetadataBuilder,
                                           compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.utils import async_tensor_h2d, make_tensor_with_pad

if TYPE_CHECKING:
    from vllm_ascend.model_runner import ModelInputForNPUBuilder

SHARE_MASK_TRIL_PREFIX_CACHE = None
SHARE_MASK_TRIL = None


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

    @staticmethod
    def get_builder_cls() -> Type["AscendMetadataBuilder"]:
        return AscendMetadataBuilder

    @classmethod
    def make_metadata_builder(cls, *args, **kwargs) -> "AscendMetadataBuilder":
        return cls.get_builder_cls()(*args, **kwargs)


@dataclass
class AscendMetadata(AttentionMetadata):
    """Metadata for Ascendbackend.
        * modified from XFormersbackend
    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|


    # FIXME: It is for flash attn.
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    block_tables: Optional[torch.Tensor]

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["AscendMetadata"] = None
    _cached_decode_metadata: Optional["AscendMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    attn_mask: Optional[torch.Tensor] = None
    pse_shift: Optional[torch.Tensor] = None
    sparse_mode: int = 0

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    # slot_mapping: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional["AscendMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))

        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])

        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = AscendMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
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
            # metadata structure
            return self._cached_decode_metadata

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])

        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = AscendMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
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


class AscendMetadataBuilder(CommonMetadataBuilder[AscendMetadata]):

    _metadata_cls = AscendMetadata


    def _add_seq_group(
            self, inter_data: "ModelInputForNPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables
        max_query_len = max(
            max(data.query_lens)
            for data in self.input_builder.inter_data_list)

        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table: List[int] = []
            prefix_cache_hit = any([
                inter_data.prefix_cache_hit
                for inter_data in self.input_builder.inter_data_list
            ])
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                if block_tables is not None:
                    block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)

            compute_slot_mapping(is_profile_run, self.slot_mapping,
                                          seq_id, seq_len, context_len,
                                          start_idx, self.block_size,
                                          inter_data.block_tables)
            range_start = max(start_idx, context_len)
            range_end = seq_len
            numel = range_end - range_start
            self.slot_mapping.extend([PAD_SLOT_ID] * (max_query_len - numel))


    def build(self, seq_lens: List[int], query_lens: List[int]):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
        """
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled)

        device = self.runner.device

        max_query_len = max(query_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens

        
        block_tables = make_tensor_with_pad(
            self.block_tables,
            pad=0,
            dtype=torch.int,
            device=device,
        )
        assert max_query_len > 0, "query_lens: {}".format(query_lens)

        assert device is not None
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.int32,
                                               device, self.runner.pin_memory)
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            self.multimodal_placeholder_maps.items()
        }

        return self._metadata_cls(  # type: ignore
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=True,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            max_query_len=max_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            block_tables=block_tables,
        )


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

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: List[torch.Tensor],
        attn_metadata: AscendMetadata,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads * head_size]
                   num_tokens = batch_size * seq_len
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache: shape = [2, num_blocks, block_size,
                               num_kv_heads * head_size]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads * head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len * num_heads * head_size]
        """
        assert layer._k_scale == 1.0 and layer._v_scale == 1.0
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")
        # view q k v to BSH
        num_tokens = query.shape[0]

        if kv_cache is not None and len(kv_cache) >= 2:
            slots = attn_metadata.slot_mapping
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            num_blocks, block_size, _ = key_cache.shape
            key_tmp = key.view(num_tokens, self.num_kv_heads, self.head_size)
            value_tmp = value.view(num_tokens, self.num_kv_heads, self.head_size)
            key_cache_tmp = key_cache.view(num_blocks, block_size, self.num_kv_heads,
                                   self.head_size)
            value_cache_tmp = value_cache.view(num_blocks, block_size,
                                       self.num_kv_heads, self.head_size)
            torch_npu.npu_reshapecache(key_tmp, value_tmp, key_cache_tmp, value_cache_tmp, slots)

        if attn_metadata.num_prefills > 0:
            if attn_metadata.attn_mask is None:
                if num_tokens > 16384:
                    attn_metadata.sparse_mode = 2
                attention_mask = gen_input_mask(
                    attn_metadata.max_prefill_seq_len, self.sliding_window,
                    num_tokens)
                attn_metadata.attn_mask = attention_mask

            if (self.alibi_slopes is not None
                    and attn_metadata.pse_shift is None):
                attn_metadata.pse_shift = _make_alibi_bias(
                    self.alibi_slopes,
                    self.num_kv_heads,
                    dtype=query.dtype,
                    seq_len=attn_metadata.max_prefill_seq_len,
                    batch_size=num_tokens,
                )

            if (len(kv_cache) == 0 or attn_metadata.block_tables is None
                    or attn_metadata.block_tables.numel() == 0):
                max_seq_len = attn_metadata.max_prefill_seq_len

                # shape of q/k/v [B,S*H] --> [B,S,N,D]
                query = query.view(-1, max_seq_len, self.num_heads,
                                   self.head_size).transpose(1, 2)
                key = key.view(-1, max_seq_len, self.num_kv_heads,
                               self.head_size).transpose(1, 2)
                value = value.view(-1, max_seq_len, self.num_kv_heads,
                                   self.head_size).transpose(1, 2)
                # FA for prefill phase
                output = torch_npu.npu_prompt_flash_attention(
                    query,
                    key,
                    value,
                    pse_shift=attn_metadata.pse_shift,
                    atten_mask=attn_metadata.attn_mask,
                    num_heads=self.num_heads,
                    scale_value=1 / math.sqrt(self.head_size),
                    input_layout="BNSD",
                    num_key_value_heads=self.num_kv_heads,
                    pre_tokens=65535,
                    next_tokens=0,
                    sparse_mode=attn_metadata.sparse_mode,
                )
                # reshape to [B,H]
                output = output.transpose(1, 2).reshape(
                    num_tokens, self.num_heads * self.head_size)
            else:
                # prefix-enabled attention
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support prefix caching")
                assert attn_metadata.seq_lens is not None
                assert kv_cache is not None
                query = query.view(query.shape[0], -1,
                                   self.num_heads * self.head_size)
                output = torch.zeros(query.shape,
                                     device="npu",
                                     dtype=query.dtype)
                # TODO (Mengqing Cao): torch_npu.npu_incre_flash_attention
                # support only when `S == 1`, OPTIMIZE ME when prefix caching
                # is supported in torch-npu ops.
                for i in range(query.shape[0]):
                    # FA for prefill phase
                    output[i] = torch_npu.npu_incre_flash_attention(
                        query[i].unsqueeze(0),
                        key_cache,
                        value_cache,
                        num_heads=self.num_heads,
                        num_key_value_heads=self.num_kv_heads,
                        scale_value=self.scale,
                        input_layout="BSH",
                        block_table=attn_metadata.block_tables,
                        block_size=key_cache.
                        shape[1],  # max val of block_size == 512
                        actual_seq_lengths=attn_metadata.seq_lens,
                    )
                # [B,S,H] --> [B,H]
                output = output.squeeze(1)

        elif attn_metadata.decode_metadata:
            # FA for decoding phase
            assert kv_cache is not None
            # shape of query [B,S*H] --> [B,S,H]
            query = query.view(
                -1,
                1,
                self.head_size * self.num_heads,
            )
            output = torch_npu.npu_incre_flash_attention(
                query,
                key_cache,
                value_cache,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                scale_value=self.scale,
                input_layout="BSH",
                block_table=attn_metadata.block_tables,
                block_size=key_cache.shape[1],  # max val of block_size == 512
                actual_seq_lengths=attn_metadata.seq_lens,
            )

            # [B,S,H] --> [B,H]
            output = output.squeeze(1)
        return output


def gen_input_mask(seq_len, sliding_window, len):
    """
    Generating lower triangular matrix
    """
    if len > 16384:
        # improve computing performance on NPU when input tokens are huge
        global SHARE_MASK_TRIL_PREFIX_CACHE
        if SHARE_MASK_TRIL_PREFIX_CACHE is None:
            SHARE_MASK_TRIL_PREFIX_CACHE = torch.triu(
                torch.ones(1, 1, 2048, 2048, dtype=bool, device="npu"),
                diagonal=1,
            )
        attention_mask = SHARE_MASK_TRIL_PREFIX_CACHE
    else:
        global SHARE_MASK_TRIL
        if SHARE_MASK_TRIL is None or SHARE_MASK_TRIL.shape[0] < seq_len:
            SHARE_MASK_TRIL = ~torch.tril(
                torch.ones(seq_len, seq_len, dtype=bool, device="npu"))

        attention_mask = SHARE_MASK_TRIL
        if sliding_window is not None:
            attention_mask = ~attention_mask
            attention_mask = torch.triu(attention_mask,
                                        diagonal=1 - sliding_window)
            attention_mask = ~attention_mask

    return attention_mask


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_len: int,
    batch_size: int,
):
    bias = torch.arange(seq_len, dtype=dtype, device=alibi_slopes.device)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(seq_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    # Calculate a matrix where each element represents ith element- jth
    # element.
    bias = bias[None, :] - bias[:, None]

    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        1,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))

    return bias
