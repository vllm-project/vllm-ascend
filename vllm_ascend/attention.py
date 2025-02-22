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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("Failed to import torch_npu.")

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType,
                                              MLAAttentionImpl)
from vllm.attention.backends.utils import (CommonAttentionState,
                                           CommonMetadataBuilder,
                                           compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.utils import async_tensor_h2d, make_tensor_with_pad

if TYPE_CHECKING:
    from vllm_ascend.model_runner import ModelInputForNPUBuilder


def generate_attn_mask(max_seq_len: int, dtype=torch.float16):
    # Construct lower triangle matrix.
    mask_flag = torch.tril(
        torch.ones((max_seq_len, max_seq_len),
                   dtype=torch.bool)).view(max_seq_len, max_seq_len)
    # Create upper triangle matrix used to mark mask positions.
    mask_flag = ~mask_flag
    # Currently for fp16 dtype, the mask value should be set to -inf.
    # TODO: Eliminate this part in the future.
    if dtype == torch.float16:
        mask_value = torch.finfo(torch.float32).min
    else:
        mask_value = 1
    attn_mask = torch.masked_fill(torch.zeros(size=(max_seq_len, max_seq_len)),
                                  mask_flag, mask_value).to(dtype)
    return attn_mask


class AttentionMaskBuilder:

    def __init__(self, attn_mask: torch.Tensor):
        self._seq_len_cached = attn_mask.shape[0]
        self.attn_mask_cache = attn_mask

    @classmethod
    def initialize_from_len(cls,
                            max_seq_len: int,
                            dtype: torch.dtype = torch.float16):
        return cls(generate_attn_mask(max_seq_len, dtype))

    def update_attn_cache(self, seqlen: int, dtype: torch.dtype,
                          device: torch.device):
        if seqlen > self._seq_len_cached or self.attn_mask_cache.dtype != dtype:
            self._seq_len_cached = seqlen
            self.attn_mask_cache = generate_attn_mask(seqlen, dtype)
        if self.attn_mask_cache.device != device:
            self.attn_mask_cache = self.attn_mask_cache.to(device)

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype,
                      device: torch.device):
        self.update_attn_cache(max_seq_len, dtype, device)
        return self.attn_mask_cache[:max_seq_len, :max_seq_len].contiguous()

    def get_decode_attn_mask(
        self,
        input_lengths: torch.tensor,
        max_s: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.update_attn_cache(max_s, dtype, device)
        return (self.attn_mask_cache.index_select(
            0, input_lengths)[:, :max_s].view(-1, 1, max_s).contiguous())


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


class AscendMLAAttentionBackend(AscendAttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["AscendMLAAttentionBackendImpl"]:
        return AscendMLAAttentionBackendImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (1, num_blocks, block_size, num_kv_heads * head_size)


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

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

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

        # Construct & cache prefill-phase attention metadata structure.
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
            # metadata structure.
            return self._cached_decode_metadata

        # Compute some attn_metadata fields which default to None.
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])

        # Construct & cache decode-phase attention metadata structure.
        self._cached_decode_metadata = AscendMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
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
    _attn_mask_builder = None  # noqa

    def __init__(self, input_builder: "ModelInputForNPUBuilder"):
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

        self.attn_mask = None
        if AscendMetadataBuilder._attn_mask_builder is None:
            AscendMetadataBuilder._attn_mask_builder = AttentionMaskBuilder.initialize_from_len(
                128, self.input_builder.runner.model_config.dtype)

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
            compute_slot_mapping(
                is_profile_run,
                self.slot_mapping,
                seq_id,
                seq_len,
                context_len,
                start_idx,
                self.block_size,
                inter_data.block_tables,
            )

    def build(
        self,
        seq_lens: List[int],
        query_lens: List[int],
    ):
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

        if self.num_prefills > 0:
            self.attn_mask = AscendMetadataBuilder._attn_mask_builder.get_attn_mask(  # type: ignore
                max_prefill_seq_len,
                self.input_builder.runner.model_config.dtype,
                self.input_builder.runner.device)
        else:
            self.attn_mask = None

        block_tables = make_tensor_with_pad(
            self.block_tables,
            pad=0,
            dtype=torch.int32,
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
            enable_kv_scales_calculation=False,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=self.num_decode_tokens,
            seq_lens=seq_lens,
            max_query_len=max_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            block_tables=block_tables,
            attn_mask=self.attn_mask,
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
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")
        # View q k v to BSH.
        num_tokens = query.shape[0]
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        # TODO: Remove this contiguous in the future.
        value = value.contiguous()

        output = torch.empty(num_tokens,
                             self.num_heads,
                             self.head_size,
                             dtype=query.dtype,
                             device=query.device)

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
                torch_npu.npu_reshapecache(key=key,
                                           value=value,
                                           keyCache=key_cache,
                                           valueCache=value_cache,
                                           slotMapping=slots,
                                           compressType=0,
                                           kvCacheCfg=0)

            if attn_metadata.num_prefills > 0:

                if (attn_metadata.block_tables is None
                        or attn_metadata.block_tables.numel() == 0):
                    assert attn_metadata.attn_mask is not None
                    mask = attn_metadata.attn_mask
                    assert attn_metadata.prefill_metadata is not None
                    self.seq_lens_tensor_cpu = torch.from_numpy(
                        np.array(
                            attn_metadata.prefill_metadata.seq_lens).astype(
                                np.int32))
                    torch_npu.npu_selfattention(
                        query=query,
                        key=key,
                        value=value,
                        mask=mask,
                        maskType=1,
                        isTriuMask=0,
                        seqLen=self.seq_lens_tensor_cpu,
                        scale=self.scale,
                        qScale=1,
                        headNum=self.num_heads,
                        kvHeadNum=self.num_kv_heads,
                        mlaVHeadSize=0,
                        calcType=3,
                        kernelType=0,
                        clampType=0,
                        scaleType=0,
                        quantType=0,
                        cacheType=0,
                        batchRunStatusEnable=False,
                        kvcacheCfg=0,
                        clampMin=0,
                        clampMax=0,
                        inputLayout=0,
                        windowSize=0,
                        outDataType=0,
                        out=output)
                else:
                    # TODO: Will support prefix cache and chunked prefill soon.
                    raise RuntimeError(
                        "Prefix cache and chunked prefill are currently not supported."
                    )
            elif attn_metadata.decode_metadata:
                assert kv_cache is not None
                self.seq_lens_tensor_cpu = torch.from_numpy(
                    np.array(attn_metadata.decode_metadata.seq_lens).astype(
                        np.int32))
                block_tables = attn_metadata.decode_metadata.block_tables
                torch_npu.npu_pagedattention(
                    query=query,
                    keyCache=key_cache,
                    valueCache=value_cache,
                    contextLens=self.seq_lens_tensor_cpu,
                    maskType=0,
                    kvHeadNum=self.num_kv_heads,
                    headNum=self.num_heads,
                    mlaVHeadSize=0,
                    qkScale=self.scale,
                    scaleType=0,
                    blockTables=block_tables,
                    batchRunStatusEnable=False,
                    hasQuantOffset=False,
                    calcType=3,
                    quantType=0,
                    compressType=0,
                    inputLayout=0,
                    outDataType=0,
                    attnOut=output)

        return output.view(num_tokens, self.hidden_size)


class AscendMLAAttentionBackendImpl(MLAAttentionImpl):

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
        **extra_impl_args,
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

        # MLA Args
        self.q_lora_rank = extra_impl_args['q_lora_rank']
        self.kv_lora_rank = extra_impl_args['kv_lora_rank']
        self.qk_nope_head_dim = extra_impl_args['qk_nope_head_dim']
        self.qk_rope_head_dim = extra_impl_args['qk_rope_head_dim']
        self.qk_head_dim = extra_impl_args['qk_head_dim']
        self.v_head_dim = extra_impl_args['v_head_dim']
        self.rotary_emb = extra_impl_args['rotary_emb']
        self.q_proj = extra_impl_args['q_proj']
        self.kv_b_proj = extra_impl_args['kv_b_proj']
        self.o_proj = extra_impl_args['o_proj']
        self.w_kc = None
        self.w_vc = None

    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_q_c: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            hidden_states_or_q_c: shape = [num_tokens, num_heads * head_size]
                                           num_tokens = batch_size * seq_len
            kv_c_normed: shape = [num_tokens, num_kv_heads * head_size]
            k_pe: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache: shape = [1, num_blocks, block_size,
                               num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len * num_heads * head_size]
        """
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        attn_type = self.attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        num_tokens = hidden_states_or_q_c.shape[0]
        q = self.q_proj(hidden_states_or_q_c)[0].view(-1, self.num_heads,
                                                      self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)

        k_pe = k_pe.view(num_tokens, self.num_kv_heads, -1)

        if self.rotary_emb.__class__.__name__ == 'RotaryEmbedding':
            ori_q_pe_shape, ori_k_pe_shape = q_pe.shape, k_pe.shape
            q_pe = q_pe.reshape(num_tokens, -1)
            k_pe = k_pe.reshape(num_tokens, -1)
            q_pe, k_pe = self.rotary_emb(attn_metadata.input_positions, q_pe,
                                         k_pe)
            q_pe = q_pe.view(ori_q_pe_shape)
            k_pe = k_pe.view(ori_k_pe_shape)
        else:
            q_pe, k_pe = self.rotary_emb(attn_metadata.input_positions, q_pe,
                                         k_pe)

        if self.w_kc is None or self.w_vc is None:
            kv_b_proj_weight = self.kv_b_proj.weight.reshape(
                self.num_heads, self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank)
            self.w_kc = kv_b_proj_weight[:, :self.
                                         qk_nope_head_dim, :].contiguous()
            self.w_vc = kv_b_proj_weight[:,
                                         self.qk_nope_head_dim:, :].transpose(
                                             1, 2).contiguous()

        if attn_metadata.num_prefills > 0:
            kv_heads_num = self.num_heads
            kv = self.kv_b_proj(kv_c_normed)[0].view(num_tokens, kv_heads_num,
                                                     -1)
            k_nope, value = kv.split([self.qk_nope_head_dim, self.v_head_dim],
                                     dim=-1)
            k_cache = torch.cat(
                [kv_c_normed.view(num_tokens, self.num_kv_heads, -1), k_pe],
                dim=2)
            k_pe = k_pe.expand(-1, self.num_heads, -1)
            key = torch.cat([k_nope.view(num_tokens, kv_heads_num, -1), k_pe],
                            dim=2)
        else:
            kv_heads_num = self.num_kv_heads
            q_nope_t = torch.transpose(q_nope, 0, 1)
            q_nope_out = torch.bmm(q_nope_t, self.w_kc)
            q_nope = torch.transpose(q_nope_out, 0, 1)
            k_cache = torch.cat(
                [kv_c_normed.view(num_tokens, self.num_kv_heads, -1), k_pe],
                dim=2)

        query = torch.cat([q_nope, q_pe], dim=-1).view(num_tokens,
                                                       self.num_heads, -1)

        if kv_cache.numel() > 0:
            key_cache = kv_cache[0]
            num_blocks, block_size, _ = key_cache.shape

            key_cache = key_cache.view(
                num_blocks, block_size, self.num_kv_heads,
                self.qk_rope_head_dim + self.kv_lora_rank)
            slots = attn_metadata.slot_mapping
            torch_npu.npu_reshapecache(key=k_cache,
                                       value=None,
                                       keyCache=key_cache,
                                       valueCache=None,
                                       slotMapping=slots,
                                       compressType=0,
                                       kvCacheCfg=1)

        if attn_metadata.num_prefills > 0:
            attn_output = torch.empty(num_tokens,
                                      self.num_heads,
                                      self.v_head_dim,
                                      dtype=query.dtype,
                                      device="npu")
            if (attn_metadata.block_tables is None
                    or attn_metadata.block_tables.numel() == 0):
                assert attn_metadata.attn_mask is not None
                mask = attn_metadata.attn_mask
                assert attn_metadata.prefill_metadata is not None
                assert attn_metadata.prefill_metadata.seq_lens is not None
                self.seq_lens_tensor_cpu = torch.from_numpy(
                    np.array(attn_metadata.prefill_metadata.seq_lens).astype(
                        np.int32))
                torch_npu.npu_selfattention(query=query,
                                            key=key,
                                            value=value,
                                            kvcacheCfg=0,
                                            mask=mask,
                                            maskType=1,
                                            isTriuMask=0,
                                            seqLen=self.seq_lens_tensor_cpu,
                                            scale=self.scale,
                                            qScale=1,
                                            scaleType=0,
                                            headNum=self.num_heads,
                                            kvHeadNum=self.num_heads,
                                            mlaVHeadSize=0,
                                            calcType=3,
                                            kernelType=0,
                                            clampType=0,
                                            quantType=0,
                                            cacheType=0,
                                            windowSize=0,
                                            clampMin=0,
                                            clampMax=0,
                                            batchRunStatusEnable=False,
                                            inputLayout=0,
                                            outDataType=0,
                                            out=attn_output)
            else:
                # TODO: Will support prefix cache and chunked prefill soon.
                raise RuntimeError(
                    "Prefix cache and chunked prefill are currently not supported."
                )
        elif attn_metadata.decode_metadata:
            assert kv_cache is not None
            attn_output = torch.empty(num_tokens,
                                      self.num_heads,
                                      self.kv_lora_rank,
                                      dtype=query.dtype,
                                      device="npu")
            self.seq_lens_tensor_cpu = torch.from_numpy(
                np.array(attn_metadata.decode_metadata.seq_lens).astype(
                    np.int32))
            block_tables = attn_metadata.decode_metadata.block_tables
            torch_npu.npu_pagedattention(query=query,
                                         keyCache=key_cache,
                                         valueCache=None,
                                         contextLens=self.seq_lens_tensor_cpu,
                                         maskType=0,
                                         kvHeadNum=self.num_kv_heads,
                                         headNum=self.num_heads,
                                         mlaVHeadSize=self.kv_lora_rank,
                                         qkScale=self.scale,
                                         blockTables=block_tables,
                                         batchRunStatusEnable=False,
                                         hasQuantOffset=False,
                                         compressType=0,
                                         calcType=0,
                                         scaleType=0,
                                         quantType=0,
                                         inputLayout=0,
                                         outDataType=-1,
                                         attnOut=attn_output)
            attn_output_t = torch.transpose(attn_output, 0, 1)
            attn_output_t = torch.bmm(attn_output_t, self.w_vc)
            attn_output = torch.transpose(attn_output_t, 0, 1)

        output, _ = self.o_proj(attn_output.reshape(num_tokens, -1))

        return output
