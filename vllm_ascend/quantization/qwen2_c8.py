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
"""C8 KV Cache quantization method for Qwen2/Qwen3 models."""

from typing import List, Optional

import torch
import torch_npu
from vllm.attention.backends.abstract import AttentionType
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, AscendDeviceType,
                               get_ascend_device_type)


def quant_per_tensor(in_tensor: torch.Tensor,
                     input_scale: torch.Tensor,
                     input_offset: Optional[torch.Tensor],
                     function=False):
    """Quantize tensor to int8."""
    return torch_npu.npu_quantize(in_tensor, input_scale, input_offset,
                                  torch.qint8, -1, function)


class Qwen2C8KVCacheMethod(BaseKVCacheMethod):
    """C8 KV Cache quantization method for Qwen2/Qwen3 models with fused qkv_proj."""

    def __init__(self, quant_config=None, prefix: str = "") -> None:
        """Initialize Qwen2 C8 KV Cache method."""
        self.antiquant_scale_comb = None
        if quant_config and hasattr(quant_config, 'quant_description'):
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
            self.params_dtype = vllm_config.model_config.dtype
        else:
            self.params_dtype = torch.bfloat16

    def create_weights(self, layer) -> None:
        """Create KV cache quantization parameters for Qwen2/Qwen3 Attention layer."""
        param_dict = {}
        scale_dtype = self.params_dtype

        param_dict["key_antiquant_scale"] = torch.empty(layer.num_kv_heads *
                                                        layer.head_size,
                                                        dtype=scale_dtype,
                                                        requires_grad=False)
        param_dict["value_antiquant_scale"] = torch.empty(layer.num_kv_heads *
                                                          layer.head_size,
                                                          dtype=scale_dtype,
                                                          requires_grad=False)

        for weight_name, weight_param in param_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            layer.register_parameter(weight_name, param)

    def process_weights_after_loading(self, layer):
        """Process weights after loading from checkpoint."""
        self.antiquant_scale_comb = torch.cat(
            (layer.key_antiquant_scale.data.unsqueeze(0),
             layer.value_antiquant_scale.data.unsqueeze(0)),
            dim=0).contiguous()

    def anti_quant_int8(self, key_cache, value_cache,
                        layer) -> List[torch.Tensor]:
        dst_type = self.params_dtype
        assert key_cache.dtype == torch.int8
        assert value_cache.dtype == torch.int8
        assert dst_type != torch.int8

        key_cache_anti_quant = torch_npu.npu_anti_quant(
            x=key_cache,
            scale=layer.key_antiquant_scale.data.view(-1),
            dst_dtype=dst_type)
        value_cache_anti_quant = torch_npu.npu_anti_quant(
            x=value_cache,
            scale=layer.value_antiquant_scale.data.view(-1),
            dst_dtype=dst_type)

        return [key_cache_anti_quant, value_cache_anti_quant]

    def apply(self, layer, query, key, value, kv_cache, attn_metadata,
              attn_type, scale, output) -> torch.Tensor:
        """Apply C8 KV cache quantization during forward pass."""
        num_tokens = query.shape[0]

        if attn_metadata is None:
            return output.view(num_tokens, layer.num_heads * layer.head_size)

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for Qwen2C8KVCacheMethod")

        quant_key = quant_per_tensor(
            key.view(-1, layer.num_kv_heads * layer.head_size),
            layer.key_antiquant_scale.data.view(-1), None, True)
        quant_value = quant_per_tensor(
            value.view(-1, layer.num_kv_heads * layer.head_size),
            layer.value_antiquant_scale.data.view(-1), None, True)

        query = query.view(-1, layer.num_heads, layer.head_size)
        key = key.view(-1, layer.num_kv_heads, layer.head_size)
        value = value.view(-1, layer.num_kv_heads, layer.head_size)
        value = value.contiguous()
        if kv_cache[0].numel() > 0:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping

            block_size = key_cache.shape[1]
            slots_indices = slots.reshape(-1, 1)
            block_indices = slots_indices // block_size
            slots_indices = slots_indices % block_size
            indices = torch.cat((block_indices, slots_indices), dim=1)

            torch_npu.npu_scatter_nd_update_(key_cache, indices, quant_key)
            torch_npu.npu_scatter_nd_update_(value_cache, indices, quant_value)

            self.key_cache = key_cache
            self.value_cache = value_cache
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            assert attn_metadata.attn_mask is not None
            mask = attn_metadata.attn_mask
            torch_npu._npu_flash_attention(query=query,
                                           key=key,
                                           value=value,
                                           mask=mask,
                                           seq_len=attn_metadata.seq_lens,
                                           scale_value=scale,
                                           num_heads=layer.num_heads,
                                           num_kv_heads=layer.num_kv_heads,
                                           out=output.reshape(query.shape))

        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            assert attn_metadata is not None
            assert attn_metadata.attn_mask is not None

            compress_mask = attn_metadata.attn_mask
            batch_size = attn_metadata.query_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _ = self.key_cache.shape  # type: ignore

            key_from_cache = self.key_cache.view(num_block, block_size, -1)
            value_from_cache = self.value_cache.view(num_block, block_size, -1)

            if key_from_cache.dtype == torch.int8:
                key_cache_anti_quant, value_cache_anti_quant = self.anti_quant_int8(
                    key_from_cache, value_from_cache, layer)
            else:
                key_cache_anti_quant = key_from_cache
                value_cache_anti_quant = value_from_cache

            max_seq_len = max(attn_metadata.seq_lens_list) if hasattr(
                attn_metadata, 'seq_lens_list'
            ) and attn_metadata.seq_lens_list is not None else 0

            if block_size == 128 and max_seq_len <= 2048:
                key = key_cache_anti_quant.view(  # type: ignore
                    num_block, block_size, -1)
                value = value_cache_anti_quant.view(  # type: ignore
                    num_block, block_size, -1)

                output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=compress_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                    num_key_value_heads=layer.num_kv_heads,
                    num_heads=layer.num_heads,
                    scale=scale,
                    sparse_mode=3,
                )
            else:
                torch_npu._npu_flash_attention_qlens(
                    query=query,
                    key_cache=key_cache_anti_quant,
                    value_cache=value_cache_anti_quant,
                    block_table=block_table,
                    mask=compress_mask,
                    seq_len=attn_metadata.query_lens,
                    context_lens=attn_metadata.seq_lens,
                    num_kv_heads=layer.num_kv_heads,
                    num_heads=layer.num_heads,
                    scale_value=scale,
                    out=output)
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            if hasattr(attn_metadata, "decode"):
                decode_meta = attn_metadata.decode
                seq_lens = decode_meta.seq_lens_list
            else:
                seq_lens = attn_metadata.seq_lens

            block_size = key_cache.shape[1]
            query = query.view(num_tokens, 1,
                               layer.num_heads * layer.head_size).contiguous()

            key = key_cache
            value = value_cache

            output = torch_npu.npu_incre_flash_attention(
                query,
                key,
                value,
                num_key_value_heads=layer.num_kv_heads,
                num_heads=layer.num_heads,
                actual_seq_lengths=seq_lens,
                scale_value=scale,
                input_layout='BSH',
                block_size=block_size,
                block_table=attn_metadata.block_tables,
                antiquant_scale=self.antiquant_scale_comb,
            )

        elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            if layer.head_size == 192:
                raise NotImplementedError(
                    "KV cache int8 quantization is not implemented for head_size == 192"
                )

            assert attn_metadata is not None
            assert attn_metadata.attn_mask is not None

            if get_ascend_device_type() == AscendDeviceType._310P:
                attn_metadata.attn_mask = \
                    torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(),
                                              ACL_FORMAT_FRACTAL_NZ)
                attn_metadata.seq_lens = \
                    attn_metadata.seq_lens.to(device=query.device)

            num_block, block_size, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)

            if key.dtype == torch.int8:
                key, value = self.anti_quant_int8(key, value, layer)

            output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=attn_metadata.block_tables,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                num_key_value_heads=layer.num_kv_heads,
                num_heads=layer.num_heads,
                scale=scale,
                sparse_mode=3,
            )
        else:
            raise NotImplementedError(
                f"KV cache int8 quantization is not implemented for other attention states: {attn_metadata.attn_state}"
            )

        return output
