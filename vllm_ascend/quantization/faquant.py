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

from typing import List

import torch
import torch_npu

from .quant_utils import (SRC_DTYPE_TO_ACL_DTYPE, TYPE_QUANT_QKV_ONLINE,
                          quant_per_tensor)


class AscendFAQuantAttentionMethod:
    """Linear method for Ascend FAQuant
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_quant_param() -> List[str]:
        return [
            "fa_q.scale", "fa_q.offset", "fa_k.scale", "fa_k.offset",
            "fa_v.scale", "fa_v.offset"
        ]

    @staticmethod
    def get_extra_module_names() -> List[str]:

        return ["fa_q", "fa_k", "fa_v"]

    @staticmethod
    def process_weights_after_loading(layer):
        fa_qscale = layer.fa_q.scale
        fa_kscale = layer.fa_k.scale
        fa_vscale = layer.fa_v.scale
        repeated_query_scale = layer.fa_q.scale.repeat(1, layer.head_size)
        layer.fa_qscale = torch.nn.Parameter(repeated_query_scale,
                                             requires_grad=False)
        repeated_query_offset = layer.fa_q.offset.repeat(1, layer.head_size)
        layer.fa_qoffset = torch.nn.Parameter(repeated_query_offset,
                                              requires_grad=False)
        repeated_fa_kscale = layer.fa_k.scale.repeat(1, layer.head_size)
        layer.fa_kscale = torch.nn.Parameter(repeated_fa_kscale,
                                             requires_grad=False)
        repeated_fa_koffset = layer.fa_k.offset.repeat(1, layer.head_size)
        layer.fa_koffset = torch.nn.Parameter(repeated_fa_koffset,
                                              requires_grad=False)
        repeated_fa_vscale = layer.fa_v.scale.repeat(1, layer.head_size)
        layer.fa_vscale = torch.nn.Parameter(repeated_fa_vscale,
                                             requires_grad=False)
        repeated_fa_voffset = layer.fa_v.offset.repeat(1, layer.head_size)
        layer.fa_voffset = torch.nn.Parameter(repeated_fa_voffset,
                                              requires_grad=False)

        if fa_kscale.shape[0] <= 0:
            raise ValueError(
                "Expected size of fa_kscale in dimension 0 should be greater than 0"
                f"but got {fa_kscale.shape[0]}.")
        gqa_size = fa_qscale.shape[0] // fa_kscale.shape[0]
        fa3_k_scale, fa3_v_scale = fa_kscale.repeat(1, gqa_size).view(
            -1, 1), fa_vscale.repeat(1, gqa_size).view(-1, 1)
        qk_scale = torch.nn.Parameter(torch.squeeze(
            fa_qscale * fa3_k_scale).to(torch.float),
                                      requires_grad=False)
        layer.register_parameter("qk_scale", qk_scale)
        fa3_v_scale = torch.nn.Parameter(
            torch.squeeze(fa3_v_scale).contiguous().to(torch.float),
            requires_grad=False)
        layer.register_parameter("fa3_v_scale", fa3_v_scale)

    @classmethod
    def apply(cls, layer: torch.nn.Module, query: torch.Tensor,
              key: torch.Tensor, value: torch.Tensor, *extra_args,
              **optional_args) -> torch.Tensor:
        key_cache, value_cache, scale, block_tables, \
            is_prefill, mask, slots, output = extra_args
        seq_lens_tensor_cpu = optional_args.get("seq_lens_tensor_cpu", None)

        query_shape = query.shape
        key_shape = key.shape
        value_shape = value.shape

        query = query.view(query.shape[0], -1)
        key = key.view(key.shape[0], -1)
        value = value.view(value.shape[0], -1)

        if is_prefill:
            if key_cache is not None:

                key_int8 = quant_per_tensor(key, layer.fa_kscale,
                                            layer.fa_koffset, True)
                value_int8 = quant_per_tensor(value, layer.fa_vscale,
                                              layer.fa_voffset, True)
                key_int8 = key_int8.view(key_shape)
                value_int8 = value_int8.view(value_shape)
                torch_npu._npu_reshape_and_cache(key_int8, value_int8,
                                                 key_cache, value_cache, slots)
            if mask is None:
                raise ValueError(
                    "attn_metadata.attn_mask is Null. Please check.")
            query = query.view(query_shape)
            key = key.view(key_shape)
            value = value.view(value_shape)
            if output is not None:
                output = output.view(query.shape)
                torch_npu._npu_flash_attention(query,
                                               key,
                                               value,
                                               mask,
                                               torch.tensor(
                                                   seq_lens_tensor_cpu,
                                                   dtype=torch.int32),
                                               scale,
                                               layer.num_heads,
                                               layer.num_kv_heads,
                                               out=output)
            else:
                query = query.view(query_shape)
                key = key.view(key_shape)
                value = value.view(value_shape)
                output = torch.empty_like(query,
                                          dtype=query.dtype).to(query.device)
                torch_npu._npu_flash_attention(query,
                                               key,
                                               value,
                                               mask,
                                               torch.tensor(
                                                   seq_lens_tensor_cpu,
                                                   dtype=torch.int32),
                                               scale,
                                               layer.num_heads,
                                               layer.num_kv_heads,
                                               out=output)

        else:
            if key_cache is None:
                raise ValueError(
                    "KV Cache can't be None in decoding phase. Got None. Please check."
                )
            query_int8 = quant_per_tensor(query, layer.fa_qscale,
                                          layer.fa_qoffset, True)
            key_int8 = quant_per_tensor(key, layer.fa_kscale, layer.fa_koffset,
                                        True)
            value_int8 = quant_per_tensor(value, layer.fa_vscale,
                                          layer.fa_voffset, True)
            query_int8 = query_int8.view(query_shape)
            key_int8 = key_int8.view(key_shape)
            value_int8 = value_int8.view(value_shape)
            query = query.view(query_shape)
            torch_npu._npu_reshape_and_cache(key_int8, value_int8, key_cache,
                                             value_cache, slots)
            if output is not None:
                output = output.view(query.shape)
                torch_npu._npu_paged_attention_quant(
                    query_int8, key_cache, value_cache, layer.num_kv_heads,
                    layer.num_heads, scale, block_tables,
                    torch.tensor(seq_lens_tensor_cpu, dtype=torch.int32),
                    TYPE_QUANT_QKV_ONLINE, SRC_DTYPE_TO_ACL_DTYPE[query.dtype],
                    layer.qk_scale, layer.fa3_v_scale, output)
            else:
                output = torch.empty_like(query,
                                          dtype=query.dtype).to(query.device)
                torch_npu._npu_paged_attention_quant(
                    query_int8, key_cache, value_cache, layer.num_kv_heads,
                    layer.num_heads, scale, block_tables,
                    torch.tensor(seq_lens_tensor_cpu, dtype=torch.int32),
                    TYPE_QUANT_QKV_ONLINE, SRC_DTYPE_TO_ACL_DTYPE[query.dtype],
                    layer.qk_scale, layer.fa3_v_scale, output)

        output = torch.flatten(output, start_dim=-2)
        return output

    @classmethod
    def create_weights(cls, layer: torch.nn.Module) -> None:
        extra_module_names = cls.get_extra_module_names()
        for name in extra_module_names:
            setattr(layer, name, torch.nn.Module())

        params_dtype = torch.get_default_dtype()

        params_dict = {}

        params_dict["fa_q.scale"] = torch.empty((layer.num_heads, 1),
                                                dtype=params_dtype)
        params_dict["fa_q.offset"] = torch.empty((layer.num_heads, 1),
                                                 dtype=torch.int8)
        params_dict["fa_k.scale"] = torch.empty((layer.num_kv_heads, 1),
                                                dtype=params_dtype)
        params_dict["fa_k.offset"] = torch.empty((layer.num_kv_heads, 1),
                                                 dtype=torch.int8)
        params_dict["fa_v.scale"] = torch.empty((layer.num_kv_heads, 1),
                                                dtype=params_dtype)
        params_dict["fa_v.offset"] = torch.empty((layer.num_kv_heads, 1),
                                                 dtype=torch.int8)

        for name, weight in params_dict.items():
            module_name, weight_name = name.split('.')
            module = getattr(layer, module_name)
            module.register_parameter(
                weight_name, torch.nn.Parameter(weight, requires_grad=False))
