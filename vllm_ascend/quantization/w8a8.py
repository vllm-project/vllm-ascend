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

from typing import Any, Dict, Optional

import torch
import torch_npu
from vllm.attention.backends.abstract import AttentionType
from vllm.forward_context import get_forward_context

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ,
                               COMPRESSED_TENSORS_METHOD, AscendDeviceType,
                               get_ascend_device_type, is_enable_nz)


def quant_per_tensor(in_tensor: torch.Tensor,
                     input_scale: torch.Tensor,
                     input_offset: torch.Tensor,
                     function=False):
    return torch_npu.npu_quantize(in_tensor, input_scale, input_offset,
                                  torch.qint8, -1, function)


class AscendW8A8LinearMethod:
    """Linear method for Ascend W8A8.

    Args:
        w_sym: whether the linear weight is symmetrically quantized.
    """

    def __init__(self) -> None:
        # aclnn quant matmul requires to transpose matrix B, set to true by default.
        self.transpose_weight = get_ascend_device_type(
        ) != AscendDeviceType._310P

    @staticmethod
    def get_weight(
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        params_dict["input_scale"] = torch.empty(1, dtype=params_dtype)
        params_dict["input_offset"] = torch.empty(1, dtype=torch.int8)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        if params_dtype == torch.bfloat16:
            params_dict["deq_scale"] = torch.empty(output_size,
                                                   dtype=torch.float32)
        elif params_dtype == torch.float16:
            params_dict["deq_scale"] = torch.empty(output_size,
                                                   dtype=torch.int64)
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=params_dtype)
        return params_dict

    def get_pergroup_param(self,
                           input_size: int,
                           output_size: int,
                           params_dtype: torch.dtype,
                           layer_type: Optional[str] = None) -> Dict[str, Any]:
        return {}

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        if x.dtype != torch.int8:
            layer_cls_name = layer.__class__.__name__
            try:
                weight_prefetch_method = get_forward_context(
                ).weight_prefetch_method
            except AssertionError:
                weight_prefetch_method = None

            # prefetch qkvo_proj.weight preprocess
            if weight_prefetch_method:
                weight_prefetch_method.maybe_prefetch_attn_weight_preprocess(
                    layer_cls_name=layer_cls_name,
                    weight=layer.weight,
                    start_flag=x,
                )
            try:
                quant_comm_config = getattr(layer, "_quant_comm_config")
            except AttributeError:
                quant_comm_config = {}
            comm_fn = quant_comm_config.get("communication_fn")
            enable_flashcomm2_quant_comm = comm_fn is not None and (
                "o_proj" in layer.prefix or "out_proj" in layer.prefix)
            if enable_flashcomm2_quant_comm:
                quant_input_x = x.contiguous().view(
                    -1, layer.aclnn_input_scale_reciprocal.size(0))
                quant_x = quant_per_tensor(
                    quant_input_x,
                    layer.aclnn_input_scale_reciprocal,
                    layer.aclnn_input_offset,
                )
                comm_input = quant_x.view(x.size(0), -1)
                assert comm_fn is not None
                x = comm_fn(comm_input)
            else:
                # quant
                x = quant_per_tensor(
                    x,
                    layer.aclnn_input_scale_reciprocal,
                    layer.aclnn_input_offset,
                )

            # prefetch qkvo_proj.weight postprocess
            if weight_prefetch_method:
                weight_prefetch_method.maybe_prefetch_attn_weight_postprocess(
                    layer_cls_name=layer_cls_name,
                    stop_flag=x,
                )

        quant_bias = layer.quant_bias if tp_rank == 0 else None

        try:
            ascend_quant_method = getattr(layer, "ascend_quant_method")
        except AttributeError:
            ascend_quant_method = ""
        if ascend_quant_method == COMPRESSED_TENSORS_METHOD:
            quant_bias = bias

        if get_ascend_device_type() == AscendDeviceType._310P:
            # On 300I Duo platform, we need transpose again if
            # using nz. This transpose can be skipped in torchair.
            output = torch_npu.npu_quant_matmul(
                x,
                layer.weight.data.transpose(1, 0),
                layer.deq_scale,
                bias=quant_bias,
                output_dtype=layer.params_dtype,
            )
        else:
            output = torch_npu.npu_quant_matmul(
                x,
                layer.weight,
                layer.deq_scale,
                bias=quant_bias,
                output_dtype=layer.params_dtype,
            )
        return output

    def process_weights_after_loading(self, layer):
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False)
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor),
            requires_grad=False).to(layer.aclnn_input_scale.dtype)
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        if is_enable_nz():
            layer.weight.data = torch_npu.npu_format_cast(
                layer.weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
        ascend_quant_method = getattr(layer, "ascend_quant_method", "")
        if ascend_quant_method == COMPRESSED_TENSORS_METHOD:
            deq_scale = layer.input_scale.data * layer.weight_scale.data
            layer.deq_scale = torch.nn.Parameter(deq_scale,
                                                 requires_grad=False)


class AscendC8KVCacheMethod:

    def __init__(self) -> None:
        self.antiquant_scale_comb = None

    @staticmethod
    def create_weights(layer) -> None:
        param_dict = {}  # num_kv_heads * head_size
        param_dict["key_antiquant_scale"] = torch.empty(layer.num_kv_heads *
                                                        layer.head_size,
                                                        dtype=torch.float16,
                                                        requires_grad=False)
        param_dict["value_antiquant_scale"] = torch.empty(layer.num_kv_heads *
                                                          layer.head_size,
                                                          dtype=torch.float16,
                                                          requires_grad=False)
        for weight_name, weight_param in param_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            layer.register_parameter(weight_name, param)

    def process_weights_after_loading(self, layer):
        self.antiquant_scale_comb = torch.cat(
            (layer.key_antiquant_scale.data.unsqueeze(0),
             layer.value_antiquant_scale.data.unsqueeze(0)),
            dim=0).to(torch.float16).contiguous()

    def apply(self, layer, query, key, value, kv_cache, attn_metadata,
              attn_type, scale, output) -> torch.Tensor:
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.view(num_tokens, layer.num_heads * layer.head_size)
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        # C8
        quant_key = quant_per_tensor(
            key.view(-1, layer.num_kv_heads * layer.head_size),
            layer.key_antiquant_scale.data.view(-1), None, True)
        quant_value = quant_per_tensor(
            value.view(-1, layer.num_kv_heads * layer.head_size),
            layer.value_antiquant_scale.data.view(-1), None, True)

        # View q k v to BSH.
        query = query.view(-1, layer.num_heads, layer.head_size)
        key = key.view(-1, layer.num_kv_heads, layer.head_size)
        value = value.view(-1, layer.num_kv_heads, layer.head_size)
        # TODO: Remove this contiguous in the future.
        value = value.contiguous()

        if kv_cache[0].numel() > 0:
            # if key_cache is None:
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping

            block_size = key_cache.shape[1]
            slots_indices = slots.reshape(-1, 1)
            block_indices = slots_indices // block_size
            slots_indices = slots_indices % block_size
            indices = torch.cat((block_indices, slots_indices), dim=1)

            # C8
            torch_npu.npu_scatter_nd_update_(key_cache, indices, quant_key)
            torch_npu.npu_scatter_nd_update_(value_cache, indices, quant_value)

        # V0-Style scheduler situation.
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            assert attn_metadata is not None
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
            raise NotImplementedError("kv cache int8 are not "
                                      "implemented for "
                                      "PrefillCacheHit")
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:  # changed attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            if hasattr(attn_metadata, "decode"):
                # torch_air
                decode_meta = attn_metadata.decode
                seq_lens = decode_meta.seq_lens_list
            else:
                seq_lens = attn_metadata.seq_lens
            block_size = key_cache.shape[1]
            query = query.view(num_tokens, 1, layer.num_heads *
                               layer.head_size).contiguous()  # changed

            # [num_blocks, block_size, N, D] --> [num_blocks, N, block_size, D]
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

        # Normal V1 situation.
        else:
            raise NotImplementedError("kv cache int8 are not "
                                      "implemented for "
                                      "other case")
        return output
