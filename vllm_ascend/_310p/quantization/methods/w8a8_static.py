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

from typing import Any

import torch
import torch_npu

from vllm_ascend.quantization.methods.base import AscendLinearScheme
from vllm_ascend.utils import COMPRESSED_TENSORS_METHOD, get_weight_prefetch_method


class AscendW8A8LinearMethod310P(AscendLinearScheme):
    def get_weight(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype = torch.float16,
    ) -> dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}

    def get_pertensor_param(self, params_dtype: torch.dtype) -> dict[str, Any]:
        return {
            "input_scale": torch.empty(1, dtype=params_dtype),
            "input_offset": torch.empty(1, dtype=torch.int8),
        }

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params: dict[str, Any] = {}
        params["quant_bias"] = torch.empty(output_size, dtype=torch.int32)

        # TODO-csx
        if params_dtype == torch.bfloat16:
            params["deq_scale"] = torch.empty(output_size, dtype=torch.float32)
        else:
            params["deq_scale"] = torch.empty(output_size, dtype=torch.int64)

        params["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None, tp_rank: int | None = 0
    ) -> torch.Tensor:
        if x.dtype != torch.int8:
            layer_cls_name = layer.__class__.__name__
            weight_prefetch_method = get_weight_prefetch_method()
            if weight_prefetch_method:
                weight_prefetch_method.maybe_prefetch_attn_weight_preprocess(
                    layer_cls_name=layer_cls_name,
                    weight=layer.weight,
                    start_flag=x,
                )

            try:
                quant_comm_config = layer._quant_comm_config
            except AttributeError:
                quant_comm_config = {}
            comm_fn = quant_comm_config.get("communication_fn")
            enable_flashcomm2_quant_comm = comm_fn is not None and (
                "o_proj" in layer.prefix or "out_proj" in layer.prefix
            )

            if enable_flashcomm2_quant_comm:
                quant_input_x = x.contiguous().view(-1, layer.aclnn_input_scale_reciprocal.size(0))
                quant_x = torch.ops.vllm.quantize(
                    quant_input_x,
                    layer.aclnn_input_scale,
                    layer.aclnn_input_scale_reciprocal,
                    layer.aclnn_input_offset,
                )
                comm_input = quant_x.view(x.size(0), -1)
                x = comm_fn(comm_input)
            else:
                x = torch.ops.vllm.quantize(
                    x,
                    layer.aclnn_input_scale,
                    layer.aclnn_input_scale_reciprocal,
                    layer.aclnn_input_offset,
                )

            if weight_prefetch_method:
                weight_prefetch_method.maybe_prefetch_attn_weight_postprocess(
                    layer_cls_name=layer_cls_name,
                    stop_flag=x,
                )

        quant_bias = layer.quant_bias if tp_rank == 0 else None
        ascend_quant_method = getattr(layer, "ascend_quant_method", "")
        if ascend_quant_method == COMPRESSED_TENSORS_METHOD:
            quant_bias = bias

        return torch_npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=layer.params_dtype,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # input quant params broadcast
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False,
        )
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor),
            requires_grad=False,
        ).to(layer.aclnn_input_scale.dtype)

        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()

        # TODO-NZ

        # scale/offset flatten
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)

        ascend_quant_method = getattr(layer, "ascend_quant_method", "")
        if ascend_quant_method == COMPRESSED_TENSORS_METHOD:
            deq_scale = layer.input_scale.data * layer.weight_scale.data
            layer.deq_scale = torch.nn.Parameter(deq_scale, requires_grad=False)
