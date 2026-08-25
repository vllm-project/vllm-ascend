#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from vllm_ascend.utils import maybe_trans_nz

from .registry import register_scheme
from .w8a8_base import AscendW8A8Linear310pScheme


@register_scheme("W8A8", "linear")
class AscendW8A8LinearMethod310(AscendW8A8Linear310pScheme):
    """310P-only W8A8 static linear scheme.

    Notes:
      - This scheme is discovered via 310P local registry.
    """

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params: dict[str, Any] = {}
        params["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        # ModelSlim W8A8 stores float32 ``deq_scale``. On 310P, ACL accepts float32
        # scales for ``npu_quant_matmul``. Allocating int64 (mainline fp16 path)
        # causes float32→int64 load casts that zero tiny scales and garbles output
        # (seen on Qwen3-8B-W8A8 with ``--dtype float16``).
        params["deq_scale"] = torch.empty(output_size, dtype=torch.float32)
        params["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if x.dtype != torch.int8:
            x = torch.ops.vllm.quantize(
                x,
                layer.aclnn_input_scale,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
            )

        quant_bias = layer.quant_bias if tp_rank == 0 else None

        # 310P QuantBatchMatmulV3 requires transpose_x2=True: keep FRACTAL_NZ
        # weight as [N, K] and pass a transpose view (same as W8A8S).
        return torch_npu.npu_quant_matmul(
            x,
            layer.weight.data.transpose(0, 1),
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=layer.params_dtype,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        expanding_factor = layer.weight.data.shape[1]

        # ---- quant stage tensors ----
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor),
            requires_grad=False,
        )
        layer.aclnn_input_scale_reciprocal = torch.nn.Parameter(
            1.0 / layer.aclnn_input_scale.data,
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor),
            requires_grad=False,
        ).to(layer.aclnn_input_scale.dtype)

        # ---- matmul stage tensor ----
        layer.weight.data = maybe_trans_nz(layer.weight.data)

        # ---- dequant stage tensors ----
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        layer.weight_offset.data = torch.flatten(layer.weight_offset.data)
        if layer.deq_scale.dtype != torch.float32:
            layer.deq_scale = torch.nn.Parameter(
                layer.deq_scale.data.to(torch.float32),
                requires_grad=False,
            )
