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

from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_ascend.ops.linear import AscendReplicatedLinear


class AscendGateLinear(GateLinear):
    """Ascend GateLinear: FP32 router weight/compute on NPU.

    Skips GateLinear.__init__ (CUDA/ROCm GEMM probes). Signature matches
    upstream for drop-in replacement; unused CUDA knobs are ignored.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        skip_bias_add: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        return_bias: bool = True,
    ):
        AscendReplicatedLinear.__init__(
            self,
            input_size,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=torch.float32 if quant_config is None or force_fp32_compute else params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
        )
        self.out_dtype = out_dtype
        self.is_unquantized = isinstance(self.quant_method, UnquantizedLinearMethod)
        self.allow_cublas_router_gemm = False
        self._router_gemm_cublas_capable = False
        self.allow_specialized_router_gemm = False

    def forward(self, x: torch.Tensor):
        # TODO: Remove this workaround after upgrading to a vLLM version that
        # no longer forces router logits to bf16 via
        # self.gate.set_out_dtype(torch.bfloat16).
        if self.is_unquantized and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        return ReplicatedLinear.forward(self, x)
