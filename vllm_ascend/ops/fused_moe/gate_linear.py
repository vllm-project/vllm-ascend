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
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

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
        params_dtype: torch.dtype | None = None,  # noqa: ARG002
        force_fp32_compute: bool = False,  # noqa: ARG002
        prefix: str = "",
    ):
        AscendReplicatedLinear.__init__(
            self,
            input_size,
            output_size,
            bias=bias,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=prefix,
        )
        self.out_dtype = out_dtype

    def forward(self, x: torch.Tensor):
        # TODO: Remove this workaround after upgrading to a vLLM version that
        # no longer forces router logits to bf16 via
        # self.gate.set_out_dtype(torch.bfloat16).
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Use F.linear for FP32 router logits. AscendUnquantizedLinearMethod
        # dispatches vllm::unquantized_gemm (PrivateUse1-only), which breaks
        # cpu-ut and is unnecessary for this small gate GEMM.
        bias = self.bias if not self.skip_bias_add else None
        output = F.linear(x, self.weight, bias)
        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
