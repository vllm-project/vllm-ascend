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
from vllm.model_executor.layers.linear import ReplicatedLinear


class AscendGateLinear(GateLinear):
    """Ascend replacement for vLLM GateLinear.
    Router logits are sensitive to numerical precision because they directly
    affect expert selection in MoE models. On NPU, computing the router gate in
    lower precision may lead to accuracy issues in some agent workloads.
    Therefore, this layer forces the gate input and weights to fp32 for the
    router linear computation, and keeps the router logits in fp32.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            params_dtype=torch.float32,
            out_dtype=out_dtype,
            force_fp32_compute=True,
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor):
        # TODO: Remove this workaround after upgrading to a vLLM version that
        # no longer forces router logits to bf16 via
        # self.gate.set_out_dtype(torch.bfloat16).
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        output, output_bias = ReplicatedLinear.forward(self, x)

        return output, output_bias
