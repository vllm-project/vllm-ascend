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

from vllm_ascend.ops.linear import AscendReplicatedLinear


class AscendGateLinear(GateLinear):
    """Ascend GateLinear: NPU counterpart of upstream's GEMM dispatch.

    Upstream tiers 1-3 are CUDA/ROCm cuteDSL kernels (unavailable on NPU);
    this override keeps tier 4 and the tier 5 fallback:

    - Tier 4: bf16 x bf16 with fp32 accumulation via torch.mm's out_dtype
      epilogue (aclnnMm MatMulV3 on NPU, cf. the cuBLAS/hipBLASLt epilogue
      upstream dispatches to). Each bf16 x bf16 product has at most 16
      significant bits and is exact in the fp32 accumulator, so for
      bf16-origin operands this is numerically equivalent to the fp32
      upcast path, and ~6x faster at decode-time router shapes (e.g.
      K=7168 / N=896 on Kimi K3).
    - Tier 5: the ReplicatedLinear fallback with upstream's dtype
      semantics: cast x to the weight dtype, compute, cast the output to
      out_dtype.

    ``out_dtype`` defaults to fp32 on NPU when the model passes None
    (upstream leaves None and computes natively): the fp32-logits
    contract keeps downstream NPU topk kernels on their validated path
    and lets bf16-weight models (upstream DeepSeek-V2 / GLM5Next wiring)
    take tier 4 directly.

    A forced fp32 router stays fp32: models opt in explicitly via
    params_dtype=torch.float32 / force_fp32_compute (e.g. MiniMax-M3), or
    via the Ascend-specific precast_fp32_weight path (DSV4), which the MoE
    runner routes around this forward entirely.
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
        # Mirror upstream GateLinear.__init__: with no NPU specialized
        # kernel available, force_fp32_compute means storing the weight in
        # fp32 so the fallback tier computes in fp32.
        if force_fp32_compute:
            params_dtype = torch.float32
        # Skip GateLinear.__init__ (CUDA/ROCm GEMM probes); weights follow
        # the model dtype unless explicitly upcast.
        AscendReplicatedLinear.__init__(
            self,
            input_size,
            output_size,
            bias=bias,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
        )
        # NPU default: fp32 logits when out_dtype is None (see class
        # docstring); out_dtype is immutable afterwards.
        self.out_dtype = out_dtype if out_dtype is not None else torch.float32

    def forward(self, x: torch.Tensor):
        # Tier 4: bf16 x bf16 -> fp32 accumulation. Eligibility mirrors
        # upstream: bf16 weight (i.e. no fp32 upcast requested), fp32
        # out_dtype, no bias (torch.mm has no bias term).
        if (
            self.weight.dtype == torch.bfloat16
            and x.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
            and self.bias is None
        ):
            return torch.mm(x, self.weight.t(), out_dtype=torch.float32), None
        # Tier 5: cast x to the weight dtype, compute, cast the output to
        # out_dtype (upstream fallback semantics; out_dtype is always set
        # after __init__).
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        output, output_bias = ReplicatedLinear.forward(self, x)
        if output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias
