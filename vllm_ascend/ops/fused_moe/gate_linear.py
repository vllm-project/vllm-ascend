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
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear


class AscendGateLinear(GateLinear):
    """Ascend replacement for vLLM GateLinear.

    Router logits are sensitive to numerical precision because they directly
    affect expert selection in MoE models. On NPU, computing the router gate
    in lower precision may lead to accuracy issues in some agent workloads
    (observed with bf16×bf16→bf16 accumulation). This layer forces the gate
    input and weights to fp32 for the router linear computation, and keeps
    the router logits in fp32.

    When the checkpoint router weight is stored as bfloat16 (detected at
    load time via ``weight_loader``), the router GEMM can instead run on
    the Cube bf16 data path with fp32 accumulation
    (``torch.mm(x_bf16, w_bf16.t(), out_dtype=…)`` → aclnnMm MatMulV3).
    For bf16-origin operands the two paths are numerically equivalent:
    each bf16×bf16 product has at most 16 significant bits and is
    representable exactly in the fp32 accumulator, so the only difference
    from the upcast fp32 path is the K-dim summation order (~1e-5).
    The bf16 path is ~6x faster at decode-time router shapes (small M,
    e.g. K=7168 / N=896 on Kimi K3).

    The fast path is enabled automatically when the original checkpoint
    weight dtype is bfloat16; otherwise the fp32 fallback is used,
    preserving the original behaviour for models with true fp32 router
    weights.
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
        # Original dtype of the checkpoint weight, recorded by weight_loader.
        # None until the first weight shard is loaded.
        self._checkpoint_weight_dtype: torch.dtype | None = None
        # Pre-materialised bf16 weight cache (populated lazily on first
        # forward, after weights are loaded). Invalidated by weight_loader
        # so RL in-place weight updates are reflected.
        self._bf16_weight: torch.Tensor | None = None

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # Record the original checkpoint dtype before copy_ upcasts to fp32.
        # This is the only reliable observation point: after copy_ the
        # Parameter is fp32, and after process_weights_after_loading the
        # data may be NZ-repacked making bit-level inspection unreliable.
        if self._checkpoint_weight_dtype is None:
            self._checkpoint_weight_dtype = loaded_weight.dtype
        # Invalidate the bf16 cache: RL weight-update flows reload weights
        # in-place through weight_loader; re-materialise on next forward.
        self._bf16_weight = None
        super().weight_loader(param, loaded_weight)

    def _use_bf16_fast_path(self, x: torch.Tensor) -> bool:
        """Whether the bf16 router fast path can be used.

        Requires:
        - Checkpoint weight dtype is bfloat16 (recorded by weight_loader).
        - Input dtype is bfloat16.
        - Output dtype is fp32 (so torch.mm can use out_dtype=fp32, the
          only out_dtype supported by aclnnMm MatMulV3 on NPU).
        - No bias (fast path does not handle bias).
        """
        return (
            self._checkpoint_weight_dtype == torch.bfloat16
            and x.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
            and self.bias is None
        )

    def _forward_bf16_router(self, x: torch.Tensor):
        # The fp32 params hold bf16-origin values; the downcast is a
        # lossless round-to-nearest-even round trip
        # (bf16 → fp32 → bf16 = identity).
        if self._bf16_weight is None or self._bf16_weight.device != self.weight.device:
            # Keep the bf16 cache in ND layout: a layout micro-benchmark
            # (M=3/24) showed bit-identical outputs for ND vs FRACTAL_NZ
            # caches, no measurable benefit from NZ on this aclnnMm
            # MatMulV3 path, and a pre-transposed-NZ cache ~40% slower.
            # `.t()` on ND is a free stride view (aclnnMm handles transB).
            self._bf16_weight = self.weight.to(device=self.weight.device, dtype=torch.bfloat16)
        w_bf = self._bf16_weight
        # bf16 × bf16 products accumulated in fp32 (MatMulV3).
        return torch.mm(
            x,
            w_bf.t(),
            out_dtype=torch.float32,
        ), None

    def forward(self, x: torch.Tensor):
        if self._use_bf16_fast_path(x):
            return self._forward_bf16_router(x)

        # TODO: Remove this workaround after upgrading to a vLLM version that
        # no longer forces router logits to bf16 via
        # self.gate.set_out_dtype(torch.bfloat16).
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        output, output_bias = ReplicatedLinear.forward(self, x)

        return output, output_bias
