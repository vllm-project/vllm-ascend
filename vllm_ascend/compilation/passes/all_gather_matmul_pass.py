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
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern


class AllGatherDynamicQuantMatmulPattern(BasePattern):
    """
    Pattern that matches sequence-parallel all-gather followed by W8A8 dynamic
    quantized matmul and replaces it with the NPU all-gather-matmul primitive.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.randn(2, 256, device="npu", dtype=self.dtype)
        weight = torch.randint(-8, 8, (256, 128), device="npu", dtype=torch.int8)
        weight_scale = torch.ones(128, device="npu", dtype=torch.float32)
        return [x, weight, weight_scale]

    def get_pattern(self):
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ):
            gathered = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(x, label=True)
            quantized_x, pertoken_scale = torch.ops.npu.npu_dynamic_quant(gathered, dst_type=torch.int8)
            return torch.ops.npu.npu_quant_matmul(
                quantized_x,
                weight,
                weight_scale,
                pertoken_scale=pertoken_scale,
                bias=None,
                output_dtype=self.dtype,
            )

        return pattern

    def get_replacement(self):
        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ):
            return torch.ops.vllm.all_gather_dynamic_quant_matmul(x, weight, weight_scale)

        return replacement


class AllGatherUnquantizedMatmulPattern(BasePattern):
    """
    Pattern that matches sequence-parallel all-gather followed by an
    unquantized linear matmul and replaces it with AllGatherMatmulV2.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.randn(2, 256, device="npu", dtype=self.dtype)
        weight = torch.randn(128, 256, device="npu", dtype=self.dtype)
        return [x, weight]

    def get_pattern(self):
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ):
            gathered = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(x, label=True)
            return torch.ops.vllm.unquantized_gemm(gathered, weight, None)

        return pattern

    def get_replacement(self):
        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
        ):
            return torch.ops.vllm.all_gather_unquantized_matmul(x, weight, None)

        return replacement


class AllGatherUnquantizedMatmulWithBiasPattern(BasePattern):
    """
    Pattern that matches sequence-parallel all-gather followed by an
    unquantized linear matmul with bias and replaces it with AllGatherMatmulV2.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.randn(2, 256, device="npu", dtype=self.dtype)
        weight = torch.randn(128, 256, device="npu", dtype=self.dtype)
        bias = torch.randn(128, device="npu", dtype=self.dtype)
        return [x, weight, bias]

    def get_pattern(self):
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            gathered = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(x, label=True)
            return torch.ops.vllm.unquantized_gemm(gathered, weight, bias)

        return pattern

    def get_replacement(self):
        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            return torch.ops.vllm.all_gather_unquantized_matmul(x, weight, bias)

        return replacement


class AllGatherMatmulFusionPass(VllmInductorPass):
    """
    Fuse sequence-parallel all-gather + dynamic quant matmul for column-parallel
    linear layers.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="all_gather_matmul_fusion_pass",
        )

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            logger.debug("AllGatherMatmul fusion not enabled: unsupported dtype %s", dtype)
            return

        AllGatherDynamicQuantMatmulPattern(vllm_config).register(self.pattern_match_passes)
        AllGatherUnquantizedMatmulPattern(vllm_config).register(self.pattern_match_passes)
        AllGatherUnquantizedMatmulWithBiasPattern(vllm_config).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph) -> None:  # type: ignore[override]
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s all_gather_matmul patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
