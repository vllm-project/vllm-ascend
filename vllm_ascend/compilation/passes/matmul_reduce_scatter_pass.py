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

MM_REDUCE_SCATTER_PATTERN_TOKENS = 8


class DynamicQuantMatmulReduceScatterPattern(BasePattern):
    """
    Match dynamic W8A8 row-parallel matmul followed by sequence-parallel
    reduce-scatter.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.randn(MM_REDUCE_SCATTER_PATTERN_TOKENS, 256, device="npu", dtype=self.dtype)
        weight = torch.randint(-8, 8, (256, 128), device="npu", dtype=torch.int8)
        weight_scale = torch.ones(128, device="npu", dtype=torch.float32)
        return [x, weight, weight_scale]

    def get_pattern(self):
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ):
            quantized_x, pertoken_scale = torch.ops.npu.npu_dynamic_quant(x, dst_type=torch.int8)
            output = torch.ops.npu.npu_quant_matmul(
                quantized_x,
                weight,
                weight_scale,
                pertoken_scale=pertoken_scale,
                bias=None,
                output_dtype=self.dtype,
            )
            return torch.ops.vllm.maybe_pad_and_reduce(output)

        return pattern

    def get_replacement(self):
        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
        ):
            return torch.ops.vllm.dynamic_quant_matmul_reduce_scatter(x, weight, weight_scale)

        return replacement


class QuantMatmulReduceScatterPattern(BasePattern):
    """
    Match static W8A8 row-parallel matmul followed by sequence-parallel
    reduce-scatter.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.randn(MM_REDUCE_SCATTER_PATTERN_TOKENS, 256, device="npu", dtype=self.dtype)
        weight = torch.randint(-8, 8, (256, 128), device="npu", dtype=torch.int8)
        deq_scale = torch.ones(128, device="npu", dtype=torch.float32)
        input_scale = torch.ones(256, device="npu", dtype=self.dtype)
        input_scale_reciprocal = torch.ones(256, device="npu", dtype=self.dtype)
        input_offset = torch.zeros(256, device="npu", dtype=self.dtype)
        return [x, weight, deq_scale, input_scale, input_scale_reciprocal, input_offset]

    def get_pattern(self):
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            deq_scale: torch.Tensor,
            input_scale: torch.Tensor,
            input_scale_reciprocal: torch.Tensor,
            input_offset: torch.Tensor,
        ):
            quantized_x = torch.ops.vllm.quantize(x, input_scale, input_scale_reciprocal, input_offset)
            output = torch.ops.npu.npu_quant_matmul(
                quantized_x,
                weight,
                deq_scale,
                bias=None,
                output_dtype=self.dtype,
            )
            return torch.ops.vllm.maybe_pad_and_reduce(output)

        return pattern

    def get_replacement(self):
        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            deq_scale: torch.Tensor,
            input_scale: torch.Tensor,
            input_scale_reciprocal: torch.Tensor,
            input_offset: torch.Tensor,
        ):
            return torch.ops.vllm.quant_matmul_reduce_scatter(
                x,
                weight,
                deq_scale,
                input_scale,
                input_scale_reciprocal,
                input_offset,
                None,
            )

        return replacement


class QuantMatmulBiasReduceScatterPattern(BasePattern):
    """
    Match static W8A8 row-parallel matmul with quant bias followed by
    sequence-parallel reduce-scatter.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.randn(MM_REDUCE_SCATTER_PATTERN_TOKENS, 256, device="npu", dtype=self.dtype)
        weight = torch.randint(-8, 8, (256, 128), device="npu", dtype=torch.int8)
        deq_scale = torch.ones(128, device="npu", dtype=torch.float32)
        input_scale = torch.ones(256, device="npu", dtype=self.dtype)
        input_scale_reciprocal = torch.ones(256, device="npu", dtype=self.dtype)
        input_offset = torch.zeros(256, device="npu", dtype=self.dtype)
        quant_bias = torch.zeros(128, device="npu", dtype=torch.int32)
        return [x, weight, deq_scale, input_scale, input_scale_reciprocal, input_offset, quant_bias]

    def get_pattern(self):
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            deq_scale: torch.Tensor,
            input_scale: torch.Tensor,
            input_scale_reciprocal: torch.Tensor,
            input_offset: torch.Tensor,
            quant_bias: torch.Tensor,
        ):
            quantized_x = torch.ops.vllm.quantize(x, input_scale, input_scale_reciprocal, input_offset)
            output = torch.ops.npu.npu_quant_matmul(
                quantized_x,
                weight,
                deq_scale,
                bias=quant_bias,
                output_dtype=self.dtype,
            )
            return torch.ops.vllm.maybe_pad_and_reduce(output)

        return pattern

    def get_replacement(self):
        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            deq_scale: torch.Tensor,
            input_scale: torch.Tensor,
            input_scale_reciprocal: torch.Tensor,
            input_offset: torch.Tensor,
            quant_bias: torch.Tensor,
        ):
            return torch.ops.vllm.quant_matmul_reduce_scatter(
                x,
                weight,
                deq_scale,
                input_scale,
                input_scale_reciprocal,
                input_offset,
                quant_bias,
            )

        return replacement


class UnquantizedMatmulReduceScatterPattern(BasePattern):
    """
    Match unquantized row-parallel matmul followed by sequence-parallel
    reduce-scatter.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.randn(MM_REDUCE_SCATTER_PATTERN_TOKENS, 256, device="npu", dtype=self.dtype)
        weight = torch.randn(128, 256, device="npu", dtype=self.dtype)
        return [x, weight]

    def get_pattern(self):
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ):
            output = torch.ops.vllm.unquantized_gemm(x, weight, None)
            return torch.ops.vllm.maybe_pad_and_reduce(output)

        return pattern

    def get_replacement(self):
        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
        ):
            return torch.ops.vllm.unquantized_matmul_reduce_scatter(x, weight, None)

        return replacement


class UnquantizedMatmulBiasReduceScatterPattern(BasePattern):
    """
    Match unquantized row-parallel matmul with bias followed by
    sequence-parallel reduce-scatter.
    """

    def get_inputs(self) -> list[torch.Tensor]:
        x = torch.randn(MM_REDUCE_SCATTER_PATTERN_TOKENS, 256, device="npu", dtype=self.dtype)
        weight = torch.randn(128, 256, device="npu", dtype=self.dtype)
        bias = torch.randn(128, device="npu", dtype=self.dtype)
        return [x, weight, bias]

    def get_pattern(self):
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            output = torch.ops.vllm.unquantized_gemm(x, weight, bias)
            return torch.ops.vllm.maybe_pad_and_reduce(output)

        return pattern

    def get_replacement(self):
        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            return torch.ops.vllm.unquantized_matmul_reduce_scatter(x, weight, bias)

        return replacement


class MatmulReduceScatterFusionPass(VllmInductorPass):
    """
    Route sequence-parallel row-parallel matmul + reduce-scatter through the
    fused NPU primitive.

    The fusion is intentionally limited to the tensor-level pattern emitted by
    the ordinary sequence-parallel row path. Layer-specific wrappers such as
    DSA-CP keep their original ``matmul_and_reduce`` behavior.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.enabled = vllm_config.model_config.dtype in (torch.float16, torch.bfloat16)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="matmul_reduce_scatter_fusion_pass",
        )
        if not self.enabled:
            logger.debug(
                "Matmul reduce-scatter fusion not enabled: unsupported dtype %s",
                vllm_config.model_config.dtype,
            )
            return

        try:
            DynamicQuantMatmulReduceScatterPattern(vllm_config).register(self.pattern_match_passes)
            QuantMatmulReduceScatterPattern(vllm_config).register(self.pattern_match_passes)
            QuantMatmulBiasReduceScatterPattern(vllm_config).register(self.pattern_match_passes)
            UnquantizedMatmulReduceScatterPattern(vllm_config).register(self.pattern_match_passes)
            UnquantizedMatmulBiasReduceScatterPattern(vllm_config).register(self.pattern_match_passes)
        except ModuleNotFoundError as e:
            logger.debug("Skipping matmul reduce-scatter tensor patterns: %s", e)

    def __call__(self, graph: torch.fx.Graph) -> None:  # type: ignore[override]
        if not self.enabled:
            return

        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s matmul_reduce_scatter patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
