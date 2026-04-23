#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.utils import enable_custom_op


class AddRMSNormQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, None, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps
            )
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        return replacement


class AddRMSNormQuantPatternWithBias(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset, rmsnorm_bias]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, bias, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps, beta=bias
            )
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        return replacement


class AddRMSNormQuantSPPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, None, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            out0 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(out0, True)
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps
            )
            quantized_output = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(quantized_output, True)
            return quantized_output, out1

        return replacement


class AddRMSNormQuantSPPatternWithBias(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        scale = torch.ones(4, device="npu", dtype=self.dtype)
        scale_reciprocal = torch.ones(4, device="npu", dtype=self.dtype)
        offset = torch.zeros(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset, rmsnorm_bias]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, bias, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            out0 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(out0, True)
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps, beta=bias
            )
            quantized_output = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(quantized_output, True)
            return quantized_output, out1

        return replacement


class AddRMSNormDynamicQuantPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, True]
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


class AddRMSNormDynamicQuantPatternWithBias(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, rmsnorm_bias]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, bias, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, True], beta=bias
            )
            return (
                output[0],
                output[3],
                output[2],
            )

        return replacement


class AddRMSNormDynamicQuantSPPattern(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight]

    def get_pattern(self):
        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            out0 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(out0, True)
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor, rms_norm_weight: torch.Tensor):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, True]
            )
            out3 = output[3]
            quantized_output = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(output[0], True)
            out3 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(out3, True)
            return quantized_output, out3, output[2]

        return replacement


class AddRMSNormDynamicQuantSPPatternWithBias(BasePattern):
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        super().__init__(vllm_config, eps)

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu", dtype=self.dtype)
        residual = torch.randn(2, 4, device="npu", dtype=self.dtype)
        rms_norm_weight = torch.randn(4, device="npu", dtype=self.dtype)
        rmsnorm_bias = torch.randn(4, device="npu", dtype=self.dtype)
        return [rms_norm_input, residual, rms_norm_weight, rmsnorm_bias]

    def get_pattern(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops._C_ascend.npu_add_rms_norm_bias(
                rms_norm_input, residual, rms_norm_weight, bias, self.eps
            )
            out0 = output[0]
            out1 = output[2]
            out0 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(out0, True)
            quantized_output = torch.ops.npu.npu_dynamic_quant(out0)
            return quantized_output[0], quantized_output[1], out1

        return pattern

    def get_replacement(self):
        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            bias: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
                rms_norm_input, residual, rms_norm_weight, epsilon=self.eps, output_mask=[True, True], beta=bias
            )
            out3 = output[3]
            quantized_output = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(output[0], True)
            out3 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(out3, True)
            return quantized_output, out3, output[2]

        return replacement


class AddRMSNormQuantFusionPass(VllmInductorPass):
    """
    A pass for fusing AddRMSNorm and W8A8 quantization operations on Ascend.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="rmsnorm_quant_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug("Quant fusion not enabled: unsupported dtype %s", dtype)
            return

        common_epsilons = [1e-5, 1e-6]
        for eps in common_epsilons:
            AddRMSNormDynamicQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
            AddRMSNormDynamicQuantSPPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
            if enable_custom_op():
                AddRMSNormQuantPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormQuantSPPattern(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormQuantPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormQuantSPPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormDynamicQuantPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)
                AddRMSNormDynamicQuantSPPatternWithBias(vllm_config, eps=eps).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        return True
