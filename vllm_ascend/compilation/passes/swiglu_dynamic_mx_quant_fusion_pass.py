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
import torch_npu
from torch._inductor.pattern_matcher import PatternMatcherPass, PatternPrettyPrinter
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern


class SwiGluDynamicMXQuantPattern(BasePattern):
    """
    Pattern for fusing SwiGlu and DynamicMXQuant operations.

    Matches the pattern:
        swiglu_out = torch_npu.npu_swiglu(x)
        quantized_out, scale = torch_npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch.float8_e4m3fn)

    Replaces with:
        quantized_out, scale = torch_npu.npu_swiglu_mx_quant(x, dst_type=torch.float8_e4m3fn)
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config, eps=1e-6)  # eps not used for SwiGlu

    def get_inputs(self):
        """
        Generate example inputs for the SwiGluDynamicMXQuant fusion pattern.
        """
        x = torch.randn(2, 64 * 2, device="npu", dtype=self.dtype)  # SwiGlu splits input in half
        return [x]

    def get_pattern(self):
        """
        Pattern function that matches SwiGlu followed by DynamicMXQuant.
        """
        def pattern(x: torch.Tensor):
            """
            Pattern for SwiGluDynamicMXQuant fusion.
            """
            swiglu_out = torch.ops.npu.npu_swiglu(x)
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch.float8_e4m3fn)
            return quantized_output[0], quantized_output[1]

        return pattern

    def get_replacement(self):
        """
        Replacement function using the fused npu_swiglu_mx_quant operator.
        """
        def replacement(x: torch.Tensor):
            """
            Replacement for the SwiGluDynamicMXQuant fusion.
            """
            output = torch.ops.npu.npu_swiglu_mx_quant(x, activation_left=True, dst_type=torch.float8_e4m3fn)
            return output[0], output[1]

        return replacement


class SwiGluDynamicMXQuantSPPattern(BasePattern):
    """
    Pattern for fusing SwiGlu and DynamicMXQuant operations with sequence parallelism.

    Matches the pattern:
        swiglu_out = torch_npu.npu_swiglu(x)
        swiglu_out = maybe_all_gather_and_maybe_unpad(swiglu_out, True)
        quantized_out, scale = torch_npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch.float8_e4m3fn)

    Replaces with:
        quantized_out, scale = torch_npu.npu_swiglu_mx_quant(x, dst_type=torch.float8_e4m3fn)
        quantized_out = maybe_all_gather_and_maybe_unpad(quantized_out, True)
        scale = maybe_all_gather_and_maybe_unpad(scale, True)
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config, eps=1e-6)  # eps not used for SwiGlu

    def get_inputs(self):
        """
        Generate example inputs for the SwiGluDynamicMXQuant fusion pattern with SP.
        """
        x = torch.randn(2, 64 * 2, device="npu", dtype=self.dtype)  # SwiGlu splits input in half
        return [x]

    def get_pattern(self):
        """
        Pattern function that matches SwiGlu followed by all_gather and DynamicMXQuant.
        """
        def pattern(x: torch.Tensor):
            """
            Pattern for SwiGluDynamicMXQuant fusion with sequence parallelism.
            """
            swiglu_out = torch.ops.npu.npu_swiglu(x)
            swiglu_out = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(swiglu_out, True)
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch.float8_e4m3fn)
            return quantized_output[0], quantized_output[1]

        return pattern

    def get_replacement(self):
        """
        Replacement function using the fused npu_swiglu_mx_quant operator with SP.
        """
        def replacement(x: torch.Tensor):
            """
            Replacement for the SwiGluDynamicMXQuant fusion with sequence parallelism.
            """
            output = torch.ops.npu.npu_swiglu_mx_quant(x, dst_type=torch.float8_e4m3fn)
            quantized_output = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(output[0], True)
            mxscale = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(output[1], True)
            return quantized_output, mxscale

        return replacement


class SwiGluDynamicMXQuantFloat4Pattern(BasePattern):
    """
    Pattern for fusing SwiGlu and DynamicMXQuant operations with float4_e2m1fn_x2 dtype.

    Matches the pattern:
        swiglu_out = torch_npu.npu_swiglu(x)
        quantized_out, scale = torch_npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch_npu.float4_e2m1fn_x2)

    Replaces with:
        quantized_out, scale = torch_npu.npu_swiglu_mx_quant(x, dst_type=torch_npu.float4_e2m1fn_x2)
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config, eps=1e-6)  # eps not used for SwiGlu

    def get_inputs(self):
        """
        Generate example inputs for the SwiGluDynamicMXQuant fusion pattern.
        """
        x = torch.randn(2, 64 * 2, device="npu", dtype=self.dtype)  # SwiGlu splits input in half
        return [x]

    def get_pattern(self):
        """
        Pattern function that matches SwiGlu followed by DynamicMXQuant with float4.
        """
        def pattern(x: torch.Tensor):
            """
            Pattern for SwiGluDynamicMXQuant fusion with float4_e2m1fn_x2.
            """
            swiglu_out = torch.ops.npu.npu_swiglu(x)
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch_npu.float4_e2m1fn_x2)
            return quantized_output[0], quantized_output[1]

        return pattern

    def get_replacement(self):
        """
        Replacement function using the fused npu_swiglu_mx_quant operator with float4.
        """
        def replacement(x: torch.Tensor):
            """
            Replacement for the SwiGluDynamicMXQuant fusion with float4_e2m1fn_x2.
            """
            output = torch.ops.npu.npu_swiglu_mx_quant(x, dst_type=torch_npu.float4_e2m1fn_x2)
            return output[0], output[1]

        return replacement


class SwiGluDynamicMXQuantSPFloat4Pattern(BasePattern):
    """
    Pattern for fusing SwiGlu and DynamicMXQuant operations with sequence parallelism and float4_e2m1fn_x2 dtype.

    Matches the pattern:
        swiglu_out = torch_npu.npu_swiglu(x)
        swiglu_out = maybe_all_gather_and_maybe_unpad(swiglu_out, True)
        quantized_out, scale = torch_npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch_npu.float4_e2m1fn_x2)

    Replaces with:
        quantized_out, scale = torch_npu.npu_swiglu_mx_quant(x, dst_type=torch_npu.float4_e2m1fn_x2)
        quantized_out = maybe_all_gather_and_maybe_unpad(quantized_out, True)
        scale = maybe_all_gather_and_maybe_unpad(scale, True)
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config, eps=1e-6)  # eps not used for SwiGlu

    def get_inputs(self):
        """
        Generate example inputs for the SwiGluDynamicMXQuant fusion pattern with SP.
        """
        x = torch.randn(2, 64 * 2, device="npu", dtype=self.dtype)  # SwiGlu splits input in half
        return [x]

    def get_pattern(self):
        """
        Pattern function that matches SwiGlu followed by all_gather and DynamicMXQuant with float4.
        """
        def pattern(x: torch.Tensor):
            """
            Pattern for SwiGluDynamicMXQuant fusion with sequence parallelism and float4_e2m1fn_x2.
            """
            swiglu_out = torch.ops.npu.npu_swiglu(x)
            swiglu_out = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(swiglu_out, True)
            quantized_output = torch.ops.npu.npu_dynamic_mx_quant(swiglu_out, dst_type=torch_npu.float4_e2m1fn_x2)
            return quantized_output[0], quantized_output[1]

        return pattern

    def get_replacement(self):
        """
        Replacement function using the fused npu_swiglu_mx_quant operator with SP and float4.
        """
        def replacement(x: torch.Tensor):
            """
            Replacement for the SwiGluDynamicMXQuant fusion with sequence parallelism and float4_e2m1fn_x2.
            """
            output = torch.ops.npu.npu_swiglu_mx_quant(x, dst_type=torch_npu.float4_e2m1fn_x2)
            quantized_output = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(output[0], True)
            mxscale = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(output[1], True)
            return quantized_output, mxscale

        return replacement


class SwiGluDynamicMXQuantFusionPass(VllmInductorPass):
    """
    A pass for fusing SwiGlu and DynamicMXQuant operations on Ascend.

    This pass optimizes models like Qwen3-8B and Qwen3-32B that use the
    SwiGlu activation function followed by MX quantization.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="swiglu_dynamic_mx_quant_fusion_pass"
        )

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug("SwiGlu DynamicMXQuant fusion not enabled: unsupported dtype %s", dtype)
            return

        # Check if the required npu_swiglu_mx_quant operator is available with float8
        try:
            # Test if the operator exists with float8
            test_tensor = torch.empty(1, 2, device="npu", dtype=dtype)
            _ = torch.ops.npu.npu_swiglu_mx_quant(test_tensor, dst_type=torch.float8_e4m3fn)
            float8_available = True
        except (AttributeError, RuntimeError) as e:
            logger.debug(
                "SwiGlu DynamicMXQuant fusion (float8) not enabled: npu_swiglu_mx_quant operator unavailable: %s", e
            )
            float8_available = False

        # Check if the required npu_swiglu_mx_quant operator is available with float4
        try:
            # Test if the operator exists with float4
            test_tensor = torch.empty(1, 2, device="npu", dtype=dtype)
            _ = torch.ops.npu.npu_swiglu_mx_quant(test_tensor, dst_type=torch_npu.float4_e2m1fn_x2)
            float4_available = True
        except (AttributeError, RuntimeError) as e:
            logger.debug(
                "SwiGlu DynamicMXQuant fusion (float4) not enabled: npu_swiglu_mx_quant operator unavailable: %s", e
            )
            float4_available = False

        if not float8_available and not float4_available:
            return

        # Register the float8 patterns
        if float8_available:
            SwiGluDynamicMXQuantPattern(vllm_config).register(self.pattern_match_passes)
            SwiGluDynamicMXQuantSPPattern(vllm_config).register(self.pattern_match_passes)
            logger.debug("SwiGlu DynamicMXQuant float8 fusion patterns registered")

        # Register the float4 patterns
        if float4_available:
            SwiGluDynamicMXQuantFloat4Pattern(vllm_config).register(self.pattern_match_passes)
            SwiGluDynamicMXQuantSPFloat4Pattern(vllm_config).register(self.pattern_match_passes)
            logger.debug("SwiGlu DynamicMXQuant float4 fusion patterns registered")

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s SwiGlu DynamicMXQuant patterns", self.matched_count)
        logger.debug("Patterns registered for replacement:")
        pattern_idx = 0
        for pattern_entry in self.pattern_match_passes.patterns.values():
            for p in pattern_entry:
                p_str = PatternPrettyPrinter.run(p.pattern)
                logger.debug("Pattern %d: %s", pattern_idx, p_str)
                pattern_idx += 1
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        return True