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
import torchair
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.npugraph_ex_passes.utils.npugraph_ex_utils_check import extra_stream_scope_check


class GraphEXAddRMSNormQuantPattern:
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu")
        residual = torch.randn(2, 4, device="npu")
        rms_norm_weight = torch.randn(4, device="npu")
        scale = torch.ones(4, device="npu")
        scale_reciprocal = torch.ones(4, device="npu")
        offset = torch.zeros(4, device="npu")
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset]

    def register(self):
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
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

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

        torchair.register_replacement(
            search_fn=pattern,
            replace_fn=replacement,
            example_inputs=self.get_inputs(),
            extra_check=extra_stream_scope_check,
        )


class GraphEXAddRMSNormQuantPatternWithBias:
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuantWithBias fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu")
        residual = torch.randn(2, 4, device="npu")
        rms_norm_weight = torch.randn(4, device="npu")
        rmsnorm_bias = torch.randn(4, device="npu")
        scale = torch.ones(4, device="npu")
        scale_reciprocal = torch.ones(4, device="npu")
        offset = torch.zeros(4, device="npu")
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset, rmsnorm_bias]

    # The replacement registered here will be actually executed after AOT.
    def register(self):
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
            Pattern for AddRMSNormQuantWithBias fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            out0 = out0 + bias
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

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
            Replacement for AddRMSNormQuantWithBias fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps, beta=bias
            )
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        torchair.register_replacement(
            search_fn=pattern,
            replace_fn=replacement,
            example_inputs=self.get_inputs(),
            extra_check=extra_stream_scope_check,
        )


class GraphEXAddRMSNormQuantSPPattern:
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuantSPPattern fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu")
        residual = torch.randn(2, 4, device="npu")
        rms_norm_weight = torch.randn(4, device="npu")
        scale = torch.ones(4, device="npu")
        scale_reciprocal = torch.ones(4, device="npu")
        offset = torch.zeros(4, device="npu")
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset]

    # The replacement registered here will be actually executed after AOT.
    def register(self):
        def pattern(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Pattern for AddRMSNormQuantSPPattern fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            out0 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(out0, True)
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

        def replacement(
            rms_norm_input: torch.Tensor,
            residual: torch.Tensor,
            rms_norm_weight: torch.Tensor,
            scale: torch.Tensor,
            scale_reciprocal: torch.Tensor,
            offset: torch.Tensor,
        ):
            """
            Replacement for the AddRMSNormQuantSPPattern fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps
            )
            quantized_output = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(quantized_output, True)
            return quantized_output, out1

        torchair.register_replacement(
            search_fn=pattern,
            replace_fn=replacement,
            example_inputs=self.get_inputs(),
            extra_check=extra_stream_scope_check,
        )


class GraphEXAddRMSNormQuantSPPatternWithBias:
    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.dtype = vllm_config.model_config.dtype
        self.eps = eps

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuantSPPatternWithBias fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu")
        residual = torch.randn(2, 4, device="npu")
        rms_norm_weight = torch.randn(4, device="npu")
        rmsnorm_bias = torch.randn(4, device="npu")
        scale = torch.ones(4, device="npu")
        scale_reciprocal = torch.ones(4, device="npu")
        offset = torch.zeros(4, device="npu")
        return [rms_norm_input, residual, rms_norm_weight, scale, scale_reciprocal, offset, rmsnorm_bias]

    # The replacement registered here will be actually executed after AOT.
    def register(self):
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
            Pattern for AddRMSNormQuantSPPatternWithBias fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            out0 = out0 + bias
            out0 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(out0, True)
            quantized_output = torch.ops.vllm.quantize(out0, scale, scale_reciprocal, offset)
            return quantized_output, out1

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
            Replacement for the AddRMSNormQuantSPPatternWithBias fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input, residual, rms_norm_weight, scale, offset, epsilon=self.eps, beta=bias
            )
            quantized_output = output[0]
            out1 = output[2]
            quantized_output = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(quantized_output, True)
            return quantized_output, out1

        torchair.register_replacement(
            search_fn=pattern,
            replace_fn=replacement,
            example_inputs=self.get_inputs(),
            extra_check=extra_stream_scope_check,
        )


class GraphEXAddRMSNormFusionPass:
    """
    A pass for fusing AddRMSNorm and W8A8 quantization operations on Ascend.
    """

    def __init__(self, vllm_config: VllmConfig):
        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.debug("Quant fusion not enabled: unsupported dtype %s", dtype)
            return

        common_epsilons = [1e-5, 1e-6]
        for eps in common_epsilons:
            GraphEXAddRMSNormQuantPattern(vllm_config, eps=eps).register()
            GraphEXAddRMSNormQuantPatternWithBias(vllm_config, eps=eps).register()
            GraphEXAddRMSNormQuantSPPattern(vllm_config, eps=eps).register()
            GraphEXAddRMSNormQuantSPPatternWithBias(vllm_config, eps=eps).register()

    def __call__(self, graph: torch.fx.Graph):
        pass

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        return True
