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
import functools

import torch
from torch._inductor.pattern_matcher import Match
from vllm.logger import logger


def _extra_stream_scope_check(match: Match) -> bool:
    """
    Checks if all nodes in the same stream.
    """
    non_default_streams = set()
    has_default = False

    for node in match.nodes:
        if node.op == "call_function":
            current_stream = node.meta.get("stream_label")
            if current_stream is None:
                has_default = True
            else:
                non_default_streams.add(current_stream)
                if len(non_default_streams) > 1:
                    logger.debug(
                        f"Cross-stream operation detected in pattern match for AddRMSNormQuant. "
                        f"Multiple streams found: {non_default_streams}. "
                        f"Fusion is not supported for cross-stream operations."
                    )
                    return False

    if has_default and len(non_default_streams) > 0:
        logger.debug(
            f"Cross-stream operation detected in pattern match for AddRMSNormQuant. "
            f"Multiple streams found: {non_default_streams}. "
            f"Fusion is not supported for cross-stream operations.")
        return False

    return True


@functools.lru_cache(None)
# The replacement registered here will be actually executed after AOT.
def replacement_add_rms_norm_quant(epsilon):

    def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor,
                rms_norm_weight: torch.Tensor, scale: torch.Tensor,
                offset: torch.Tensor):
        """
        Pattern for AddRMSNormQuant fusion.
        """
        output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual,
                                                rms_norm_weight, epsilon)
        out0 = output[0]
        out1 = output[2]
        quantized_output = torch.ops.npu.npu_quantize(out0, scale, offset,
                                                      torch.qint8, -1, False)
        return quantized_output, out1

    def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor,
                    rms_norm_weight: torch.Tensor, scale: torch.Tensor,
                    offset: torch.Tensor):
        """
        Replacement for the AddRMSNormQuant fusion.
        """
        output = torch.ops.npu.npu_add_rms_norm_quant(
            rms_norm_input,
            residual,
            rms_norm_weight,
            # The inverse of scale is required by npu_add_rms_norm_quant kernel which is opposite to the npu_quantize kernel.
            1. / scale,
            offset,
            epsilon=epsilon)
        quantized_output = output[0]
        out1 = output[2]
        return quantized_output, out1

    def get_inputs():
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu")
        residual = torch.randn(2, 4, device="npu")
        rms_norm_weight = torch.randn(4, device="npu")
        scale = torch.tensor([1.0], device="npu")
        offset = torch.tensor([0.0], device="npu")
        return [rms_norm_input, residual, rms_norm_weight, scale, offset]

    import torchair

    torchair.register_replacement(search_fn=pattern,
                                  replace_fn=replacement,
                                  example_inputs=get_inputs(),
                                  extra_check=_extra_stream_scope_check)


# The replacement registered here will be actually executed after AOT.
def replacement_add_rms_norm_quant_with_bias(epsilon):

    def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor,
                rms_norm_weight: torch.Tensor, scale: torch.Tensor,
                offset: torch.Tensor, bias: torch.Tensor):
        """
        Pattern for AddRMSNormQuantWithBias fusion.
        """
        output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual,
                                                rms_norm_weight, epsilon)
        out0 = output[0]
        out1 = output[2]
        out0 = out0 + bias
        quantized_output = torch.ops.npu.npu_quantize(out0, scale, offset,
                                                      torch.qint8, -1, False)
        return quantized_output, out1

    def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor,
                    rms_norm_weight: torch.Tensor, scale: torch.Tensor,
                    offset: torch.Tensor, bias: torch.Tensor):
        """
        Replacement for AddRMSNormQuantWithBias fusion.
        """
        output = torch.ops.npu.npu_add_rms_norm_quant(
            rms_norm_input,
            residual,
            rms_norm_weight,
            # The inverse of scale is required by npu_add_rms_norm_quant kernel which is opposite to the npu_quantize kernel.
            1. / scale,
            offset,
            epsilon=epsilon,
            beta=bias)
        quantized_output = output[0]
        out1 = output[2]
        return quantized_output, out1

    def get_inputs():
        """
        Generate example inputs for the AddRMSNormQuantWithBias fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu")
        residual = torch.randn(2, 4, device="npu")
        rms_norm_weight = torch.randn(4, device="npu")
        rmsnorm_bias = torch.randn(4, device="npu")
        scale = torch.ones(4, device="npu")
        offset = torch.zeros(4, device="npu")
        return [
            rms_norm_input, residual, rms_norm_weight, scale, offset,
            rmsnorm_bias
        ]

    import torchair

    torchair.register_replacement(search_fn=pattern,
                                  replace_fn=replacement,
                                  example_inputs=get_inputs(),
                                  extra_check=_extra_stream_scope_check)


# The replacement registered here will be actually executed after AOT.
def replacement_add_rms_norm_quant_sp_pattern(epsilon):

    def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor,
                rms_norm_weight: torch.Tensor, scale: torch.Tensor,
                offset: torch.Tensor):
        """
        Pattern for AddRMSNormQuantSPPattern fusion.
        """
        output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual,
                                                rms_norm_weight, epsilon)
        out0 = output[0]
        out1 = output[2]
        out0 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(out0, True)
        quantized_output = torch.ops.npu.npu_quantize(out0, scale, offset,
                                                      torch.qint8, -1, False)
        return quantized_output, out1

    def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor,
                    rms_norm_weight: torch.Tensor, scale: torch.Tensor,
                    offset: torch.Tensor):
        """
        Replacement for the AddRMSNormQuantSPPattern fusion.
        """
        output = torch.ops.npu.npu_add_rms_norm_quant(
            rms_norm_input,
            residual,
            rms_norm_weight,
            # The inverse of scale is required by npu_add_rms_norm_quant kernel which is opposite to the npu_quantize kernel.
            1. / scale,
            offset,
            epsilon=epsilon)
        quantized_output = output[0]
        out1 = output[2]
        quantized_output = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            quantized_output, True)
        return quantized_output, out1

    def get_inputs():
        """
        Generate example inputs for the AddRMSNormQuantSPPattern fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu")
        residual = torch.randn(2, 4, device="npu")
        rms_norm_weight = torch.randn(4, device="npu")
        scale = torch.ones(4, device="npu")
        offset = torch.zeros(4, device="npu")
        return [rms_norm_input, residual, rms_norm_weight, scale, offset]

    import torchair

    torchair.register_replacement(search_fn=pattern,
                                  replace_fn=replacement,
                                  example_inputs=get_inputs(),
                                  extra_check=_extra_stream_scope_check)


# The replacement registered here will be actually executed after AOT.
def replacement_add_rms_norm_quant_sp_pattern_with_bias(epsilon):

    def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor,
                rms_norm_weight: torch.Tensor, scale: torch.Tensor,
                offset: torch.Tensor, bias: torch.Tensor):
        """
        Pattern for AddRMSNormQuantSPPatternWithBias fusion.
        """
        output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual,
                                                rms_norm_weight, epsilon)
        out0 = output[0]
        out1 = output[2]
        out0 = out0 + bias
        out0 = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(out0, True)
        quantized_output = torch.ops.npu.npu_quantize(out0, scale, offset,
                                                      torch.qint8, -1, False)
        return quantized_output, out1

    def replacement(rms_norm_input: torch.Tensor, residual: torch.Tensor,
                    rms_norm_weight: torch.Tensor, scale: torch.Tensor,
                    offset: torch.Tensor, bias: torch.Tensor):
        """
        Replacement for the AddRMSNormQuantSPPatternWithBias fusion.
        """
        output = torch.ops.npu.npu_add_rms_norm_quant(
            rms_norm_input,
            residual,
            rms_norm_weight,
            # The inverse of scale is required by npu_add_rms_norm_quant kernel which is opposite to the npu_quantize kernel.
            1. / scale,
            offset,
            epsilon=epsilon,
            beta=bias)
        quantized_output = output[0]
        out1 = output[2]
        quantized_output = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            quantized_output, True)
        return quantized_output, out1

    def get_inputs():
        """
        Generate example inputs for the AddRMSNormQuantSPPatternWithBias fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu")
        residual = torch.randn(2, 4, device="npu")
        rms_norm_weight = torch.randn(4, device="npu")
        rmsnorm_bias = torch.randn(4, device="npu")
        scale = torch.ones(4, device="npu")
        offset = torch.zeros(4, device="npu")
        return [
            rms_norm_input, residual, rms_norm_weight, scale, offset,
            rmsnorm_bias
        ]

    import torchair

    torchair.register_replacement(search_fn=pattern,
                                  replace_fn=replacement,
                                  example_inputs=get_inputs(),
                                  extra_check=_extra_stream_scope_check)


# register converter for pass
common_epsilons = [1e-5, 1e-6]
for eps in common_epsilons:
    logger.info(
        f"Start register fusion pattern for AddRMSNormQuant with epsilons={eps}"
    )
    replacement_add_rms_norm_quant(eps)
    replacement_add_rms_norm_quant_with_bias(eps)
    replacement_add_rms_norm_quant_sp_pattern(eps)
    replacement_add_rms_norm_quant_sp_pattern_with_bias(eps)
