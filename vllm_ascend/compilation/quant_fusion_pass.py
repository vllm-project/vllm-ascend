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
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.vllm_inductor_pass import VllmInductorPass


class AddRMSNormQuantPattern:

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config

    def get_inputs(self):
        """
        Generate example inputs for the AddRMSNormQuant fusion pattern.
        """
        rms_norm_input = torch.randn(2, 4, device="npu")
        residual = torch.randn(2, 4, device="npu")
        rms_norm_weight = torch.randn(4, device="npu")
        scale = torch.tensor([1.0], device="npu")
        offset = torch.tensor([0.0], device="npu")
        return [rms_norm_input, residual, rms_norm_weight, scale, offset]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(rms_norm_input, residual, rms_norm_weight, scale, offset):
            """
          Pattern for AddRMSNormQuant fusion.
          """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual,
                                                    rms_norm_weight, 1e-6)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_quantize(
                out0, scale, offset, torch.qint8, -1, False)
            return quantized_output, out1

        def replacement(rms_norm_input, residual, rms_norm_weight, scale,
                        offset):
            """
          Replacement for the AddRMSNormQuant fusion.
          """
            output = torch.ops.npu.npu_add_rms_norm_quant(
                rms_norm_input,
                residual,
                rms_norm_weight,
                1. /
                scale,  # The inverse of scale is required by npu_add_rms_norm_quant kernel which is opposite to the npu_quantize kernel.
                offset,
                epsilon=1e-6)
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class AscendQuantFusionPass(VllmInductorPass):
    """
    A pass for fusing AddRMSNorm and W8A8 quantization operations on Ascend.
    """

    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rmsnorm_quant_fusion_pass")
        AddRMSNormQuantPattern(vllm_config).register(self.patterns)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        matched_count = self.patterns.apply(graph)
        self.end_and_log()

    def is_applicable(self, **kwargs):
        """
        Check if the pass is applicable for the current configuration.
        """
        arg_dtypes = kwargs.get("arg_dtypes", None)
        if arg_dtypes is None:
            return False
        # We assume the first tensor's dtype is the data type of this model, update this solution when there is
        # better solution.
        dtype = arg_dtypes[0] if isinstance(
            arg_dtypes, list) and len(arg_dtypes) > 0 else arg_dtypes
        # We found that the kernel npu_add_rms_norm_quant accept varying data format for different dtypes, therefore, we only
        # provide the solution on bfloat16 here.
        return dtype in (torch.bfloat16, )
