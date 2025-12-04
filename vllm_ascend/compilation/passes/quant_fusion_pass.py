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
import logging

import torch
import torch._inductor.pattern_matcher as pm
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig


class AddRMSNormQuantPattern:

    def __init__(self, vllm_config: VllmConfig, eps: float = 1e-6):
        self.vllm_config = vllm_config
        self.eps = eps

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

        def pattern(rms_norm_input: torch.Tensor, residual: torch.Tensor,
                    rms_norm_weight: torch.Tensor, scale: torch.Tensor,
                    offset: torch.Tensor):
            """
            Pattern for AddRMSNormQuant fusion.
            """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual,
                                                    rms_norm_weight, self.eps)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_quantize(
                out0, scale, offset, torch.qint8, -1, False)
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
                1. /
                scale,  # The inverse of scale is required by npu_add_rms_norm_quant kernel which is opposite to the npu_quantize kernel.
                offset,
                epsilon=self.eps)
            quantized_output = output[0]
            out1 = output[2]
            return quantized_output, out1

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class AddRMSNormQuantFusionPass(VllmInductorPass):
    """
    A pass for fusing AddRMSNorm and W8A8 quantization operations on Ascend.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="rmsnorm_quant_fusion_pass")

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logging.info("Quant fusion not enabled: unsupported dtype %s",
                         dtype)
            return

        common_epsilons = [1e-5, 1e-6]
        for eps in common_epsilons:
            AddRMSNormQuantPattern(vllm_config,
                                   eps=eps).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logging.debug("Replaced %s patterns", self.matched_count)
        self.end_and_log()

    def is_applicable(self, runtime_shape: int | None = None) -> bool:
        """
        Check if the pass is applicable for the current configuration.
        """
        return True
