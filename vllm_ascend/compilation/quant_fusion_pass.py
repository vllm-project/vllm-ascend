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

from typing import Callable, List, Tuple

import torch
from torch.fx.subgraph_rewriter import replace_pattern
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
  

class AddRMSNormQuantPattern:

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config

    def register(self, patterns: List[Tuple[Callable, Callable]]):

        def pattern(rms_norm_input, residual, rms_norm_weight, scale, offset):
            """
          Pattern for AddRMSNormQuant fusion.
          """
            output = torch.ops.npu.npu_add_rms_norm(rms_norm_input, residual, rms_norm_weight, 1e-6)
            out0 = output[0]
            out1 = output[2]
            quantized_output = torch.ops.npu.npu_quantize(
                out0, scale, offset, torch.qint8, -1, False)
            return quantized_output, out1

        def replace(rms_norm_input, residual, rms_norm_weight, scale, offset):
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

        patterns.append((pattern, replace))


class AscendQuantFusionPass(VllmInductorPass):
    """
    A pass for fusing AddRMSNorm and W8A8 quantization operations on Ascend.
    """

    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self.patterns: List[Tuple[Callable, Callable]] = []
        # Register the AddRMSNormQuant fusion pattern into the graph rewriter pattern list
        AddRMSNormQuantPattern(vllm_config).register(self.patterns)
               
    def __call__(self, graph: torch.fx.Graph):
        self.begin()
        for pattern, replace in self.patterns:
            replace_pattern(graph, pattern, replace)     
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