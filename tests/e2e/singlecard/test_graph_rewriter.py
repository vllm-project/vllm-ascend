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
import torch.nn as nn
import torch_npu
import random
import copy
from vllm.config import VllmConfig
from vllm_ascend.compilation.quant_fusion_pass import AscendQuantFusionPass
from vllm_ascend.compilation.graph_rewrite_pass_manager import GraphRewritePassManager
from vllm_ascend.quantization.w8a8 import quant_per_tensor


class ModelWithRMSNormQuant(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, quant_config=None, prefix=""):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.quant_config = quant_config
        self.prefix = prefix
        self.former_linear = nn.Linear(hidden_size, hidden_size)
        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.quant_scale = nn.Parameter(torch.Tensor(hidden_size))
        self.quant_offset = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, x):
        hidden_states = self.former_linear(x)
        x, residual = torch_npu.npu_add_rms_norm(hidden_states, x, self.weight, self.eps)
        quantized_output = quant_per_tensor(x, self.quant_scale, self.quant_offset)
        return quantized_output, residual



class CustomizeCompilationInterface:
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.graph_rewriter_manager = GraphRewritePassManager()
        self.graph_rewriter_manager.configure(vllm_config)

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        gm = self.graph_rewriter_manager(gm)
        gm.recompile()
        return gm


def test_fusion_pass(
    num_tokens: int = 20,
    hidden_size: int = 4096,
):
    # Create a random input tensor
    input_tensor = torch.randn(num_tokens, hidden_size)
    vllm_config = VllmConfig()
    # Open the compilation fusion config and enable the graph rewriter on quantization
    vllm_config.additional_config.ascend_compilation_config.enable_graph_rewriter = True
    vllm_config.additional_config.ascend_compilation_config.enable_quantization_fusion = True
    compilation_interface = CustomizeCompilationInterface(vllm_config)
    for pass_ in compilation_interface.graph_rewriter_manager.passes:
        

    # Initialize the model with RMSNorm quantization
    model = ModelWithRMSNormQuant(hidden_size=hidden_size)
    new_model = copy.deepcopy(model)
    compiled_model = torch.compile(model, backend=CustomizeCompilationInterface(vllm_config))
    for i in range(3):
        output = compiled_model(input_tensor)
    # Check if the output is as expected
    reference_output = model(input_tensor)
    compiled_output = compiled_model(input_tensor)
    assert torch.allclose(reference_output[0], compiled_output[0]), "Outputs do not match"

    print("Test passed successfully!")
