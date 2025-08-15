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
from vllm_ascend.quantization.w8a8 import quant_per_tensor

class ModelWithRMSNormQuant(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, quant_config=None, prefix=""):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.quant_config = quant_config
        self.prefix = prefix
        self.former_linear = nn.Linear(hidden_size, hidden_size)  # float
        self.post_linear = nn.Linear(hidden_size, hidden_size, dtype=torch.int8)    # quantized
        self.deq_scale = 0.7
        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.quant_scale = 0.83
        self.quant_offset = 3

    def forward(self, x):
        hidden_states = self.former_linear(x)
        x, residual = torch_npu.npu_add_rms_norm(hidden_states, x, self.weight, self.eps)
        quantized_output = quant_per_tensor(x, self.quant_scale, self.quant_offset)
        return quantized_output, residual


def custom_graph_rewriter_backend(gm: torch.fx.GraphModule, example_inputs):
    from torch.fx.subgraph_rewriter import replace_pattern
    print("before fusion graph:", gm.graph)
    def pattern(npu_quant_matmul, output_parallel, rms_norm_weight, scale, offset):
        output = torch.ops.npu_add_rms_norm(npu_quant_matmul, output_parallel, rms_norm_weight, 1e-6)
        out0 = output[0]
        out1 = output[2]
        new_out = torch.ops.npu.npu_quantize(out0, scale, offset, torch.qint8, -1, False)
        return new_out, out1
    
    def replace(npu_quant_matmul, output_parallel, rms_norm_weight, scale, offset):
        output = torch.ops.npu.npu_add_rms_norm_quantize(npu_quant_matmul, output_parallel, rms_norm_weight, scale, offset, epsilon=1e-6)
        return output[0], output[2]

    replace_pattern(gm, pattern, replace)
    gm.recompile()
    print("after fusion graph:", gm.graph)
    return gm

def test_graph_rewriter():
    # Create a random input tensor
    num_tokens = 20
    hidden_size = 4096
    input_tensor = torch.randn(num_tokens, hidden_size)

    # Initialize the model with RMSNorm quantization
    model = ModelWithRMSNormQuant(hidden_size=hidden_size)
    new_model = copy.deepcopy(model)
    compiled_model = torch.compile(model, backend=custom_graph_rewriter_backend)
    for i in range(3):
        output = compiled_model(input_tensor)
    # Check if the output is as expected
    reference_output = model(input_tensor)
    compiled_output = compiled_model(input_tensor)
    assert torch.allclose(reference_output[0], compiled_output[0]), "Outputs do not match"

    print("Test passed successfully!")
