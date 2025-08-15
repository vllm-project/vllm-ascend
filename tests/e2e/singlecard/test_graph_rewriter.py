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

import pytest
import torch
import torch.nn as nn
import torch_npu
import random
import copy
from vllm.config import VllmConfig
from vllm import LLM, SamplingParams
from vllm_ascend.compilation.quant_fusion_pass import AscendQuantFusionPass
from vllm_ascend.compilation.graph_rewrite_pass_manager import GraphRewritePassManager
from vllm_ascend.quantization.w8a8 import quant_per_tensor
from tests.e2e.model_utils import check_outputs_equal

NUM_TOKENS = [4, 32, 57]
HIDDEN_SIZES = [128, 512, 1024, 2048, 4096]
MODELS = ["Qwen/Qwen3-30B-A3B"]

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
    def __init__(self, vllm_config, checking_fusion_pass: str = "torch.ops"):
        self.vllm_config = vllm_config
        self.graph_rewriter_manager = GraphRewritePassManager()
        self.graph_rewriter_manager.configure(vllm_config)
        self.checking_string_for_fusion_pass = checking_fusion_pass

    def string_checking_for_op_name(self, gm: torch.fx.GraphModule, op_names: list[str]) -> bool:
        for op_name in op_names:
            if not any(op_name in node.target for node in gm.graph.nodes):
                return False
        return True

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        for pass_name, (op_names, _) in self.checking_string_for_fusion_pass.items():
            assert self.string_checking_for_op_name(gm, op_names), f"Expected to find {op_names} in the graph, but not found."
        gm = self.graph_rewriter_manager(gm)
        gm.recompile()
        for pass_name, (_, replace_op_names) in self.checking_string_for_fusion_pass.items():
            assert self.string_checking_for_op_name(gm, replace_op_names), f"Expected to find {replace_op_names} in the graph after pass {pass_name}, but not found."
        return gm

@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
def test_quant_fusion_pass(
    num_tokens: int,
    hidden_size: int,
) -> None:
    checking_string_for_fusion_pass = {
        "quant_fusion_pass": (["torch.ops.npu.npu_add_rms_norm", "torch.ops.npu_quantize"],["torch.ops.npu.npu_add_rms_norm_quant"])
    }
    # Create a random input tensor
    input_tensor = torch.randn(num_tokens, hidden_size)
    vllm_config = VllmConfig()
    # Open the compilation fusion config and enable the graph rewriter on quantization
    vllm_config.additional_config.ascend_compilation_config.enable_graph_rewriter = True
    vllm_config.additional_config.ascend_compilation_config.enable_quantization_fusion = True

    # 1. Checking if the pass is added to the pass manager when related config is enabled
    compilation_interface = CustomizeCompilationInterface(vllm_config, checking_fusion_pass=checking_string_for_fusion_pass)
    quant_fusion_pass_found = False
    for pass_ in compilation_interface.graph_rewriter_manager.passes:
        if isinstance(pass_, AscendQuantFusionPass):
            quant_fusion_pass_found = True
            break
    assert quant_fusion_pass_found, "AscendQuantFusionPass not found in the pass manager"

    # 2, Check if the pass is applied correctly，the checking process happens in the `__call__` method of `CustomizeCompilationInterface`
    # Initialize the model with RMSNorm quantization
    model = ModelWithRMSNormQuant(hidden_size=hidden_size)
    new_model = copy.deepcopy(model)
    compiled_model = torch.compile(model, backend=compilation_interface)
    for i in range(3):
        output = compiled_model(input_tensor)

    # 3. Check if the output is as expected, we use the original model to get the reference output
    reference_output = model(input_tensor)
    compiled_output = compiled_model(input_tensor)
    assert torch.allclose(reference_output[0], compiled_output[0]), "Outputs do not match"

    print("Test passed successfully!")

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
def test_whole_model_with_quant_fusion_pass(
        model: str,
        max_tokens: int,
):
    prompts = [
        "Hello, my name is", "The president of the United States is",
        "The capital of France is", "The future of AI is"
    ]

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    # TODO: change to use vllmrunner when the registry of custom op is solved
    # while running pytest
    vllm_model = LLM(model,
                     max_model_len=1024,
                     additional_config={
                        'ascend_compilation_config': {
                            'enable_graph_rewriter': True,
                            'enable_quantization_fusion': True}
                        })
    
    vllm_aclgraph_outputs = vllm_model.generate(prompts, sampling_params)
    del vllm_model
    torch.npu.empty_cache()

    vllm_model = LLM(model, enforce_eager=True, max_model_len=1024)
    vllm_eager_outputs = vllm_model.generate(prompts, sampling_params)
    del vllm_model
    torch.npu.empty_cache()

    vllm_aclgraph_outputs_list = []
    for output in vllm_aclgraph_outputs:
        vllm_aclgraph_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    vllm_eager_outputs_list = []
    for output in vllm_eager_outputs:
        vllm_eager_outputs_list.append(
            (output.outputs[0].index, output.outputs[0].text))

    check_outputs_equal(
        outputs_0_lst=vllm_eager_outputs_list,
        outputs_1_lst=vllm_aclgraph_outputs_list,
        name_0="vllm_eager_outputs",
        name_1="vllm_aclgraph_outputs",
    )