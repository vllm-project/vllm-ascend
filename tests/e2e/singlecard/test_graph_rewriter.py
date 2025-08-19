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

import copy
import random
import re
from unittest.mock import patch
import pytest
import torch
import torch.nn as nn
import torch_npu
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from tests.ut.base import PytestBase
from tests.e2e.model_utils import check_outputs_equal
from vllm_ascend.compilation.graph_rewrite_pass_manager import \
    GraphRewritePassManager
from vllm_ascend.compilation.quant_fusion_pass import AscendQuantFusionPass
from vllm_ascend.quantization.w8a8 import quant_per_tensor

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
        x, _, residual = torch_npu.npu_add_rms_norm(hidden_states, x, self.weight,
                                                 self.eps)
        quantized_output = quant_per_tensor(x, self.quant_scale,
                                            self.quant_offset)
        return quantized_output, residual


class CustomizeCompilationInterface:

    def __init__(self, vllm_config, checking_fusion_pass: str = "torch.ops"):
        self.vllm_config = vllm_config
        self.graph_rewriter_manager = GraphRewritePassManager()
        self.graph_rewriter_manager.configure(vllm_config)
        self.checking_string_for_fusion_pass = checking_fusion_pass

    def string_checking_for_op_name(self, gm: torch.fx.GraphModule,
                                    regex_pattern) -> bool:
        match = regex_pattern.search(str(gm.graph))
        if match:
            return True
        return False

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        for pass_name, (op_names,
                        _) in self.checking_string_for_fusion_pass.items():
            assert self.string_checking_for_op_name(
                gm, op_names
            ), f"Expected to find {op_names} in the graph, but not found."
        kwargs = {"arg_dtypes": [torch.bfloat16]}
        gm = self.graph_rewriter_manager(gm, **kwargs)
        gm.recompile()
        for pass_name, (_, replace_op_names
                        ) in self.checking_string_for_fusion_pass.items():
            assert self.string_checking_for_op_name(
                gm, replace_op_names
            ), f"Expected to find {replace_op_names} in the graph after pass {pass_name}, but not found."
        return gm


class TestGraphRewriter(PytestBase):

    @pytest.mark.parametrize("num_tokens", NUM_TOKENS)
    @pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
    def test_quant_fusion_pass(
        num_tokens: int,
        hidden_size: int,
    ) -> None:
        fusion_pattern_regex = re.compile(
            r".*: \[num_users=2\] = call_function\[target=torch\.ops\.npu\.npu_add_rms_norm\]\(.*\)"
            r".*: \[num_users=1\] = call_function\[target=operator\.getitem\]\(.*\)"
            r".*: \[num_users=1\] = call_function\[target=operator\.getitem\]\(.*\)"
            r".*: \[num_users=1\] = call_function\[target=torch\.ops\.npu\.npu_quantize\]\(.*\)",
            re.DOTALL)

        fusion_replace_regex = re.compile(
            r".*: \[num_users=2\] = call_function\[target=torch\.ops\.npu\.npu_add_rms_norm_quant\]\(.*\)"
            r".*: \[num_users=1\] = call_function\[target=operator\.getitem\]\(.*\)"
            r".*: \[num_users=1\] = call_function\[target=operator\.getitem\]\(.*\)",
            re.DOTALL)

        checking_string_for_fusion_pass = {
            "quant_fusion_pass": (fusion_pattern_regex, fusion_replace_regex)
        }
        # Create a random input tensor
        input_tensor = torch.randn(num_tokens,
                                hidden_size).to("npu").to(torch.bfloat16)
        vllm_config = VllmConfig()
        # Open the compilation fusion config and enable the graph rewriter on quantization
        config = {
            "ascend_compilation_config": {
                "enable_graph_rewriter": True,
                "fx_graph_eager": True,
                "enable_quantization_fusion": True
            }
        }
        vllm_config.additional_config.update(config)

        # 1. Checking if the pass is added to the pass manager when related config is enabled
        compilation_interface = CustomizeCompilationInterface(
            vllm_config, checking_fusion_pass=checking_string_for_fusion_pass)
        quant_fusion_pass_found = False
        for pass_ in compilation_interface.graph_rewriter_manager.passes:
            if isinstance(pass_, AscendQuantFusionPass):
                quant_fusion_pass_found = True
                break
        assert quant_fusion_pass_found, "AscendQuantFusionPass not found in the pass manager"

        # 2, Check if the pass is applied correctly，the checking process happens in the `__call__` method of `CustomizeCompilationInterface`
        # Initialize the model with RMSNorm quantization
        model = ModelWithRMSNormQuant(hidden_size=hidden_size).to("npu").to(
            torch.bfloat16)
        new_model = copy.deepcopy(model)
        compiled_model = torch.compile(model, backend=compilation_interface)
        for i in range(3):
            output = compiled_model(input_tensor)

        # 3. Check if the output is as expected, we use the original model to get the reference output
        reference_output = new_model(input_tensor)
        compiled_output = compiled_model(input_tensor)
        assert torch.allclose(reference_output[0],
                            compiled_output[0], rtol=1, atol=1), "Outputs do not match"

        print("Test passed successfully!")

