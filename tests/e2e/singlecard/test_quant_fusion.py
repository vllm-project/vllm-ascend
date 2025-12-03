#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch_npu
import vllm.config
from torch._inductor.decomposition import select_decomp_table
from vllm.compilation.fx_utils import OpOverload
from vllm.config import ModelConfig, VllmConfig, get_current_vllm_config

from vllm_ascend.compilation.compiler_interface import compile_fx
from vllm_ascend.compilation.passes.quant_fusion_pass import \
    AddRMSNormQuantFusionPass


class TestModel(nn.Module):
    """
    A minimal test model that simulates the pattern:
        AddRMSNorm → Quantization
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, device="npu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.rms_norm_weight = nn.Parameter(
            torch.randn(hidden_size, device=device))
        self.quant_scale = torch.tensor([1.0], device=device)
        self.quant_offset = torch.tensor([0.0], device=device)

    def forward(self, x):
        """
        Forward pass:
          1. Perform npu_add_rms_norm
          2. Quantize the normalized output to int8
        Returns both quantized output and updated residual.
        """
        residual = torch.zeros_like(x)

        norm_output, _, new_residual = torch_npu.npu_add_rms_norm(
            x, residual, self.rms_norm_weight, self.eps)

        quantized_output = torch_npu.npu_quantize(norm_output,
                                                  self.quant_scale,
                                                  self.quant_offset,
                                                  torch.qint8, -1, False)

        return quantized_output, new_residual

    def ops_in_model_before(self) -> List[OpOverload]:
        """Return the list of expected operators BEFORE fusion."""
        return [
            torch.ops.npu.npu_add_rms_norm.default,
            torch.ops.npu.npu_quantize.default
        ]

    def ops_in_model_after(self) -> List[OpOverload]:
        """Return the list of expected operators AFTER successful fusion."""
        return [torch.ops.npu.npu_add_rms_norm_quant.default]


class TestBackend:
    """
    A custom compilation backend for testing operator fusion passes.
    It applies the AddRMSNormQuantFusionPass during graph compilation and
    records the FX graph before and after the transformation.
    """

    def __init__(self):
        vllm_config = get_current_vllm_config()
        compile_config = vllm_config.compilation_config
        self.custom_passes = [
            AddRMSNormQuantFusionPass(vllm_config=vllm_config)
        ]
        self.inductor_config = compile_config.inductor_compile_config
        self.inductor_config["graph_fusion_manager"] = self.post_pass

        # Placeholders to store FX graphs for verification
        self.graph_pre_pass = None
        self.graph_post_pass = None

    def post_pass(self,
                  graph: fx.Graph,
                  runtime_shape: int | None = None) -> fx.Graph:
        """
        Apply custom graph transformation passes.
        """
        self.graph_pre_pass = deepcopy(graph)
        for pass_ in self.custom_passes:
            pass_(graph)
        self.graph_post_pass = deepcopy(graph)
        return graph

    def compile(
            self,
            graph: fx.GraphModule,
            example_inputs: list[Any],
            compiler_config: dict[str, Any],
            runtime_shape: Optional[int] = None,
            key: Optional[str] = None
    ) -> tuple[Optional[Callable], Optional[Any]]:
        """
        Compile the FX graph using vLLM's Ascend compiler interface.
        Wraps the post-pass logic into the inner_compile callback.
        """

        def compile_inner(graph, example_inputs):
            current_pass_manager = compiler_config["graph_fusion_manager"]
            return current_pass_manager(graph, runtime_shape)

        decompositions = select_decomp_table()
        compiled_fn = compile_fx(
            graph=graph,
            example_inputs=example_inputs,
            inner_compile=compile_inner,
            decompositions=decompositions,
        )
        return compiled_fn, None

    def __call__(self, gm: fx.GraphModule, example_inputs: List[Any]):
        """
        Make the backend callable by torch.compile().
        Returns a compiled executable function.
        """
        compiled_fn, _ = self.compile(
            gm,
            example_inputs,
            compiler_config={"graph_fusion_manager": self.post_pass},
            runtime_shape=None,
            key=None,
        )
        return compiled_fn

    def find_nodes_by_target(self, graph: fx.GraphModule,
                             target: OpOverload) -> List[fx.Node]:
        """Helper to find all FX nodes that call a specific operator."""
        return [
            node for node in graph.graph.nodes
            if hasattr(node, 'target') and node.target == target
        ]

    def check_before_ops(self,
                         ops: Sequence[OpOverload],
                         fully_replaced: bool = True):
        """
        Verify that the original (unfused) operators exist before the pass
        and are fully removed afterward (if fully_replaced=True).
        """
        for op in ops:
            num_pre = len(self.find_nodes_by_target(self.graph_pre_pass, op))
            num_post = len(self.find_nodes_by_target(self.graph_post_pass, op))
            print(f"Op {op}: pre={num_pre}, post={num_post}")

            assert num_pre > 0, f"Op {op} not found in pre-pass graph"
            if fully_replaced:
                assert num_post == 0, f"Unexpected op {op} in post-pass graph: {num_post} nodes remain"

    def check_after_ops(self, ops: Sequence[OpOverload]):
        """Verify that the fused operator appears in the transformed graph."""
        for op in ops:
            num_post = len(self.find_nodes_by_target(self.graph_post_pass, op))
            print(f"Op {op}: post={num_post}")
            assert num_post > 0, f"Op {op} not found in post-pass graph"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_rmsnorm_quant_fusion(dtype: torch.dtype, hidden_size: int,
                              num_tokens: int, eps: float):
    """
    End-to-end test for AddRMSNorm+Quantize fusion.
    Compares: Operator presence/absence before and after graph transformation
    """
    torch.set_default_dtype(dtype)
    torch.manual_seed(1)

    vllm_config = VllmConfig(model_config=ModelConfig(dtype=dtype))

    with vllm.config.set_current_vllm_config(vllm_config):
        backend = TestBackend()
        model = TestModel(hidden_size, eps, device="npu")
        model = model.to("npu")

        x = torch.rand(num_tokens,
                       hidden_size,
                       device="npu",
                       dtype=dtype,
                       requires_grad=False)

        result_unfused = model(x)
        print("Unfused result:", [t.shape for t in result_unfused])
        model_fused = torch.compile(model, backend=backend)
        result_fused = model_fused(x)
        print("Fused result:", [t.shape for t in result_fused])

        print("=== Checking operator fusion ===")
        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())
