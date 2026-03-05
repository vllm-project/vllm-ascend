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
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import torch.fx as fx
from torch._inductor.decomposition import select_decomp_table
from vllm.compilation.passes.fx_utils import OpOverload
from vllm.config import get_current_vllm_config

from vllm_ascend.compilation.compiler_interface import compile_fx


class TestBackend:
    """
    A custom compilation backend for testing operator fusion passes.
    It applies the AddRMSNormQuantFusionPass during graph compilation and
    records the FX graph before and after the transformation.
    """

    def __init__(self, custom_passes: list[Any] | None = None):
        vllm_config = get_current_vllm_config()
        compile_config = vllm_config.compilation_config
        self.inductor_config = compile_config.inductor_compile_config
        self.inductor_config["graph_fusion_manager"] = self.post_pass
        self.custom_passes = custom_passes

        # Placeholders to store FX graphs for verification
        self.graph_pre_pass = None
        self.graph_post_pass = None

    def post_pass(self, graph: fx.Graph, runtime_shape: int | None = None) -> fx.Graph:
        """
        Apply custom graph transformation passes.
        """
        self.graph_pre_pass = deepcopy(graph)
        if self.custom_passes is not None:
            for pass_ in self.custom_passes:
                pass_(graph)
        self.graph_post_pass = deepcopy(graph)
        return graph

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: int | None = None,
        key: str | None = None,
    ) -> tuple[Callable | None, Any | None]:
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

    def __call__(self, gm: fx.GraphModule, example_inputs: list[Any] | None):
        """
        Make the backend callable by torch.compile().
        Returns a compiled executable function.
        """
        assert example_inputs is not None
        compiled_fn, _ = self.compile(
            gm,
            example_inputs,
            compiler_config={"graph_fusion_manager": self.post_pass},
            runtime_shape=None,
            key=None,
        )
        return compiled_fn

    def find_nodes_by_target(self, graph: fx.GraphModule, target: OpOverload) -> list[fx.Node]:
        """Helper to find all FX nodes that call a specific operator."""
        return [node for node in graph.graph.nodes if hasattr(node, "target") and node.target == target]

    def check_before_ops(self, ops: Sequence[OpOverload], fully_replaced: bool = True):
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
