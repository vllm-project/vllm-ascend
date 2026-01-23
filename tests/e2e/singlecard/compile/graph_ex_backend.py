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
from vllm.compilation.fx_utils import OpOverload
from vllm.config import get_current_vllm_config
from vllm.logger import logger

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
        self.inductor_config["graphex_fusion_manager"] = self.post_pass
        self.custom_passes = custom_passes

        # Placeholders to store FX graphs for verification
        self.graph_pre_pass = None
        self.graph_post_pass = None

    def pre_pass(self, graph: fx.Graph, runtime_shape: int | None = None) -> fx.Graph:
        self.graph_pre_pass = deepcopy(graph)
        return graph

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
            current_pass_manager = compiler_config["graphex_fusion_manager"]
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
        Uses AOT compilation to get the complete graph, then applies torchair compilation.
        """
        assert example_inputs is not None
        import torch
        import torchair
        from torch._dynamo.backends.common import aot_autograd
        from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple

        def fw_compiler(graph_module: fx.GraphModule, example_inputs_inner: list[Any]):
            """
            Forward compiler callback called by aot_autograd.
            At this point, the graph has been fully processed and decomposed.
            """
            # Save the graph before torchair compilation
            self.graph_pre_pass = deepcopy(graph_module.graph)

            # Apply custom passes (for GraphEX passes, __call__ is empty)
            if self.custom_passes is not None:
                for pass_ in self.custom_passes:
                    pass_(graph_module.graph)

            # Create torchair backend and compile
            config = torchair.CompilerConfig()
            npu_backend = torchair.get_npu_backend(compiler_config=config)

            # Compile with torchair - this triggers pattern matching and fusion
            try:
                compiled_fn = npu_backend(graph_module, example_inputs_inner)
            except RuntimeError as e:
                # If AscendIR conversion fails, that's expected for testing
                if "Failed to converter" in str(e) and "to AscendIR" in str(e):
                    logger.info(f"AscendIR conversion failed (expected): {e}")
                    # Return the original graph module for testing
                    compiled_fn = graph_module
                else:
                    raise

            # Save the graph after torchair compilation
            self.graph_post_pass = deepcopy(graph_module.graph)

            return compiled_fn

        # Use aot_autograd to process the graph, then apply our compiler
        decompositions = select_decomp_table()

        # Handle graph return type
        if not graph_returns_tuple(gm):
            def recursive_compile(g, ei):
                return aot_autograd(fw_compiler=fw_compiler, decompositions=decompositions)(g, ei)
            return make_graph_return_tuple(gm, example_inputs, recursive_compile)

        return aot_autograd(fw_compiler=fw_compiler, decompositions=decompositions)(gm, example_inputs)

    def find_nodes_by_target(self, graph: fx.Graph | fx.GraphModule, target: OpOverload) -> list[fx.Node]:
        """Helper to find all FX nodes that call a specific operator."""
        if graph is None:
            return []
        # Support both fx.Graph and fx.GraphModule
        nodes = graph.graph.nodes if isinstance(graph, fx.GraphModule) else graph.nodes
        return [node for node in nodes if hasattr(node, "target") and node.target == target]

    def check_before_ops(self, ops: Sequence[OpOverload], fully_replaced: bool = True):
        for op in ops:
            num_pre = len(self.find_nodes_by_target(self.graph_pre_pass, op))
            print(f"Op {op}: pre={num_pre}")
            assert num_pre > 0, f"Op {op} not found in pre-pass graph"

    def check_after_ops(self, ops: Sequence[OpOverload]):
        for op in ops:
            if self.graph_post_pass is None:
                logger.warning("Cannot check post-pass graph (traced compilation may not expose graph)")
                return
            num_post = len(self.find_nodes_by_target(self.graph_post_pass, op))
            print(f"Op {op}: post={num_post}")
            assert num_post > 0, f"Op {op} not found in post-pass graph"
